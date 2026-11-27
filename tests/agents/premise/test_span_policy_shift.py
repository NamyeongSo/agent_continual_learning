# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Regression tests for saved-action evaluation, without environment replay."""

from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import litellm
import pytest

from exgentic.agents.ace.prompts.generator import build_generator_system_prompt
from exgentic.agents.premise.pipeline import PremisePipeline, write_json
from exgentic.agents.premise.selection import select_official_score_beam, select_policy_shift_beam
from exgentic.agents.premise.span_policy_shift import (
    HISTORY_FILE,
    SpanPolicyShiftEvaluator,
    normalize_action,
    replace_policy,
    strict_f1,
)
from scripts.utils.policy_shift_metrics import score_actions


def call(name="lookup", **arguments):
    return {"name": name, "arguments": arguments}


def message(calls):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {
                    "name": item["name"],
                    "arguments": json.dumps(item["arguments"]),
                },
            }
            for index, item in enumerate(calls)
        ],
    }


def response(calls, *, finish="tool_calls", inputs=100, outputs=10):
    return litellm.ModelResponse(
        choices=[{"finish_reason": finish, "message": message(calls)}],
        usage={"prompt_tokens": inputs, "completion_tokens": outputs},
    )


def saved_source(tmp_path, *, success=True, count=4, name="source"):
    policy = build_generator_system_prompt("original playbook")
    records = []
    history = [{"role": "system", "content": policy}, {"role": "user", "content": "task and environment policy"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {"value": {"type": "integer"}}},
            },
        }
    ]
    for index in range(count):
        action_message = message([call(value=index)])
        records.append(
            {
                "action_index": index,
                "request": {
                    "messages": copy.deepcopy(history),
                    "tools": tools,
                    "tool_choice": "auto",
                },
                "message": action_message,
            }
        )
        history.extend([action_message, {"role": "tool", "tool_call_id": "call_0", "content": f"observed {index}"}])
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return {
        "benchmark": "appworld",
        "task_id": name,
        "success": success,
        "score": 0.75,
        "actions": [{"action": call(value=i)} for i in range(count)],
        "policy_shift_history": str(path),
        "actor_policy": policy,
    }, records


def evaluator(tmp_path, **kwargs):
    return SpanPolicyShiftEvaluator(model="openai/test", base_url="http://localhost/v1", output=tmp_path, **kwargs)


def test_matched_probe_preserves_actual_recorded_generation_settings(tmp_path, monkeypatch):
    source, records = saved_source(tmp_path, count=1)
    settings = {"model": "openai/test", "temperature": 1.0, "max_tokens": 2048,
                "top_p": 0.87, "reasoning_effort": "low",
                "extra_body": {"top_k": 17, "chat_template_kwargs": {"enable_thinking": False}}}
    records[0]["generation_settings"] = settings
    Path(source["policy_shift_history"]).write_text(json.dumps(records[0]) + "\n")
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return response([call(value=0)])

    monkeypatch.setattr(litellm, "completion", complete)
    result = evaluator(tmp_path, temperature=1, match_source_settings=True).evaluate(
        {"playbook": {"playbook": "candidate"}}, source, {"span": {"start": 0, "end": 1}}, "matched",
    )
    assert result["strict_f1"] == 1 and len(calls) == 1
    for name, value in settings.items():
        assert calls[0][name] == value
    assert calls[0]["messages"][1:] == records[0]["request"]["messages"][1:]
    assert calls[0]["tools"] == records[0]["request"]["tools"]


@pytest.mark.parametrize("change", [{"temperature": 0}, {"max_tokens": 1024}, {"model": "openai/other"}])
def test_matched_probe_rejects_sampler_changes_before_inference(tmp_path, monkeypatch, change):
    _source, records = saved_source(tmp_path, count=1)
    request = records[0]["request"]
    request["generation_settings"] = {"model": "openai/test", "temperature": 1, "max_tokens": 2048, **change}
    monkeypatch.setattr(litellm, "completion", lambda **kw: pytest.fail("Mismatched settings must not execute"))
    with pytest.raises(ValueError, match="must match"):
        evaluator(tmp_path, temperature=1, match_source_settings=True)._complete(
            request, benchmark="appworld", action_index=0, log_path=tmp_path / "unused.jsonl",
        )


def test_unlimited_probe_uses_live_reasoning_sampler_and_accepts_more_than_2048_output_tokens(tmp_path, monkeypatch):
    _source, records = saved_source(tmp_path, count=1)
    request = records[0]["request"]
    request["generation_settings"] = {
        "model": "openai/test", "temperature": 1, "max_tokens": None, "top_p": 0.95,
        "reasoning_effort": "xhigh", "presence_penalty": 0,
        "extra_body": {"top_k": 20, "min_p": 0, "repetition_penalty": 1,
                       "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}},
    }
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return response([call(value=0)], outputs=5000)

    monkeypatch.setattr(litellm, "completion", complete)
    message, usage = evaluator(tmp_path, temperature=1, max_tokens=None, match_source_settings=True)._complete(
        request, benchmark="appworld", action_index=0, log_path=tmp_path / "unlimited.jsonl",
    )
    assert message["tool_calls"] and usage[0]["completion_tokens"] == 5000
    assert "max_tokens" not in calls[0] and "max_completion_tokens" not in calls[0]
    assert calls[0]["reasoning_effort"] == "xhigh" and calls[0]["top_p"] == 0.95
    assert calls[0]["extra_body"] == request["generation_settings"]["extra_body"]


@pytest.mark.parametrize("past_count", [3, 6, 9])
def test_past_count_config_supports_nine_and_preserves_previous_choices(tmp_path, past_count):
    manifest = tmp_path / "split.json"
    write_json(manifest, {"bfcl_42": {"train": ["one"]}})
    pipeline = PremisePipeline(
        output=tmp_path / "run",
        manifest=manifest,
        benchmarks=["bfcl"],
        mode="isolated",
        seed=42,
        count=None,
        past_count=past_count,
        model="openai/test",
        base_url="http://localhost/v1",
        candidate_evaluation="span_policy_shift",
    )
    assert pipeline.past_count == past_count
    assert json.loads((pipeline.output / "state.json").read_text())["config"]["past_count"] == past_count


@pytest.mark.parametrize(
    "predicted,reference",
    [
        ([call("a", v=1), call("b")], [call("a", v=1), call("b"), call("a", v=1)]),
        ([call("a", v=2)], [call("a", v=1)]),
        ([call(a=1, b=2)], [call(b=2, a=1)]),
        ([], []),
        ([], [call()]),
    ],
)
def test_strict_f1_is_identical_to_existing_cp_metric(predicted, reference):
    assert strict_f1(predicted, reference) == pytest.approx(
        score_actions(predicted, reference, "appworld")["strict_f1"]
    )


@pytest.mark.parametrize("success", [True, False])
def test_four_action_span_uses_four_independent_original_prefixes_and_stored_actions(tmp_path, monkeypatch, success):
    source, records = saved_source(tmp_path, success=success, count=6)
    before = copy.deepcopy(records)
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        index = (len(kwargs["messages"]) - 2) // 2
        # The first two positions agree; the last two change arguments.
        return response([call(value=index if index <= 2 else 999)])

    monkeypatch.setattr(litellm, "completion", complete)
    result = evaluator(tmp_path).evaluate(
        {"playbook": {"playbook": "candidate playbook"}},
        source,
        {"span": {"start": 1, "end": 5, "type": "failure" if success else "success"}},
        "candidate",
    )
    assert len(calls) == 4  # No original-policy inference and no span-end continuation.
    assert result["strict_f1"] == 0.5
    assert result["evaluation_score"] == 0.5
    assert result["environment_execution"] is False
    for kwargs in calls:
        offset = (len(kwargs["messages"]) - 2) // 2
        assert kwargs["messages"][0]["content"] == build_generator_system_prompt("candidate playbook")
        assert kwargs["messages"][1:] == before[offset]["request"]["messages"][1:]
        assert kwargs["tools"] == before[offset]["request"]["tools"]
        assert kwargs["max_tokens"] == 2048
        assert kwargs["temperature"] == 0
        assert kwargs["max_retries"] == kwargs["num_retries"] == 0
    assert records == before
    assert [row["reference"] for row in result["actions"]] == [[call(value=i)] for i in range(1, 5)]
    assert len((tmp_path / "policy_shift_calls/candidate.jsonl").read_text().splitlines()) == 4


def test_span_actions_overlap_and_out_of_order_completions_restore_original_position_order(tmp_path, monkeypatch):
    source, _ = saved_source(tmp_path, count=4)
    barrier = threading.Barrier(4)
    finished = [threading.Event() for _ in range(4)]
    completion_order = []

    def complete(**kwargs):
        index = (len(kwargs["messages"]) - 2) // 2
        barrier.wait(timeout=5)
        if index < 3:
            assert finished[index + 1].wait(timeout=5)
        completion_order.append(index)
        finished[index].set()
        return response([call(value=index)])

    monkeypatch.setattr(litellm, "completion", complete)
    runner = evaluator(tmp_path)
    result = runner.evaluate(
        {"playbook": {"playbook": "candidate"}},
        source,
        {"span": {"start": 0, "end": 4}},
        "parallel",
        max_workers=4,
    )
    assert completion_order == [3, 2, 1, 0]
    assert [action["action_index"] for action in result["actions"]] == [0, 1, 2, 3]
    assert result["strict_f1"] == 1
    assert runner.last_parallelism == {
        "queued_action_jobs": 4,
        "peak_action_workers": 4,
        "cancelled_action_jobs": 0,
    }
    records = [json.loads(line) for line in (tmp_path / "policy_shift_calls/parallel.jsonl").read_text().splitlines()]
    assert sorted(row["action_index"] for row in records) == [0, 1, 2, 3]


def test_pipeline_flattens_all_candidate_current_and_past_actions_into_one_bounded_queue(tmp_path, monkeypatch):
    manifest = tmp_path / "split.json"
    write_json(manifest, {"appworld_42": {"train": ["one"]}})
    pipeline = PremisePipeline(
        output=tmp_path / "run",
        manifest=manifest,
        benchmarks=["appworld"],
        mode="isolated",
        seed=42,
        count=None,
        past_count=3,
        model="openai/test",
        base_url="http://localhost/v1",
        candidate_evaluation="span_policy_shift",
        replay_workers=8,
        llm_calls=4,
    )
    current, _ = saved_source(tmp_path, success=True, count=4, name="current")
    past, _ = saved_source(tmp_path, success=False, count=4, name="past")
    reflection = {"span": {"start": 0, "end": 4}}
    candidates = [
        {
            "id": f"candidate_{i}",
            "valid": True,
            "parent_rank": 1 if i <= 2 else 2,
            "playbook": {"playbook": f"candidate {i}"},
            "reflection": reflection,
        }
        for i in range(1, 5)
    ]
    pipeline._score_job = lambda *_a, **_kw: pytest.fail("No per-span executor or environment replay")
    barrier, lock = threading.Barrier(4), threading.Lock()
    active = peak = requests = 0

    def complete(**kwargs):
        nonlocal active, peak, requests
        with lock:
            active += 1
            peak = max(peak, active)
            requests += 1
        try:
            barrier.wait(timeout=5)
            index = (len(kwargs["messages"]) - 2) // 2
            return response([call(value=index)])
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(litellm, "completion", complete)
    pipeline._score_candidates(1, candidates, [current, current], [{"trace": past, "reflection": reflection}])
    assert requests == 4 * 2 * 4
    assert peak == 4  # One shared --llm-calls limit, below the eight action workers.
    stats = pipeline.policy_shift_evaluator.last_parallelism
    assert stats["queued_action_jobs"] == 32
    assert 4 <= stats["peak_action_workers"] <= 8
    assert stats["cancelled_action_jobs"] == 0
    assert all(
        candidate["current_evaluation_score"] == 1 and candidate["past_evaluation_score"] == 0
        for candidate in candidates
    )
    logs = list((pipeline.output / "policy_shift_calls").glob("*.jsonl"))
    assert len(logs) == 8
    assert all(len(path.read_text().splitlines()) == 4 for path in logs)


def test_budget_failure_cancels_queued_span_actions_and_charges_in_flight_results(tmp_path, monkeypatch):
    from concurrent.futures import Future

    from exgentic.core.experiment_limits import ExperimentBudgetExceededError

    source, _ = saved_source(tmp_path, success=False, count=6, name="failed")
    other, _ = saved_source(tmp_path, success=True, count=1, name="other")
    runner = evaluator(tmp_path)
    started, lock = [], threading.Lock()
    barrier, cancelled = threading.Barrier(2), threading.Event()
    original_cancel = Future.cancel

    def cancel(future):
        result = original_cancel(future)
        if result:
            cancelled.set()
        return result

    def action(span, row):
        index = row["action_index"]
        with lock:
            started.append((span.key, index))
        if span.key == "failed":
            if index < 2:
                barrier.wait(timeout=5)
            if index == 0:
                failure = ExperimentBudgetExceededError("output_truncated", limit=2048, actual=2048, turn=1)
                failure.usage_records = [{"prompt_tokens": 10, "completion_tokens": 2048}]
                raise failure
            assert cancelled.wait(timeout=5)
        return {"action_index": index, "strict_f1": 1, "usage": [{"prompt_tokens": 10, "completion_tokens": 5}]}

    monkeypatch.setattr(Future, "cancel", cancel)
    monkeypatch.setattr(runner, "_evaluate_action", action)
    candidate = {"playbook": {"playbook": "candidate"}}
    outcomes = runner.evaluate_batch(
        [
            ("failed", candidate, source, {"span": {"start": 0, "end": 6}}, "failed"),
            ("other", candidate, other, {"span": {"start": 0, "end": 1}}, "other"),
        ],
        max_workers=2,
    )
    failed = outcomes["failed"]
    assert failed["evaluation_score"] == 0
    assert failed["status"] == "budget_exceeded"
    assert failed["action_generations"] == sum(key == "failed" for key, _ in started)
    assert len(failed["actions"]) + len(failed["failed_action_usage"]) == failed["action_generations"]
    assert runner.last_parallelism["cancelled_action_jobs"] > 0
    assert outcomes["other"]["evaluation_score"] == 1


@pytest.mark.parametrize("success,expected", [(True, 1.0), (False, 0.0)])
def test_direction_uses_original_official_success_not_span_type_or_partial_score(
    tmp_path, monkeypatch, success, expected
):
    source, _ = saved_source(tmp_path, success=success, count=1)
    monkeypatch.setattr(litellm, "completion", lambda **_kwargs: response([call(value=0)]))
    reflection = {"span": {"start": 0, "end": 1, "type": "failure" if success else "success"}}
    result = evaluator(tmp_path).evaluate({"playbook": {"playbook": "candidate"}}, source, reflection, "c")
    assert result["evaluation_score"] == expected
    source.pop("success")
    with pytest.raises(ValueError, match="official trajectory success"):
        evaluator(tmp_path).evaluate({}, source, {}, "missing")


@pytest.mark.parametrize("failure", ["truncated", "input", "output", "turns"])
def test_actor_budget_failure_cannot_earn_shift_credit_or_retry(tmp_path, monkeypatch, failure):
    source, _ = saved_source(tmp_path, success=False, count=41 if failure == "turns" else 1)
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return response(
            [call(value=999)],
            finish="length" if failure == "truncated" else "tool_calls",
            inputs=100001 if failure == "input" else 100,
            outputs=2049 if failure == "output" else 2048,
        )

    monkeypatch.setattr(litellm, "completion", complete)
    start = 40 if failure == "turns" else 0
    result = evaluator(tmp_path, num_retries=5).evaluate(
        {"playbook": {"playbook": "candidate"}},
        source,
        {"span": {"start": start, "end": start + 1}},
        "budget",
    )
    assert result["evaluation_score"] == 0
    assert result["success"] is False
    assert result["status"] == "budget_exceeded"
    assert len(calls) == (0 if failure == "turns" else 1)
    assert result["action_generations"] == len(calls)
    assert len(result["failed_action_usage"]) == len(calls)


def test_transport_retries_are_finite_logged_and_do_not_regenerate_reference(tmp_path, monkeypatch):
    source, _ = saved_source(tmp_path, count=1)
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("temporary connection failure")
        return response([call(value=0)])

    monkeypatch.setattr(litellm, "completion", complete)
    result = evaluator(tmp_path, num_retries=1).evaluate(
        {"playbook": {"playbook": "candidate"}},
        source,
        {"span": {"start": 0, "end": 1}},
        "retry",
    )
    assert len(calls) == 2
    assert result["strict_f1"] == 1
    records = [json.loads(line) for line in (tmp_path / "policy_shift_calls/retry.jsonl").read_text().splitlines()]
    assert records[0]["error_type"] == "RuntimeError"
    assert records[1]["attempt"] == 2
    assert result["actions"][0]["usage"][0]["prompt_tokens"] == 100


def test_invalid_output_is_not_rewarded_as_novelty(tmp_path, monkeypatch):
    source, _ = saved_source(tmp_path, success=False, count=1)
    invalid = response([call()])
    invalid.choices[0].message.tool_calls[0].function.arguments = "malformed JSON"
    monkeypatch.setattr(litellm, "completion", lambda **_kwargs: invalid)
    with pytest.raises(ValueError):
        evaluator(tmp_path).evaluate(
            {"playbook": {"playbook": "candidate"}}, source, {"span": {"start": 0, "end": 1}}, "invalid"
        )


def test_policy_replacement_preserves_multimodal_content_environment_policy_and_quoted_history():
    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "task OLD policy"},
                    {"type": "image_url", "image_url": {"url": "data:image/test"}},
                ],
            },
            {"role": "assistant", "content": "quoted OLD"},
        ]
    }
    changed = replace_policy(request, "OLD", "NEW")
    assert changed["messages"][0]["content"][0]["text"] == "task NEW policy"
    assert changed["messages"][0]["content"][1] == request["messages"][0]["content"][1]
    assert changed["messages"][1:] == request["messages"][1:]


def test_native_strict_f1_uses_exact_commands_duration_and_completion_not_analysis():
    parser = SimpleNamespace(
        parse_response=lambda text: SimpleNamespace(
            error="",
            commands=[SimpleNamespace(keystrokes="pwd\n", duration=0.1)],
            is_task_complete=text == "done",
        )
    )
    reference = normalize_action({"content": "analysis one"}, "terminalbench2", parser)
    same = normalize_action({"content": "analysis two"}, "terminalbench2", parser)
    complete = normalize_action({"content": "done"}, "terminalbench2", parser)
    assert strict_f1(same, reference) == 1
    assert strict_f1(complete, reference) == pytest.approx(2 / 3)
    assert strict_f1([call("terminal_command", keystrokes="pwd\n", duration=1)], reference) == 0


def test_policy_shift_pareto_retains_all_current_past_fronts_and_tie_breakers():
    candidates = [
        {"id": f"candidate_{i}", "valid": True, "current_evaluation_score": now, "past_evaluation_score": past}
        for i, now, past in [(1, 1, 0), (2, 0.7, 0.7), (3, 0, 1), (4, 0.6, 0.6)]
    ]
    selected, details = select_policy_shift_beam(candidates)
    official = [
        {
            **c,
            "current_official_score": c["current_evaluation_score"],
            "past_official_score": c["past_evaluation_score"],
        }
        for c in candidates
    ]
    assert [c["id"] for c in selected] == [c["id"] for c in select_official_score_beam(official)[0]]
    assert details["performance_fronts"] == [["candidate_1", "candidate_2", "candidate_3"], ["candidate_4"]]
    assert details["criterion"] == "span_policy_shift_current_past"


@pytest.mark.parametrize("retrieval_mode", ["none", "a", "b", "ab", "random"])
def test_two_task_pipeline_evaluates_only_actions_and_keeps_original_memory_on_resume(
    tmp_path, monkeypatch, retrieval_mode,
):
    benchmark = "appworld" if retrieval_mode == "none" else "bfcl"
    monkeypatch.setattr("exgentic.agents.premise.pipeline.MiniLMClient", lambda *_args: SimpleNamespace(
        identity={"model": "test MiniLM"}, encode=lambda texts: [
            {"vector": [1.0, 0.0], "input_wordpieces": 5, "truncated": False} for _ in texts]))
    manifest = tmp_path / "split.json"
    write_json(manifest, {f"{benchmark}_42": {"train": ["one", "two"]}})
    options = {
        "output": tmp_path / "run",
        "manifest": manifest,
        "benchmarks": [benchmark],
        "mode": "isolated",
        "seed": 42,
        "count": None,
        "past_count": 3,
        "model": "openai/test",
        "base_url": "http://localhost/v1",
        "proposal_retrieval": retrieval_mode,
    }
    variant = {"candidate_evaluation": "span_policy_shift", "playbook_edit_mode": "add_update_delete"}
    pipeline = PremisePipeline(**options, **variant)
    parent_calls = []

    def parent(**kwargs):
        assert "replay_source" not in kwargs
        parent_calls.append(kwargs)
        trace, _ = saved_source(
            tmp_path,
            success=kwargs["task_id"] == pipeline.order[0][1],
            count=1,
            name=kwargs["label"],
        )
        trace["task_id"] = kwargs["task_id"]
        trace["benchmark"] = benchmark
        trace["task"] = kwargs["task_id"]
        return trace

    pipeline._run_agent = parent
    pipeline.llm = SimpleNamespace(
        reflect=lambda *_a, **_kw: {"span": {"start": 0, "end": 1}, "entry_assessments": [],
                                   "analysis": "Check recorded action inputs.",
                                   "expected_behavior": "Use the requested input."},
        mutate=lambda *_, **_kwargs: {
            "operations": [
                {"type": "ADD", "section": "strategies_and_insights", "content": "Check inputs."},
            ]
        },
        judge=lambda *_: pytest.fail("No alignment judging"),
    )
    monkeypatch.setattr(litellm, "completion", lambda **_kwargs: response([call(value=0)]))
    first = pipeline.step()
    memory_before = copy.deepcopy(pipeline.states[benchmark]["memory"][0])
    second = pipeline.step()
    assert len(parent_calls) == 4
    assert len(second["past_sample"]) == 1
    assert first["candidates"][0]["current_evaluation_score"] == 1
    assert second["candidates"][0]["current_evaluation_score"] == 0
    assert second["candidates"][0]["past_evaluation_score"] == 1
    assert second["parallelism"]["replay_jobs"] == second["parallelism"]["peak_replay_sessions"] == 0
    assert second["parallelism"]["policy_shift_action_generations"] == 8
    if retrieval_mode != "none":
        assert [candidate["proposal_retrieval"]["selected"] for candidate in second["candidates"]] == [1] * 4
    assert all("current_official_score" not in c and "current_alignment" not in c for c in second["candidates"])
    memory_after = pipeline.states[benchmark]["memory"][0]
    assert memory_before["trace_artifact"] == memory_after["trace_artifact"]
    assert memory_before["reflection"] == memory_after["reflection"]
    restored = PremisePipeline(**options, **variant)
    assert restored.cursor == 2
    original = restored._memory_trace(memory_after)
    assert original["success"] is True
    assert original["actor_policy"] == build_generator_system_prompt("original playbook")
    for overrides in [{}, {"candidate_evaluation": "span_policy_shift", "policy_shift_temperature": 0.7}]:
        with pytest.raises(ValueError, match="configuration differs"):
            PremisePipeline(**options, playbook_edit_mode="add_update_delete", **overrides)


def test_recording_keeps_exact_acting_tools_and_ignores_shortlisting(tmp_path, monkeypatch):
    from exgentic.agents.litellm_tool_calling.instance import LiteLLMToolCallingAgentInstance
    from exgentic.agents.premise.instance import PremiseAgentInstance

    actor = object.__new__(PremiseAgentInstance)
    actor.record_policy_shift, actor.enable_thinking = True, False
    actor.messages = [{"role": "system", "content": "policy"}]
    actor.model_settings = SimpleNamespace(reasoning_effort=None)
    actor._paths = SimpleNamespace(agent_dir=tmp_path)
    actor._step_count = 1
    monkeypatch.setattr(LiteLLMToolCallingAgentInstance, "_completion", lambda *_a, **_kw: response([call(value=1)]))
    tools = [{"name": "exact shortlisted tool"}]
    actor._completion(messages=[{"role": "developer", "content": "select tools"}])
    assert not (tmp_path / HISTORY_FILE).exists()
    actor._completion(messages=actor.messages, tools=tools)
    actor.messages.append({"role": "user", "content": "later observation"})
    saved = json.loads((tmp_path / HISTORY_FILE).read_text())
    assert saved["request"]["messages"] == [{"role": "system", "content": "policy"}]
    assert saved["request"]["tools"] == tools
    assert normalize_action(saved["message"], "bfcl") == [call(value=1)]


def test_parent_adapter_records_policy_history_and_uses_official_appworld_tgc(tmp_path, monkeypatch):
    import exgentic.agents.premise.pipeline as module

    manifest = tmp_path / "split.json"
    write_json(manifest, {"appworld_42": {"train": ["one"]}})
    pipeline = PremisePipeline(
        output=tmp_path / "run",
        manifest=manifest,
        benchmarks=["appworld"],
        mode="isolated",
        seed=42,
        count=None,
        past_count=3,
        model="openai/test",
        base_url="http://localhost/v1",
        candidate_evaluation="span_policy_shift",
        appworld_policy_override="Pinned environment policy",
    )
    configs = []

    def execute(*, config):
        configs.append(config)
        return SimpleNamespace(session_results=[SimpleNamespace(status=SimpleNamespace(value="completed"))])

    monkeypatch.setattr(module, "execute", execute)
    monkeypatch.setattr(
        module,
        "trajectory_from_session",
        lambda *_a: {
            "session_dir": str(tmp_path / "session"),
            "success": False,
            "score": 0.9999,
        },
    )
    trace = pipeline._run_agent(
        benchmark="appworld",
        task_id="one",
        playbook={"playbook": "playbook"},
        label="parent",
    )
    assert trace["score"] == 0
    assert trace["partial_score"] == 0.9999
    assert trace["score_source"] == "appworld_tgc_v1"
    assert trace["actor_policy"] == build_generator_system_prompt("playbook")
    assert trace["policy_shift_history"] == str(tmp_path / "session/agent" / HISTORY_FILE)
    assert configs[0].agent_kwargs["record_policy_shift"] is True
    assert configs[0].benchmark_kwargs["policy_override"] == "Pinned environment policy"
    assert configs[0].replay_prefix is None


def test_native_prefix_recording_and_evaluation_need_no_terminal_or_environment(tmp_path, monkeypatch):
    import asyncio

    pytest.importorskip("harbor")
    from harbor.agents.terminus_2 import Terminus2
    from harbor.llms.base import LLMResponse
    from harbor.llms.chat import Chat

    from exgentic.benchmarks.terminalbench2.terminus_limits import BudgetedTerminus2

    class FakeLLM:
        async def call(self, **_kwargs):
            return LLMResponse(
                content=json.dumps(
                    {
                        "analysis": "work",
                        "plan": "verify",
                        "commands": [
                            {"keystrokes": "pwd\n", "duration": 0.1},
                        ],
                    }
                ),
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=10,
                    cache_tokens=0,
                    cost_usd=0,
                ),
            )

    monkeypatch.setattr(Terminus2, "_init_llm", lambda *_a, **_kw: FakeLLM())
    actor = BudgetedTerminus2(
        logs_dir=tmp_path,
        model_name="openai/test",
        max_turns=40,
        record_policy_shift=True,
        enable_summarize=False,
        record_terminal_session=False,
    )
    chat = Chat(FakeLLM())
    asyncio.run(actor._query_llm(chat, "task\noriginal playbook"))
    asyncio.run(actor._query_llm(chat, "original observation"))
    records = [json.loads(line) for line in (tmp_path / HISTORY_FILE).read_text().splitlines()]
    assert [row["action_index"] for row in records] == [0, 1]
    assert records[1]["request"]["messages"][1]["content"] == actor.episodes[0]["response"]
    source = {
        "benchmark": "terminalbench2",
        "task_id": "native",
        "success": True,
        "actions": [{}, {}],
        "policy_shift_history": str(tmp_path / HISTORY_FILE),
        "actor_policy": "original playbook",
    }
    generated = []

    def complete(**kwargs):
        generated.append(kwargs)
        return litellm.ModelResponse(
            choices=[
                {"finish_reason": "stop", "message": {"role": "assistant", "content": actor.episodes[0]["response"]}}
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 10},
        )

    monkeypatch.setattr(litellm, "completion", complete)
    result = evaluator(tmp_path).evaluate(
        {"playbook": {"playbook": "candidate"}},
        source,
        {"span": {"start": 0, "end": 2}},
        "native",
    )
    assert result["strict_f1"] == result["evaluation_score"] == 1
    assert len(generated) == 2
    later = max(generated, key=lambda request: len(request["messages"]))
    assert later["messages"][0]["content"] == "task\ncandidate"
    assert later["messages"][1:] == records[1]["request"]["messages"][1:]
