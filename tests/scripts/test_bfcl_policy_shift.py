# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import copy
import json

import pytest

from scripts.premise.run_bfcl_policy_shift import (
    METRICS,
    POINTS,
    decode_record,
    freeze_history_bank,
    normalize_calls,
    probe_request,
    response_message,
    resume_config,
    score_calls,
    summarize_predictions,
    task_balanced,
)
from scripts.premise.schedule_bfcl_policy_shift import dependency_state


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


@pytest.mark.parametrize("explicit_override", [True, False])
def test_runtime_endpoint_override_routes_every_stage_without_changing_preparation(tmp_path, monkeypatch,
                                                                                  explicit_override):
    from scripts.premise import run_bfcl_policy_shift as study

    manifest = tmp_path / "split.json"
    manifest.write_text("{}")
    output = tmp_path / "gemma"
    output.mkdir()
    model = {"key": "gemma", "model": "openai/google/gemma-4-26B-A4B-it",
             "base_url": "http://localhost:18008/v1", "sources": [{"seed": seed} for seed in (42, 43, 44)]}
    prepared = {"model": model, "runs": []}
    path = output / "prepared.json"
    path.write_text(json.dumps(prepared))
    original = path.read_bytes()
    override = "http://localhost:18009/v1"
    if not explicit_override:
        (output / "runtime_manifest.json").write_text(json.dumps({"model": {"base_url": override}}))

    def prepare(_output, declared_model, _hash):
        assert declared_model == model
        return prepared

    def stage(_output, runtime, *args):
        assert runtime["model"]["base_url"] == override
        return []

    async def probes(_output, _anchors, _contexts, runtime_model, _workers):
        assert runtime_model["base_url"] == override

    def analyze(_output, _anchors, runtime_model):
        assert runtime_model["base_url"] == override
        return {"coverage": {"failed_or_missing_probes": 0}}

    monkeypatch.setattr(study, "prepare_model", prepare)
    for name in ("finish_training", "collect_contexts", "build_reference_rollouts"):
        monkeypatch.setattr(study, name, stage)
    monkeypatch.setattr(study, "freeze_history_bank", lambda *_args: [])
    monkeypatch.setattr(study, "run_probes", probes)
    monkeypatch.setattr(study, "analyze", analyze)
    study.run({"output": str(tmp_path), "checkpoints": list(POINTS), "workers": 150, "reference_max_steps": 20,
               "manifest": str(manifest), "manifest_sha256": study.digest(manifest.read_bytes()), "models": [model]},
              "gemma", base_url=override if explicit_override else None)
    assert path.read_bytes() == original
    assert model["base_url"] == "http://localhost:18008/v1"
    assert study.read_json(output / "runtime_manifest.json")["model"]["base_url"] == override


def test_scheduler_observes_external_controller_and_requires_final_analysis(tmp_path, monkeypatch):
    from scripts.premise import schedule_bfcl_policy_shift as schedule

    fingerprint = ["same-start"]
    monkeypatch.setattr(schedule, "process_fingerprint", lambda _pid: fingerprint[0])
    controller = schedule.ExistingController(tmp_path, 123, "same-start")
    (tmp_path / "status.json").write_text(json.dumps({"phase": "complete"}))
    assert controller.poll() is None
    fingerprint[0] = None
    assert controller.poll() == 1
    (tmp_path / "analysis").mkdir()
    (tmp_path / "analysis/summary.json").write_text("{}")
    assert controller.poll() == 0
    fingerprint[0] = "reused-pid"
    assert controller.poll() == 0


def test_scheduler_rejects_reused_pid_when_adopting_an_external_controller(tmp_path, monkeypatch):
    from scripts.premise import schedule_bfcl_policy_shift as schedule

    (tmp_path / "runtime_manifest.json").write_text(json.dumps({
        "model": {"key": "gemma"}, "output": str(tmp_path), "controller_pid": 123,
        "controller_start_ticks": "original-start",
    }))
    monkeypatch.setattr(schedule, "process_fingerprint", lambda _pid: "other-start")
    assert schedule.existing_controller(tmp_path, "gemma") is None


def test_order_changes_accuracy_but_not_multiset_f1():
    reference = [call("a", x=1), call("b")]
    result = score_calls(reference[::-1], reference)
    assert result["strict_accuracy"] == result["loose_accuracy"] == 0
    assert result["strict_f1"] == result["loose_f1"] == 1


def test_duplicate_calls_and_wrong_arguments_are_not_silently_collapsed():
    result = score_calls([call("a", x=1)] * 2 + [call("b", x=9)], [call("a", x=1), call("b", x=2)])
    assert result["strict_precision"] == pytest.approx(1 / 3)
    assert result["strict_recall"] == 0.5
    assert result["strict_f1"] == pytest.approx(0.4)
    assert result["loose_precision"] == pytest.approx(2 / 3)
    assert result["loose_recall"] == 1
    assert result["loose_f1"] == pytest.approx(0.8)


def test_empty_predictions_are_valid_zero_agreement_when_reference_has_calls():
    assert all(value == 0 for value in score_calls([], [call("a")]).values())
    assert all(value == 1 for value in score_calls([], []).values())


def test_safe_decoder_supports_actual_litellm_choices_repr_and_canonical_arguments():
    response = {"choices": ["Choices(finish_reason='tool_calls', index=0, "
                            "message=Message(content=None, role='assistant', tool_calls=["
                            "ChatCompletionMessageToolCall(function=Function(arguments='"
                            '{"b": 2, "a": 1}'
                            "', name='find'), id='id1', type='function')]))"]}
    calls = normalize_calls(response_message(response))
    assert score_calls(calls, [call("find", a=1, b=2)])["strict_accuracy"] == 1
    with pytest.raises(ValueError, match="Unsupported trace"):
        decode_record("__import__('os').system('false')")


@pytest.mark.parametrize("response", [
    {"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "x"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": None}}]},
])
def test_truncation_and_empty_protocol_response_are_errors_not_empty_actions(response):
    with pytest.raises(ValueError):
        response_message(response)


def test_probe_changes_only_context_and_keeps_history_tools_and_low_reasoning():
    anchor = {"messages": [{"role": "system", "content": "initial"}, {"role": "user", "content": "task"},
                            {"role": "assistant", "tool_calls": [{"function": {"name": "a", "arguments": "{}"}}]},
                            {"role": "tool", "content": "done", "tool_call_id": "same-id"}],
              "tools": [{"type": "function", "function": {"name": "a", "parameters": {"type": "object"}}}]}
    original = copy.deepcopy(anchor)
    request = probe_request(anchor, "", {"model": "openai/openai/gpt-oss-20b", "reasoning_effort": "low"})
    assert request["messages"][1:] == anchor["messages"][1:]
    assert request["tools"] == anchor["tools"]
    assert request["messages"][0]["content"] != "initial"
    assert request["model"] == "openai/gpt-oss-20b"
    assert request["reasoning_effort"] == "low"
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    assert anchor == original


def test_task_balancing_does_not_give_long_tasks_extra_weight():
    rows = [{"task_id": "long", **dict.fromkeys(METRICS, 1)} for _ in range(9)]
    rows.append({"task_id": "short", **dict.fromkeys(METRICS, 0)})
    assert task_balanced(rows)["strict_accuracy"] == 0.5


def test_actual_reference_requests_are_frozen_even_when_later_rollout_fails(tmp_path):
    from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK
    from exgentic.agents.ace.prompts.generator import build_generator_system_prompt

    trace = tmp_path / ("reference/sessions/s42/alignment_s42_cp000_parent_reference_000/"
                        "sessions/session/agent/litellm/trace.jsonl")
    trace.parent.mkdir(parents=True)
    request = {"messages": [{"role": "system", "content": build_generator_system_prompt(DEFAULT_PLAYBOOK)},
                             {"role": "user", "content": "unchanged observation"}],
               "tools": [{"type": "function", "function": {"name": "a", "parameters": {"type": "object"}}}]}
    response = {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [
        {"type": "function", "id": "id", "function": {"name": "a", "arguments": "{}"}},
    ]}}]}
    trace.write_text("\n".join(json.dumps(row) for row in [
        {"request": {"messages": []}, "response": {}},
        {"request": request, "response": response},
        {"request": request, "response": {}},
    ]) + "\n")
    rollout = {"key": "s42_000", "seed": 42, "task_id": "task", "checkpoint_key": "s42_cp000",
               "context": "parent", "partition": "reference", "index": 0, "group": "s42", "status": "error",
               "score": 0}
    anchors = freeze_history_bank(tmp_path, [rollout])
    assert len(anchors) == 1
    history = json.loads((tmp_path / "histories/s42_000_a0001.json").read_text())
    assert history["messages"] == request["messages"] and history["tools"] == request["tools"]
    assert history["reference_calls"] == [call("a")]
    assert history["reference_rollout_status"] == "error"
    audit = json.loads((tmp_path / "reference_audit.json").read_text())
    assert len(audit["excluded_acting_requests"]) == 1
    assert freeze_history_bank(tmp_path, [rollout]) == anchors


def test_failed_probe_excluded_from_same_complete_bank_at_every_checkpoint():
    anchors = [{"key": "long", "task_id": "long", "seed": 42},
               {"key": "short", "task_id": "short", "seed": 42}]
    records = [{"anchor_key": anchor["key"], "task_id": anchor["task_id"], "seed": 42,
                "checkpoint": cp, "status": "completed", "metrics": dict.fromkeys(METRICS, value),
                "predicted_calls": [call("a")]}
               for anchor, value in zip(anchors, (1, 0), strict=True) for cp in POINTS]
    bad = next(row for row in records if row["anchor_key"] == "long" and row["checkpoint"] == 10)
    bad.update(status="error")
    rows, pairs, coverage = summarize_predictions(anchors, records)
    assert coverage["common_anchors_all_checkpoints"] == 1
    assert coverage["failed_or_missing_probes"] == 1
    assert all(row["common_anchors"] == 1 and row["strict_accuracy"] == 0
               for row in rows if row["seed"] == 42)
    assert next(row for row in rows if row["seed"] == 42 and row["checkpoint"] == 5)[
        "available_strict_accuracy"] == 0.5
    assert len([row for row in pairs if row["seed"] == 42]) == 55
    assert all(row["strict_accuracy"] is None for row in rows if row["seed"] == 43)


def test_sequential_continuation_preserves_order_and_sampler_and_records_step_change():
    original = {"benchmarks": ["bfcl", "appworld"], "mode": "sequential", "past_count": 3,
                "order": [["bfcl", str(i)] for i in range(50)] + [["appworld", "a"]],
                "benchmark_configs": {"bfcl": {"subset": "multi_turn_base"}, "appworld": {}},
                "max_steps": {"bfcl": 50, "appworld": 50}}
    prior = copy.deepcopy(original)
    config = resume_config(original)
    assert config["order"] == original["order"][:50]
    assert config["past_count"] == original["past_count"]
    assert config["mode"] == "isolated" and config["max_steps"]["bfcl"] == 20
    assert original == prior


def dependency_fixture(root):
    analysis = root / "analysis"
    analysis.mkdir()
    (root / "analysis_pipeline_status.json").write_text(json.dumps({
        "phase": "complete", "runner_exit_code": 0, "analysis_exit_code": 0, "finished_at": "2026-10-11",
    }))
    (root / "status.json").write_text(json.dumps({
        "phase": "complete_with_errors", "total_evaluations": 100, "processed_tasks": 100,
        "active_tasks": 0, "execution_errors": 7,
    }))
    (analysis / "correlation_summary.json").write_text("{}")
    (analysis / "figure.pdf").write_bytes(b"pdf")
    return {"name": "model", "output": str(root), "analysis": "analysis", "required_figures": ["figure.pdf"]}


def test_gate_accepts_reported_execution_errors_only_after_final_figures_exist(tmp_path):
    dependency = dependency_fixture(tmp_path)
    assert dependency_state(dependency)["ready"]
    (tmp_path / "analysis/figure.pdf").unlink()
    assert not dependency_state(dependency)["ready"]


def test_gate_does_not_start_on_interim_summary_or_incomplete_evaluation(tmp_path):
    dependency = dependency_fixture(tmp_path)
    summary = tmp_path / "analysis/correlation_summary.json"
    summary.write_text('{"partial_progress": {}}')
    assert not dependency_state(dependency)["ready"]
    summary.write_text("{}")
    status = tmp_path / "status.json"
    progress = json.loads(status.read_text())
    progress["processed_tasks"] = 99
    status.write_text(json.dumps(progress))
    assert not dependency_state(dependency)["ready"]


def test_checked_subprocess_wrapper_can_signal_analyzer_success_through_finished_phase(tmp_path):
    dependency = dependency_fixture(tmp_path)
    path = tmp_path / "analysis_pipeline_status.json"
    pipeline = json.loads(path.read_text())
    del pipeline["analysis_exit_code"]
    path.write_text(json.dumps(pipeline))
    assert dependency_state(dependency)["ready"]
    pipeline.pop("finished_at")
    path.write_text(json.dumps(pipeline))
    assert not dependency_state(dependency)["ready"]
