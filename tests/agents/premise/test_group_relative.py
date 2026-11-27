# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Offline integration checks for fixed-bank group-relative prompt optimization."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import litellm
import pytest

from exgentic.actor_context import shared_actor_prompt
from exgentic.agents.premise.action_agreement import CanonicalActionAgreement, bound
from exgentic.agents.premise.group_relative import GroupRelativePipeline, agreement, partial_reward, relative_advantages
from exgentic.agents.premise.pipeline import write_json
from scripts.premise.run_group_relative import build_parser


def options(tmp_path, **overrides):
    manifest = tmp_path / "split.json"
    if not manifest.exists():
        write_json(manifest, {"seed_scope": "within_split_order_only", "bfcl_42": {"train": ["one", "two"]}})
    return {
        "output": tmp_path / "run", "manifest": manifest, "benchmarks": ["bfcl"],
        "mode": "isolated", "seed": 42, "count": None,
        "model": "openai/test", "base_url": "http://localhost/v1",
        "metric": "strict_f1",  # Legacy protocol tests retain their exact historical behavior.
        **overrides,
    }


def tool_message(name, step):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": f"call_{step}", "type": "function",
        "function": {"name": name, "arguments": json.dumps({"step": step})},
    }]}


def install_fake_actor(pipeline, monkeypatch, rewards=(1, 0, 0.5, 0, 1)):
    calls = []

    def run_agent(*, benchmark, task_id, playbook, label, **kwargs):
        assert not kwargs, "Only original initial-policy execution is allowed"
        generation = int(label.split("_g")[1].split("_r")[0])
        if generation == 0:
            assert playbook == pipeline._empty_state()["beam"][0]
        calls.append((task_id, label))
        rollout = int(label.rsplit("_r", 1)[1]) - 1
        reward = rewards(generation, rollout) if callable(rewards) else rewards[rollout]
        name = "good" if reward == 1 else "bad" if reward == 0 else "neutral"
        policy = shared_actor_prompt(benchmark, playbook["playbook"])
        history = [{"role": "system", "content": policy},
                   {"role": "user", "content": f"{task_id}; reference={name}"}]
        tools = [{"type": "function", "function": {"name": n, "parameters": {
            "type": "object", "properties": {"step": {"type": "integer"}}, "required": ["step"],
        }}} for n in ("good", "bad", "neutral")]
        rows, actions = [], []
        for step in range(2 + rollout % 2):
            message = tool_message(name, step)
            rows.append({"action_index": step, "request": {
                "messages": copy.deepcopy(history), "tools": tools, "tool_choice": "auto",
            }, "message": message, "generation_settings": {
                "model": pipeline.model, "temperature": pipeline.actor_temperature, "max_tokens": 2048,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }})
            actions.append({"action": {"name": name, "arguments": {"step": step}},
                            "observation": f"observation {step}"})
            history.extend([message, {"role": "tool", "tool_call_id": f"call_{step}", "content": f"obs {step}"}])
        path = pipeline.output / "fake_histories" / f"{label}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return {
            "benchmark": benchmark, "task_id": task_id, "task": task_id, "context": {},
            "actions": actions, "score": float(reward == 1), "success": reward == 1,
            "details": {"session_metadata": {"auxiliary_turn_metric": {
                "metric": "bfcl_turn_pass_rate_v1", "score": reward,
            }}},
            "actor_policy": policy, "actor_context": pipeline.actor_context,
            "policy_shift_history": str(path), "prefix_length": 0,
        }

    monkeypatch.setattr(pipeline, "_run_agent", run_agent)
    return calls


def install_fake_model(pipeline, monkeypatch, *, truncate_candidate=False):
    probes, prompts = [], []

    def ask(prompt, *, stage, **kwargs):
        prompts.append((stage, prompt))
        # Exercise the actual PEACE reflection/mutation prompt construction.
        if stage == "reflection":
            assert "Analyze this completed benchmark task" in prompt
            return {"span": {"start": 0, "end": 1, "type": "failure"},
                    "analysis": "Use the supported behavior", "expected_behavior": "Use good", "entry_assessments": []}
        assert stage == "mutation"
        assert "BEST rollout" in prompt and "WORST rollout" in prompt
        assert "<historical_training_evidence>" in prompt
        return {"operations": [{"type": "ADD", "section": "others", "content": "BFCL: GOODRULE"}]}

    def complete(**kwargs):
        probes.append(kwargs)
        assert "reference=neutral" not in kwargs["messages"][1]["content"]
        step = (len(kwargs["messages"]) - 2) // 2
        use_good = "GOODRULE" in kwargs["messages"][0]["content"]
        return litellm.ModelResponse(
            choices=[{"finish_reason": "length" if truncate_candidate and use_good else "tool_calls",
                      "message": tool_message("good" if use_good else "bad", step)}],
            usage={"prompt_tokens": 100, "completion_tokens": 10},
        )

    monkeypatch.setattr(pipeline.llm, "ask", ask)
    monkeypatch.setattr(litellm, "completion", complete)
    return probes, prompts


def test_advantages_are_relative_and_equal_rewards_have_no_signal():
    assert relative_advantages([0, 0, 0, 0, 0]) == [0] * 5
    assert relative_advantages([1, 1, 1, 1, 1]) == [0] * 5
    values = relative_advantages([1, 0, 0.5, 0, 1])
    assert values[0] > 0 > values[1] and values[2] == 0
    assert sum(values) == pytest.approx(0)
    for rewards in ([float("nan")], [float("inf")], [-1], [2], []):
        with pytest.raises(ValueError):
            relative_advantages(rewards)


def test_accuracy_preserves_order_and_f1_preserves_duplicate_counts():
    a, b = {"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}
    assert agreement([a, b], [b, a], "strict_accuracy") == 0
    assert agreement([a, b], [b, a], "strict_f1") == 1
    assert agreement([a, a], [a], "strict_f1") == pytest.approx(2 / 3)


class OfflineText:
    identity: ClassVar[dict] = {"kind": "offline-test-text", "revision": "pinned-test"}

    def prepare(self):
        return self.identity

    def compare(self, a, b):
        pytest.fail("This BFCL fixture has no natural-language arguments")


def canonical_options(tmp_path, **overrides):
    return options(tmp_path, metric="canonical_f1", action_comparator=CanonicalActionAgreement(
        tmp_path / "comparisons", text=OfflineText(),
    ), **overrides)


def test_canonical_optimizer_selects_by_saved_prefixes_without_environment_replay(tmp_path, monkeypatch):
    pipeline = GroupRelativePipeline(**canonical_options(tmp_path))
    actors = install_fake_actor(pipeline, monkeypatch)
    probes, _ = install_fake_model(pipeline, monkeypatch)
    pipeline.run()
    assert len(actors) == 10 and probes
    assert pipeline.config["method"] == "group-relative-canonical-action-lower-v1"
    assert pipeline.config["sampling_ratio"] == "canonical_agreement_proxy_not_likelihood_ratio"
    assert pipeline.config["canonical_score_mode"] == pipeline.config["sampling_ratio_bound"] == "lower"
    first = json.loads((pipeline.output / "iterations/iter_0001.json").read_text())
    assert first["train_update_applied"] and first["environment_replay_sessions"] == 0
    selected = next(c for c in first["candidates"] if c["id"] == first["selection"]["selected"])
    assert selected["selection_score"] > first["candidates"][0]["selection_score"]
    assert all(e["total_prefixes"] == e["resolved_prefixes"] for e in selected["prefix_evaluations"])
    assert (pipeline.output / "agreement_runtime.json").exists()
    resumed = GroupRelativePipeline(**canonical_options(tmp_path))
    monkeypatch.setattr(resumed, "_run_agent", lambda **kw: pytest.fail("Completed rollout must be reused"))
    resumed.run()


@pytest.mark.parametrize("score_mode", ["lower", "interval"])
def test_canonical_lower_compares_actual_scores_while_interval_requires_disjoint_bounds(
    tmp_path, monkeypatch, score_mode,
):
    pipeline = GroupRelativePipeline(**canonical_options(tmp_path, count=1, canonical_score_mode=score_mode))
    actors = install_fake_actor(pipeline, monkeypatch)
    install_fake_model(pipeline, monkeypatch)
    original = pipeline.action_comparator.compare

    def uncertain(a, b, **kwargs):
        result = original(a, b, **kwargs)
        return {**result, "lower": 0., "upper": 1., "resolved": False} if a != b else result

    monkeypatch.setattr(pipeline.action_comparator, "compare", uncertain)
    pipeline.run()
    artifact = json.loads((pipeline.output / "iterations/iter_0001.json").read_text())
    assert len(actors) == 5
    parent, child = artifact["candidates"][:2]
    assert child["selection_lower"] <= parent["selection_upper"]
    assert child["selection_upper"] > child["selection_lower"]
    assert any(e["resolved_prefixes"] < e["total_prefixes"] for e in child["prefix_evaluations"])
    if score_mode == "lower":
        assert artifact["selection"]["selected"] == "candidate_1"
        assert child["selection_score"] > parent["selection_score"]
        # Lower is applied before either sign of advantage. Unknown negative-A
        # matches contribute A*0, not A*upper as in the interval objective lower.
        assert child["surrogate"] > child["surrogate_lower"]
        for evaluation in child["prefix_evaluations"]:
            assert evaluation["agreement"] == evaluation["sampling_ratio_lower"]
            for prefix in evaluation["comparisons"]:
                assert prefix["sampling_ratio_value"] == prefix["sampling_ratio"]["lower"]
    else:
        assert artifact["selection"]["selected"] == "incumbent"
        assert pipeline.config["method"] == "group-relative-canonical-action-v1"
        assert "canonical_score_mode" not in pipeline.config  # Historical resume configuration is preserved.


@pytest.mark.parametrize("score_mode", ["lower", "interval"])
def test_canonical_score_intervals_keep_prefix_and_group_denominators(tmp_path, monkeypatch, score_mode):
    pipeline = GroupRelativePipeline(**canonical_options(tmp_path, count=1, shift_penalty=.2,
                                                         canonical_score_mode=score_mode))
    history = tmp_path / "hist.jsonl"
    history.write_text(json.dumps({"action_index": 0, "request": {"messages": []}}) + "\n")
    trace = {"benchmark": "bfcl", "policy_shift_history": str(history)}
    group = {"index": 0}
    active = [(group, 0, 1., trace), (group, 1, -1., trace)]
    candidates = [{"id": i, "valid": True, "playbook": {"playbook": i}} for i in ("incumbent", "child")]

    def row(predicted, reference):
        return {"status": "completed", "actions": [{"action_index": 0,
                "predicted": predicted, "reference": reference}]}

    results = {("incumbent", 0, 0): row("good", "good"),
               ("incumbent", 0, 1): row("bad", "bad"),
               ("child", 0, 0): row("good", "good"),
               ("child", 0, 1): row("unknown", "bad")}

    def compare(a, b, **kwargs):
        value = bound(1, reason="same") if a == b else bound(0, 1, reason="unknown")
        return {**value, "resolved": a == b}

    monkeypatch.setattr(pipeline.action_comparator, "compare", compare)
    pipeline._score_canonical(candidates, [group], active,
                              {"incumbent": "incumbent", "child": "child"}, results)
    child = candidates[1]
    # (+1 * 1 + -1 * [0,1]) / original G=5 => [0,.2].
    assert child["surrogate_lower"] == 0 and child["surrogate_upper"] == pytest.approx(.2)
    assert child["shift_lower"] == 0 and child["shift_upper"] == .5
    assert child["selection_lower"] == -.1 and child["selection_upper"] == pytest.approx(.2)
    assert child["surrogate"] == pytest.approx(.2 if score_mode == "lower" else 0)
    assert child["selection_score"] == pytest.approx(.1 if score_mode == "lower" else -.1)
    pipeline.max_policy_shift = .4
    pipeline._score_canonical(candidates, [group], active,
                              {"incumbent": "incumbent", "child": "child"}, results)
    assert not child["valid"] and "bound" in child["errors"][-1]


def test_canonical_resume_pins_comparator_and_runtime_assets(tmp_path):
    pipeline = GroupRelativePipeline(**canonical_options(tmp_path))
    pipeline._prepare_action_comparator()
    path = pipeline.output / "agreement_runtime.json"
    runtime = json.loads(path.read_text())
    runtime["text_assets"]["revision"] = "tampered"
    write_json(path, runtime)
    with pytest.raises(ValueError, match="runtime changed"):
        pipeline._prepare_action_comparator()
    changed = canonical_options(tmp_path)
    changed["action_comparator"].text.identity = {"kind": "different"}
    changed["action_comparator"] = CanonicalActionAgreement(tmp_path / "other", text=changed["action_comparator"].text)
    with pytest.raises(ValueError, match=r"configuration|config"):
        GroupRelativePipeline(**changed)


@pytest.mark.parametrize("initial_mode", ["lower", "interval"])
def test_canonical_score_mode_is_pinned_on_resume(tmp_path, initial_mode):
    initial = GroupRelativePipeline(**canonical_options(tmp_path, canonical_score_mode=initial_mode))
    resumed = GroupRelativePipeline(**canonical_options(tmp_path, canonical_score_mode=initial_mode))
    assert resumed.config == initial.config
    other_mode = "interval" if initial_mode == "lower" else "lower"
    with pytest.raises(ValueError, match="configuration differs"):
        GroupRelativePipeline(**canonical_options(tmp_path, canonical_score_mode=other_mode))


def test_partial_rewards_for_all_benchmarks_never_fall_back_to_binary(tmp_path):
    appworld = {"benchmark": "appworld", "score": 0, "success": False, "partial_score": 0.75}
    assert partial_reward(appworld)["score"] == 0.75 and appworld["score"] == 0
    with pytest.raises(ValueError, match="AppWorld"):
        partial_reward({"benchmark": "appworld", "score": 1, "success": True})
    bfcl = {"benchmark": "bfcl", "score": 0, "details": {"session_metadata": {
        "auxiliary_turn_metric": {"metric": "bfcl_turn_pass_rate_v1", "score": 0.5},
    }}}
    assert partial_reward(bfcl)["score"] == 0.5
    with pytest.raises(ValueError, match="BFCL"):
        partial_reward({"benchmark": "bfcl", "score": 1})
    report = tmp_path / "benchmark/harbor/verifier/ctrf.json"
    write_json(report, {"results": {"summary": {"tests": 4, "passed": 1},
                                   "tests": [{"status": s} for s in ("passed", "failed", "skipped", "failed")]}})
    terminal = {"benchmark": "terminalbench2", "score": 0, "session_dir": str(tmp_path)}
    assert partial_reward(terminal)["score"] == 0.25
    for benchmark in ("bfcl", "appworld", "terminalbench2"):
        trace = {"benchmark": benchmark, "score": 0, "partial_score": 0.9,
                 "details": {"session_metadata": {"experiment_budget_exceeded": True}}}
        assert partial_reward(trace)["score"] == 0


def test_live_model_settings_reach_actor_and_probe_with_unlimited_budget(tmp_path, monkeypatch):
    settings = {"temperature": 1, "top_p": 0.95, "presence_penalty": 0, "max_tokens": None,
                "timeout": 600, "num_retries": 1, "reasoning_effort": "xhigh",
                "extra_body": {"top_k": 20, "min_p": 0, "repetition_penalty": 1,
                               "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}}
    monkeypatch.setenv("EXGENTIC_ACTOR_OUTPUT_UNLIMITED", "1")
    pipeline = GroupRelativePipeline(**options(tmp_path, count=1, model_call_settings=settings))
    configs = []

    def execute(*, config):
        configs.append(config)
        return SimpleNamespace(session_results=[SimpleNamespace(status=SimpleNamespace(value="completed"))])

    monkeypatch.setattr("exgentic.agents.premise.pipeline.execute", execute)
    monkeypatch.setattr("exgentic.agents.premise.pipeline.trajectory_from_session", lambda *a: {
        "benchmark": "bfcl", "task_id": "one", "success": False, "score": 0,
        "session_dir": str(tmp_path), "actions": [], "details": {},
    })
    pipeline._run_agent(benchmark="bfcl", task_id="one", playbook=pipeline.states["bfcl"]["beam"][0], label="check")
    kwargs = configs[0].agent_kwargs
    assert kwargs["model_settings"].model_dump(include=set(settings)) == settings
    assert kwargs["enable_thinking"] and kwargs["allow_shared_thinking"]
    assert kwargs["native_model_settings"] == settings
    assert pipeline.policy_shift_evaluator.max_tokens is None
    assert pipeline.config["experiment_limits"]["bfcl"]["output_tokens"] is None
    assert pipeline.config["actor_context_metadata"]["enable_thinking"] is True
    assert pipeline.llm.generation_overrides == settings


def make_first_task_zero(pipeline, monkeypatch):
    calls = install_fake_actor(pipeline, monkeypatch)
    original = pipeline._run_agent
    refreshed_contexts = []

    def actor(**kwargs):
        trace = original(**kwargs)
        if kwargs["task_id"] == "one" and "_g0000_" in kwargs["label"]:
            trace["details"]["session_metadata"]["auxiliary_turn_metric"]["score"] = 0.5
        if "_g0001_" in kwargs["label"]:
            refreshed_contexts.append(kwargs["playbook"]["playbook"])
        return trace

    monkeypatch.setattr(pipeline, "_run_agent", actor)
    return calls, refreshed_contexts


def test_zero_advantage_task_is_resampled_after_epoch_with_learned_context(tmp_path, monkeypatch):
    pipeline = GroupRelativePipeline(**options(tmp_path, epochs=2))
    calls, contexts = make_first_task_zero(pipeline, monkeypatch)
    install_fake_model(pipeline, monkeypatch)
    pipeline.run()
    assert len(calls) == 15 and len(contexts) == 5
    assert all("GOODRULE" in context for context in contexts)
    assert len(pipeline.bank_history["0"]) == 2 and len(pipeline.bank_history["1"]) == 1
    assert pipeline._load_group(0)["generation"] == 1
    old = pipeline._read_reference(pipeline.bank_history["0"][0])
    assert old["advantages"] == [0] * 5
    third = json.loads((pipeline.output / "iterations/iter_0003.json").read_text())
    assert third["group_generation"] == 1 and third["status"] == "completed"
    assert third["reward_metric"] == "bfcl_turn_pass_rate_v1"
    assert pipeline.config["benchmark_configs"]["bfcl"]["auxiliary_turn_metrics"] is True


def test_refresh_recovers_finished_generation_after_interruption(tmp_path, monkeypatch):
    pipeline = GroupRelativePipeline(**options(tmp_path, epochs=2))
    calls, _ = make_first_task_zero(pipeline, monkeypatch)
    install_fake_model(pipeline, monkeypatch)
    collect = pipeline._collect_group

    def interrupt(index, generation, playbook):
        result = collect(index, generation, playbook)
        if generation == 1:
            raise RuntimeError("refresh interrupted before state commit")
        return result

    monkeypatch.setattr(pipeline, "_collect_group", interrupt)
    with pytest.raises(RuntimeError, match="refresh interrupted"):
        pipeline.run()
    assert len(calls) == 15
    resumed = GroupRelativePipeline(**options(tmp_path, epochs=2))
    assert resumed.cursor == 2 and resumed.pending_refresh == {0}
    monkeypatch.setattr(resumed, "_run_agent", lambda **kw: pytest.fail("Reuse completed refreshed generation"))
    install_fake_model(resumed, monkeypatch)
    resumed.run()
    assert resumed.cursor == 4 and len(resumed.bank_history["0"]) == 2
    assert not resumed.pending_refresh


def test_full_bank_precedes_updates_and_epochs_reuse_it_without_environment_calls(tmp_path, monkeypatch):
    pipeline = GroupRelativePipeline(**options(tmp_path, epochs=2, past_count=1))
    actors = install_fake_actor(pipeline, monkeypatch)
    probes, prompts = install_fake_model(pipeline, monkeypatch)
    original_proposal = pipeline._proposal

    def proposal(*args):
        assert len(actors) == 10 and len(pipeline.bank) == 2
        return original_proposal(*args)

    monkeypatch.setattr(pipeline, "_proposal", proposal)
    pipeline.run()
    assert len(actors) == 10
    assert pipeline.cursor == 4
    assert "GOODRULE" in pipeline.states["bfcl"]["beam"][0]["playbook"]
    first = json.loads((pipeline.output / "iterations/iter_0001.json").read_text())
    assert len(first["candidates"]) == 6  # C=5 proposals, plus the current prompt.
    incumbent, candidate = first["candidates"][:2]
    assert first["selection"]["selected"] == "candidate_1"
    assert incumbent["surrogate"] < 0 < candidate["surrogate"]
    assert candidate["surrogate"] == pytest.approx(2 * first["advantages"][0] / 5)
    assert len(candidate["prefix_evaluations"]) == 4  # The middle rollout has A=0.
    assert first["environment_replay_sessions"] == 0
    assert first["prefix_queue"]["queued_action_jobs"] == 20  # 10 positions x 2 distinct policies.
    second = json.loads((pipeline.output / "iterations/iter_0002.json").read_text())
    assert len(second["evaluated_groups"]) == 2
    assert len(second["candidates"][1]["prefix_evaluations"]) == 8
    assert second["selection"]["selected"] == "incumbent" and not second["train_update_applied"]
    assert len([p for stage, p in prompts if stage == "mutation"]) == 20
    assert len([p for stage, p in prompts if stage == "reflection"]) == 40
    assert all(p["temperature"] == 1 and p["max_tokens"] == 2048 for p in probes)
    assert (pipeline.output / "training_complete.json").is_file()
    assert len(list((pipeline.output / "checkpoints").glob("checkpoint_*.json"))) == 4

    resumed = GroupRelativePipeline(**options(tmp_path, epochs=2, past_count=1))
    monkeypatch.setattr(resumed, "_run_agent", lambda **kw: pytest.fail("Finished bank must not execute again"))
    before = len(probes)
    resumed.run()
    assert len(probes) == before and resumed.cursor == 4


def test_equal_reward_group_skips_proposals_and_probes(tmp_path, monkeypatch):
    pipeline = GroupRelativePipeline(**options(tmp_path, count=1))
    actors = install_fake_actor(pipeline, monkeypatch, rewards=(0, 0, 0, 0, 0))
    monkeypatch.setattr(pipeline.llm, "reflect", lambda *a, **kw: pytest.fail("Zero advantage cannot update"))
    monkeypatch.setattr(litellm, "completion", lambda **kw: pytest.fail("No prefix probes for zero advantage"))
    pipeline.run()
    artifact = json.loads((pipeline.output / "iterations/iter_0001.json").read_text())
    assert len(actors) == 10 and artifact["advantages"] == [0] * 5
    assert artifact["status"] == "skipped" and not artifact["train_update_applied"]
    assert len(pipeline.bank_history["0"]) == 2 and pipeline.refreshed_epochs == {1}


@pytest.mark.parametrize("settings", [{"shift_penalty": 2}, {"max_policy_shift": 0}])
def test_trust_controls_retain_incumbent_when_change_is_too_large(tmp_path, monkeypatch, settings):
    pipeline = GroupRelativePipeline(**options(tmp_path, count=1, **settings))
    install_fake_actor(pipeline, monkeypatch)
    install_fake_model(pipeline, monkeypatch)
    pipeline.run()
    artifact = json.loads((pipeline.output / "iterations/iter_0001.json").read_text())
    assert artifact["selection"]["selected"] == "incumbent"
    assert not artifact["train_update_applied"]
    assert artifact["candidates"][1]["shift_from_incumbent"] == 1


def test_truncated_candidate_gets_no_novelty_credit_and_keeps_usage(tmp_path, monkeypatch):
    pipeline = GroupRelativePipeline(**options(tmp_path, count=1, replay_workers=1))
    install_fake_actor(pipeline, monkeypatch)
    install_fake_model(pipeline, monkeypatch, truncate_candidate=True)
    pipeline.run()
    artifact = json.loads((pipeline.output / "iterations/iter_0001.json").read_text())
    assert artifact["selection"]["selected"] == "incumbent"
    child = artifact["candidates"][1]
    assert not child["valid"] and "selection_score" not in child
    assert any(e["result"].get("failed_action_usage") for e in child["prefix_evaluations"])


def test_interrupted_commit_is_recovered_without_new_inference(tmp_path, monkeypatch):
    pipeline = GroupRelativePipeline(**options(tmp_path, count=1))
    install_fake_actor(pipeline, monkeypatch)
    probes, _ = install_fake_model(pipeline, monkeypatch)
    pipeline.collect_rollouts()

    def crash():
        raise RuntimeError("Simulated crash after writing the iteration artifact")

    monkeypatch.setattr(pipeline, "_save", crash)
    with pytest.raises(RuntimeError, match="Simulated crash"):
        pipeline.step()
    resumed = GroupRelativePipeline(**options(tmp_path, count=1))
    assert resumed.cursor == 0
    monkeypatch.setattr(resumed, "_proposal", lambda *a: pytest.fail("Recover committed proposals"))
    before = len(probes)
    artifact = resumed.step()
    assert resumed.cursor == 1 and artifact["selection"]["selected"] == "candidate_1"
    assert len(probes) == before


def test_partial_collection_reuses_finished_rollouts_and_preserves_infrastructure_errors(tmp_path, monkeypatch):
    pipeline = GroupRelativePipeline(**options(tmp_path, count=1, parent_workers=1))
    measured = install_fake_actor(pipeline, monkeypatch)
    original = pipeline._run_agent

    def fail_one(**kwargs):
        if kwargs["label"].endswith("r02"):
            raise ConnectionError("Inference unavailable")
        return original(**kwargs)

    monkeypatch.setattr(pipeline, "_run_agent", fail_one)
    with pytest.raises(ConnectionError):
        pipeline.collect_rollouts()
    assert not pipeline.bank
    resumed = GroupRelativePipeline(**options(tmp_path, count=1, parent_workers=1))
    new_measured = install_fake_actor(resumed, monkeypatch)
    resumed.collect_rollouts()
    assert len(measured) + len(new_measured) == 5
    assert Counter(label for _, label in measured + new_measured).most_common(1)[0][1] == 1
    assert resumed._load_group(0)["rewards"] == [1, 0, 0.5, 0, 1]


def test_changed_history_is_rejected_before_new_proposals(tmp_path, monkeypatch):
    pipeline = GroupRelativePipeline(**options(tmp_path, count=1))
    install_fake_actor(pipeline, monkeypatch)
    pipeline.collect_rollouts()
    group = pipeline._load_group(0)
    history = group["traces"][0]["policy_shift_history"]
    Path(history).write_text("tampered\n")
    with pytest.raises(ValueError, match="Frozen action history changed"):
        pipeline.step()


def test_measured_zero_without_actions_keeps_reward_but_has_no_prefix_evidence(tmp_path, monkeypatch):
    pipeline = GroupRelativePipeline(**options(tmp_path, count=1))
    install_fake_actor(pipeline, monkeypatch)
    original = pipeline._run_agent

    def empty_failure(**kwargs):
        trace = original(**kwargs)
        if kwargs["label"].endswith("r02"):
            trace["actions"] = []
            trace["policy_shift_history"] = str(pipeline.output / "never_recorded.jsonl")
        return trace

    monkeypatch.setattr(pipeline, "_run_agent", empty_failure)
    install_fake_model(pipeline, monkeypatch)
    pipeline.run()
    group = pipeline._load_group(0)
    assert group["rewards"][1] == 0 and group["advantages"][1] < 0
    assert not group["has_prefixes"][1]
    assert Path(group["traces"][1]["policy_shift_history"]).read_text() == ""
    artifact = json.loads((pipeline.output / "iterations/iter_0001.json").read_text())
    assert len(artifact["candidates"][1]["prefix_evaluations"]) == 3
    assert artifact["worst_rollout"] == 3


@pytest.mark.parametrize("setting", [
    {"candidate_count": 6}, {"rollout_count": 6}, {"epochs": 2}, {"past_count": 1},
    {"metric": "strict_accuracy"}, {"shift_penalty": 0.1},
    {"actor_temperature": 0, "policy_shift_temperature": 0},
])
def test_resume_pins_method_and_sampling_configuration(tmp_path, setting):
    GroupRelativePipeline(**options(tmp_path))
    with pytest.raises(ValueError, match="configuration differs"):
        GroupRelativePipeline(**options(tmp_path, **setting))


def test_cli_defaults_and_validation(tmp_path):
    args = build_parser().parse_args(["--task-split-manifest", "split.json", "--output-dir", "run"])
    assert args.rollouts == args.candidates == 5 and args.past == 0
    assert args.epochs == 1 and args.metric == "canonical_f1"
    assert args.canonical_score_mode == "lower"
    assert args.proposal_workers == 5
    assert args.action_workers == args.llm_calls == 128
    assert args.actor_temperature == args.policy_shift_temperature == 1
    for setting in ({"candidate_count": 0}, {"rollout_count": 1}, {"epochs": 0}, {"past_count": -1},
                    {"shift_penalty": float("nan")}, {"max_policy_shift": 2}, {"canonical_score_mode": "upper"}):
        with pytest.raises(ValueError):
            GroupRelativePipeline(**options(tmp_path, **setting))


def test_group_relative_default_queue_uses_full_endpoint_capacity_and_five_proposals(tmp_path, monkeypatch):
    from exgentic.agents.premise import llm

    limits = []
    original = llm.configure_llm_slots

    def configure(max_calls):
        limits.append(max_calls)
        original(max_calls)

    monkeypatch.setattr(llm, "configure_llm_slots", configure)
    pipeline = GroupRelativePipeline(**options(tmp_path))
    assert pipeline.candidate_count == pipeline.mutation_workers == pipeline.parent_workers == 5
    assert pipeline.replay_workers == 128 and limits == [128]


def test_explicit_queue_limits_override_group_relative_defaults(tmp_path, monkeypatch):
    from exgentic.agents.premise import llm

    limits = []
    monkeypatch.setattr(llm, "configure_llm_slots", limits.append)
    pipeline = GroupRelativePipeline(**options(tmp_path, replay_workers=7, llm_calls=11, mutation_workers=3))
    assert pipeline.replay_workers == 7 and pipeline.mutation_workers == 3
    assert pipeline.candidate_count == 5 and limits == [11]


def test_subclass_source_protocol_is_pinned_before_resume_validation(tmp_path):
    class ImportedSourcePipeline(GroupRelativePipeline):
        source_version = "one"

        def _configuration_extensions(self):
            return {"imported_source_version": self.source_version}

    original = ImportedSourcePipeline(**options(tmp_path))
    assert original.config["imported_source_version"] == "one"
    assert ImportedSourcePipeline(**options(tmp_path)).config == original.config
    ImportedSourcePipeline.source_version = "two"
    with pytest.raises(ValueError, match="configuration differs"):
        ImportedSourcePipeline(**options(tmp_path))
