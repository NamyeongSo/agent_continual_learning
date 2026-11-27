# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Check holdout isolation, full trajectory reconstruction and single-decision replay."""

import copy
import json
import threading
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from exgentic.agents.premise.reward_data import (
    BENCHMARKS,
    FeedbackTruncator,
    _trace_choice_message,
    digest,
    holdout_tasks,
    prepare_dataset,
    public_trace,
    reconstruct_pair,
    training_order,
    write_json,
)
from exgentic.agents.premise.reward_pipeline import RewardLLM, RewardPipeline, RewardRequestError, mutate_rules
from exgentic.agents.premise.reward_prompts import analysis_messages, judge_messages
from scripts.premise.run_reward_agent import RewardValidationQueue


def session(root, benchmark, steps, prefix=(), reward=0):
    root.mkdir(parents=True)
    write_json(
        root / "session.json",
        {
            "task": "Fulfill the user request",
            "context": {"policy": "Use public evidence", "benchmark": benchmark, "task_id": "hidden_task"},
            "actions": [],
        },
    )
    write_json(
        root.parent.parent / "run/config.json",
        {"replay_prefix": {"actions": [{"action": step["action"]} for step in prefix]}},
    )
    if benchmark == "terminalbench2":
        write_json(
            root / "benchmark/harbor/agent/trajectory.json",
            {
                "steps": [
                    {"source": "user", "message": "Injected old playbook should not be forwarded"},
                    *[{"source": "agent", "message": text, "metrics": {"reward": reward}} for text in steps],
                ],
                "final_metrics": {"reward": reward, "private_expected_answer": "secret"},
            },
        )
    else:
        events = []
        for i, step in enumerate(steps):
            events.extend(
                [
                    {"event": "action", "step": i, "action": step["action"]},
                    {"event": "observation", "step": i, "observation": step["observation"]},
                ]
            )
        (root / "trajectory.jsonl").write_text("\n".join(json.dumps(event) for event in events))


def test_reconstruct_pair_restores_full_tool_prefix_and_removes_provenance(tmp_path):
    prefix = {"action": {"name": "inspect"}, "observation": {"result": "observed state"}}
    old_end = {"action": {"name": "modify"}, "observation": {"result": "wrong state"}}
    new_end = {"action": {"name": "modify"}, "observation": {"result": "desired state"}}
    old = tmp_path / "old/sessions/one"
    new = tmp_path / "new/sessions/two"
    session(old, "appworld", [prefix, old_end])
    session(new, "appworld", [new_end], prefix=[prefix])
    pair = reconstruct_pair({"benchmark": "appworld", "original_session_dir": str(old), "replay_session_dir": str(new)})
    assert pair["original"] == [prefix, old_end]
    assert pair["replay"] == [prefix, new_end]
    assert pair["context"] == {"policy": "Use public evidence"}
    assert "score" not in pair
    public_trace.cache_clear()


def test_native_pair_uses_agent_steps_without_verifier_rewards_or_double_prefix(tmp_path):
    old = tmp_path / "old/sessions/one"
    new = tmp_path / "new/sessions/two"
    session(old, "terminalbench2", ["inspect", "wrong edit"], reward=0)
    session(new, "terminalbench2", ["inspect", "correct edit"], reward=1)
    pair = reconstruct_pair(
        {"benchmark": "terminalbench2", "original_session_dir": str(old), "replay_session_dir": str(new)}
    )
    assert pair["replay"] == [{"message": "inspect"}, {"message": "correct edit"}]
    assert pair["public_tools"] == []
    assert "secret" not in json.dumps(pair)
    assert "metrics" not in json.dumps(pair)
    assert "Injected" not in json.dumps(pair)
    public_trace.cache_clear()


def test_reconstruct_pair_preserves_public_argument_schema_and_optional_finish(tmp_path):
    old, new = tmp_path / "old/sessions/one", tmp_path / "new/sessions/two"
    step = {"action": {"name": "modify", "arguments": {"target": "item"}}, "observation": "Done"}
    session(old, "appworld", [step])
    session(new, "appworld", [step])
    path = old / "session.json"
    manifest = json.loads(path.read_text())
    schema = {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]}
    finish_schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    manifest["actions"] = [
        {"name": "modify", "description": "Modify a target", "arguments_schema": schema},
        {"name": "finish", "is_finish": True, "arguments_schema": finish_schema},
        {"name": "unobserved", "arguments_schema": schema},
        {"name": "modify", "is_hidden": True, "arguments_schema": {"private": "secret"}},
    ]
    write_json(path, manifest)
    public_trace.cache_clear()
    pair = reconstruct_pair({"benchmark": "appworld", "original_session_dir": str(old), "replay_session_dir": str(new)})
    tools = pair["public_tools"]
    assert len(tools) == 2
    assert tools[0]["arguments_schema"] == schema
    assert tools[1]["arguments_schema"] == finish_schema
    assert "required" not in tools[1]["arguments_schema"]
    assert "secret" not in json.dumps(tools)
    public_trace.cache_clear()


@pytest.mark.parametrize("benchmark", ["bfcl", "appworld"])
def test_public_trace_recovers_legacy_composite_parameters_by_call_identity(tmp_path, benchmark):
    root = tmp_path / "old/sessions/one"
    action = {"id": "call", "name": "modify", "arguments": {}, "validation": {"valid": True}}
    step = {"action": {"actions": [action]}, "observation": {"result": "public result"}}
    session(root, benchmark, [step])
    manifest = json.loads((root / "session.json").read_text())
    manifest["actions"] = [{"name": "modify", "arguments_schema": {"required": ["target"]}}]
    write_json(root / "session.json", manifest)
    if benchmark == "bfcl":
        write_json(
            root / "benchmark/bfcl_trace.json",
            {"turn_step_actions": [[[{**action, "arguments": {"target": "item"}}]]]},
        )
    else:
        path = root / "agent/litellm/trace.jsonl"
        path.parent.mkdir(parents=True)
        message = {
            "role": "assistant",
            "tool_calls": [{"id": "call", "function": {"name": "modify", "arguments": '{"target": "item"}'}}],
        }
        rows = [
            {"record_type": "definition", "id": "message", "value": message},
            {"trace_format": "dedup_v1", "request": {"messages_refs": ["message"]}},
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows))
    write_json(root / "benchmark/scores.json", {"private_expected_answer": "must not leak"})
    public_trace.cache_clear()
    restored = public_trace(str(root), benchmark)
    assert restored["steps"][0]["action"]["actions"][0]["arguments"] == {"target": "item"}
    assert restored["steps"][0]["observation"] == step["observation"]
    assert "must not leak" not in json.dumps(restored)
    assert json.loads((root / "trajectory.jsonl").read_text().splitlines()[0])["action"] == step["action"]
    public_trace.cache_clear()


def test_unrecoverable_required_composite_parameters_reject_new_dataset(tmp_path):
    root = tmp_path / "old/sessions/one"
    step = {
        "action": {"actions": [{"id": "missing", "name": "modify", "arguments": {}, "validation": {"valid": True}}]},
        "observation": "Done",
    }
    session(root, "appworld", [step])
    manifest = json.loads((root / "session.json").read_text())
    manifest["actions"] = [{"name": "modify", "arguments_schema": {"required": ["target"]}}]
    write_json(root / "session.json", manifest)
    public_trace.cache_clear()
    with pytest.raises(ValueError, match="Cannot recover required public parameters"):
        public_trace(str(root), "appworld")
    public_trace.cache_clear()


def test_legacy_last_response_recovers_tool_arguments_without_running_constructors(tmp_path):
    root = tmp_path / "old/sessions/one"
    step = {"action": {"actions": [{"id": "call", "name": "modify", "arguments": {}}]}, "observation": "Done"}
    session(root, "appworld", [step])
    path = root / "agent/litellm/trace.jsonl"
    path.parent.mkdir(parents=True)
    choice = (
        "Choices(finish_reason='tool_calls', index=0, message=Message(role='assistant', "
        "tool_calls=[ChatCompletionMessageToolCall(id='call', "
        "function=Function(name='modify', arguments='{\"target\": \"item\"}'))]))"
    )
    path.write_text(json.dumps({"request": {}, "response": {"choices": [choice]}}))
    public_trace.cache_clear()
    assert public_trace(str(root), "appworld")["steps"][0]["action"]["actions"][0]["arguments"] == {"target": "item"}
    assert _trace_choice_message("__import__('os').system('must-not-run')") == {}
    assert _trace_choice_message("Choices(**unknown)") == {}
    public_trace.cache_clear()


def test_reconstruct_pair_recovers_prefix_parameters_and_still_checks_prefix(tmp_path):
    child = {"id": "call", "name": "modify", "arguments": {}, "validation": {"valid": True}}
    prefix = {"action": {"actions": [child]}, "observation": "public result"}
    end = {"action": {"name": "finish", "arguments": {}}, "observation": "Done"}
    old, new = tmp_path / "old/sessions/one", tmp_path / "new/sessions/two"
    session(old, "bfcl", [prefix, end])
    session(new, "bfcl", [end], prefix=[prefix])
    write_json(
        old / "benchmark/bfcl_trace.json", {"turn_step_actions": [[[{**child, "arguments": {"target": "item"}}]]]}
    )
    history_path = new / "agent/litellm/trace.jsonl"
    history_path.parent.mkdir(parents=True)
    actual_prefix = [{"index": 0, "action": prefix["action"], "observation": '["Error: target is required"]'}]
    content = (
        "The environment has been restored by executing the following recorded actions in a fresh session.\n"
        + json.dumps(actual_prefix)
    )
    history_path.write_text(json.dumps({"request": {"messages": [{"role": "user", "content": content}]}}))
    row = {"benchmark": "bfcl", "original_session_dir": str(old), "replay_session_dir": str(new)}
    public_trace.cache_clear()
    pair = reconstruct_pair(row)
    assert pair["original"][0]["action"]["actions"][0]["arguments"] == {"target": "item"}
    assert pair["replay"][0]["action"]["actions"][0]["arguments"] == {}
    assert "validation" not in pair["replay"][0]["action"]["actions"][0]
    assert pair["replay"][0]["observation"] == '["Error: target is required"]'
    assert pair["replay"][0]["observation"] != pair["original"][0]["observation"]
    config_path = new.parent.parent / "run/config.json"
    config = json.loads(config_path.read_text())
    config["replay_prefix"]["actions"][0]["action"]["actions"][0]["arguments"] = {"target": "other"}
    write_json(config_path, config)
    with pytest.raises(ValueError, match="Recorded replay prefix differs"):
        reconstruct_pair(row)
    public_trace.cache_clear()


def test_task_holdouts_hit_pair_target_without_leaking_shared_tasks():
    rows = [{"task_id": task} for task, size in (("x", 5), ("y", 3), ("z", 2)) for _ in range(size)]
    tasks, counts = holdout_tasks(rows, 0.2, 42)
    assert tasks == {"z"}
    assert counts["val_pairs"] == 2
    assert counts["train_pairs"] == 8
    assert holdout_tasks(rows, 0.2, 42) == (tasks, counts)


def test_prepare_dataset_filters_seed_balances_letters_and_freezes_disjoint_tasks(tmp_path, monkeypatch):
    from exgentic.agents.premise import reward_data

    rows = [
        {
            "seed": seed,
            "benchmark": slug,
            "task_id": f"{slug}-{i}",
            "original_session_dir": f"/{seed}/{slug}/{i}/old",
            "replay_session_dir": f"/{seed}/{slug}/{i}/new",
            "original_score": 0,
            "replay_score": 1,
        }
        for seed in (42, 43)
        for slug in BENCHMARKS
        for i in range(10)
    ]
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    common = [{"action": "shared"}]
    monkeypatch.setattr(
        reward_data,
        "reconstruct_pair",
        lambda _row: {
            "task": "public task",
            "context": {},
            "public_tools": [],
            "original": [*common, {"action": "failed"}],
            "replay": [*common, {"action": "succeeded"}],
        },
    )
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "test": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    result = prepare_dataset(path, tmp_path / "data", tokenizer_path=tokenizer_path)
    assert len(result["records"]) == 30
    assert all(record["provenance"]["seed"] == 42 for record in result["records"])
    for slug in BENCHMARKS:
        assert result["counts"][slug]["val_pairs"] == 2
        for split in ("train", "val"):
            bucket = [r for r in result["records"] if r["benchmark"] == slug and r["split"] == split]
            assert sum(r["label"] == "A" for r in bucket) == sum(r["label"] == "B" for r in bucket)
            for record in bucket:
                visible = json.loads((tmp_path / "data/pairs" / f"{record['id']}.json").read_text())
                assert visible["common_prefix"] == common
                winner_suffix = visible["trajectory_a" if record["label"] == "A" else "trajectory_b"]
                assert winner_suffix == [{"action": "succeeded"}]
                assert not any(key in visible for key in ("score", "label", "benchmark", "task_id", "provenance"))
    train = {r["task_id"] for r in result["records"] if r["split"] == "train"}
    val = {r["task_id"] for r in result["records"] if r["split"] == "val"}
    assert not train & val
    assert set(training_order(result["records"], "sequential", 42)) == set(
        training_order(result["records"], "interleaved", 42)
    )
    with pytest.raises(FileExistsError):
        prepare_dataset(path, tmp_path / "data", tokenizer_path=tokenizer_path)


def test_mutation_can_correct_rules_and_rejects_benchmark_memorization():
    parent = [{"id": "rule-00001", "content": "Check observed effects."}]
    rules, next_id = mutate_rules(
        parent,
        2,
        {
            "operations": [
                {
                    "type": "UPDATE",
                    "id": "rule-00001",
                    "content": "Prefer observed effects matching the requested outcomes.",
                },
                {"type": "ADD", "content": "Prefer verified recovery over an unresolved rejected action."},
            ]
        },
        [],
    )
    assert next_id == 3
    assert len(rules) == 2
    assert parent[0]["content"] == "Check observed effects."
    for bad in (
        "For AppWorld, prefer completion.",
        "Prefer tool do_thing.",
        "Trust task task-xyz.",
        "Verify state with show_drafts.",
        "Do not penalize 'finish'.",
        "Check constraints, e.g. nationality.",
        "Prefer trying several solver algorithms.",
        "When an error occurs, the agent should execute another operation.",
    ):
        with pytest.raises(ValueError):
            mutate_rules(parent, 2, {"operations": [{"type": "ADD", "content": bad}]}, ["do_thing", "task-xyz"])


def test_mutation_has_no_playbook_capacity_limit_and_cannot_delete():
    parent = [{"id": f"rule-{i:05d}", "content": "Prefer confirmed outcomes. " * 20} for i in range(1, 101)]
    before = copy.deepcopy(parent)
    rules, next_id = mutate_rules(
        parent,
        101,
        {"operations": [{"type": "ADD", "content": "Prefer confirmed recovery over unresolved errors."}]},
        [],
    )
    assert len(rules) == 101
    assert sum(len(rule["content"]) for rule in rules) > 12000
    assert next_id == 102
    assert rules[:100] == before
    with pytest.raises(ValueError, match="deletion is disabled"):
        mutate_rules(parent, 101, {"operations": [{"type": "DELETE", "id": "rule-00001"}]}, [])
    assert parent == before


def test_judge_prompt_has_no_training_feedback_and_reflection_is_single_turn():
    visible = {"task": "public request", "common_prefix": [], "trajectory_a": [], "trajectory_b": []}
    messages = judge_messages(visible, [])
    assert json.loads(messages[-1]["content"]) == {"criteria": [], "comparison": visible}
    feedback = json.loads(
        analysis_messages("reflection", visible, [], {"choice": "A", "reason": "evidence"}, "B")[-1]["content"]
    )
    assert feedback["training_feedback"] == {"preferred": "B", "reward": 0}
    assert "No span selection" in analysis_messages("reflection", visible, [], {"choice": "A"}, "B")[0]["content"]


class FakeLLM:
    model, timeout, request_retries, output_retries = "fake", 600, 1, 5

    def __init__(self):
        self.calls = []

    def ask(self, messages, stage, **_kwargs):
        self.calls.append((stage, messages))
        if stage == "judge":
            return {"choice": "A", "reason": "The requested state is visibly achieved."}
        if stage == "reflection":
            return dict.fromkeys(("diagnosis", "evidence", "criterion", "limits", "uncertainty"), "grounded")
        return {"operations": [{"type": "ADD", "content": "Prefer visible effects matching requested outcomes."}]}


def small_dataset(tmp_path):
    dataset = tmp_path / "data"
    records = []
    for slug in BENCHMARKS:
        for split in ("train", "val"):
            pair_id = f"{slug}-{split}"
            visible = {
                "task": "public task",
                "context": {},
                "public_tools": [],
                "common_prefix": [],
                "trajectory_a": [{"action": "achieves task"}],
                "trajectory_b": [{"action": "fails task"}],
            }
            write_json(dataset / "pairs" / f"{pair_id}.json", visible)
            records.append(
                {
                    "id": pair_id,
                    "benchmark": slug,
                    "task_id": pair_id,
                    "split": split,
                    "label": "A",
                    "input_sha256": digest(visible),
                }
            )
    write_json(
        dataset / "manifest.json",
        {
            "records": records,
            "source_seed": 42,
            "counts": {slug: {"train_pairs": 1, "val_pairs": 1} for slug in BENCHMARKS},
        },
    )
    return dataset


def test_bandit_replays_current_and_past_decisions_without_validation_training(tmp_path):
    llm = FakeLLM()
    pipeline = RewardPipeline(small_dataset(tmp_path), tmp_path / "run", "sequential", 42, llm)
    first = pipeline.step()
    second = pipeline.step()
    assert first["status"] == second["status"] == "updated"
    assert len(second["candidates"]) == 9
    assert len(pipeline.beam) == 3
    assert [sum(c["parent_rank"] == rank for c in second["candidates"]) for rank in (1, 2, 3)] == [3, 3, 3]
    assert second["past_ids"] == [first["pair_id"]]
    assert all(child["current_result"]["reward"] == 1 for child in second["candidates"])
    assert all(child["past_results"][0]["pair_id"] == first["pair_id"] for child in second["candidates"])
    assert all(pipeline.records[pair_id]["split"] == "train" for pair_id in pipeline.memory)
    resumed = RewardPipeline(pipeline.dataset, pipeline.output, "sequential", 42, llm)
    assert resumed.cursor == 2
    assert resumed.beam == pipeline.beam


@pytest.mark.parametrize("parent_count,children_per_parent", [(2, 2), (3, 3)])
def test_requested_beams_replay_ten_past_pairs_and_use_only_official_preference_reward(
    tmp_path, parent_count, children_per_parent
):
    from collections import Counter

    dataset = small_dataset(tmp_path)
    manifest = json.loads((dataset / "manifest.json").read_text())
    original = next(row for row in manifest["records"] if row["split"] == "train")
    visible = json.loads((dataset / "pairs" / f"{original['id']}.json").read_text())
    for i in range(10):
        record = {**original, "id": f"extra-{i}", "task_id": f"extra-{i}"}
        manifest["records"].append(record)
        write_json(dataset / "pairs" / f"{record['id']}.json", visible)
    write_json(dataset / "manifest.json", manifest)
    llm = FakeLLM()
    pipeline = RewardPipeline(
        dataset,
        tmp_path / "run",
        "sequential",
        42,
        llm,
        parent_count=parent_count,
        children_per_parent=children_per_parent,
    )
    for _ in range(11):
        pipeline.step()
    before_calls = len(llm.calls)
    result = pipeline.step()
    assert len(result["past_ids"]) == len(set(result["past_ids"])) == 10
    assert all(pipeline.records[pair_id]["split"] == "train" for pair_id in result["past_ids"])
    candidates = parent_count * children_per_parent
    assert len(result["candidates"]) == candidates
    assert len(pipeline.beam) == parent_count
    assert Counter(stage for stage, _messages in llm.calls[before_calls:]) == {
        "judge": parent_count + candidates * 11,
        "reflection": candidates,
        "mutation": candidates,
    }
    assert all(set(value) == {"current", "past"} for value in result["selection"]["objectives"].values())
    assert pipeline.config["alignment_scoring"] is False
    assert pipeline.config["max_rules"] is pipeline.config["max_rule_chars"] is None
    pair_id = result["pair_id"]
    pipeline.records[pair_id]["label"] = "B"
    assert pipeline.scored_judge(pair_id, [])["reward"] == 0
    pipeline.records[pair_id]["label"] = "A"
    assert pipeline.scored_judge(pair_id, [])["reward"] == 1


def test_prompt_revision_preserves_state_and_rejects_other_config_changes(tmp_path, monkeypatch):
    from exgentic.agents.premise import reward_prompts

    llm = FakeLLM()
    pipeline = RewardPipeline(small_dataset(tmp_path), tmp_path / "run", "sequential", 42, llm)
    pipeline.step()
    before = (pipeline.output / "state.json").read_bytes()
    monkeypatch.setattr(reward_prompts, "REFLECTOR_SYSTEM", reward_prompts.REFLECTOR_SYSTEM + " Keep responses short.")
    with pytest.raises(ValueError, match="configuration differs"):
        RewardPipeline(pipeline.dataset, pipeline.output, "sequential", 42, llm)
    assert (pipeline.output / "state.json").read_bytes() == before
    revised = RewardPipeline(pipeline.dataset, pipeline.output, "sequential", 42, llm, revise_prompts=True)
    assert revised.cursor == pipeline.cursor
    assert revised.beam == pipeline.beam
    assert revised.memory == pipeline.memory
    assert revised.rng.getstate() == pipeline.rng.getstate()
    revision = json.loads(next((pipeline.output / "prompt_revisions").glob("*.json")).read_text())
    assert revision["previous_state"] == json.loads(before)
    assert revision["new_config"] == revised.config
    with pytest.raises(ValueError, match="configuration differs"):
        RewardPipeline(pipeline.dataset, pipeline.output, "sequential", 43, llm, revise_prompts=True)


def test_rejected_mutations_record_reason_and_do_not_change_beam(tmp_path):
    class InvalidMutationLLM(FakeLLM):
        def ask(self, messages, stage, **kwargs):
            if stage == "mutation":
                return {"operations": [{"type": "ADD", "content": "Prefer effects, e.g. a concrete case."}]}
            return super().ask(messages, stage, **kwargs)

    pipeline = RewardPipeline(small_dataset(tmp_path), tmp_path / "run", "sequential", 42, InvalidMutationLLM())
    beam = copy.deepcopy(pipeline.beam)
    result = pipeline.step()
    assert result["status"] == "skipped"
    assert pipeline.beam == beam
    assert len(result["errors"]) == 9
    assert all(row["message"] == "Rule contains a concrete example or function identifier" for row in result["errors"])


def test_validation_reports_per_benchmark_accuracy_and_cannot_change_training(tmp_path):
    llm = FakeLLM()
    pipeline = RewardPipeline(small_dataset(tmp_path), tmp_path / "run", "sequential", 42, llm)
    before = (pipeline.output / "state.json").read_bytes()
    beam = copy.deepcopy(pipeline.beam)
    queue = RewardValidationQueue(workers=2)
    queue.enqueue(pipeline, pipeline.output / "checkpoints/checkpoint_00000.json")
    queue.pool.shutdown(wait=True)
    queue.poll()
    result = json.loads((pipeline.output / "validation/cp00000.json").read_text())
    assert result["complete"]
    assert all(result["by_benchmark"][slug]["accuracy"] == 1 for slug in BENCHMARKS)
    assert pipeline.beam == beam
    assert (pipeline.output / "state.json").read_bytes() == before
    assert not any(stage != "judge" for stage, _messages in llm.calls)


def test_output_truncation_gets_five_additional_unchanged_calls(tmp_path):
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="length")], usage=None)

    llm = RewardLLM("http://unused", "fake", tmp_path, threading.BoundedSemaphore(1))
    llm.local.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    with pytest.raises(RewardRequestError, match="six identical requests"):
        llm.ask([{"role": "user", "content": "test"}], "judge")
    assert len(calls) == 6
    assert all(call == calls[0] for call in calls)
    assert calls[0]["max_tokens"] == 2048


def test_feedback_budget_is_per_turn_includes_marker_and_preserves_agent_actions():
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "test": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    truncator = FeedbackTruncator(tokenizer)
    action = {"name": "call", "arguments": {"text": "agent input " * 1000}}
    steps = [
        {"action": action, "observation": "test " * 800},
        {"action": action, "observation": "test " * 700},
        {"action": action, "observation": "test " * 400},
    ]
    clipped = truncator.steps(steps)
    assert all(step["action"] == action for step in clipped)
    assert all(len(tokenizer.encode(step["observation"], add_special_tokens=False).ids) <= 500 for step in clipped)
    assert sum("환경 feedback 생략됨" in step["observation"] for step in clipped) == 2
    assert clipped[2]["observation"] == steps[2]["observation"]
    assert sum(len(tokenizer.encode(step["observation"]).ids) for step in clipped) > 1000
    assert truncator.truncated_turns == 2


def test_controller_finishes_both_settings_and_all_frozen_validation_checkpoints(tmp_path, monkeypatch):
    from scripts.premise import run_reward_agent as controller

    dataset = small_dataset(tmp_path)
    monkeypatch.setattr(controller, "RewardLLM", lambda *_args, **_kwargs: FakeLLM())
    original_sleep = controller.time.sleep
    monkeypatch.setattr(controller.time, "sleep", lambda _seconds: original_sleep(0.001))
    output = tmp_path / "experiment"
    assert controller.run(output, dataset, "http://unused", "fake", 42, 2, 600) == 0
    complete = json.loads((output / "complete.json").read_text())
    assert complete["success"]
    for mode in ("sequential", "interleaved"):
        directory = output / f"s42_{mode}"
        assert json.loads((directory / "state.json").read_text())["cursor"] == 3
        summaries = sorted((directory / "validation").glob("cp*.json"))
        assert len(summaries) == 4
        for summary in summaries:
            value = json.loads(summary.read_text())
            assert value["complete"]
            assert all(row["accuracy"] == 1 for row in value["by_benchmark"].values())


def test_comparison_shares_request_limit_and_frozen_data_between_four_training_runs(tmp_path, monkeypatch):
    from scripts.premise import run_reward_agent as controller

    slots_seen = []

    def fake_llm(_base_url, _model, _traces, slots, **_kwargs):
        slots_seen.append(slots)
        return FakeLLM()

    monkeypatch.setattr(controller, "RewardLLM", fake_llm)
    original_sleep = controller.time.sleep
    monkeypatch.setattr(controller.time, "sleep", lambda _seconds: original_sleep(0.001))
    dataset = small_dataset(tmp_path)
    output = tmp_path / "comparison"
    assert controller.run_comparison(output, dataset, "http://unused", "fake", 42, 4, 600) == 0
    assert len(slots_seen) == 4
    assert all(slots is slots_seen[0] for slots in slots_seen)
    assert json.loads((output / "complete.json").read_text())["success"]
    configs = {}
    for parents, children in [(2, 2), (3, 3)]:
        for mode in ("sequential", "interleaved"):
            directory = output / f"p{parents}_c{children}/s42_{mode}"
            state = json.loads((directory / "state.json").read_text())
            configs[parents, mode] = state["config"]
            assert state["cursor"] == 3
            assert len(state["beam"]) == parents
            assert state["config"]["children_per_parent"] == children
            assert state["config"]["past_count"] == 10
            assert state["config"]["mutation_operations"] == ["ADD", "UPDATE"]
            assert state["config"]["alignment_scoring"] is False
            assert len(list((directory / "validation").glob("cp*.json"))) == 4
    for mode in ("sequential", "interleaved"):
        assert {k: v for k, v in configs[2, mode].items() if k not in ("parent_count", "children_per_parent")} == {
            k: v for k, v in configs[3, mode].items() if k not in ("parent_count", "children_per_parent")
        }
