# SPDX-License-Identifier: Apache-2.0
"""Check w/o-GT isolation, saved sign selection, stage counts and Gemma history rules."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ace import run_gemma4_train_only as study


def test_stage_queues_contain_only_without_gt_and_match_the_3500_actor_budget():
    tasks = [f"task_{i}" for i in range(50)]
    counts = []
    for stage, per_mode in ((1, 250), (2, 250), (3, 750)):
        jobs = study.stage_jobs(tasks, stage)
        assert len(jobs) == per_mode
        assert len({j["id"] for j in jobs}) == per_mode
        assert {j["task_id"] for j in jobs} == set(tasks)
        if stage > 1:
            assert {j["arm"] for j in jobs} == {"without_gt"}
        counts.append(2 * len(jobs))
    assert counts == [500, 500, 1500]
    assert counts[0] + counts[2] + 30 * 50 == 3500


def test_modes_share_the_requested_sampler_without_inheriting_qwen_penalties():
    for mode in study.MODES:
        settings = study.settings_for(mode)
        assert settings["temperature"] == 1 and settings["top_p"] == 0.95
        assert settings["extra_body"]["top_k"] == 64
        assert "presence_penalty" not in settings and "reasoning_effort" not in settings
        assert settings["extra_body"]["chat_template_kwargs"] == {
            "enable_thinking": mode == "thinking", "preserve_thinking": mode == "thinking",
        }
        assert settings["max_tokens"] is None


def test_optional_arm_scope_preserves_both_arm_defaults_and_validates_subsets():
    assert study.accumulation.protocol_arms({}) == ("without_gt", "with_gt")
    assert len(study.accumulation.conditions(42)) == 6
    assert len(study.accumulation.conditions(42, ("without_gt",))) == 3
    for arms in ([], ["wrong"], ["without_gt", "without_gt"]):
        with pytest.raises(ValueError, match="GT arm scope"):
            study.accumulation.protocol_arms({"arms": arms})


def test_advantage_subset_trains_12_contexts_and_skips_all_tied_tasks(tmp_path):
    tasks = [f"task_{i}" for i in range(50)]
    bank = {"without_gt": {task: [
        {"task_id": task, "source_repeat": i, "score_numerator": i if task != tasks[0] else 0,
         "score_denominator": 5, "mean_pct": i / 5 if task != tasks[0] else 0.0}
        for i in range(5)] for task in tasks}}
    runs = {}
    for mode in study.MODES:
        path = tmp_path / mode / "accumulation"
        runs[mode] = path
        study.base.write(path / "reflection_bank.json", bank)
        study.base.write(path / "protocol.json", {
            "arms": ["without_gt"], "task_orders": {str(seed): {
                "train": tasks, "val": ["val_only"], "test": ["test_only"],
            } for seed in study.SEEDS},
        })
    target = tmp_path / "advantage"
    jobs = study.advantage.prepare(target, runs)
    assert len(jobs) == 12
    for job in jobs:
        assert job["steps"] == 49 and job["condition"]["arm"] == "without_gt"
        choices = study.base.read(Path(job["path"]) / "conditions" / job["condition"]["id"] / "selections.json")
        assert {r["candidate"]["task_id"] for r in choices} == set(tasks[1:])
        assert all(r["advantage"] >= 0 for r in choices) if job["condition"]["mode"] == "best_advantage" else (
            all(r["advantage"] <= 0 for r in choices)
        )
    assert not (target / "thinking/with_gt").exists()


def test_no_evaluation_queue_is_created_before_all_learners_finish(tmp_path):
    job = {"path": str(tmp_path), "condition": {"id": "unfinished"}, "steps": 50}
    with pytest.raises(RuntimeError, match="Evaluation barrier"):
        study.evaluation_queue(tmp_path, [job])
    assert not (tmp_path / "control/evaluation_queue.json").exists()


def test_pinned_official_template_strips_old_thoughts_except_tool_calls():
    path = (study.ROOT / "outputs/ace_appworld_gemma4_26b_a4b_it_wo_gt_train_only_20261124"
            "/serving/chat_template.jinja")
    if not path.exists():
        pytest.skip("Requires the frozen Gemma official template used in this local experiment")
    rendered = study.history_canary(path.read_text())
    assert "KEEP_TOOL_CALL_THOUGHT" in rendered
    assert "DROP_ORDINARY_THOUGHT" not in rendered
    assert "DROP_FINISHED_THOUGHT" not in rendered


def test_actual_frozen_ace_retains_tool_thoughts_but_drops_ordinary_thoughts():
    from litellm import ModelResponse

    path = (study.ROOT / "outputs/ace_appworld_gemma4_26b_a4b_it_wo_gt_train_only_20261124"
            "/thinking/reflection/source_snapshot/src/exgentic/agents/ace/ace_instance.py")
    if not path.exists():
        pytest.skip("Requires the prepared Gemma frozen actor")
    spec = importlib.util.spec_from_file_location("exgentic.agents.ace.gemma_history_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actor = module.ACEAgentInstance.__new__(module.ACEAgentInstance)
    actor._step_count, actor._all_actions, actor._action_log, actor.messages = 0, [], [], []
    actor.model, actor._task_context_error = "openai/google/gemma-4-26B-A4B-it", None
    actor._observe = actor._log_observation = lambda *_: None
    actor._assistant_tools = lambda: []
    actor._add_message = lambda message: actor.messages.append(message)
    actor._registry = SimpleNamespace(tool_calls_to_action=lambda calls: [])
    actor._logger = SimpleNamespace(info=lambda *args: None)
    actor._cost = SimpleNamespace(update_cost_from_tokens=lambda *args: None)
    responses = iter([
        ModelResponse(choices=[{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "reasoning_content": "KEEP_TOOL_THOUGHT",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "report", "arguments": '{"value":"hello"}'}}],
        }}], usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
        ModelResponse(choices=[{"finish_reason": "stop", "message": {
            "role": "assistant", "content": "FINAL_REPLY", "reasoning_content": "DROP_ORDINARY_THOUGHT",
        }}], usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
    ])
    actor._completion = lambda **kwargs: next(responses)
    actor.react(None)
    assert actor.messages[-1]["reasoning_content"] == "KEEP_TOOL_THOUGHT"
    actor.react(None)
    assert actor.messages[-1]["content"] == "FINAL_REPLY"
    assert not actor.messages[-1].get("reasoning_content")


def test_100_worker_continuation_preserves_a_saved_official_result(tmp_path, monkeypatch):
    study.base.write(tmp_path / "plan.json", {"workers": 100})
    (tmp_path / "control").mkdir()
    item = {"root": str(tmp_path), "job": {"stage": 1, "id": "saved"}, "sampler": "thinking"}
    path = study.base.job_dir(item["root"], item["job"]) / "result.json"
    row = {"pct": 0.0, "success": False}
    study.base.write(path, row)
    original = study.training.digest(path)
    pools = []

    class Pool:
        def __init__(self, max_workers, **kwargs):
            pools.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, *args):
            pytest.fail("A completed official outcome must not be rerun")

    monkeypatch.setattr(study, "ProcessPoolExecutor", Pool)
    result = study.run_queue(tmp_path, "initial_rollouts", [item], lambda _: pytest.fail("No new actor"),
                             lambda _: study.base.read(path))
    assert pools == [100] and list(result.values()) == [row]
    assert study.training.digest(path) == original
    assert study.base.read(tmp_path / "status.json")["worker_limit"] == 100
    study.base.write(tmp_path / "plan.json", {"workers": 50})
    assert study.worker_limit(tmp_path) == 50
