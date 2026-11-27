# SPDX-License-Identifier: Apache-2.0
"""Regression checks for pairing, native GT reflection, and honest progress."""

from types import SimpleNamespace

import pytest

from exgentic.agents.ace.ace_instance import ACEAgentInstance
from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK, PlaybookStore
from scripts.ace import run_context_proposal_bottleneck as study


def test_reflection_analysis_handles_empty_initial_ceiling(tmp_path):
    from scripts.ace.analyze_context_proposal_bottleneck import ARMS, build_tables, plot, summarize

    tasks = ["partial", "mixed"]
    common = {"appworld_policy_sha256": "policy", "total_tests": 5}
    initial = [
        {**common, "task_id": task, "source_repeat": repeat, "passed_tests": count, "pct": count / 5,
         "success": count == 5}
        for task, counts in zip(tasks, ([1] * 5, [5, 4, 4, 4, 4]), strict=True)
        for repeat, count in enumerate(counts)
    ]
    evaluated = [
        {**common, "task_id": task, "source_repeat": repeat, "evaluation_repeat": evaluation_repeat,
         "arm": arm, "passed_tests": count, "pct": count / 5, "success": count == 5}
        for arm in ARMS
        for task, count in zip(tasks, (2, 4), strict=True)
        for repeat in range(5)
        for evaluation_repeat in range(3)
    ]
    task_rows, candidates = build_tables(tasks, initial, evaluated, "policy")
    summary = summarize(task_rows, candidates, initial, evaluated)
    assert summary["subgroups"]["initial_ceiling"] == {"tasks": 0}
    assert summary["subgroups"]["initial_below_ceiling"]["tasks"] == 2
    plot(tmp_path, task_rows, summary)
    assert (tmp_path / "selection_and_subgroups.png").is_file()


def test_queues_share_five_sources_and_keep_every_context_repeat():
    tasks = [f"task_{i}" for i in range(50)]
    initial, proposals, evaluation = (study.jobs_for_stage(tasks, stage) for stage in (1, 2, 3))
    assert [len(initial), len(proposals), len(evaluation)] == [250, 500, 2500]
    assert len(initial) + len(evaluation) == 2750
    for jobs in (initial, proposals, evaluation):
        assert len({j["id"] for j in jobs}) == len(jobs)
    for source in initial:
        paired = [j for j in proposals if j["source_id"] == source["id"]]
        assert {j["arm"] for j in paired} == set(study.ARMS)
        for proposal in paired:
            repeats = [j for j in evaluation if j["context_id"] == proposal["id"]]
            assert {j["evaluation_repeat"] for j in repeats} == set(range(5))
            assert all(j["task_id"] == source["task_id"] for j in repeats)


def test_pct_uses_exact_test_counts_and_keeps_official_success_separate(tmp_path):
    study.write(tmp_path / "benchmark/scores.json", {"test_tracker": {
        "num_tests": 3, "passes": [1, 2], "failures": [3], "success": False, "difficulty": 2,
    }})
    session = SimpleNamespace(details={}, status="unsuccessful", score=0.6667, success=False)
    result = study.pct_outcome(session, tmp_path)
    assert result["pct"] == 2 / 3
    assert result["success"] is False


def test_budget_failures_are_measured_zero_and_infrastructure_is_not(tmp_path):
    budget = SimpleNamespace(details={"session_metadata": {
        "experiment_budget_exceeded": True, "experiment_budget_failure": {"reason": "input_tokens"},
    }}, status="error", score=0.0, success=False)
    assert study.pct_outcome(budget, tmp_path)["pct"] == 0
    with pytest.raises(RuntimeError, match="No valid official outcome"):
        study.pct_outcome(SimpleNamespace(details={}, status="error", score=None, success=None), tmp_path)


def test_turn_cap_pct_outcome_keeps_partial_credit(tmp_path):
    study.write(tmp_path / "benchmark/scores.json", {"test_tracker": {
        "num_tests": 3, "passes": [1, 2], "failures": [3], "success": False, "difficulty": 2,
    }})
    session = SimpleNamespace(details={"session_metadata": {"turn_limit_reached": True,
                              "turn_cap_details": {"reason": "max_turns"}}},
                              status="unsuccessful", score=0.6667, success=False)
    result = study.pct_outcome(session, tmp_path)
    assert result["pct"] == 2 / 3 and result["turn_limit_reached"] is True


def test_rescored_reflector_feedback_replaces_artificial_zero_without_changing_history():
    from scripts.ace.migrate_turn_cap_pct import rescore_feedback

    original = {"actions": [{"step": 40, "action": "execute_python"}], "observations": ["saved output"],
                "training_feedback": {"score": 0, "session_metadata": {"experiment_budget_exceeded": True}}}
    outcome = {"pct": 7 / 24, "success": False, "passed_tests": 7, "total_tests": 24, "difficulty": 3,
               "evaluation": {"num_tests": 24}, "turn_cap_details": {"reason": "max_turns", "limit": 40}}
    updated = rescore_feedback(original, outcome, {"actor_limits": {"policy_version": "turncap-pct-test"}})
    assert updated["actions"] == original["actions"] and updated["observations"] == original["observations"]
    assert original["training_feedback"]["score"] == 0
    assert updated["training_feedback"]["score"] == 7 / 24
    assert updated["training_feedback"]["session_metadata"]["turn_limit_reached"] is True
    assert "experiment_budget_exceeded" not in updated["training_feedback"]["session_metadata"]


@pytest.mark.parametrize("with_gt", [False, True])
def test_native_reflection_only_adds_gt_to_requested_arm(monkeypatch, with_gt):
    instance = ACEAgentInstance("proposal-test", training_time=True)
    instance._store = PlaybookStore("proposal-test", DEFAULT_PLAYBOOK)
    instance.task = "Do the task"
    instance._action_log = [{"step": 1, "action": "execute_python", "arguments": "model trajectory"}]
    instance.receive_training_feedback({"score": 0.5, "success": False})
    captured = []
    monkeypatch.setattr(instance, "_llm_call_simple", lambda *args, **kwargs: (
        captured.append(args[1]) or '{"reasoning":"Native reflection","bullet_tags":[]}'
    ))
    reflection = instance._run_post_session_reflection(gt_trajectory="successful reference" if with_gt else None)
    assert reflection == "Native reflection"
    assert "model trajectory" in captured[0]
    assert '"score": 0.5' in captured[0]
    assert ("successful reference" in captured[0]) is with_gt
    assert ("**Reference Trajectory (GT):**" in captured[0]) is with_gt
    assert instance._store.playbook == DEFAULT_PLAYBOOK


def test_stage_barrier_stops_after_unresolved_jobs(tmp_path):
    from concurrent.futures import Future

    job = {"id": "task_r0", "stage": 1, "task_id": "task", "source_id": "task_r0", "source_repeat": 0}
    study.write(tmp_path / "stage1_queue.json", [job])
    protocol = {"infrastructure_attempts": 2, "progress_logging": {"heartbeat_interval_seconds": 30},
                "stages": [{"name": "initial_rollouts", "total_jobs": 1}]}

    class FailingPool:
        def __init__(self):
            self.attempts = []

        def submit(self, fn, output, job, attempt):
            self.attempts.append(attempt)
            future = Future()
            future.set_result({"ok": False, "error": "Unavailable environment"})
            return future

    pool = FailingPool()
    with pytest.raises(RuntimeError, match="next stage will not start"):
        study.run_stage(tmp_path, protocol, 1, pool, study.Progress(tmp_path, protocol))
    assert pool.attempts == [1, 2]
    status = study.read(tmp_path / "status.json")
    assert status["state"] == "error" and status["completed"] == 0 and status["errors"] == 1


def test_progress_counts_measured_zero_as_completed(tmp_path):
    protocol = {"stages": [{"name": "initial_rollouts", "total_jobs": 250}]}
    study.Progress(tmp_path, protocol).emit(1, [{"pct": 0.0}], [1] * 50, [1] * 199, [])
    status = study.read(tmp_path / "status.json")
    assert status["completed"] == 1 and status["percent"] == 0.4 and status["errors"] == 0
    assert "done=1/250" in (tmp_path / "progress.log").read_text()
