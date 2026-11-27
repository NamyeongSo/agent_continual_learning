# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import json
from types import SimpleNamespace

import pytest

from scripts.premise import run_bfcl_alignment as runner


def frozen_plan(output, checkpoints=runner.CHECKPOINTS):
    points = []
    for seed, checkpoint in checkpoints:
        key = f"s{seed}_cp{checkpoint:03d}"
        points.append({"key": key, "seed": seed, "checkpoint": checkpoint, "model": "openai/model",
                       "train_task_ids": [f"train-{index}" for index in range(50)],
                       "seen_task_ids": [f"train-{index}" for index in range(checkpoint)],
                       "past_task_ids": [f"train-{index}" for index in range(10)],
                       "validation_task_ids": [f"val-{index}" for index in range(10)],
                       "validation_full_task_ids": [f"val-{index}" for index in range(50)]})
        for context in ("parent", *(f"candidate_{index:02d}" for index in range(1, 6))):
            runner.save_immutable(output / "frozen" / key / f"{context}.json",
                                  {"playbook": {"playbook": context}})
    return {"checkpoints": points, "max_steps": 20}


def test_queue_requires_completed_proposals_and_schedules_every_candidate(tmp_path):
    plan = frozen_plan(tmp_path)
    with pytest.raises(ValueError, match="Generate all candidates"):
        runner.make_jobs(tmp_path, plan)
    runner.save_immutable(tmp_path / "generation_complete.json", {"candidates": 25})
    jobs = runner.make_jobs(tmp_path, plan)
    assert len(jobs) == len({job["key"] for job in jobs}) == 2750
    assert sum(job["context"] == "parent" for job in jobs) == 250
    assert sum(job["partition"] == "val" for job in jobs) == 1250
    for point in plan["checkpoints"]:
        for context in (f"candidate_{index:02d}" for index in range(1, 6)):
            group = [job for job in jobs if job["checkpoint_key"] == point["key"] and job["context"] == context]
            assert {job["task_id"] for job in group if job["partition"] == "train"} == set(point["train_task_ids"])
            assert {job["task_id"] for job in group if job["partition"] == "val"} == set(
                point["validation_full_task_ids"]
            )


@pytest.mark.parametrize("effort", [None, "low"])
def test_worker_uses_twenty_steps_and_does_not_mix_contexts_at_the_same_checkpoint(tmp_path, monkeypatch, effort):
    from exgentic.interfaces.lib import api

    plan = frozen_plan(tmp_path)
    runner.save_immutable(tmp_path / "generation_complete.json", {"candidates": 25})
    jobs = [job for job in runner.make_jobs(tmp_path, plan) if job["checkpoint_key"] == "s42_cp010"
            and job["partition"] == "train" and job["index"] == 0]
    monkeypatch.setattr(runner, "_OUTPUT", tmp_path)
    monkeypatch.setattr(runner, "_CONTEXTS", {})
    monkeypatch.setattr(runner, "_SETTINGS", {"max_tokens": 8192, "transport_additional_retries": 5,
                                            "temperature": 0.0, "request_timeout": 600,
                                            **({"reasoning_effort": effort} if effort is not None else {})})
    captured = []

    def execute(*, config):
        captured.append(config)
        return SimpleNamespace(session_results=[SimpleNamespace(
            session_id="session", status=SimpleNamespace(value="completed"), steps=20,
            execution_time=1, score=1.0,
        )])

    monkeypatch.setattr(api, "execute", execute)
    rows = [runner.evaluate_pair(job) for job in jobs]
    assert all(row["status"] == "completed" for row in rows)
    assert all(config.max_steps == 20 and config.max_workers == 1 for config in captured)
    assert all(config.agent_kwargs["enable_thinking"] is False for config in captured)
    assert all(config.agent_kwargs["model_settings"].reasoning_effort == effort for config in captured)
    assert {config.agent_kwargs["playbook"] for config in captured} == {job["context"] for job in jobs}
    assert len({config.run_id for config in captured}) == 6
    assert all((tmp_path / "results" / f"{job['key']}.json").exists() for job in jobs)


def test_frozen_plan_rejects_step_budget_changes_on_resume(tmp_path):
    path = tmp_path / "launch_manifest.json"
    runner.save_immutable(path, {"max_steps": 20})
    with pytest.raises(ValueError, match="Frozen inputs differ"):
        runner.save_immutable(path, {"max_steps": 50})
    assert json.loads(path.read_text()) == {"max_steps": 20}


def test_tie_selection_uses_uniform_expectation_and_constant_scores_have_no_correlation():
    assert runner.selected_improvement([0.8, 0.8, 0.7], [0.1, 0.3, 0.9]) == pytest.approx(0.2)
    assert runner.correlation([0.1, 0.3, 0.9], [0.5, 0.5, 0.5]) is None
    assert runner.ranks([0.1, 0.1, 0.3]) == [1.5, 1.5, 3.0]
    assert runner.correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)


def test_full_validation_evaluations_retain_the_shared_ten_task_comparison(tmp_path):
    plan = frozen_plan(tmp_path)
    runner.save_immutable(tmp_path / "generation_complete.json", {"candidates": 25})
    records = {}
    for job in runner.make_jobs(tmp_path, plan):
        score = 0.4 if job["context"] == "parent" else 0.5
        if job["partition"] == "val":
            score = float(job["index"] < 10)
        records[job["key"]] = {**job, "score": score, "status": "completed"}
    summary = runner.analyze(tmp_path, plan, records)
    candidate = summary["checkpoints"][0]["candidates"][0]
    assert candidate["s_val"] == 1.0
    assert candidate["s_val_full"] == pytest.approx(0.2)
    assert candidate["delta_train"] == pytest.approx(0.1)
    assert candidate["delta_train_without_past"] == pytest.approx(0.1)
    assert candidate["delta_seen_without_past"] is None
    assert summary["execution_errors"] == 0


def test_additional_checkpoints_schedule_225_candidates_without_the_original_points(tmp_path):
    prior = set(runner.CHECKPOINTS)
    additional = []
    for seed in (42, 43, 44):
        pool = [step for step in range(10, 50) if (seed, step) not in prior]
        additional.extend((seed, pool[round(index * (len(pool) - 1) / 14)]) for index in range(15))
    assert len(set(additional)) == 45 and not set(additional) & prior
    plan = frozen_plan(tmp_path, tuple(additional))
    runner.save_immutable(tmp_path / "generation_complete.json", {"candidates": 225})
    jobs = runner.make_jobs(tmp_path, plan)
    assert len(jobs) == len({job["key"] for job in jobs}) == 24750
    assert sum(job["context"] == "parent" for job in jobs) == 2250
    assert sum(job["partition"] == "val" for job in jobs) == 11250


def test_generation_reuses_every_frozen_candidate_and_reports_the_actual_count(tmp_path):
    plan = frozen_plan(tmp_path, ((42, 11), (43, 13)))
    plan.update(base_url="http://localhost:12345/v1", request_timeout=600)
    for point in plan["checkpoints"]:
        runner.save_immutable(tmp_path / "frozen" / point["key"] / "proposal_trace.json", {})
        for index in range(1, 6):
            path = tmp_path / "frozen" / point["key"] / f"candidate_{index:02d}.json"
            path.write_text(json.dumps({"valid": True, "parent_rank": 1, "playbook": {"playbook": "unchanged"}}))
    runner.generate_candidates(tmp_path, plan)
    assert json.loads((tmp_path / "generation_complete.json").read_text())["candidates"] == 10
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["candidate_count"] == status["completed_candidates"] == 10


def test_generation_resume_only_retries_the_failed_mutation_with_its_frozen_reflection(tmp_path, monkeypatch):
    from exgentic.agents.premise.pipeline import PremisePipeline

    plan = frozen_plan(tmp_path, ((42, 11),))
    plan.update(base_url="http://localhost:12345/v1", request_timeout=600)
    directory = tmp_path / "frozen" / "s42_cp011"
    (directory / "candidate_05.json").unlink()
    runner.save_immutable(directory / "proposal_trace.json", {"benchmark": "bfcl"})
    reflection = {"parent_rank": 1, "id": "reflection_5"}
    runner.save_immutable(directory / "reflection_05.json", reflection)
    runner.save_immutable(directory / "candidate_05_error.json", {
        "valid": False, "errors": ["mutator: ValueError: LLM response has no JSON object"],
    })
    for index in range(1, 5):
        path = directory / f"candidate_{index:02d}.json"
        path.write_text(json.dumps({"valid": True, "parent_rank": 1, "playbook": {"playbook": "unchanged"}}))
    called = []

    def candidate(self, frozen_reflection, trace, parent):
        called.append((frozen_reflection, trace, parent))
        return {"valid": True, "parent_rank": 1, "playbook": {"playbook": "new"}}

    monkeypatch.setattr(PremisePipeline, "_one_candidate", candidate)
    runner.generate_candidates(tmp_path, plan)
    assert len(called) == 1 and called[0][0] == reflection
    saved = json.loads((directory / "candidate_05.json").read_text())
    assert saved["generation_additional_attempts"] == 1
    assert json.loads((tmp_path / "generation_complete.json").read_text())["candidates"] == 5


def test_sequential_source_parent_requires_a_bfcl_only_history_and_next_task():
    snapshot = {"cursor": 10, "config": {"mode": "sequential", "order": [["bfcl", str(i)] for i in range(11)]},
                "states": {"global": {"beam": [{"playbook": "gemma-parent"}], "memory": []}}}
    assert runner.checkpoint_parent(snapshot, 10) == {"playbook": "gemma-parent"}
    snapshot["config"]["order"][10][0] = "appworld"
    with pytest.raises(ValueError, match="only BFCL"):
        runner.checkpoint_parent(snapshot, 10)
    snapshot["config"]["order"][10][0] = "bfcl"
    snapshot["states"]["global"]["memory"] = [{"benchmark": "appworld"}]
    with pytest.raises(ValueError, match="another benchmark"):
        runner.checkpoint_parent(snapshot, 10)


@pytest.mark.parametrize(("model", "error", "accept_failure"), [
    ("openai/openai/gpt-oss-20b", "LiteLLM completion missing assistant content.", True),
    ("openai/openai/gpt-oss-20b", "InternalServerError: unexpected tokens remaining in message header", True),
    ("openai/openai/gpt-oss-20b", "InternalServerError: upstream service unavailable", False),
    ("openai/other-model", "unexpected tokens remaining in message header", False),
])
def test_proposal_resume_keeps_gpt_protocol_failures_but_rejects_server_outages(
    tmp_path, monkeypatch, model, error, accept_failure,
):
    from concurrent.futures import ThreadPoolExecutor

    from exgentic.agents.premise import pipeline

    plan = frozen_plan(tmp_path, ((42, 10),))
    plan.update(base_url="http://localhost:12345/v1", max_tokens=8192, temperature=0.0,
                request_timeout=600, transport_additional_retries=5)
    point = plan["checkpoints"][0]
    point.update(model=model, proposal_task="train-10", proposal_trace_mode="rerun_parent_with_requested_step_budget")
    parent = tmp_path / "frozen" / point["key"] / "parent.json"
    job = {
        "key": f"{point['key']}/parent/proposal/000", "group": f"{point['key']}/parent/proposal",
        "index": 0, "checkpoint_key": point["key"], "seed": 42, "checkpoint": 10,
        "context": "parent", "partition": "proposal", "task_id": "train-10",
        "context_path": str(parent), "context_sha256": runner.execution.digest(parent.read_bytes()),
        "model": model, "max_steps": 20,
    }
    saved = {**job, "status": "error", "score": 0.0, "steps": 3, "session_id": "failed-model", "error": error}
    runner.save_immutable(tmp_path / "proposal_rollouts" / "results" / f"{job['key']}.json", saved)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", lambda **kwargs: ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(pipeline, "trajectory_from_session", lambda *args: {"score": 0.0, "actions": []})

    def reject_new_rollout(*args):
        pytest.fail("Resume must preserve the existing failed trajectory instead of resampling it")

    monkeypatch.setattr(runner, "evaluate_pair", reject_new_rollout)
    trace_path = parent.with_name("proposal_trace.json")
    if accept_failure:
        runner.materialize_proposal_traces(tmp_path, plan)
        trace = json.loads(trace_path.read_text())
        assert trace["score"] == 0.0 and trace["agent_protocol_error"] == error
    else:
        with pytest.raises(RuntimeError, match="Proposal parent rollout failed"):
            runner.materialize_proposal_traces(tmp_path, plan)
        assert not trace_path.exists()
