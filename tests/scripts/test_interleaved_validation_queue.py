# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import json
from collections import Counter
from types import SimpleNamespace

import pytest

from scripts.utils import evaluate_interleaved_validation_queue as runner


@pytest.fixture
def launch(tmp_path, monkeypatch):
    splits = {}
    for seed in (42, 43, 44):
        for benchmark, count in [("bfcl", 50), ("appworld", 50), ("terminalbench2", 30)]:
            splits[f"{benchmark}_{seed}"] = {
                "train": [f"{benchmark}-s{seed}-train-{i}" for i in range(count)],
                "val": [f"{benchmark}-s{seed}-val-{i}" for i in range(count)],
                "test": [f"{benchmark}-s{seed}-test-{i}" for i in range(count - (benchmark == "terminalbench2"))],
            }
    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps(splits))
    monkeypatch.setattr(runner, "SPLITS", split_path)
    ace_root, premise_root = tmp_path / "ace", tmp_path / "premise"
    monkeypatch.setattr(runner, "ACE_ROOT", ace_root)
    monkeypatch.setattr(runner, "PREMISE_ROOT", premise_root)
    for seed in (42, 43, 44):
        for point in runner.POINTS:
            ace = ace_root / f"test_evaluation/snapshots/interleaved_s{seed}/checkpoint_{point:03d}.json"
            ace.parent.mkdir(parents=True, exist_ok=True)
            ace.write_text(json.dumps({
                "store_id": "ace_interleaved_global", "session_count": point,
                "training_task_ordinal": point, "playbook": f"ACE seed {seed} checkpoint {point}",
            }))
            premise = premise_root / f"s{seed}_interleaved_t3/checkpoints/checkpoint_{point:03d}.json"
            premise.parent.mkdir(parents=True, exist_ok=True)
            premise.write_text(json.dumps({
                "config": {"mode": "interleaved", "seed": seed}, "cursor": point,
                "states": {"global": {"beam": [
                    {"playbook": f"PREMiSE rank 1 seed {seed} checkpoint {point}"},
                    {"playbook": f"PREMiSE rank 2 seed {seed} checkpoint {point}"},
                ]}},
            }))
    output = tmp_path / "evaluation"
    output.mkdir()
    return runner.prepare(output, 150, "http://localhost:18006/v1")


def test_common_queue_covers_all_validation_tasks_and_six_checkpoints(launch):
    groups = launch["groups"]
    records = {g["group"]: [None] * len(g["task_ids"]) for g in groups}
    jobs = list(runner.queue_jobs(groups, records))
    assert launch["total_tasks"] == len(jobs) == 780
    assert launch["checkpoint_count"] == 6
    assert Counter(job["benchmark"] for job in jobs) == {"bfcl": 300, "appworld": 300, "terminalbench2": 180}
    assert Counter((job["method"], job["checkpoint"]) for job in jobs) == {
        (method, point): 130 for method in runner.METHODS for point in runner.POINTS
    }
    assert len({(job["method"], job["checkpoint"], job["benchmark"], job["task_id"]) for job in jobs}) == 780
    assert {job["benchmark"] for job in jobs[:150]} == set(runner.BENCHMARKS)
    assert all(job["split"] == "val" and "-val-" in job["task_id"] for job in jobs)
    assert jobs == list(runner.queue_jobs(groups, records))


@pytest.fixture
def multi_seed_launch(launch, tmp_path):
    output = tmp_path / "seeds42_43"
    output.mkdir()
    return runner.prepare(output, 150, "http://localhost:18006/v1", seeds=(42, 43),
                          max_tokens=8192, model_num_retries=5)


def test_seed42_and_43_share_one_queue_with_their_own_validation_splits(multi_seed_launch):
    groups = multi_seed_launch["groups"]
    rows = {g["group"]: [None] * len(g["task_ids"]) for g in groups}
    jobs = list(runner.queue_jobs(groups, rows))
    assert len(jobs) == multi_seed_launch["total_tasks"] == 1560
    assert multi_seed_launch["checkpoint_count"] == 12
    assert multi_seed_launch["seeds"] == [42, 43]
    assert Counter(job["benchmark"] for job in jobs) == {"bfcl": 600, "appworld": 600, "terminalbench2": 360}
    assert Counter((job["method"], job["seed"], job["checkpoint"]) for job in jobs) == {
        (method, seed, point): 130 for method in runner.METHODS for seed in (42, 43) for point in runner.POINTS
    }
    assert len({(job["method"], job["seed"], job["checkpoint"], job["benchmark"], job["task_id"])
                for job in jobs}) == 1560
    assert all(f"-s{job['seed']}-val-" in job["task_id"] for job in jobs)


@pytest.mark.parametrize("method", ["ace", "premise"])
def test_reused_worker_keeps_policy_separate_for_each_seed(multi_seed_launch, tmp_path, monkeypatch, method):
    monkeypatch.setattr(runner, "_PLAYBOOKS", {})
    policies = []
    for seed in (42, 43):
        group = next(g for g in multi_seed_launch["groups"] if (
            g["method"], g["seed"], g["checkpoint"], g["benchmark"]
        ) == (method, seed, 50, "bfcl"))
        job = {**group, "index": 0, "task_id": group["task_ids"][0]}
        config = runner.task_config(job, tmp_path, f"val{seed}-test")
        policy = config.agent_kwargs["initial_playbook" if method == "ace" else "playbook"]
        assert f"seed {seed}" in policy
        policies.append(policy)
        settings = config.agent_kwargs["model_settings"].model_dump(exclude_none=True)
        expected_tokens = 8192 if method == "ace" else 2048
        assert (settings["max_tokens"], settings["temperature"], settings["num_retries"]) == (expected_tokens, 0.0, 5)
    assert policies[0] != policies[1]
    assert len(runner._PLAYBOOKS) == 2


def test_multi_seed_scores_keep_130_task_denominator_for_every_checkpoint(multi_seed_launch, tmp_path):
    groups = multi_seed_launch["groups"]
    records = {g["group"]: [{"status": "completed", "score": float(g["seed"] == 42),
                             "partial_score": float(g["seed"] == 42)} for _ in g["task_ids"]] for g in groups}
    runner.save_scores(tmp_path, groups, records)
    scores = runner.shared.read_json(tmp_path / "scores.json")["scores"]
    assert len(scores) == 12
    assert len({(s["method"], s["seed"], s["checkpoint"]) for s in scores}) == 12
    assert all(s["processed"] == s["total"] == 130 and s["complete"] for s in scores)
    assert all(s["hard"] == float(s["seed"] == 42) for s in scores)
    assert all(s["by_benchmark"]["terminalbench2"]["total"] == 30 for s in scores)


def test_resume_rejects_result_from_another_seed(multi_seed_launch, tmp_path):
    group = multi_seed_launch["groups"][0]
    row = {key: value for key, value in group.items() if key != "task_ids"}
    row.update(seed=43 if group["seed"] == 42 else 42, task_id=group["task_ids"][0], status="completed")
    runner.shared.write_json(tmp_path / "results" / group["group"] / "000.json", row)
    with pytest.raises(ValueError, match="seed or checkpoint identity differs"):
        runner.run(tmp_path, multi_seed_launch)


def test_resume_does_not_repeat_completed_or_failed_tasks(launch):
    groups = launch["groups"]
    records = {g["group"]: [None] * len(g["task_ids"]) for g in groups}
    removed = {(groups[0]["group"], 0), (groups[-1]["group"], 29)}
    records[groups[0]["group"]][0] = {"status": "completed"}
    records[groups[-1]["group"]][29] = {"status": "error"}
    jobs = list(runner.queue_jobs(groups, records))
    assert len(jobs) == 778
    assert removed.isdisjoint((job["group"], job["index"]) for job in jobs)


def test_ten_checkpoint_queue_excludes_existing_50_100_130(monkeypatch, request):
    points = tuple(point for point in range(10, 131, 10) if point not in (50, 100, 130))
    monkeypatch.setattr(runner, "POINTS", points)
    launch = request.getfixturevalue("launch")
    groups = launch["groups"]
    records = {g["group"]: [None] * len(g["task_ids"]) for g in groups}
    jobs = list(runner.queue_jobs(groups, records))
    assert launch["checkpoint_count"] == 20
    assert launch["total_tasks"] == len(jobs) == 2600
    assert {job["checkpoint"] for job in jobs} == set(points)
    assert Counter(job["benchmark"] for job in jobs) == {"bfcl": 1000, "appworld": 1000, "terminalbench2": 600}
    assert Counter((job["method"], job["checkpoint"]) for job in jobs) == {
        (method, point): 130 for method in runner.METHODS for point in points
    }


def test_premise_uses_global_rank_one_and_rejects_wrong_seed():
    snapshot = {"config": {"mode": "interleaved", "seed": 44}, "cursor": 50,
                "states": {"global": {"beam": [{"playbook": "rank one"}, {"playbook": "rank two"}]}}}
    assert runner.checkpoint_playbook("premise", snapshot, 50) == "rank one"
    snapshot["config"]["seed"] = 43
    with pytest.raises(ValueError, match="seed/mode/checkpoint"):
        runner.checkpoint_playbook("premise", snapshot, 50)


@pytest.mark.parametrize("limit", [2048, 8192])
def test_frozen_configs_use_shared_actor_limits(launch, tmp_path, monkeypatch, limit):
    monkeypatch.setattr(runner, "_PLAYBOOKS", {})
    for group in launch["groups"]:
        job = {**group, "task_id": group["task_ids"][0], "max_tokens": limit}
        config = runner.task_config(job, tmp_path, "test-run")
        assert config.task_ids == [group["task_ids"][0]]
        assert config.max_workers == 1
        assert "system_prompt_override" not in config.agent_kwargs
        if group["method"] == "ace":
            assert config.agent_kwargs["evaluation_mode"]
            assert not config.agent_kwargs["training_time"]
        else:
            assert "rank 1" in config.agent_kwargs["playbook"]
        settings = config.agent_kwargs["model_settings"]
        if hasattr(settings, "model_dump"):
            settings = settings.model_dump(exclude_none=True)
        if group["benchmark"] == "terminalbench2":
            assert settings.get("max_tokens") == (None if group["method"] == "ace" else 2048)
            assert config.benchmark_kwargs["actor"] == "terminus2"
            assert config.benchmark_kwargs["max_interactions"] == config.max_steps == 40
        else:
            assert settings["max_tokens"] == (limit if group["method"] == "ace" else min(limit, 2048))
            assert config.max_steps == (30 if group["benchmark"] == "bfcl" else 40)


@pytest.mark.parametrize("method", ["ace", "premise"])
def test_worker_caps_legacy_output_settings_in_actual_completion(launch, tmp_path, monkeypatch, method):
    from exgentic.interfaces.lib import api

    monkeypatch.setattr(runner, "_OUTPUT", tmp_path)
    monkeypatch.setattr(runner, "_PLAYBOOKS", {})
    group = next(g for g in launch["groups"] if g["method"] == method and g["benchmark"] == "bfcl")
    job = {**group, "index": 0, "task_id": group["task_ids"][0], "max_tokens": 8192,
           "evaluation_label": "max8192", "model_num_retries": 5, "transport_additional_retries": 5,
           "transport_retry_after": 0}
    import litellm

    from exgentic.core.experiment_limits import ActorTurnBudget, experiment_limits

    requests = []

    def complete(**kwargs):
        requests.append(kwargs)
        return litellm.ModelResponse(
            choices=[{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            usage={"prompt_tokens": 100, "completion_tokens": 100},
        )

    def execute(*, config):
        settings = config.agent_kwargs["model_settings"]
        if hasattr(settings, "model_dump"):
            settings = settings.model_dump(exclude_none=True)
        assert settings["max_tokens"] == (8192 if method == "ace" else 2048)
        assert settings["temperature"] == 0.0 and settings["num_retries"] == 5
        assert config.run_id.endswith("_max8192")
        actor = SimpleNamespace(_actor_turn_budget=ActorTurnBudget(experiment_limits("bfcl")), _step_count=1)
        runner.shared.actor_request_completion(actor, {"max_tokens": 8192})
        session = SimpleNamespace(session_id="session1", status=SimpleNamespace(value="success"), score=1.0, steps=1)
        return SimpleNamespace(session_results=[session])

    monkeypatch.setattr(litellm, "completion", complete)
    monkeypatch.setattr(api, "execute", execute)
    row = runner.evaluate_task(job)
    assert row["status"] == "completed", row.get("error")
    assert row["score"] == 1.0
    assert requests[0]["max_tokens"] == 2048
    audit = runner.shared.read_json(tmp_path / "request_audits" / group["group"] / "000.json")
    assert audit["max_tokens"] == 2048 and audit["model_calls"] == 1
    assert audit["transport_additional_retries"] == 5


def test_queue_rejects_manifest_and_worker_token_limit_mismatch(launch, tmp_path):
    launch["max_tokens"]["appworld"] = 8192
    with pytest.raises(ValueError, match="output limit differs from the launch manifest"):
        runner.run(tmp_path, launch)


def test_new_queue_uses_only_free_slots_until_existing_queue_finishes(tmp_path):
    runner.shared.write_json(tmp_path / "launch_manifest.json", {"workers": 150, "total_tasks": 387})
    dependency = {"output": str(tmp_path), "workers": 150, "global_workers": 150, "total_tasks": 387,
                  "launch_sha256": runner.shared.digest((tmp_path / "launch_manifest.json").read_bytes())}
    launch = {"workers": 150, "capacity_dependency": dependency}
    runner.shared.write_json(tmp_path / "status.json", {
        "phase": "running", "workers": 150, "active_tasks": 49, "queued_tasks": 0,
        "processed_tasks": 338, "total_tasks": 387,
    })
    assert runner.available_workers(launch) == 101
    runner.shared.write_json(tmp_path / "status.json", {
        "phase": "running", "workers": 150, "active_tasks": 120, "queued_tasks": 10,
        "processed_tasks": 257, "total_tasks": 387,
    })
    assert runner.available_workers(launch) == 0
    runner.shared.write_json(tmp_path / "status.json", {
        "phase": "complete", "workers": 150, "active_tasks": 0, "queued_tasks": 0,
        "processed_tasks": 387, "total_tasks": 387,
    })
    assert runner.available_workers(launch) == 150
    runner.shared.write_json(tmp_path / "launch_manifest.json", {"workers": 151, "total_tasks": 387})
    with pytest.raises(ValueError, match="inputs changed"):
        runner.available_workers(launch)
