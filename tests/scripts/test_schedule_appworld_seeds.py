# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from types import SimpleNamespace

import pytest

from scripts.utils import schedule_appworld_seeds as scheduler


def test_partition_retains_two_methods_per_seed_and_does_not_mutate_jobs():
    jobs = [{"method": method, "seed": seed, "status": "running"}
            for method in ("ace", "premise") for seed in (42, 43, 44)]
    groups = scheduler.partition(jobs)
    assert {seed: len(group) for seed, group in groups.items()} == {42: 2, 43: 2, 44: 2}
    groups[43][0]["status"] = "paused"
    assert all(j["status"] == "running" for j in jobs)
    with pytest.raises(ValueError, match="matched seeds"):
        scheduler.partition(jobs[:-1])


def test_legacy_completion_requires_training_and_every_final_test_record(tmp_path):
    job = {"status": "completed", "output": str(tmp_path), "train_dir": str(tmp_path / "train"),
           "test_ids": ["first", "second"]}
    scheduler.write(tmp_path / "train/training_complete.json", {"complete": True})
    scheduler.write(tmp_path / "evaluation/summary.json", {"complete": True, "tasks": 2, "execution_errors": 0})
    scheduler.write(tmp_path / "evaluation/records/000.json", {"status": "completed", "score": 0})
    assert not scheduler.completed(job, "legacy")
    scheduler.write(tmp_path / "evaluation/records/001.json", {"status": "completed", "score": 1})
    assert scheduler.completed(job, "legacy")
    job["status"] = "evaluating"
    assert not scheduler.completed(job, "legacy")


def test_release_waits_for_all_six_seed42_experiments(tmp_path, monkeypatch):
    batches = []
    for index in range(3):
        group = tmp_path / str(index)
        jobs = [{"id": f"{index}-{method}", "status": "completed"} for method in range(2)]
        scheduler.write(group / "progress.json", {"jobs": jobs})
        batches.append({"kind": "native", "groups": {"42": str(group)}})
    monkeypatch.setattr(scheduler, "completed", lambda j, kind: j["status"] == "completed")
    plan = {"batches": batches}
    assert scheduler.first_seed_finished(plan)
    progress = scheduler.read(tmp_path / "2/progress.json")
    progress["jobs"][1]["status"] = "running"
    scheduler.write(tmp_path / "2/progress.json", progress)
    assert not scheduler.first_seed_finished(plan)


def test_pause_resume_does_not_charge_a_launch_failure():
    progress = {"jobs": [{"status": "paused", "pause_previous_status": "training", "attempts": 1},
                         {"status": "paused", "pause_previous_status": "running", "launches": 1}]}
    scheduler.resumed_jobs(progress)
    assert all(j["status"] == "pending" for j in progress["jobs"])
    assert progress["jobs"][0]["attempts"] == 0
    assert progress["jobs"][1]["launches"] == 0


def test_spawn_identity_waits_for_the_child_to_finish_exec(monkeypatch):
    identities = iter([{"pid": 12, "command": ["parent"]}, {"pid": 12, "command": ["child"]}])
    monkeypatch.setattr(scheduler, "identify", lambda pid: next(identities))
    monkeypatch.setattr(scheduler.time, "sleep", lambda seconds: None)
    process = SimpleNamespace(pid=12, poll=lambda: None)
    assert scheduler.spawned_identity(process, ["child"])["command"] == ["child"]
