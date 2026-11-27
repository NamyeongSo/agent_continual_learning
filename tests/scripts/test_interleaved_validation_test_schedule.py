# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import fcntl

import pytest

from scripts.utils import evaluate_interleaved_validation_test_schedule as schedule


def candidates():
    return [{"method": method, "checkpoint": point, "complete": True, "total": 130,
             "hard": 0.4, "soft": 0.6}
            for method in schedule.runner.METHODS for point in range(10, 131, 10)]


def test_selection_considers_existing_checkpoints_and_uses_hard_then_soft():
    rows = candidates()
    for row in rows:
        if row["method"] == "ace" and row["checkpoint"] == 50:
            row.update(hard=0.8, soft=0.85)
        if row["method"] == "premise" and row["checkpoint"] in (100, 120):
            row.update(hard=0.6, soft=0.7 if row["checkpoint"] == 100 else 0.8)
        if row["method"] == "premise" and row["checkpoint"] == 130:
            row.update(hard=0.5, soft=0.95)
    assert {r["method"]: r["checkpoint"] for r in schedule.select_winners(rows)} == {"premise": 120, "ace": 50}


def test_test_selection_requires_all_26_complete_validation_scores():
    rows = candidates()
    rows[0]["complete"] = False
    with pytest.raises(ValueError, match="complete 130-task"):
        schedule.select_winners(rows)
    with pytest.raises(ValueError, match="all 26"):
        schedule.select_winners(candidates()[1:])


def test_predecessor_must_complete_and_release_workers_even_with_task_errors(tmp_path):
    shared = schedule.shared
    shared.write_json(tmp_path / "launch_manifest.json", {"total_tasks": 100})
    (tmp_path / "controller.lock").touch()
    digest = shared.digest((tmp_path / "launch_manifest.json").read_bytes())
    dependency = {"output": str(tmp_path), "launch_sha256": digest,
                  "total_tasks": 100, "checkpoint_count": 2}
    assert not schedule.predecessor_ready(dependency)
    shared.write_json(tmp_path / "validation_complete.json", {
        "all_tasks_terminal": True, "processed_tasks": 100, "checkpoint_count": 2, "execution_errors": 5,
    })
    shared.write_json(tmp_path / "status.json", {"phase": "complete", "processed_tasks": 100})
    with (tmp_path / "controller.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not schedule.predecessor_ready(dependency)
    assert schedule.predecessor_ready(dependency)


def test_two_selected_checkpoints_use_only_129_test_tasks_each(tmp_path):
    shared = schedule.shared
    splits, groups = {}, []
    winners = [{"method": "premise", "checkpoint": 120}, {"method": "ace", "checkpoint": 50}]
    for benchmark, count in [("bfcl", 50), ("appworld", 50), ("terminalbench2", 29)]:
        splits[f"{benchmark}_44"] = {
            "train": [f"{benchmark}-train"], "val": [f"{benchmark}-val"],
            "test": [f"{benchmark}-test-{i}" for i in range(count)],
        }
        for winner in winners:
            groups.append({**winner, "benchmark": benchmark, "group": "validation-group", "split": "val",
                           "task_ids": [f"{benchmark}-val"]})
    shared.write_json(tmp_path / "validation/task_split_manifest.json", splits)
    launch = schedule.prepare_test(tmp_path, {}, groups, winners)
    records = {g["group"]: [None] * len(g["task_ids"]) for g in launch["groups"]}
    jobs = list(schedule.runner.queue_jobs(launch["groups"], records))
    assert launch["total_tasks"] == len(jobs) == 258
    assert launch["checkpoint_count"] == 2
    assert all(job["split"] == "test" and "-test-" in job["task_id"] for job in jobs)
    assert launch["selected_checkpoints"] == {"premise": 120, "ace": 50}
