# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import json

from scripts.litellm.evaluate_bfcl_appworld_after_terminalbench import (
    ROOT,
    make_tasks,
    predecessor_finished,
    summarize,
    write,
)


def test_queue_has_150_unique_tasks_per_benchmark_without_split_resampling():
    manifest = json.loads((ROOT / "scripts/utils/task_splits/"
                           "bfcl_appworld_terminalbench2_fixed_balanced_order_seeds42_44_v1.json").read_text())
    tasks = make_tasks(manifest)
    assert len(tasks) == len({task["id"] for task in tasks}) == 300
    for benchmark in ("bfcl", "appworld"):
        assert sum(task["benchmark"] == benchmark for task in tasks) == 150
        for split in ("train", "val", "test"):
            selected = [task["task_id"] for task in tasks if task["benchmark"] == benchmark and task["split"] == split]
            assert selected == manifest[f"{benchmark}_42"][split]
    assert {task["split"] for task in tasks[:100]} == {"test"}


def test_dependency_requires_all_89_attempts_to_finish_and_preserves_errors(tmp_path):
    tasks = [{"task_id": f"task-{index}"} for index in range(89)]
    write(tmp_path / "config.json", {"tasks": tasks})
    write(tmp_path / "status.json", {"phase": "all_splits_parallel"})
    for task in tasks[:-1]:
        write(tmp_path / "records" / f"{task['task_id']}.json", {"status": "completed"})
    assert not predecessor_finished(tmp_path)
    write(tmp_path / "status.json", {"phase": "completed"})
    assert not predecessor_finished(tmp_path)
    write(tmp_path / "records/task-88.json", {"status": "infrastructure_error"})
    write(tmp_path / "status.json", {"phase": "infrastructure_errors"})
    assert predecessor_finished(tmp_path)
    write(tmp_path / "status.json", {"phase": "paused"})
    assert not predecessor_finished(tmp_path)


def test_missing_and_infrastructure_outcomes_are_not_scored_as_failures():
    tasks = [{"id": "bfcl:a", "benchmark": "bfcl", "task_id": "a", "split": "test"},
             {"id": "appworld:b", "benchmark": "appworld", "task_id": "b", "split": "test"},
             {"id": "appworld:c", "benchmark": "appworld", "task_id": "c", "split": "test"}]
    records = {"bfcl:a": {"status": "completed", "success": False},
               "appworld:b": {"status": "completed", "success": True},
               "appworld:c": {"status": "infrastructure_error"}}
    report = summarize({"tasks": tasks}, records)
    assert report["measured"] == 2 and report["complete"] is False
    assert report["benchmarks"]["appworld"]["all"]["score_percent"] == 100
    assert report["benchmarks"]["appworld"]["all"]["measured"] == 1
    assert report["benchmarks"]["bfcl"]["all"]["score_percent"] == 0
