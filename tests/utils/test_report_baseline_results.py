# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Regression checks for the read-only experiment score report."""

import json

import pytest

from scripts.utils import report_baseline_results as report_module
from scripts.utils.report_baseline_results import (
    aggregate,
    multibenchmark_runs,
    offline_row,
    online_windows,
    premise_scores,
    render_table,
)


def test_online_windows_use_checkpoint_intervals_and_wait_for_checkpoint():
    order = [("bfcl", "a"), ("appworld", "b"), ("bfcl", "c"), ("appworld", "d")]
    scores = {1: 1.0, 2: 0.2, 3: 0.0}
    rows = online_windows(order, scores, (2, 4), cursor=3)
    by_key = {(row["checkpoint"], row["benchmark"]): row for row in rows}

    assert by_key[2, "bfcl"]["score"] == 1.0
    assert by_key[2, "appworld"]["score"] == 0.0
    assert by_key[2, "bfcl"]["complete"]
    assert by_key[4, "bfcl"]["score"] == 0.0
    assert not by_key[4, "bfcl"]["complete"]  # Its rollout exists, but checkpoint 4 does not yet exist.
    assert by_key[4, "appworld"]["scored"] == 0
    finished = online_windows(order, scores, (2, 4), cursor=4)
    assert next(row for row in finished if row["checkpoint"] == 4 and row["benchmark"] == "bfcl")["complete"]
    assert next(row for row in finished if row["checkpoint"] == 4 and row["benchmark"] == "appworld") == {
        "checkpoint": 4,
        "benchmark": "appworld",
        "score": None,
        "scored": 0,
        "planned": 1,
        "complete": True,
    }


def test_cumulative_online_keeps_earlier_scores_and_waits_for_checkpoint():
    order = [("bfcl", "a"), ("appworld", "b"), ("bfcl", "c"), ("appworld", "d")]
    scores = {1: 1.0, 2: 0.2, 3: 0.0}
    rows = online_windows(order, scores, (2, 4), cursor=3, cumulative=True)
    by_key = {(row["checkpoint"], row["benchmark"]): row for row in rows}

    assert by_key[2, "bfcl"]["score"] == 1.0
    assert by_key[2, "appworld"]["score"] == 0.0
    assert by_key[2, "bfcl"]["complete"]
    assert by_key[4, "bfcl"] == {
        "checkpoint": 4, "benchmark": "bfcl", "score": 0.5,
        "scored": 2, "planned": 2, "complete": False,
    }
    assert by_key[4, "appworld"] == {
        "checkpoint": 4, "benchmark": "appworld", "score": 0.0,
        "scored": 1, "planned": 2, "complete": False,
    }
    finished = online_windows(order, scores, (2, 4), cursor=4, cumulative=True)
    assert all(row["complete"] for row in finished)
    assert next(row for row in finished if row["checkpoint"] == 4 and row["benchmark"] == "appworld")["scored"] == 1
    with pytest.raises(ValueError, match="Invalid checkpoint"):
        online_windows(order, scores, (4, 2), cursor=4, cumulative=True)


def test_online_appworld_tgc_preserves_coverage_and_other_benchmarks():
    order = [("appworld", "full"), ("appworld", "partial"),
             ("bfcl", "partial"), ("terminalbench2", "partial"), ("appworld", "skipped")]
    rows = online_windows(order, {1: 1.0, 2: 0.9, 3: 0.6, 4: 0.5}, (5,), cursor=5)
    by_benchmark = {row["benchmark"]: row for row in rows}
    assert by_benchmark["appworld"] == {
        "checkpoint": 5, "benchmark": "appworld", "score": 0.5,
        "scored": 2, "planned": 3, "complete": True,
    }
    assert by_benchmark["bfcl"]["score"] == 0.6
    assert by_benchmark["terminalbench2"]["score"] == 0.5


def test_premise_appworld_tgc_counts_any_fully_successful_parent(tmp_path):
    folder = tmp_path / "iterations"
    folder.mkdir()
    for index, parents in enumerate(([0.8, 1.0], [0.5, 0.9]), start=1):
        (folder / f"iter_{index:04d}.json").write_text(json.dumps({
            "benchmark": "appworld", "task_id": str(index),
            "parents": [{"score": score} for score in parents],
        }))
    order = [("appworld", "1"), ("appworld", "2")]
    scores = premise_scores(tmp_path, 2, order)
    assert online_windows(order, scores, (2,), cursor=2)[0]["score"] == 0.5


@pytest.mark.parametrize("method", ["ace", "premise", "ps_st_pareto"])
@pytest.mark.parametrize("mode", ["interleaved", "sequential"])
def test_online_report_uses_cumulative_ranges_only_for_interleaved(tmp_path, monkeypatch, method, mode):
    root = tmp_path / method
    root.mkdir()
    order = [(slug, f"{slug}-{index}") for slug, count in
             (("appworld", 50), ("bfcl", 50), ("terminalbench2", 30)) for index in range(count)]
    scores = {index: 1.0 if index <= 50 else 0.0 if index <= 100 else 0.5 for index in range(1, 131)}
    (root / "launch_manifest.json").write_text(json.dumps({
        "method": method,
        "jobs": [{"mode": mode, "seed": 42, "output_dir": str(tmp_path / "run"), "task_order": order}],
    }))
    config = tmp_path / "monitor.json"
    config.write_text(json.dumps({"learning_roots": [str(root)]}))
    monkeypatch.setattr(report_module, "ace_scores", lambda _run, _order: scores)
    monkeypatch.setattr(report_module, "premise_scores", lambda _run, _cursor, _order: scores)
    monkeypatch.setattr(report_module.monitor, "collect", lambda _config: {
        "learning": [{"method": method, "mode": mode, "seed": 42, "status": "trained",
                      "cursor": 130, "total": 130, "counts": {"appworld": 50, "bfcl": 50, "terminalbench2": 30}}],
        "checkpoint_evaluations": [],
    })

    run = multibenchmark_runs(config, "official")[0]
    by_key = {(row["checkpoint"], row["benchmark"]): row for row in run["online"]}
    expected = {(50, "appworld"): (1.0, 50), (100, "bfcl"): (0.0, 50), (130, "terminalbench2"): (0.5, 30)}
    if mode == "interleaved":
        expected.update({(100, "appworld"): (1.0, 50), (130, "appworld"): (1.0, 50), (130, "bfcl"): (0.0, 50)})
    assert set(by_key) == set(expected)
    for key, (score, count) in expected.items():
        assert by_key[key]["score"] == score
        assert by_key[key]["scored"] == by_key[key]["planned"] == count
        assert by_key[key]["complete"]


def test_premise_online_uses_best_available_parent_and_skips_only_when_both_missing(tmp_path):
    folder = tmp_path / "iterations"
    folder.mkdir()
    (folder / "iter_0001.json").write_text(
        json.dumps({"benchmark": "bfcl", "task_id": "a", "parents": [{"score": 0.25}, {"score": 1.0}]})
    )
    (folder / "iter_0002.json").write_text(
        json.dumps({"benchmark": "bfcl", "task_id": "b", "parents": [None, {"score": 1.0}]})
    )
    (folder / "iter_0003.json").write_text(
        json.dumps({"benchmark": "bfcl", "task_id": "c", "parents": [{"score": 0.4}, None]})
    )
    (folder / "iter_0004.json").write_text(json.dumps({"benchmark": "bfcl", "task_id": "d", "parents": [None, None]}))

    order = [("bfcl", task) for task in "abcd"]
    assert premise_scores(tmp_path, 4, order) == {1: 1.0, 2: 1.0, 3: 0.4}
    with pytest.raises(ValueError, match="differs from planned task order"):
        premise_scores(tmp_path, 2, [("bfcl", "a"), ("bfcl", "wrong")])


def test_aggregate_excludes_partial_online_and_error_bearing_offline():
    runs = []
    for seed, score, clean in ((42, 0.5, True), (43, 0.7, True), (44, 0.9, False)):
        runs.append(
            {
                "method": "ace",
                "setting": "sequential",
                "train_benchmark": None,
                "seed": seed,
                "online": [{"checkpoint": 50, "benchmark": "bfcl", "score": score, "complete": clean}],
                "offline": [{"checkpoint": 50, "benchmark": "bfcl", "score": score, "clean_complete": clean}],
            }
        )

    for metric in ("online", "offline"):
        row = aggregate(runs, metric)[0]
        assert row["seeds"] == [42, 43]
        assert row["n"] == 2
        assert row["mean"] == pytest.approx(0.6)
        assert row["std"] == pytest.approx(0.14142135623730948)


def test_offline_interim_mean_includes_error_and_partial_seed_records():
    runs = []
    for seed, task_mean, processed, errors, clean in (
        (42, 0.3, 2, 1, False), (43, 0.5, 3, 1, False), (44, 0.7, 3, 0, True)
    ):
        runs.append({
            "method": "ace", "setting": "sequential", "train_benchmark": None, "seed": seed,
            "offline": [{
                "checkpoint": 130, "benchmark": "terminalbench2",
                "score": task_mean if clean else None, "task_mean_score": task_mean,
                "processed": processed, "total": 3, "errors": errors,
                "clean_complete": clean, "protocol_errors": [],
            }],
        })

    summary = aggregate(runs, "offline")[0]
    assert summary["n"] == 1
    assert summary["mean"] == pytest.approx(0.7)
    assert summary["interim_n"] == 3
    assert summary["interim_task_mean"] == pytest.approx(0.5)
    assert summary["interim_task_std"] == pytest.approx(0.2)
    assert (summary["interim_processed"], summary["interim_total"], summary["interim_errors"]) == (8, 9, 2)
    table = render_table({"runs": runs, "offline_aggregates": [summary]}, "offline")
    assert any("중간 task 평균 50.00 ± 20.00% (n=3/3, task=8/9, errors=2)" in line for line in table)


def test_monitor_reports_observed_task_mean_before_summary_exists(tmp_path):
    for session, task, score, status in (("a", "a", 1.0, "success"), ("b", "b", 0.0, "error")):
        path = tmp_path / "run" / "sessions" / session / "results.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"task_id": task, "score": score, "status": status}))

    group = report_module.monitor.group_progress(
        [tmp_path / "summary.json"], ["a", "b", "c"], identity={}, checkpoint_hash=None
    )

    assert group["processed"] == 2
    assert group["total"] == 3
    assert group["errors"] == 1
    assert group["score"] == pytest.approx(0.5)


def test_interleaved_offline_keeps_terminalbench_when_checkpoint_task_is_bfcl(tmp_path, monkeypatch):
    root = tmp_path / "ace"
    root.mkdir()
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"benchmark_score": 9 / 29}))
    (root / "launch_manifest.json").write_text(json.dumps({
        "agent": "ace",
        "jobs": [{"mode": "interleaved", "seed": 42, "output_dir": str(tmp_path / "run"),
                  "task_order": [["bfcl", f"task-{index}"] for index in range(130)]}],
    }))
    config = tmp_path / "monitor.json"
    config.write_text(json.dumps({"learning_roots": [str(root)]}))
    monkeypatch.setattr(report_module, "ace_scores", lambda _run, _order: {})
    monkeypatch.setattr(report_module.monitor, "collect", lambda _config: {
        "learning": [{"method": "ace", "mode": "interleaved", "seed": 42,
                      "status": "trained", "cursor": 130, "total": 130, "counts": {"bfcl": 130}}],
        "checkpoint_evaluations": [{
            "method": "ace", "mode": "interleaved", "seed": 42,
            "checkpoint": 130, "benchmark": "terminalbench2", "result_path": str(summary),
            "score": 9 / 29, "processed": 29, "total": 29, "errors": 0,
            "complete": True, "clean_complete": True, "owner_pids": [],
        }],
    })

    runs = multibenchmark_runs(config, "official")

    assert runs[0]["own_benchmark_at_checkpoint"][130] == "bfcl"
    assert runs[0]["offline"][0]["benchmark"] == "terminalbench2"
    assert runs[0]["offline"][0]["score"] == pytest.approx(9 / 29)


def test_offline_table_shows_error_only_checkpoint_result():
    run = {
        "method": "ace", "setting": "sequential", "train_benchmark": None, "seed": 42,
        "offline": [{"checkpoint": 130, "benchmark": "terminalbench2", "score": None,
                     "processed": 29, "total": 29, "errors": 2, "clean_complete": False}],
    }
    table = render_table({"runs": [run], "offline_aggregates": aggregate([run], "offline")}, "offline")

    assert any("| ace | sequential | 130 | terminalbench2 | — [29/29] errors=2" in line for line in table)


def test_offline_official_appworld_score_differs_from_task_mean(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"benchmark_score": 0.25, "mean_task_score": 0.7}))
    group = {
        "checkpoint": 50,
        "benchmark": "appworld",
        "result_path": str(path),
        "score": 0.7,
        "processed": 4,
        "total": 4,
        "errors": 0,
        "complete": True,
        "clean_complete": True,
    }
    assert offline_row(group, "ace", "official")["score"] == 0.25
    assert offline_row(group, "ace", "task-mean")["score"] == 0.7

    path.write_text(json.dumps({"results": [{"score": 1.0}, {"score": 0.75}, {"score": 0.9}, {"score": 0.7}]}))
    assert offline_row(group, "premise", "official")["score"] == 0.25
