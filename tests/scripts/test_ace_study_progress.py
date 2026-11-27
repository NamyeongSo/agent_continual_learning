# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the independent ACE study monitor."""

import json
import time

import pytest

from scripts.research_journal.ace_study_progress import ace_studies
from scripts.research_journal.live_progress import LiveProgress
from scripts.research_journal.store import JournalStore


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def study(tmp_path, monkeypatch):
    root = tmp_path / "outputs/gemma31"
    store = JournalStore(tmp_path / "journal", [tmp_path])
    collector = LiveProgress(store, base_dir=tmp_path)
    write(root / "ace_study_monitor.json", {"kind": "ace_context_proposal_train_only_v1",
                                            "journal_id": "gemma-record", "label": "Gemma31",
                                            "initial_metrics_file": "analysis/initial.json"})
    write(root / "plan.json", {"model": "google/gemma-4-31B-it", "workers": 50,
                                "training_contexts": 30, "train_evaluations": 1500,
                                "modes": ["thinking", "nonthinking"]})
    write(root / "control/process.json", {"pid": 123})
    write(root / "status.json", {"state": "running", "phase": "updated_context_rollouts",
                                  "completed": 290, "running": 50, "errors": [],
                                  "messages": "DO_NOT_EXPOSE"})
    for mode in ("thinking", "nonthinking"):
        for stage in (1, 2):
            write(root / mode / f"reflection/stage{stage}_results.json", [{"private": "DO_NOT_EXPOSE"}] * 250)
    monkeypatch.setattr("scripts.research_journal.live_progress.live_pid", lambda pid, _: pid == 123)
    return root, collector


def read(study):
    root, collector = study
    return ace_studies(root.parent.parent, collector.study_files, collector.allowed, lambda pid, _: pid == 123)[0]


def test_reports_phase_barriers_without_counting_missing_scores_as_zero(study):
    row = read(study)
    assert row["journal_id"] == "gemma-record" and row["controller_alive"]
    assert [s["done"] for s in row["stages"]] == [500, 500, 290, 0, 0]
    assert row["stages"][2]["current"] and row["stages"][2]["running"] == 50
    assert row["initial_scores"] == row["final_scores"] == []
    assert "DO_NOT_EXPOSE" not in json.dumps(row)
    assert "DO_NOT_EXPOSE" not in str(study[1].study_files.entries)


def test_saved_official_tgc_is_distinct_from_pct(study):
    root, _ = study
    write(root / "analysis/initial.json", [{"mode": "thinking", "pct_percent": 99.99,
                                          "tgc_percent": 0.0, "pass_at_5_percent": 0.0,
                                          "private": "DO_NOT_EXPOSE"}])
    scores = read(study)["initial_scores"]
    assert scores[0]["pct_percent"] == 99.99 and scores[0]["tgc_percent"] == 0.0
    assert "private" not in scores[0]


def test_study_progress_refreshes_while_legacy_snapshot_is_cached(study, monkeypatch):
    root, collector = study
    collector.cached = {"generated_at": "old"}
    collector.cached_at = time.monotonic()
    monkeypatch.setattr(collector, "checkpoint_snapshot", lambda: [])
    monkeypatch.setattr("scripts.research_journal.live_progress.evaluations", lambda *_: [])
    assert collector.snapshot()["ace_studies"][0]["stages"][2]["done"] == 290
    write(root / "status.json", {"state": "running", "phase": "updated_context_rollouts", "completed": 333})
    result = collector.snapshot()
    assert result["generated_at"] == "old"
    assert result["ace_studies"][0]["stages"][2]["done"] == 333


def test_dead_controller_is_not_running(study):
    root, _ = study
    write(root / "control/process.json", {"pid": 999})
    row = read(study)
    assert row["state"] == "stopped" and not row["controller_alive"]
    assert all(s["running"] == 0 for s in row["stages"])


@pytest.mark.parametrize("field", ["running_contexts", "active_contexts"])
def test_training_counts_context_receipts_instead_of_curator_steps(study, field):
    root, _ = study
    write(root / "status.json", {"state": "running", "phase": "all_context_training",
                                 "completed": 123, "completed_steps": 123, field: 2, "training_worker_limit": 2})
    jobs = [{"path": str(root / "thinking/accumulation"), "condition": {"id": name}}
            for name in ("s42_without_gt_best", "s42_without_gt_worst")]
    write(root / "control/training_plan.json", jobs)
    write(root / "thinking/accumulation/conditions/s42_without_gt_best/training_completed.json", {"done": True})
    stage = read(study)["stages"][3]
    assert stage["done"] == 1 and stage["total"] == 30 and stage["running"] == 2
    assert read(study)["workers"] == 2


def test_completed_requires_receipt_and_seed_summary(study):
    root, _ = study
    write(root / "status.json", {"state": "completed", "phase": "train_only_complete"})
    write(root / "control/process.json", {"pid": 999})
    assert read(study)["state"] != "completed"
    write(root / "completed.json", {"contexts": 30, "train_evaluations": 1500})
    write(root / "train_final_mean_std.json", [{"pct_mean": 0.5, "tgc_mean": 0.1}] * 10)
    assert read(study)["state"] == "completed"


def test_initial_artifact_cannot_escape_allowed_roots(study, tmp_path):
    root, _ = study
    write(root / "ace_study_monitor.json", {"kind": "ace_context_proposal_train_only_v1",
                                           "initial_metrics_file": str(tmp_path.parent / "outside.json")})
    assert read(study)["initial_scores"] == []
