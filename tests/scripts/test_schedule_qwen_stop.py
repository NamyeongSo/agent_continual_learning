# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from datetime import UTC, datetime

import pytest

from scripts.utils import schedule_qwen_stop as stop
from scripts.utils.schedule_qwen_pause import resumed_jobs


def test_stop_deadline_uses_the_explicit_korean_offset():
    expected = datetime(2026, 10, 26, 8, 30, tzinfo=UTC).timestamp()
    assert stop.deadline("2026-10-26T17:30:00+09:00") == expected
    with pytest.raises(ValueError, match="explicit UTC offset"):
        stop.deadline("2026-10-26T17:30:00")


def test_stop_covers_later_seed_groups_even_if_seed42_finishes_before_the_deadline(tmp_path):
    groups = {str(seed): str(tmp_path / f"seed{seed}") for seed in (42, 43, 44)}
    stop.write(tmp_path / "seed_schedule/schedule.json", {"batches": [{"groups": groups}]})
    roots, _ = stop.roots_and_cohort({"cohort": str(tmp_path), "original_roots": [str(tmp_path / "original")]})
    assert set(roots) == {tmp_path / "original", *(tmp_path / f"seed{seed}" for seed in (42, 43, 44))}


def test_pause_preserves_completed_and_already_waiting_results_and_launch_budgets(tmp_path, monkeypatch):
    monkeypatch.setattr(stop, "preserve_interrupted", lambda *args: None)
    done = {"id": "done", "status": "completed", "score": 0}
    waiting = {"id": "waiting", "status": "paused", "pause_previous_status": "running", "launches": 1}
    progress = {"jobs": [done.copy(), waiting.copy(),
                         {"id": "active", "status": "training", "attempts": 2, "pid": 11},
                         {"id": "pending", "status": "pending", "attempts": 0}]}
    paused = stop.pause_jobs(progress, tmp_path, 0)
    assert paused["jobs"][:2] == [done, waiting]
    assert all(j["status"] == "paused" and "pid" not in j for j in paused["jobs"][2:])
    resumed = resumed_jobs(paused)
    assert resumed["jobs"][2] == {"id": "active", "status": "pending", "attempts": 1}
    assert resumed["jobs"][3] == {"id": "pending", "status": "pending", "attempts": 0}


def test_stale_controller_pid_is_never_used_for_an_unrelated_process(monkeypatch, tmp_path):
    def unrelated(*args):
        raise ValueError("PID belongs to another command")

    monkeypatch.setattr(stop, "identify", unrelated)
    assert stop.controller({"pid": 123}, tmp_path) is None


def test_stop_is_one_shot_and_never_runs_a_resume(tmp_path, monkeypatch):
    plan = {"output": str(tmp_path), "pause_at": "2000-01-01T17:30:00+09:00"}
    calls = []

    def pause(_plan):
        calls.append("pause")
        stop.write(tmp_path / "state.json", {"phase": "paused"})

    monkeypatch.setattr(stop, "pause", pause)
    stop.run(plan)
    stop.run(plan)
    assert calls == ["pause"]
