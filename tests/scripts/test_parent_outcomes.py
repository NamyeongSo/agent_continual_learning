# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import json

import pytest

from scripts.research_journal.live_progress import FileCache


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def evidence(folder, benchmark="appworld", *, label="parent1", namespace="abc", protocol=False):
    run = f"premise_{namespace}_i0001_{label}_{benchmark}"
    session = folder / "sessions" / run / "sessions" / "session"
    write(session / "config.json", {"run_id": run, "benchmark": benchmark, "task_id": "task"})
    raw = {"session_id": "session", "task_id": "task", "status": "success", "score": 1, "success": True}
    if protocol:
        raw.update(status="error", score=0, success=False, details={"session_metadata": {
            "error_source": "agent", "error": "LiteLLM completion missing assistant content."}})
    write(session / "results.json", raw)
    name = "bfcl_score.json" if benchmark == "bfcl" else "results.json"
    write(session / "benchmark" / name, {
        "session_id": "session", "task_id": "task", "score": .99 if benchmark == "appworld" else 0,
        "success": False,
    })
    return session


def iteration(folder):
    return write(folder / "iterations/iter_0001.json", {
        "iteration": 1, "task_id": "task", "benchmark": "appworld", "status": "skipped",
        "reason": "later replay connection error", "parents": [None, {"score": 1, "success": True}],
    })


@pytest.mark.parametrize("benchmark", ["bfcl", "appworld", "terminalbench2"])
def test_later_replay_error_does_not_erase_measured_first_parent(tmp_path, benchmark):
    path = iteration(tmp_path)
    evidence(tmp_path, benchmark)
    before = path.read_bytes()
    cache = FileCache()
    result = cache.parent_score(path, benchmark)
    assert result["score"] == 1
    assert "parent1" in result["score_source"]
    assert path.read_bytes() == before
    assert "later replay" not in repr(cache.entries)


@pytest.mark.parametrize("benchmark", ["bfcl", "appworld"])
def test_empty_model_response_with_official_failure_counts_as_measured_zero(tmp_path, benchmark):
    path = iteration(tmp_path)
    session = evidence(tmp_path, benchmark, protocol=True)
    before = (session / "results.json").read_bytes()
    result = FileCache().parent_score(path, benchmark)
    assert result["score"] == 0  # AppWorld .99 partial score is still official TGC zero.
    assert "/benchmark/" in result["score_source"]
    assert (session / "results.json").read_bytes() == before


@pytest.mark.parametrize("file,patch", [
    ("results.json", {"task_id": "foreign"}),
    ("results.json", {"session_id": "foreign"}),
    ("config.json", {"task_id": "foreign"}),
    ("config.json", {"benchmark": "bfcl"}),
    ("config.json", {"run_id": "foreign"}),
    ("results.json", {"details": {"session_metadata": {
        "error_source": "execution", "error": "Connection error"}}}),
    ("results.json", {"status": "cancelled"}),
    ("benchmark/results.json", {"task_id": "foreign"}),
    ("benchmark/results.json", {"session_id": "foreign"}),
    ("benchmark/results.json", {"score": float("nan")}),
    ("benchmark/results.json", {"success": None}),
])
def test_missing_identity_or_infrastructure_is_not_a_measured_failure(tmp_path, file, patch):
    path = iteration(tmp_path)
    session = evidence(tmp_path, protocol=True)
    target = session / file
    write(target, {**json.loads(target.read_text()), **patch})
    assert FileCache().parent_score(path, "appworld")["score"] is None


def test_never_substitute_second_parent_or_choose_among_multiple_attempts(tmp_path):
    path = iteration(tmp_path)
    evidence(tmp_path, label="parent2")
    cache = FileCache()
    assert cache.parent_score(path, "appworld")["score"] is None
    evidence(tmp_path, namespace="abc")
    evidence(tmp_path, namespace="def")
    assert cache.parent_score(path, "appworld")["score"] is None


def test_missing_proof_can_arrive_later_and_iteration_edits_invalidate_cache(tmp_path):
    path = iteration(tmp_path)
    cache = FileCache()
    assert cache.parent_score(path, "appworld")["score"] is None
    evidence(tmp_path)
    assert cache.parent_score(path, "appworld")["score"] == 1
    data = json.loads(path.read_text())
    write(path, {**data, "parents": [{"score": 0, "success": False}]})
    assert cache.parent_score(path, "appworld")["score"] == 0


def test_symlink_outside_training_folder_is_not_reporting_evidence(tmp_path):
    folder = tmp_path / "train"
    path = iteration(folder)
    outside = tmp_path / "other"
    session = evidence(outside)
    link = folder / "sessions" / session.parents[1].name
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(session.parents[1], target_is_directory=True)
    assert FileCache().parent_score(path, "appworld")["score"] is None
