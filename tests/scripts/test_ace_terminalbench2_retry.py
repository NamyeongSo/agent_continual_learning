# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Check that Terminal-Bench recovery preserves original task order and scores."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def test_merge_replaces_only_error_sessions():
    script = Path(__file__).resolve().parents[2] / "scripts/ace/retry_terminalbench2_test.py"
    spec = importlib.util.spec_from_file_location("ace_terminalbench2_retry", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = {
        "sessions": [
            {"task_id": "one", "score": 1.0, "status": "success", "skip": None},
            {"task_id": "two", "score": 0.0, "status": "error", "skip": None},
            {"task_id": "three", "score": 0.0, "status": "unfinished", "skip": None},
        ],
        "error_count": 1, "mean_task_score": 1 / 3, "benchmark_score": 0.5,
    }
    recovered = [{"task_id": "two", "score": 1.0, "status": "success", "skip": None}]
    result = module.merge_sessions(original, recovered)
    assert [entry["task_id"] for entry in result["sessions"]] == ["one", "two", "three"]
    assert result["sessions"][0] == original["sessions"][0]
    assert result["sessions"][2] == original["sessions"][2]
    assert result["mean_task_score"] == pytest.approx(2 / 3)
    assert result["benchmark_score"] == pytest.approx(2 / 3)
    assert result["error_count"] == 0
    assert result["original_error_count"] == 1
    assert result["retried_error_task_ids"] == ["two"]

    with pytest.raises(ValueError, match="Retried task IDs"):
        module.merge_sessions(original, [{"task_id": "wrong", "score": 1.0, "status": "success"}])


def test_context_skipped_tasks_are_not_retried():
    script = Path(__file__).resolve().parents[2] / "scripts/ace/retry_terminalbench2_test.py"
    spec = importlib.util.spec_from_file_location("ace_terminalbench2_context_retry", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = {"sessions": [
        {"task_id": "context", "status": "error", "skip": {"context_error": True}},
        {"task_id": "network", "status": "error", "skip": None},
        {"task_id": "official-zero", "status": "success", "score": 0},
    ]}
    assert module.retry_task_ids(summary) == ["network"]
