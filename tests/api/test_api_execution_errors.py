# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import errno
import json

import pytest

from exgentic import evaluate
from exgentic.core.orchestrator import execution
from exgentic.core.orchestrator.termination import RunCancelError
from exgentic.core.types import RunConfig, SessionOutcomeStatus


def make_config(tmp_path, max_workers):
    return RunConfig(
        benchmark="test_benchmark",
        agent="test_agent",
        task_ids=["task-1", "task-2", "task-3"],
        run_id="skip-task-error",
        output_dir=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        max_workers=max_workers,
        benchmark_kwargs={"tasks": ["task-1", "task-2", "task-3"]},
        agent_kwargs={"policy": "good_then_finish", "finish_after": 2},
    )


@pytest.mark.parametrize("max_workers", [1, 2])
@pytest.mark.parametrize("error", [OSError(errno.E2BIG, "Argument list too long"), RuntimeError("setup failed")])
def test_task_execution_error_is_recorded_and_remaining_tasks_finish(tmp_path, monkeypatch, max_workers, error):
    original = execution.run_session_config

    def fail_one_task(*, session_config, tracker):
        if session_config.task_id == "task-2":
            raise error
        return original(session_config=session_config, tracker=tracker)

    monkeypatch.setattr(execution, "run_session_config", fail_one_task)
    config = make_config(tmp_path, max_workers)
    results = evaluate(config)
    sessions = {session.task_id: session for session in results.session_results}
    assert results.total_sessions == 3
    assert sessions["task-1"].status == SessionOutcomeStatus.SUCCESS
    assert sessions["task-3"].status == SessionOutcomeStatus.SUCCESS
    failed = sessions["task-2"]
    assert failed.status == SessionOutcomeStatus.ERROR
    assert failed.success is False
    assert failed.steps == 0
    assert failed.details["session_metadata"]["error"] == str(error)
    assert failed.details["session_metadata"]["skipped"] is True
    if isinstance(error, OSError):
        assert failed.details["session_metadata"]["errno"] == errno.E2BIG

    root = tmp_path / config.run_id
    saved_run = json.loads((root / "results.json").read_text(encoding="utf-8"))
    assert saved_run["total_sessions"] == 3  # Saved after all workers, not on the first error.
    failed_root = root / "sessions" / config.to_session_config("task-2").get_session_id()
    assert json.loads((failed_root / "results.json").read_text(encoding="utf-8"))["status"] == "error"
    assert str(error) in (failed_root / "error.log").read_text(encoding="utf-8")


def test_transient_error_is_retried_before_skipping(tmp_path, monkeypatch):
    original = execution.run_session_config
    attempts = 0

    def fail_once(*, session_config, tracker):
        nonlocal attempts
        if session_config.task_id == "task-2":
            attempts += 1
            if attempts == 1:
                raise RuntimeError("Venv service failed to start")
        return original(session_config=session_config, tracker=tracker)

    monkeypatch.setattr(execution, "run_session_config", fail_once)
    results = evaluate(make_config(tmp_path, 1))
    assert attempts == 2
    assert results.total_sessions == 3
    assert all(session.status == SessionOutcomeStatus.SUCCESS for session in results.session_results)


def test_user_cancellation_still_stops_run(tmp_path, monkeypatch):
    tasks = []

    def cancel(*, session_config, tracker):
        tasks.append(session_config.task_id)
        raise RunCancelError()

    monkeypatch.setattr(execution, "run_session_config", cancel)
    results = evaluate(make_config(tmp_path, 1))
    assert tasks == ["task-1"]
    assert results.total_sessions == 0


def test_cleanup_error_does_not_duplicate_completed_result(tmp_path, monkeypatch):
    original = execution.run_session_config

    def fail_after_completion(*, session_config, tracker):
        original(session_config=session_config, tracker=tracker)
        if session_config.task_id == "task-2":
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(execution, "run_session_config", fail_after_completion)
    results = evaluate(make_config(tmp_path, 2))
    assert results.total_sessions == 3
    assert len({session.session_id for session in results.session_results}) == 3
    assert all(session.status == SessionOutcomeStatus.SUCCESS for session in results.session_results)
