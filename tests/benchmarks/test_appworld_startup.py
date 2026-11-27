# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import time

import pytest

from exgentic.benchmarks.appworld import appworld_benchmark, appworld_eval
from exgentic.benchmarks.appworld.appworld_benchmark import AppWorldBenchmark
from exgentic.benchmarks.appworld.appworld_eval import AppWorldEvaluator, _startup_clock
from exgentic.core.types import SessionIndex


@pytest.mark.parametrize("separator", [".", "__"])
@pytest.mark.parametrize("use_cache", [False, True])
def test_local_session_metadata_matches_isolated_evaluator_without_loading_appworld(monkeypatch, separator, use_cache):
    monkeypatch.setattr(appworld_benchmark, "get_run_id", lambda: "unique-run")
    monkeypatch.setattr(appworld_eval, "get_run_id", lambda: "unique-run")
    root_updates = []
    monkeypatch.setattr(AppWorldEvaluator, "_ensure_appworld_root", lambda self: root_updates.append(True))
    env = {"experiment_name": "old-name", "max_interactions": 999, "timeout_seconds": 321}
    benchmark = AppWorldBenchmark(
        env_kwargs=env, max_interactions=50, tool_name_separator=separator, use_cache=use_cache,
    )
    evaluator = AppWorldEvaluator(**benchmark._get_evaluator_kwargs())
    for session_id in ("first", "second"):
        index = SessionIndex(task_id="task", session_id=session_id)
        before = len(root_updates)
        local = benchmark.get_session_kwargs(index)
        assert len(root_updates) == before
        assert local == evaluator.get_session_kwargs(index)
        assert local["env_kwargs"]["experiment_name"] == f"unique-run__{session_id}"
        assert local["env_kwargs"]["max_interactions"] == 50
    assert env == {"experiment_name": "old-name", "max_interactions": 999, "timeout_seconds": 321}


def test_startup_clock_keeps_real_time_when_appworld_replaces_monotonic(monkeypatch):
    before = _startup_clock()
    monkeypatch.setattr(time, "monotonic", lambda: 1_234_567_890.0)
    assert _startup_clock() >= before
    assert _startup_clock() != time.monotonic()


def test_policy_override_reaches_both_local_and_isolated_sessions(monkeypatch):
    monkeypatch.setattr(appworld_benchmark, "get_run_id", lambda: "policy-run")
    monkeypatch.setattr(appworld_eval, "get_run_id", lambda: "policy-run")
    monkeypatch.setattr(AppWorldEvaluator, "_ensure_appworld_root", lambda self: None)
    index = SessionIndex(task_id="task", session_id="session")
    default = AppWorldBenchmark().get_session_kwargs(index)
    assert "policy_override" not in default
    policy = "Applications: {allowed}. Complete with finish {}."
    benchmark = AppWorldBenchmark(policy_override=policy)
    evaluator = AppWorldEvaluator(**benchmark._get_evaluator_kwargs())
    assert benchmark.get_session_kwargs(index) == evaluator.get_session_kwargs(index)
    assert benchmark.get_session_kwargs(index)["policy_override"] == policy


def test_code_generation_is_the_default_and_can_be_disabled(monkeypatch):
    monkeypatch.setattr(appworld_benchmark, "get_run_id", lambda: "mode-run")
    monkeypatch.setattr(appworld_eval, "get_run_id", lambda: "mode-run")
    monkeypatch.setattr(AppWorldEvaluator, "_ensure_appworld_root", lambda self: None)
    index = SessionIndex(task_id="task", session_id="session")

    default = AppWorldBenchmark()
    assert default.interaction_mode == "code_generation"
    assert default._get_evaluator_kwargs()["interaction_mode"] == "code_generation"
    assert default.get_session_kwargs(index)["interaction_mode"] == "code_generation"

    legacy = AppWorldBenchmark(interaction_mode="function_calling")
    evaluator = AppWorldEvaluator(**legacy._get_evaluator_kwargs())
    assert legacy.get_session_kwargs(index) == evaluator.get_session_kwargs(index)
    assert legacy.get_session_kwargs(index)["interaction_mode"] == "function_calling"


def test_policy_template_renders_actual_apps_and_preserves_other_context():
    from datetime import datetime
    from types import SimpleNamespace

    session = object.__new__(appworld_eval.AppWorldSession)
    session._interaction_mode = "code_generation"
    session._world = SimpleNamespace(task=SimpleNamespace(
        allowed_apps=["api_docs", "supervisor", "gmail"],
        supervisor={"first_name": "Alex"}, datetime=datetime(2026, 10, 5),
    ))
    original = session.context
    session._policy_override = "Apps: {allowed}. Use finish {} without a summary."
    updated = session.context
    assert updated["policy"] == "Apps: supervisor, gmail. Use finish {} without a summary."
    assert updated["interaction_mode"] == "code_generation"
    assert "execute_python" in updated["code_execution_instructions"]
    assert updated["supervisor"] == original["supervisor"]
    assert updated["datetime"] == original["datetime"]
