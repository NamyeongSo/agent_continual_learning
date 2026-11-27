# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from exgentic.core.orchestrator import execution


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("failed_setup", [False, True])
def test_local_metadata_bypasses_evaluator_and_fallback_cleans_up(monkeypatch, local, failed_setup):
    events = []
    session = SimpleNamespace(session_id="session", task_id="task")

    def kwargs():
        if failed_setup:
            raise ValueError("metadata failure")
        return {"task_id": "task"}

    class FakeEvaluator:
        def get_session_kwargs(self, _index):
            events.append("evaluator_kwargs")
            return kwargs()

        def close(self):
            events.append("evaluator_close")

    class FakeBenchmark:
        static_action_space = True

        def __init__(self, **_kwargs):
            pass

        def get_session_kwargs(self, _index):
            events.append("local_kwargs")
            return kwargs() if local else None

        def get_evaluator(self):
            assert not local
            events.append("evaluator")
            return FakeEvaluator()

        def get_session(self, **received):
            assert received == {"task_id": "task"}
            events.append("session")
            return session

        def close(self):
            events.append("benchmark_close")

    class FakeAgent:
        def __init__(self, **_kwargs):
            pass

        def close(self):
            events.append("agent_close")

    monkeypatch.setattr(execution, "_get_benchmark_class", lambda _: FakeBenchmark)
    monkeypatch.setattr(execution, "_get_agent_class", lambda _: FakeAgent)

    def run(_config, replay, _agent, **_kwargs):
        assert replay._session is session
        assert replay._static_actions is True
        events.append("run")

    monkeypatch.setattr(execution, "run_session", run)
    config = SimpleNamespace(
        benchmark="fake", agent="fake", task_id="task", benchmark_kwargs={}, agent_kwargs={},
        replay_prefix={"task_id": "task", "actions": []}, get_session_id=lambda: "session",
    )
    if failed_setup:
        with pytest.raises(ValueError, match="metadata failure"):
            execution.run_session_config(session_config=config, tracker=SimpleNamespace())
        assert "session" not in events
    else:
        execution.run_session_config(session_config=config, tracker=SimpleNamespace())
        assert "run" in events
    assert ("evaluator" in events) is not local
    assert ("evaluator_close" in events) is not local
    assert events[-2:] == ["benchmark_close", "agent_close"]
