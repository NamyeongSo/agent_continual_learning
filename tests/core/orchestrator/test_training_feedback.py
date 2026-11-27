# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

from typing import ClassVar

import pytest

from exgentic.core.context import Context, run_scope
from exgentic.core.orchestrator.session import run_session
from exgentic.core.orchestrator.termination import BenchmarkError
from exgentic.core.types import SessionScore


class _Session:
    session_id = "session-1"
    task_id = "task-1"
    task = "Do the task"
    context: ClassVar[dict] = {}
    actions: ClassVar[list] = []

    def __init__(self, events):
        self.events = events

    def start(self):
        return None

    def done(self):
        return False

    def score(self):
        self.events.append("score")
        return SessionScore(
            score=0.25,
            success=False,
            session_metadata={"checker_result": {"valid": False}},
        )

    def close(self):
        self.events.append("session.close")


class _AgentInstance:
    def __init__(self, events):
        self.events = events
        self.feedback = None

    def start(self, **kwargs):
        pass

    def receive_training_feedback(self, feedback):
        self.feedback = feedback
        self.events.append("feedback")

    def close(self):
        self.events.append("agent.close")


class _Agent:
    def __init__(self, instance, training_time):
        self.instance = instance
        self.training_time = training_time

    def get_instance(self, session_id):
        return self.instance


class _Tracker:
    def __init__(self, events):
        self.events = events

    def on_session_start(self, *args):
        pass

    def on_session_scoring(self, *args):
        pass

    def on_session_success(self, *args):
        self.events.append("session.success")


@pytest.mark.parametrize("training_time", [False, True])
def test_official_score_is_only_sent_to_opt_in_agent(tmp_path, training_time):
    events = []
    instance = _AgentInstance(events)
    agent = _Agent(instance, training_time)
    session = _Session(events)
    tracker = _Tracker(events)

    context = Context(
        run_id="training-feedback-test",
        output_dir=str(tmp_path),
        cache_dir=str(tmp_path),
    )
    with run_scope(context):
        run_session(None, session, agent, tracker=tracker)

    assert events.index("score") < events.index("agent.close")
    if training_time:
        assert events.index("score") < events.index("feedback") < events.index("agent.close")
        assert instance.feedback["score"] == 0.25
        assert instance.feedback["session_metadata"]["checker_result"]["valid"] is False
    else:
        assert "feedback" not in events
        assert instance.feedback is None


def test_benchmark_error_closes_session_and_agent(tmp_path):
    events = []

    class FailingSession(_Session):
        def start(self):
            return object()

        def step(self, _action):
            raise RuntimeError("benchmark failed")

    class ActingInstance(_AgentInstance):
        def react(self, _observation):
            return object()

    class ErrorTracker(_Tracker):
        def on_react_success(self, *_args):
            pass

        def on_step_error(self, _session, error):
            raise BenchmarkError(error)

        def on_session_error(self, *_args):
            events.append("session.error")

    context = Context(run_id="benchmark-error-cleanup", output_dir=str(tmp_path), cache_dir=str(tmp_path))
    with run_scope(context):
        run_session(None, FailingSession(events), _Agent(ActingInstance(events), False), tracker=ErrorTracker(events))

    assert events == ["session.error", "session.close", "agent.close"]


def test_scoring_error_still_closes_session_and_agent(tmp_path):
    events = []

    class FailingScoreSession(_Session):
        def score(self):
            raise RuntimeError("scoring failed")

    context = Context(run_id="scoring-error-cleanup", output_dir=str(tmp_path), cache_dir=str(tmp_path))
    with run_scope(context), pytest.raises(RuntimeError, match="scoring failed"):
        run_session(None, FailingScoreSession(events), _Agent(_AgentInstance(events), False), tracker=_Tracker(events))

    assert events == ["session.close", "agent.close"]


def test_agent_start_error_still_closes_started_session(tmp_path):
    events = []

    class FailingStartInstance(_AgentInstance):
        def start(self, **_kwargs):
            raise RuntimeError("agent start failed")

    context = Context(run_id="agent-start-cleanup", output_dir=str(tmp_path), cache_dir=str(tmp_path))
    with run_scope(context), pytest.raises(RuntimeError, match="agent start failed"):
        run_session(None, _Session(events), _Agent(FailingStartInstance(events), False), tracker=_Tracker(events))

    assert events == ["session.close", "agent.close"]
