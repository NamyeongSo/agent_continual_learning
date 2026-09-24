from __future__ import annotations

import pytest

from exgentic.core.context import Context, run_scope
from exgentic.core.orchestrator.session import run_session
from exgentic.core.types import SessionScore


class _Session:
    session_id = "session-1"
    task_id = "task-1"
    task = "Do the task"
    context = {}
    actions = []

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
