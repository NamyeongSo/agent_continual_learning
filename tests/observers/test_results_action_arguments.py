# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Recorded composite actions must retain the parameters sent to the environment."""

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from exgentic.core.orchestrator.tracker import ResultsObserver
from exgentic.core.types.action import ParallelAction, SequentialAction, SingleAction


class MoveArguments(BaseModel):
    source: str
    destination: str


class MoveAction(SingleAction):
    arguments: MoveArguments


@pytest.mark.parametrize("composite", [ParallelAction, SequentialAction])
def test_composite_trajectory_preserves_executed_tool_parameters(tmp_path, composite):
    trajectory = tmp_path / "trajectory.jsonl"
    session = SimpleNamespace(session_id="session", task_id="task", paths=SimpleNamespace(trajectory=trajectory))
    child = MoveAction(name="move", arguments=MoveArguments(source="report", destination="archive"), id="call")
    observer = ResultsObserver(run_id="run")
    observer.on_session_start(session, None, None)
    observer.on_react_success(session, composite(actions=[child]))

    event = json.loads(trajectory.read_text())
    recorded = event["action"]["actions"][0]
    assert recorded["arguments"] == {"source": "report", "destination": "archive"}
    assert recorded["id"] == child.id
    assert recorded["name"] == child.name
    assert child.arguments.destination == "archive"
