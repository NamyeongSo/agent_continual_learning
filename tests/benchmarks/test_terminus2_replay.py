# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Exercise Harbor's parser, command execution and conversation loop during restore."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest


def test_prefix_runs_commands_without_llm_and_continues_with_fresh_observations(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    from harbor.agents.terminus_2 import Terminus2
    from harbor.llms.base import LLMResponse
    from harbor.models.agent.context import AgentContext

    from exgentic.benchmarks.terminalbench2.terminus_replay import ReplayTerminus2

    calls = []

    class FakeLLM:
        async def call(self, prompt, message_history, **kwargs):
            calls.append((prompt, list(message_history)))
            return LLMResponse(content=json.dumps({"analysis": "continue", "plan": "verify",
                                                   "commands": [{"keystrokes": "pwd\n", "duration": 0.1}],
                                                   "task_complete": False}))

    class Terminal:
        def __init__(self):
            self.keys = []

        async def is_session_alive(self):
            return True

        async def send_keys(self, keys, **kwargs):
            self.keys.append(keys)

        async def get_incremental_output(self):
            return f"FRESH observed after {self.keys!r}"

    monkeypatch.setattr(Terminus2, "_init_llm", lambda *_args, **_kwargs: FakeLLM())
    prefix_response = json.dumps({"analysis": "prepare", "plan": "setup",
                                  "commands": [{"keystrokes": "cd /workspace\n", "duration": 0.1},
                                               {"keystrokes": "export EXAMPLE=value\n", "duration": 0.1}],
                                  "task_complete": False})
    agent = ReplayTerminus2(logs_dir=tmp_path, model_name="openai/test", max_turns=2,
                           enable_summarize=False, record_terminal_session=False,
                           replay_episodes=[{"response": prefix_response}])
    terminal = Terminal()
    agent._session = terminal
    context = AgentContext()
    asyncio.run(agent.run("TASK\nCANDIDATE PLAYBOOK", SimpleNamespace(), context))
    assert terminal.keys == ["cd /workspace\n", "export EXAMPLE=value\n", "pwd\n"]
    assert len(calls) == 1
    prompt, history = calls[0]
    assert "FRESH" in prompt and "export EXAMPLE=value" in prompt
    assert "CANDIDATE PLAYBOOK" in history[0]["content"]
    assert history[1]["content"] == prefix_response
    assert context.metadata["n_episodes"] == 2
    recorded = json.loads((tmp_path / "premise_episodes.json").read_text())
    assert recorded["prefix_length"] == 1
    assert len(recorded["episodes"]) == 2
    steps = json.loads((tmp_path / "trajectory.json").read_text())["steps"]
    assert steps[1]["tool_calls"][0]["arguments"]["keystrokes"] == "cd /workspace\n"
    assert "FRESH" in steps[1]["observation"]["results"][0]["content"]


def test_prefix_must_leave_at_least_one_generation_turn(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    from harbor.agents.terminus_2 import Terminus2

    from exgentic.benchmarks.terminalbench2.terminus_replay import ReplayTerminus2

    monkeypatch.setattr(Terminus2, "_init_llm", lambda *_args, **_kwargs: SimpleNamespace())
    with pytest.raises(ValueError, match="turn budget"):
        ReplayTerminus2(logs_dir=tmp_path, model_name="openai/test", max_turns=1,
                        replay_episodes=[{"response": "{}"}])


def test_load_recorded_prefix_rejects_continuation_source(tmp_path):
    pytest.importorskip("harbor")
    from exgentic.benchmarks.terminalbench2.terminus_replay import load_recorded_prefix

    path = tmp_path / "episodes.json"
    path.write_text(json.dumps({"prefix_length": 0, "episodes": [
        {"response": "first"}, {"response": "second"},
    ]}))
    assert load_recorded_prefix(str(path), 1) == [{"response": "first"}]
    with pytest.raises(ValueError, match="exceeds"):
        load_recorded_prefix(str(path), 3)
    path.write_text(json.dumps({"prefix_length": 1, "episodes": [{"response": "first"}]}))
    with pytest.raises(ValueError, match="original"):
        load_recorded_prefix(str(path), 1)


def test_truncation_recovery_records_only_the_executed_episode(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    from harbor.agents.terminus_2 import Terminus2
    from harbor.llms.base import LLMResponse, OutputLengthExceededError
    from harbor.models.agent.context import AgentContext

    from exgentic.benchmarks.terminalbench2.terminus_replay import ReplayTerminus2

    prompts = []

    class FakeLLM:
        async def call(self, prompt, **_kwargs):
            prompts.append(prompt)
            if len(prompts) == 1:
                raise OutputLengthExceededError("truncated", truncated_response='{"analysis": "unfinished')
            return LLMResponse(content=json.dumps({
                "analysis": "retry", "plan": "inspect", "task_complete": False,
                "commands": [{"keystrokes": "pwd\n", "duration": 0.1}],
            }))

        def get_model_output_limit(self):
            return None

    class Terminal:
        async def is_session_alive(self):
            return True

        async def send_keys(self, keys, **_kwargs):
            assert keys == "pwd\n"

        async def get_incremental_output(self):
            return "fresh observation"

    monkeypatch.setattr(Terminus2, "_init_llm", lambda *_args, **_kwargs: FakeLLM())
    agent = ReplayTerminus2(logs_dir=tmp_path, model_name="openai/test", max_turns=1,
                           enable_summarize=False, record_terminal_session=False)
    agent._session = Terminal()
    asyncio.run(agent.run("task", SimpleNamespace(), AgentContext()))
    assert len(prompts) == 2
    assert "NONE of the actions" in prompts[1]
    recorded = json.loads((tmp_path / "premise_episodes.json").read_text())
    steps = json.loads((tmp_path / "trajectory.json").read_text())["steps"]
    assert len(recorded["episodes"]) == len([step for step in steps if step["source"] == "agent"]) == 1
    events = [json.loads(line) for line in (tmp_path / "query_events.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["start", "start", "end", "end"]
    assert [event["depth"] for event in events] == [0, 1, 1, 0]
    assert agent._query_depth == 0
