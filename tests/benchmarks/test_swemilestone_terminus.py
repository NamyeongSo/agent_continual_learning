# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Exercise upstream overflow recovery in the isolated Harbor environment."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from harbor.agents.terminus_2 import Terminus2
from harbor.llms.base import ContextLengthExceededError, LLMResponse
from harbor.models.agent.context import AgentContext

from exgentic.benchmarks.swemilestone.terminus_actor import MilestoneTerminus2


class Terminal:
    def __init__(self):
        self.keys = []

    async def is_session_alive(self):
        return True

    async def send_keys(self, keys, **kwargs):
        self.keys.append(keys)

    async def get_incremental_output(self):
        return f"Fresh terminal state: {self.keys}"

    async def capture_pane(self, **kwargs):
        return await self.get_incremental_output()


@pytest.mark.parametrize("overflow", [False, True])
@pytest.mark.parametrize("summary_overflow", [False, True])
def test_only_overflow_triggers_upstream_summary_and_agent_continues(
    tmp_path, monkeypatch, overflow, summary_overflow
):
    calls = []

    class FakeLLM:
        main_calls = 0

        def get_model_context_limit(self):
            return 128000

        async def call(self, prompt, message_history, **kwargs):
            calls.append((prompt, list(message_history)))
            if "You are about to hand off" in prompt:
                if summary_overflow and len(message_history) > 1:
                    raise ContextLengthExceededError("The token estimator undercounted the summary input")
                return LLMResponse(content="The file was edited. Continue validation.")
            if "You are picking up work" in prompt:
                return LLMResponse(content="Which file was edited?")
            if "The next agent has a few questions" in prompt:
                return LLMResponse(content="The file is /testbed/implementation.")
            self.main_calls += 1
            if overflow and self.main_calls == 2:
                raise ContextLengthExceededError("context_length_exceeded")
            complete = self.main_calls > 1
            return LLMResponse(
                content=json.dumps(
                    {
                        "analysis": "Implement and validate",
                        "plan": "Continue",
                        "commands": [{"keystrokes": "edit-file\n" if not complete else "validate\n", "duration": 0.1}],
                        "task_complete": complete,
                    }
                )
            )

    monkeypatch.setattr(Terminus2, "_init_llm", lambda *_args, **_kwargs: FakeLLM())
    agent = MilestoneTerminus2(
        logs_dir=tmp_path,
        model_name="openai/test",
        max_turns=3,
        enable_summarize=True,
        proactive_summarization_threshold=0,
        record_terminal_session=False,
    )
    # A negative estimate must still wait for a real ContextLengthExceededError.
    agent._count_total_tokens = lambda _chat: 100 if overflow else 1000000
    terminal = Terminal()
    agent._session = terminal
    context = AgentContext()
    asyncio.run(agent.run("MILESTONE SPECIFICATION\nACE PLAYBOOK", SimpleNamespace(), context))
    assert terminal.keys == ["edit-file\n", "validate\n", "validate\n"]
    assert agent.completed
    assert context.metadata["summarization_count"] == int(overflow) * (1 + int(summary_overflow))
    assert len(calls) == (7 + int(summary_overflow) if overflow else 3)
    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    compactions = [step for step in trajectory["steps"] if step.get("extra", {}).get("context_management")]
    assert len(compactions) == int(overflow)
    if overflow:
        assert len(list(tmp_path.glob("trajectory.summarization-*.json"))) == 3
        assert "edit-file" in str(calls[2][1])


def test_failed_context_recovery_ends_without_consuming_turns_with_synthetic_replies(tmp_path, monkeypatch):
    calls = []

    class FakeLLM:
        def get_model_context_limit(self):
            return 128000

        async def call(self, **kwargs):
            calls.append(kwargs)
            raise ContextLengthExceededError("context_length_exceeded")

    monkeypatch.setattr(Terminus2, "_init_llm", lambda *_args, **_kwargs: FakeLLM())
    agent = MilestoneTerminus2(
        logs_dir=tmp_path,
        model_name="openai/test",
        max_turns=200,
        enable_summarize=True,
        proactive_summarization_threshold=0,
        record_terminal_session=False,
    )
    agent._count_total_tokens = lambda _chat: 100
    agent._session = Terminal()
    with pytest.raises(RuntimeError, match="Terminus-2 context recovery failed"):
        asyncio.run(agent.run("MILESTONE SPECIFICATION", SimpleNamespace(), AgentContext()))
    assert len(calls) <= 6
    assert not agent._session.keys
    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert not any(step.get("source") == "agent" for step in trajectory["steps"])


def test_turn_limit_does_not_confirm_a_single_completion_signal(tmp_path, monkeypatch):
    class FakeLLM:
        async def call(self, **kwargs):
            return LLMResponse(content='{"analysis":"done","plan":"submit","commands":[],"task_complete":true}')

    monkeypatch.setattr(Terminus2, "_init_llm", lambda *_args, **_kwargs: FakeLLM())
    agent = MilestoneTerminus2(
        logs_dir=tmp_path,
        model_name="openai/test",
        max_turns=1,
        enable_summarize=True,
        proactive_summarization_threshold=0,
        record_terminal_session=False,
    )
    agent._session = Terminal()
    asyncio.run(agent.run("MILESTONE SPECIFICATION", SimpleNamespace(), AgentContext()))
    assert not agent.completed
