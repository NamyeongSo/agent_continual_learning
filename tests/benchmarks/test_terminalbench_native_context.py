# SPDX-License-Identifier: Apache-2.0
"""Regress the real Harbor context loop without a Docker environment or GPU."""

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from harbor.agents.terminus_2 import Terminus2
from harbor.llms.base import ContextLengthExceededError, LLMResponse
from harbor.llms.chat import Chat
from harbor.models.agent.context import AgentContext

from exgentic.benchmarks.terminalbench2.native_context import (
    ContextRecoveryError,
    RecoveringTerminus2,
    ServingTokenCounter,
)


class Terminal:
    def __init__(self):
        self.keys = []

    async def is_session_alive(self):
        return True

    async def send_keys(self, keys, **kwargs):
        self.keys.append(keys)

    async def get_incremental_output(self):
        return "Fresh terminal feedback"

    async def capture_pane(self, **kwargs):
        return await self.get_incremental_output()


def test_serving_counter_preserves_reasoning_and_template_and_caches(monkeypatch):
    from io import BytesIO

    requests = []

    def respond(request, timeout):
        requests.append(json.loads(request.data))
        assert request.full_url == "http://localhost:8008/tokenize"
        assert timeout == 30
        return BytesIO(b'{"count":120123}')

    monkeypatch.setattr("urllib.request.urlopen", respond)
    counter = ServingTokenCounter("http://localhost:8008/v1/", "openai/Qwen/model",
                                  {"preserve_thinking": True, "enable_thinking": True})
    messages = [{"role": "assistant", "content": "command", "reasoning_content": "thinking"}]
    assert counter([]) == 0
    assert counter(messages) == counter(messages) == 120123
    assert len(requests) == 1
    assert requests[0]["messages"] == [{**messages[0], "reasoning": "thinking"}]
    assert "reasoning" not in messages[0]  # Do not alter the actual conversation.
    assert requests[0]["chat_template_kwargs"]["preserve_thinking"] is True
    assert requests[0]["model"] == "Qwen/model"


def test_tokenizer_failure_is_explicit_infrastructure_error(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("unavailable")

    monkeypatch.setattr("urllib.request.urlopen", broken)
    with pytest.raises(ContextRecoveryError, match="Serving tokenizer failed"):
        ServingTokenCounter("http://localhost:8008/v1", "openai/test", {})([{"role": "user", "content": "x"}])


def test_history_reduction_keeps_task_and_latest_reasoning(tmp_path, monkeypatch):
    monkeypatch.setattr(Terminus2, "_init_llm",
                        lambda *_a, **_k: SimpleNamespace(get_model_context_limit=lambda: 128000))

    def counter(messages):
        return len(messages) * 20000
    agent = RecoveringTerminus2(logs_dir=tmp_path, model_name="openai/test", serving_token_counter=counter)
    chat = Chat(agent._llm, interleaved_thinking=True)
    chat._messages = [{"role": "system", "content": "original task"}]
    for i in range(4):
        chat._messages.extend([{"role": "user", "content": str(i)},
                               {"role": "assistant", "content": "command", "reasoning_content": str(i)}])
    last_exchange = chat.messages[-2:]
    chat._last_response_id = "stale"
    agent._shrink(chat, reserve=12000)
    assert counter(chat.messages) <= 116000
    assert chat.messages[0]["content"] == "original task"
    assert chat.messages[-2:] == last_exchange
    assert chat._last_response_id is None


@pytest.mark.parametrize("summary_overflow", [False, True])
def test_real_overflow_recovers_with_native_summary_and_executes_commands(tmp_path, monkeypatch, summary_overflow):
    calls = []

    class FakeLLM:
        main_calls = 0
        summary_calls = 0

        def get_model_context_limit(self):
            return 128000

        async def call(self, prompt, message_history, **kwargs):
            calls.append((prompt, list(message_history)))
            if "You are about to hand off" in prompt:
                self.summary_calls += 1
                if summary_overflow and self.summary_calls == 1:
                    raise ContextLengthExceededError("summary overflow")
                return LLMResponse(content="Edits complete; validate them.")
            if "You are picking up work" in prompt:
                return LLMResponse(content="Which file changed?")
            if "The next agent has a few questions" in prompt:
                return LLMResponse(content="Check /app/result.")
            self.main_calls += 1
            # Build enough history to test removal of old complete exchanges.
            if self.main_calls == 4:
                raise ContextLengthExceededError("context overflow despite preflight")
            complete = self.main_calls >= 5
            return LLMResponse(content=json.dumps({"analysis": "check", "plan": "continue",
                "commands": [{"keystrokes": "validate\n" if complete else "edit\n", "duration": .01}],
                "task_complete": complete}), reasoning_content="preserved reasoning")

    monkeypatch.setattr(Terminus2, "_init_llm", lambda *_a, **_k: FakeLLM())
    agent = RecoveringTerminus2(logs_dir=tmp_path, model_name="openai/test", serving_token_counter=lambda _m: 100,
                               interleaved_thinking=True, record_terminal_session=False)
    agent._session = Terminal()
    asyncio.run(agent.run("ORIGINAL TASK", SimpleNamespace(), AgentContext()))
    assert agent._session.keys == ["edit\n"] * 3 + ["validate\n"] * 2
    assert agent._llm.summary_calls == 1 + int(summary_overflow)
    assert len(calls) == 9 + int(summary_overflow)
    assert any(m.get("reasoning_content") == "preserved reasoning" for _, hist in calls for m in hist)
    assert len(list(tmp_path.glob("trajectory.summarization-*.json"))) == 3


def test_persistent_overflow_stops_without_synthetic_agent_turns(tmp_path, monkeypatch):
    calls = []

    class BrokenLLM:
        def get_model_context_limit(self):
            return 128000

        async def call(self, **kwargs):
            calls.append(kwargs)
            raise ContextLengthExceededError("permanent overflow")

    monkeypatch.setattr(Terminus2, "_init_llm", lambda *_a, **_k: BrokenLLM())
    agent = RecoveringTerminus2(logs_dir=tmp_path, model_name="openai/test", serving_token_counter=lambda _m: 100,
                               record_terminal_session=False)
    agent._session = Terminal()
    with pytest.raises(ContextRecoveryError):
        asyncio.run(agent.run("ORIGINAL TASK", SimpleNamespace(), AgentContext()))
    assert len(calls) <= 6
    assert not agent._session.keys
    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    assert not any(step.get("source") == "agent" for step in trajectory["steps"])
