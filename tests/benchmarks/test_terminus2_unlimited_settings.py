# SPDX-License-Identifier: Apache-2.0
"""Verify native Harbor prompt, sampler, reasoning history and uncapped output together."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest


def test_native_prompt_and_preserved_reasoning_with_unlimited_output(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    from harbor.agents.terminus_2 import Terminus2
    from harbor.llms.base import LLMResponse
    from harbor.models.agent.context import AgentContext

    from exgentic.benchmarks.terminalbench2.terminus_limits import BudgetedTerminus2

    monkeypatch.setenv("EXGENTIC_ACTOR_OUTPUT_UNLIMITED", "1")
    monkeypatch.delenv("EXGENTIC_INFERENCE_PROFILE", raising=False)
    calls = []

    class FakeLLM:
        async def call(self, prompt, **kwargs):
            calls.append({"prompt": prompt, **copy.deepcopy(kwargs)})
            return LLMResponse(
                content=json.dumps({"analysis": "check", "plan": "done", "task_complete": True, "commands": []}),
                reasoning_content="retained thinking",
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50_000, cache_tokens=0, cost_usd=0),
            )

    class Terminal:
        async def is_session_alive(self):
            return True

        async def get_incremental_output(self):
            return "fresh terminal"

    monkeypatch.setattr(Terminus2, "_init_llm", lambda *_args, **_kwargs: FakeLLM())
    actor = BudgetedTerminus2(
        logs_dir=tmp_path, model_name="openai/Qwen/Qwen3.8-27B-FP8", interleaved_thinking=True,
        record_terminal_session=False, enable_summarize=False,
        llm_call_kwargs={"max_tokens": 2048, "reasoning_effort": "xhigh", "temperature": 1.0, "top_p": .95,
                         "extra_body": {"top_k": 20, "min_p": 0, "repetition_penalty": 1,
                                        "chat_template_kwargs": {"preserve_thinking": True}}},
    )
    assert actor._prompt_template == actor._get_prompt_template_path().read_text()
    assert actor._get_prompt_template_path().name == "terminus-json-plain.txt"
    actor._session = Terminal()
    asyncio.run(actor.run("original task instruction", SimpleNamespace(), AgentContext()))
    assert len(calls) == 2
    for call in calls:
        assert "max_tokens" not in call and "max_completion_tokens" not in call
        assert call["reasoning_effort"] == "xhigh" and call["temperature"] == 1.0 and call["top_p"] == .95
        assert call["allowed_openai_params"] == ["reasoning_effort"]
    assert "original task instruction" in calls[0]["prompt"]
    assert calls[1]["message_history"][1]["reasoning_content"] == "retained thinking"
