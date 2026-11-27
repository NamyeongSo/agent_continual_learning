# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from copy import deepcopy

import litellm
import pytest

from exgentic.agents.litellm_tool_calling.instance import LiteLLMToolCallingAgentInstance
from exgentic.benchmarks.terminalbench2.terminalbench2_benchmark import BashAction
from exgentic.core.types import ActionType, EmptyObservation, ModelSettings, SingleObservation


@pytest.mark.parametrize("tool_call", [True, False])
def test_react_retains_reasoning_and_provider_settings_for_the_next_turn(monkeypatch, tool_call):
    monkeypatch.setattr("exgentic.agents.litellm_tool_calling.instance.check_model_accessible_sync",
                        lambda *args, **kw: None)
    monkeypatch.setattr("exgentic.agents.litellm_tool_calling.instance.get_context", lambda: None)
    settings = ModelSettings(reasoning_effort="xhigh", temperature=1.0, top_p=.95, presence_penalty=0.0,
                             extra_body={"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
                                         "chat_template_kwargs": {"preserve_thinking": True}})
    agent = LiteLLMToolCallingAgentInstance("thinking-test", model="openai/test", model_settings=settings,
                                           enable_tool_shortlisting=False)
    agent._register_cost = lambda usage: None
    requests = []

    def completion(**kwargs):
        requests.append(deepcopy(kwargs))
        message = {"role": "assistant", "content": "visible text", "reasoning_content": "Preserved reasoning."}
        if tool_call:
            message["tool_calls"] = [{"id": "call_test", "type": "function", "function": {
                "name": "bash", "arguments": '{"command":"pwd"}'}}]
        return litellm.ModelResponse(
            choices=[{"finish_reason": "tool_calls" if tool_call else "stop", "message": message}],
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )

    monkeypatch.setattr(litellm, "completion", completion)
    agent.start("Inspect", {}, [ActionType(name="bash", description="Run bash", cls=BashAction)])
    action = agent.react(EmptyObservation())
    agent.react(SingleObservation(result="ok", invoking_actions=[action]))
    history = next(message for message in requests[1]["messages"] if message["role"] == "assistant")
    assert history["reasoning_content"] == "Preserved reasoning."
    assert history["content"] == "visible text"
    assert requests[1]["reasoning_effort"] == "xhigh"
    assert requests[1]["presence_penalty"] == 0.0
    assert requests[1]["extra_body"] == settings.extra_body


def test_disabling_timeout_passes_an_explicit_unlimited_httpx_timeout(monkeypatch):
    import httpx

    agent = object.__new__(LiteLLMToolCallingAgentInstance)
    agent.model_settings = ModelSettings(timeout=600, disable_request_timeout=True)
    agent._completion_with_retries = lambda kwargs: kwargs
    monkeypatch.setattr("exgentic.agents.litellm_tool_calling.instance.get_context", lambda: None)
    request = agent._completion(model="openai/test", messages=[])
    assert isinstance(request["timeout"], httpx.Timeout)
    assert request["timeout"].as_dict() == {"connect": None, "read": None, "write": None, "pool": None}
    assert "disable_request_timeout" not in request
    agent.model_settings = ModelSettings(timeout=600)
    assert agent._completion(model="openai/test", messages=[])["timeout"] == 600
