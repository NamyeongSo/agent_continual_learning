# SPDX-License-Identifier: Apache-2.0
"""The reproduction path must use the original Harbor loop without shared-budget interception."""

import asyncio
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("context_recovery", [False, True])
@pytest.mark.parametrize("max_turns", [None, 100])
def test_native_path_keeps_upstream_defaults_and_accepts_large_context_after_40_turns(
    tmp_path, monkeypatch, context_recovery, max_turns,
):
    pytest.importorskip("harbor")
    from harbor.agents.terminus_2 import Terminus2
    from harbor.llms.base import LLMResponse

    from exgentic.benchmarks.terminalbench2.native_context import RecoveringTerminus2

    monkeypatch.setattr("exgentic.benchmarks.terminalbench2.native_context.ServingTokenCounter",
                        lambda *_a, **_k: lambda _messages: 120000)
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:8008/v1")

    from exgentic.benchmarks.terminalbench2.task_clock import AgentTimeBudget
    from exgentic.benchmarks.terminalbench2.terminalbench2_benchmark import RunTerminus2Action, RunTerminus2Args
    from exgentic.benchmarks.terminalbench2.terminalbench2_eval import TerminalBench2Session

    calls = []
    agents = []
    original_init = Terminus2.__init__

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        agents.append(self)

    class FakeLLM:
        def get_model_context_limit(self):
            return 262144

        def get_model_output_limit(self):
            return None

        async def call(self, prompt, **kwargs):
            calls.append(kwargs)
            assert "max_tokens" not in kwargs
            assert kwargs["reasoning_effort"] == "xhigh"
            assert "disable_request_timeout" not in kwargs
            if context_recovery:
                assert kwargs["timeout"].read is None
                assert kwargs["num_retries"] == kwargs["max_retries"] == 0
            complete = max_turns is None and len(calls) >= 41
            return LLMResponse(
                content=json.dumps({"analysis": "check", "plan": "continue", "task_complete": complete,
                                    "commands": []}),
                usage=SimpleNamespace(prompt_tokens=120000, completion_tokens=5000, cache_tokens=0, cost_usd=0),
            )

    class Terminal:
        async def is_session_alive(self):
            return True

        async def get_incremental_output(self):
            return "terminal"

    async def setup(self, environment):
        self._session = Terminal()

    monkeypatch.setattr(Terminus2, "__init__", initialize)
    monkeypatch.setattr(Terminus2, "_init_llm", lambda *_args, **_kwargs: FakeLLM())
    monkeypatch.setattr(Terminus2, "setup", setup)
    session = object.__new__(TerminalBench2Session)
    session._native_terminus2 = True
    session._native_context_recovery = context_recovery
    session._loop_runner = asyncio.Runner()
    session._task_clock = AgentTimeBudget(60)
    session._task_time_accounting = None
    session._agent_deadline = None
    session._max_interactions = max_turns
    session._environment = SimpleNamespace(with_default_user=lambda user: nullcontext())
    session._trial_paths = SimpleNamespace(agent_dir=tmp_path / "agent")
    session._task_obj = SimpleNamespace(instruction="original instruction", config=SimpleNamespace(
        agent=SimpleNamespace(user="root")))
    try:
        session._handle_terminus2(RunTerminus2Action(arguments=RunTerminus2Args(
            model="openai/test", model_settings={"reasoning_effort": "xhigh", "temperature": 1.0,
                                                 "disable_request_timeout": context_recovery},
        )))
    finally:
        session._loop_runner.close()
    expected_calls = max_turns or 42
    assert len(calls) == expected_calls
    assert type(agents[0]) is (RecoveringTerminus2 if context_recovery else Terminus2)
    assert agents[0]._max_episodes == (max_turns or 1000000)
    assert agents[0].options.enable_summarize is True
    assert agents[0].options.proactive_summarization_threshold == 8000
    assert agents[0].options.record_terminal_session is True
    assert not hasattr(agents[0], "_actor_budget")
    assert session._experiment_budget_failure is None
    saved = json.loads((tmp_path / "agent/native_agent_config.json").read_text())
    assert saved["shared_actor_budget"] is False
    assert saved["native_context_recovery"] is context_recovery
    assert saved["request_timeout_disabled"] is context_recovery
    assert saved["max_input_tokens"] == 262144
    assert session._task_time_accounting["model_call_count"] == expected_calls
