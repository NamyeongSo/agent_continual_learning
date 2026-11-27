# SPDX-License-Identifier: Apache-2.0
"""Model waits must not consume task time; real execution and provider errors still count correctly."""

import asyncio
from types import SimpleNamespace

import pytest

from exgentic.benchmarks.terminalbench2.task_clock import (
    AgentTimeBudget,
    AgentTimeLimitExceededError,
    initialize_model_call_ledger,
    record_model_call,
)


def test_model_wait_can_exceed_task_budget_and_environment_still_times_out(tmp_path):
    async def run():
        clock = AgentTimeBudget(.1, report_path=tmp_path / "timing.json")

        async def actor():
            with clock.model_call():
                await asyncio.sleep(.25)
            return "model completed"

        assert await clock.run(actor()) == "model completed"
        state = clock.snapshot()
        assert state["model_call_seconds"] >= .25
        assert state["charged_seconds"] < .1
        # Subsequent command execution still consumes the remaining task budget.
        with pytest.raises(AgentTimeLimitExceededError):
            await clock.run(asyncio.sleep(.2))

    asyncio.run(run())


def test_provider_timeout_is_not_reclassified_as_task_timeout():
    async def run():
        clock = AgentTimeBudget(.1)

        async def provider_error():
            with clock.model_call():
                await asyncio.sleep(.15)
                raise TimeoutError("provider request timed out")

        with pytest.raises(TimeoutError, match="provider request timed out") as exc:
            await clock.run(provider_error())
        assert not isinstance(exc.value, AgentTimeLimitExceededError)
        assert clock.remaining() > 0

    asyncio.run(run())


def test_model_retries_are_excluded_even_on_error_and_freeze_after_stop():
    tick = [0.0]
    clock = AgentTimeBudget(20, now=lambda: tick[0])
    tick[0] = 3
    with pytest.raises(ValueError):
        with clock.model_call():
            tick[0] += 100
            with clock.model_call():
                tick[0] += 200
            raise ValueError("failed request after retries")
    tick[0] += 2
    state = clock.stop()
    assert state["wall_seconds"] == 305
    assert state["model_call_seconds"] == 300
    assert state["charged_seconds"] == 5
    assert state["remaining_seconds"] == 15
    tick[0] += 1000
    assert clock.snapshot() == state


def test_interactive_actor_shares_call_time_with_isolated_session(tmp_path, monkeypatch):
    from exgentic.agents.litellm_tool_calling.instance import LiteLLMToolCallingAgentInstance
    from exgentic.core.types import ModelSettings

    ledger = tmp_path / "model_call_timing.json"
    initialize_model_call_ledger(ledger)
    tick = [0.0]
    monkeypatch.setattr("exgentic.benchmarks.terminalbench2.task_clock.time.monotonic", lambda: tick[0])
    clock = AgentTimeBudget(20, ledger_path=ledger, now=lambda: tick[0])
    actor = object.__new__(LiteLLMToolCallingAgentInstance)
    actor._session_id = "test"
    actor._paths = SimpleNamespace(benchmark_dir=tmp_path)
    actor.model_settings = ModelSettings()
    monkeypatch.setattr("exgentic.agents.litellm_tool_calling.instance.get_context", lambda: None)

    def provider(_kwargs):
        tick[0] += 100
        assert clock.remaining() == 20
        raise TimeoutError("provider error")

    actor._completion_with_retries = provider
    with pytest.raises(TimeoutError):
        actor._completion(model="openai/test")
    tick[0] += 4
    assert clock.remaining() == 16
    with record_model_call(ledger):
        tick[0] += 200
    assert clock.snapshot()["model_call_seconds"] == 300
    assert clock.snapshot()["model_call_count"] == 2


def test_native_harbor_loop_excludes_each_model_request(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    import json

    from harbor.agents.terminus_2 import Terminus2
    from harbor.llms.base import LLMResponse
    from harbor.models.agent.context import AgentContext

    from exgentic.benchmarks.terminalbench2.terminus_limits import BudgetedTerminus2

    class FakeLLM:
        async def call(self, prompt, **kwargs):
            await asyncio.sleep(.3)
            return LLMResponse(
                content=json.dumps({"analysis": "done", "plan": "done", "task_complete": True, "commands": []}),
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10, cache_tokens=0, cost_usd=0),
            )

    class Terminal:
        async def is_session_alive(self):
            return True

        async def get_incremental_output(self):
            return "terminal"

    monkeypatch.setattr(Terminus2, "_init_llm", lambda *_args, **_kwargs: FakeLLM())
    clock = AgentTimeBudget(.2)
    actor = BudgetedTerminus2(logs_dir=tmp_path, model_name="openai/test", task_clock=clock,
                             enable_summarize=False, record_terminal_session=False)
    actor._session = Terminal()
    asyncio.run(clock.run(actor.run("task", SimpleNamespace(), AgentContext())))
    state = clock.stop()
    assert state["model_call_count"] == 2
    assert state["model_call_seconds"] >= .6
    assert state["charged_seconds"] < .2


def test_active_time_mode_removes_wall_clock_rpc_ceiling():
    from exgentic.benchmarks.terminalbench2.terminalbench2_benchmark import TerminalBench2Benchmark

    benchmark = TerminalBench2Benchmark(subset="2.1", exclude_model_call_time=True)
    assert benchmark.runner_kwargs()["transport_timeout"] is None
    assert benchmark._get_evaluator_kwargs()["exclude_model_call_time"] is True
