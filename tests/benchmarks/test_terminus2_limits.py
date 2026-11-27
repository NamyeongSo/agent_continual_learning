# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Exercise the shared budgets through the actual Harbor conversation loop."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from exgentic.core.experiment_limits import ExperimentBudgetExceededError


@pytest.mark.parametrize("failure", [None, "input", "output", "truncated", "turns", "context"])
def test_native_actor_rejects_budgets_before_commands_and_accepts_last_turn(tmp_path, monkeypatch, failure):
    pytest.importorskip("harbor")
    from harbor.agents.terminus_2 import Terminus2
    from harbor.llms.base import ContextLengthExceededError, LLMResponse, OutputLengthExceededError
    from harbor.models.agent.context import AgentContext

    from exgentic.benchmarks.terminalbench2.terminus_limits import BudgetedTerminus2

    calls, executed = [], []

    class FakeLLM:
        async def call(self, prompt, **kwargs):
            calls.append(kwargs)
            assert kwargs["max_tokens"] == 2048
            if failure == "truncated":
                raise OutputLengthExceededError("truncated", truncated_response="partial")
            if failure == "context":
                raise ContextLengthExceededError("prompt too long")
            complete = failure != "turns" and len(calls) >= 39
            return LLMResponse(
                content=json.dumps({
                    "analysis": "work", "plan": "verify", "task_complete": complete,
                    "commands": [] if complete else [{"keystrokes": "pwd\n", "duration": 0.1}],
                }),
                usage=SimpleNamespace(
                    prompt_tokens=100_001 if failure == "input" else 100_000,
                    completion_tokens=2049 if failure == "output" else 2048,
                    cache_tokens=0, cost_usd=0,
                ),
            )

    class Terminal:
        async def is_session_alive(self):
            return True

        async def send_keys(self, keys, **_kwargs):
            executed.append(keys)

        async def get_incremental_output(self):
            return "fresh observation"

    monkeypatch.setattr(Terminus2, "_init_llm", lambda *_args, **_kwargs: FakeLLM())
    actor = BudgetedTerminus2(
        logs_dir=tmp_path, model_name="openai/test", max_turns=100,
        enable_summarize=False, record_terminal_session=False,
    )
    actor._session = Terminal()
    context = AgentContext()
    if failure is None:
        asyncio.run(actor.run("task", SimpleNamespace(), context))
        assert len(calls) == 40
        assert len(executed) == 38
    else:
        with pytest.raises(ExperimentBudgetExceededError) as exc:
            asyncio.run(actor.run("task", SimpleNamespace(), context))
        expected = {"input": "input_tokens", "output": "output_tokens",
                    "truncated": "output_truncated", "turns": "max_turns",
                    "context": "input_context_exceeded"}[failure]
        assert exc.value.details["reason"] == expected
        assert len(calls) == (40 if failure == "turns" else 1)
        assert len(executed) == (40 if failure == "turns" else 0)
    saved = json.loads((tmp_path / "premise_episodes.json").read_text())
    trace = json.loads((tmp_path / "trajectory.json").read_text())
    assert len(saved["episodes"]) == len([step for step in trace["steps"] if step["source"] == "agent"])


def test_native_budget_failure_returns_zero_without_running_verifier():
    from exgentic.benchmarks.terminalbench2.terminalbench2_eval import TerminalBench2Session

    session = object.__new__(TerminalBench2Session)
    session._score = None
    session._experiment_budget_failure = ExperimentBudgetExceededError(
        "input_tokens", limit=100_000, actual=100_001, turn=1,
    ).details
    session._terminus_trace_path = "trajectory.json"
    session._terminus_usage = {"input_tokens": 100_001, "output_tokens": 20}
    session._terminus_trace = ""
    session._actor = "terminus2"
    session.save_results = lambda *_args: None
    # No environment or loop runner: a verifier invocation would fail this test.
    score = session.score()
    assert score.score == 0
    assert score.success is False
    assert score.is_finished is True
