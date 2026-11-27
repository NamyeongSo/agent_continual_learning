# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import pytest

from exgentic.core.experiment_limits import ActorTurnBudget, ExperimentBudgetExceededError, experiment_limits
from exgentic.core.types import RunConfig


def test_unlimited_output_removes_wire_caps_and_accepts_large_responses(monkeypatch):
    monkeypatch.setenv("EXGENTIC_ACTOR_OUTPUT_UNLIMITED", "1")
    monkeypatch.delenv("EXGENTIC_INFERENCE_PROFILE", raising=False)
    limits = experiment_limits("terminalbench2")
    assert limits["output_tokens"] is None
    assert limits["version"] == "actor-turn-v3-output-unlimited"
    budget = ActorTurnBudget(limits)
    budget.begin(1)
    assert budget.request_kwargs({"max_tokens": 2048, "max_completion_tokens": 2048, "temperature": 1}) == {
        "temperature": 1}
    budget.record({"prompt_tokens": 1000, "completion_tokens": 50000})
    assert budget.output_tokens == 50000
    assert budget.request_kwargs({}) == {}
    config = RunConfig(benchmark="terminalbench2", agent="tool_calling", model="openai/test",
                       agent_kwargs={"model_settings": {"max_tokens": 2048}})
    assert config.agent_kwargs["model_settings"].max_tokens is None
    with pytest.raises(ExperimentBudgetExceededError, match="input_tokens"):
        budget.record({"prompt_tokens": 100001, "completion_tokens": 10})


def test_unlimited_output_retains_turn_limit_and_truncation_detection(monkeypatch):
    monkeypatch.setenv("EXGENTIC_ACTOR_OUTPUT_UNLIMITED", "1")
    monkeypatch.delenv("EXGENTIC_INFERENCE_PROFILE", raising=False)
    budget = ActorTurnBudget(experiment_limits("terminalbench2"))
    with pytest.raises(ExperimentBudgetExceededError, match="max_turns"):
        budget.begin(41)
    budget.begin(1)
    with pytest.raises(ExperimentBudgetExceededError, match="output_truncated"):
        budget.record({"prompt_tokens": 1000, "completion_tokens": 200000}, truncated=True)


def test_existing_runs_retain_their_output_budget(monkeypatch):
    monkeypatch.delenv("EXGENTIC_ACTOR_OUTPUT_UNLIMITED", raising=False)
    monkeypatch.delenv("EXGENTIC_INFERENCE_PROFILE", raising=False)
    assert experiment_limits("terminalbench2")["output_tokens"] == 2048
