# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Regression checks for the experiment actor policy, including exact boundaries."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import litellm
import pytest

from exgentic.core.context import run_scope
from exgentic.core.experiment_limits import (
    LIMITS_CONTEXT_KEY,
    ActorTurnBudget,
    ExperimentBudgetExceededError,
    actor_completion,
    budget_error,
    experiment_limits,
)
from exgentic.core.orchestrator.session import run_session
from exgentic.core.types import EmptyObservation, Message, MessageAction, RunConfig, SessionConfig, SessionScore


def response(inputs=1, outputs=1, finish="stop"):
    return litellm.ModelResponse(
        choices=[{"finish_reason": finish, "message": {"role": "assistant", "content": "answer"}}],
        usage={"prompt_tokens": inputs, "completion_tokens": outputs},
    )


@pytest.mark.parametrize("benchmark,cap", [("bfcl", 30), ("appworld", 40), ("terminalbench2", 40)])
def test_policy_normalizes_old_launchers_and_is_in_session_identity(benchmark, cap):
    config = RunConfig(
        benchmark=benchmark, agent="premise", task_ids=["task"],
        agent_kwargs={"model_settings": {"max_tokens": 8192}}, max_steps=100,
    )
    assert config.max_steps == cap
    assert config.agent_kwargs["model_settings"].max_tokens == 2048
    session = config.to_session_config("task")
    assert session.experiment_limits == experiment_limits(benchmark)
    assert session.session_id_payload()["experiment_limits"]["max_turns"] == cap
    legacy = session.model_copy(update={"experiment_limits": None})
    assert session.get_session_id() != legacy.get_session_id()
    assert SessionConfig.model_validate(session.model_dump()).experiment_limits == session.experiment_limits


def test_analysis_settings_are_independent_of_actor_hard_cap():
    config = RunConfig(benchmark="appworld", agent="ace", agent_kwargs={"model_settings": {"max_tokens": 8192}})
    assert config.agent_kwargs["model_settings"].max_tokens == 8192
    assert config.experiment_limits["output_tokens"] == 2048


def test_turn_cap_pct_policy_is_explicit_and_appworld_only(monkeypatch):
    monkeypatch.setenv("EXGENTIC_ACTOR_OUTPUT_UNLIMITED", "1")
    monkeypatch.delenv("EXGENTIC_INFERENCE_PROFILE", raising=False)
    monkeypatch.delenv("EXGENTIC_APPWORLD_TURN_CAP_PCT", raising=False)
    original = experiment_limits("appworld")
    assert original["version"] == "actor-turn-v3-output-unlimited"
    monkeypatch.setenv("EXGENTIC_APPWORLD_TURN_CAP_PCT", "1")
    updated = experiment_limits("appworld")
    assert updated["version"] != original["version"]
    assert updated["turn_limit_scoring"] == "official_pct_at_limit"
    assert updated["max_turns"] == 40 and updated["input_tokens"] == 100000 and updated["output_tokens"] is None
    assert "turn_limit_scoring" not in experiment_limits("bfcl")
    assert "turn_limit_scoring" not in experiment_limits("terminalbench2")


@pytest.mark.parametrize("field,limit", [("input_tokens", 100_000), ("output_tokens", 2048)])
def test_provider_usage_accepts_exact_limit_and_rejects_one_more(field, limit):
    budget = ActorTurnBudget(experiment_limits("bfcl"))
    budget.begin(1)
    values = {"prompt_tokens": 1, "completion_tokens": 1}
    key = "prompt_tokens" if field == "input_tokens" else "completion_tokens"
    values[key] = limit
    budget.record(values)
    budget.begin(2)
    values[key] += 1
    with pytest.raises(ExperimentBudgetExceededError) as failure:
        budget.record(values)
    assert failure.value.details["reason"] == field
    assert failure.value.details["actual"] == limit + 1
    restored = budget_error(RuntimeError(f"RPC error: {failure.value}"))
    assert restored.details == failure.value.details


def test_selection_and_acting_share_turn_budget_and_next_turn_resets_it(monkeypatch):
    calls = []
    replies = iter([response(45_000, 500), response(55_000, 1548), response(100_000, 2048)])

    def complete(**kwargs):
        calls.append(kwargs)
        return next(replies)

    monkeypatch.setattr(litellm, "completion", complete)
    actor = SimpleNamespace(_actor_turn_budget=ActorTurnBudget(experiment_limits("appworld")), _step_count=1)
    actor_completion(actor, {"max_tokens": 8192})
    actor_completion(actor, {"max_tokens": 8192})
    assert actor._actor_turn_budget.input_tokens == 100_000
    assert [call["max_tokens"] for call in calls] == [2048, 1548]
    actor._step_count = 2
    actor_completion(actor, {"max_tokens": 8192})
    assert calls[-1]["max_tokens"] == 2048
    assert actor._actor_turn_budget.input_tokens == 100_000


def test_input_usage_is_summed_even_when_each_request_fits(monkeypatch):
    monkeypatch.setattr(litellm, "completion", lambda **_kwargs: response(50_001, 100))
    actor = SimpleNamespace(_actor_turn_budget=ActorTurnBudget(experiment_limits("appworld")), _step_count=1)
    actor_completion(actor, {})
    with pytest.raises(ExperimentBudgetExceededError, match="reason=input_tokens"):
        actor_completion(actor, {})


def test_provider_context_rejection_is_a_terminal_failure_with_unknown_usage(monkeypatch):
    def complete(**_kwargs):
        raise ValueError("maximum context length exceeded")

    monkeypatch.setattr(litellm, "completion", complete)
    actor = SimpleNamespace(_actor_turn_budget=ActorTurnBudget(experiment_limits("bfcl")), _step_count=1)
    with pytest.raises(ExperimentBudgetExceededError, match="reason=input_context_exceeded") as caught:
        actor_completion(actor, {})
    assert caught.value.details["actual"] == 0


@pytest.mark.parametrize("agent_kind", ["premise", "ace"])
def test_truncated_actor_output_is_a_terminal_failure_without_retries(monkeypatch, agent_kind):
    from exgentic.agents.ace.ace_instance import ACEAgentInstance
    from exgentic.agents.premise.instance import PremiseAgentInstance
    from exgentic.core.types import ModelSettings

    cls = PremiseAgentInstance if agent_kind == "premise" else ACEAgentInstance
    actor = object.__new__(cls)
    actor._actor_turn_budget = ActorTurnBudget(experiment_limits("appworld"))
    actor._step_count = 1
    actor.model_settings = actor._model_settings = ModelSettings(num_retries=5)
    actor.enable_thinking = False
    actor._register_cost = lambda _usage: None
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return response(100, 2048, "length")

    monkeypatch.setattr(litellm, "completion", complete)
    with pytest.raises(ExperimentBudgetExceededError, match="reason=output_truncated"):
        if agent_kind == "premise":
            actor._completion_with_retries({})
        else:
            actor._completion()
    assert len(calls) == 1


@pytest.mark.parametrize("benchmark,cap", [("bfcl", 30), ("appworld", 40), ("terminalbench2", 40)])
@pytest.mark.parametrize("finish_at_limit", [False, True])
@pytest.mark.parametrize("prefix", [0, 5])
def test_turn_limit_includes_prefix_and_allows_completion_on_last_turn(
    tmp_path, benchmark, cap, finish_at_limit, prefix,
):
    class Session:
        session_id = "test"
        task_id = "task"
        task = "task"
        context: ClassVar[dict] = {}
        actions: ClassVar[list] = []
        steps = 0

        def start(self):
            return EmptyObservation()

        def done(self):
            return finish_at_limit and self.steps + prefix == cap

        def step(self, action):
            self.steps += 1
            return EmptyObservation()

        def score(self):
            assert finish_at_limit
            return SessionScore(score=1.0, success=True)

        def close(self):
            pass

    class Actor:
        def start(self, **kwargs):
            assert kwargs["context"][LIMITS_CONTEXT_KEY]["prefix_turns"] == prefix

        def react(self, observation):
            return MessageAction(arguments=Message(content="act"))

        def close(self):
            pass

    class Tracker:
        score = None

        def __getattr__(self, name):
            return lambda *args: None

        def on_session_success(self, session, score, actor):
            self.score = score

    session, tracker = Session(), Tracker()
    config = SimpleNamespace(benchmark=benchmark, replay_prefix={"actions": [{}] * prefix})
    agent = SimpleNamespace(get_instance=lambda **_kwargs: Actor())
    with run_scope(run_id="budget-test", output_dir=str(tmp_path)):
        run_session(config, session, agent, tracker=tracker)
    assert session.steps == cap - prefix
    assert tracker.score.score == float(finish_at_limit)
    assert tracker.score.success is finish_at_limit
    if not finish_at_limit:
        assert tracker.score.is_finished is True
        assert tracker.score.session_metadata["experiment_budget_failure"]["reason"] == "max_turns"


def test_bfcl_budget_failure_survives_scoring_and_cleanup(tmp_path):
    import json

    from exgentic.benchmarks.bfcl.bfcl_eval import BFCLSession

    session = object.__new__(BFCLSession)
    session._paths = SimpleNamespace(benchmark_results=tmp_path / "results.json")
    failure = ExperimentBudgetExceededError("output_tokens", limit=2048, actual=2049, turn=1)
    # No official checker or result payload is initialized: neither may be used.
    session.fail_experiment_budget(failure.details, experiment_limits("bfcl"))
    assert session.score().score == 0
    session.close()
    payload = json.loads(session.paths.benchmark_results.read_text())
    assert payload["score"] == 0
    assert payload["success"] is False
    assert payload["session_metadata"]["experiment_budget_exceeded"] is True


def test_checkpoint_queue_override_cannot_retry_truncated_actor_output(monkeypatch):
    from scripts.utils import evaluate_isolated_checkpoint_queue as queue

    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return response(100, 2048, "length")

    monkeypatch.setattr(litellm, "completion", complete)
    monkeypatch.setattr(queue, "_AUDIT", {"max_tokens": 8192, "transport_additional_retries": 5})
    actor = SimpleNamespace(_actor_turn_budget=ActorTurnBudget(experiment_limits("bfcl")), _step_count=1)
    with pytest.raises(ExperimentBudgetExceededError, match="reason=output_truncated"):
        queue.actor_request_completion(actor, {"max_tokens": 8192})
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 2048
    assert queue._AUDIT["model_calls"] == 1
    assert queue._AUDIT.get("output_length_retries", 0) == 0
