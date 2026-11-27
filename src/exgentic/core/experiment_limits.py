# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Shared, inclusive actor budgets for all benchmark experiment methods.

A turn includes tool selection, acting, and any measured retries. Reflection,
mutation, judging, and other learning requests have independent budgets.
"""

from __future__ import annotations

import os
import re
from typing import Any

ACTOR_INPUT_TOKENS = 100_000
ACTOR_OUTPUT_TOKENS = 2_048
BENCHMARK_MAX_TURNS = {"bfcl": 30, "appworld": 40, "terminalbench2": 40}
LIMITS_CONTEXT_KEY = "_exgentic_experiment_limits"
POLICY_VERSION = "actor-turn-v1"


def experiment_limits(benchmark: str) -> dict[str, Any] | None:
    if benchmark not in BENCHMARK_MAX_TURNS:
        return None
    result = {
        "version": POLICY_VERSION,
        "input_tokens": ACTOR_INPUT_TOKENS,
        "output_tokens": ACTOR_OUTPUT_TOKENS,
        "max_turns": BENCHMARK_MAX_TURNS[benchmark],
        "token_scope": "environment_turn",
    }
    from ..inference_profile import profile

    active = profile()
    if os.environ.get("EXGENTIC_ACTOR_OUTPUT_UNLIMITED") == "1":
        result.update(version="actor-turn-v3-output-unlimited", output_tokens=None)
        if active:
            raise ValueError("Unlimited output must not inherit an inference profile with output caps")
    elif active:
        result.update(version="actor-turn-v2-nonreasoning", output_excludes_reasoning=True,
                      reasoning_tokens_per_request=active["reasoning_tokens"],
                      inference_profile_sha256=active["sha256"])
    if benchmark == "appworld" and os.environ.get("EXGENTIC_APPWORLD_TURN_CAP_PCT") == "1":
        result.update(version=result["version"] + "-appworld-turncap-pct-v1",
                      turn_limit_scoring="official_pct_at_limit")
    return result


class ExperimentBudgetExceededError(ValueError):
    """A terminal task failure, with a marker that survives runner transport."""

    def __init__(self, reason: str, *, limit: int, actual: int, turn: int,
                 input_tokens: int = 0, output_tokens: int = 0):
        self.details = {
            "reason": reason, "limit": limit, "actual": actual, "turn": turn,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
        }
        super().__init__("ExperimentBudgetExceeded " + " ".join(f"{k}={v}" for k, v in self.details.items()))


def budget_error(error: Exception) -> ExperimentBudgetExceededError | None:
    """Recover a budget failure from an exception or a serialized RPC error."""
    if isinstance(error, ExperimentBudgetExceededError):
        return error
    match = re.search(
        r"ExperimentBudgetExceeded reason=(\w+) limit=(\d+) actual=(\d+) turn=(\d+) "
        r"input_tokens=(\d+) output_tokens=(\d+)", str(error),
    )
    if match is None:
        return None
    reason, limit, actual, turn, inputs, outputs = match.groups()
    return ExperimentBudgetExceededError(
        reason, limit=int(limit), actual=int(actual), turn=int(turn),
        input_tokens=int(inputs), output_tokens=int(outputs),
    )


def budget_failure_score(details: dict[str, Any], **metadata: Any):
    from .types.session import SessionScore

    return SessionScore(
        score=0.0, success=False, is_finished=True,
        session_metadata={
            **metadata, "experiment_budget_exceeded": True,
            "experiment_budget_failure": details,
        },
    )


class ActorTurnBudget:
    """Accumulate provider-reported usage across requests in one environment turn."""

    def __init__(self, limits: dict[str, Any], *, prefix_turns: int = 0):
        self.limits = limits
        self.prefix_turns = prefix_turns
        self.turn = 0
        self.input_tokens = self.output_tokens = 0

    def begin(self, continuation_turn: int) -> None:
        turn = self.prefix_turns + continuation_turn
        if turn != self.turn:
            self.turn = turn
            self.input_tokens = self.output_tokens = 0
        if turn > self.limits["max_turns"]:
            self.fail("max_turns", self.limits["max_turns"], turn)

    def fail(self, reason: str, limit: int, actual: int) -> None:
        raise ExperimentBudgetExceededError(
            reason, limit=limit, actual=actual, turn=self.turn,
            input_tokens=self.input_tokens, output_tokens=self.output_tokens,
        )

    def request_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if self.limits["output_tokens"] is None:
            result = dict(kwargs)
            result.pop("max_tokens", None)
            result.pop("max_completion_tokens", None)
            return result
        remaining = self.limits["output_tokens"] - self.output_tokens
        if remaining <= 0:
            self.fail("output_tokens", self.limits["output_tokens"], self.output_tokens + 1)
        result = dict(kwargs)
        # Never send two competing output limits to a provider.
        requested = result.pop("max_completion_tokens", None) or result.get("max_tokens") or remaining
        result["max_tokens"] = min(requested, remaining)
        return result

    def record(self, usage: Any, *, truncated: bool = False) -> None:
        def count(name):
            return usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)

        inputs, outputs = count("prompt_tokens"), count("completion_tokens")
        if inputs is None or outputs is None:
            self.fail("missing_token_usage", 0, 1)
        if self.limits.get("output_excludes_reasoning"):
            from ..inference_profile import reasoning_tokens

            try:
                thinking = reasoning_tokens(usage)
            except ValueError:
                self.fail("missing_reasoning_usage", 0, 1)
            if not 0 <= thinking <= int(outputs):
                self.fail("invalid_reasoning_usage", int(outputs), thinking)
            outputs = int(outputs) - thinking
        self.input_tokens += int(inputs)
        self.output_tokens += int(outputs)
        for name in ("input_tokens", "output_tokens"):
            actual = getattr(self, name)
            if self.limits[name] is not None and actual > self.limits[name]:
                self.fail(name, self.limits[name], actual)
        if truncated:
            self.fail("output_truncated", self.limits["output_tokens"] or 0, self.output_tokens)


def actor_completion(agent, call_kwargs: dict[str, Any], *, completion_fn=None):
    """Call an actor model and reject over-budget output before executing actions."""
    import litellm

    complete = completion_fn or litellm.completion
    from ..inference_profile import apply_request
    budget = getattr(agent, "_actor_turn_budget", None)
    if budget is None:
        return complete(**apply_request(call_kwargs, "actor"))
    budget.begin(agent._step_count)
    kwargs = apply_request(budget.request_kwargs(call_kwargs), "actor")
    if kwargs.get("reasoning_effort") is not None:
        kwargs["allowed_openai_params"] = list(set(kwargs.get("allowed_openai_params", [])) | {"reasoning_effort"})
    # All retries must pass this function; hidden retries cannot be accounted for.
    kwargs["num_retries"] = 0
    kwargs["max_retries"] = 0
    try:
        response = complete(**kwargs)
    except Exception as exc:
        from ..utils.context_errors import is_context_length_error

        if is_context_length_error(exc):
            # Providers can reject an excessive prompt before reporting usage.
            # Keep that as a terminal actor failure instead of an infrastructure retry.
            budget.fail("input_context_exceeded", budget.limits["input_tokens"], 0)
        raise
    truncated = any(choice.get("finish_reason") == "length" for choice in response["choices"])
    try:
        budget.record(response.usage, truncated=truncated)
    except ExperimentBudgetExceededError:
        # Successful calls are charged by the actor. Rejected calls never reach
        # that code, but still incurred provider usage.
        register = getattr(agent, "_register_cost", None)
        if register is not None and response.usage is not None:
            register(response.usage)
        elif (cost := getattr(agent, "_cost", None)) is not None and response.usage is not None:
            cost.update_cost_from_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)
        raise
    return response
