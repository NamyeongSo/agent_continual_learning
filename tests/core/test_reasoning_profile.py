# SPDX-License-Identifier: Apache-2.0
"""Reasoning must be billed, but it must not consume the visible actor budget."""

import hashlib
import json

import pytest

from exgentic.core.experiment_limits import ActorTurnBudget, ExperimentBudgetExceededError, experiment_limits
from exgentic.inference_profile import apply_request, profile
from scripts.utils.run_shared_reflection_matrix import release_delay
from scripts.utils.serve_reasoning_bridge import repeated, split_tokens


@pytest.fixture
def active(tmp_path, monkeypatch):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "actor_temperature": 0.0,
                "analysis_temperature": 0.7,
                "reasoning_effort": "medium",
                "reasoning_tokens": 8192,
            }
        )
    )
    monkeypatch.setenv("EXGENTIC_INFERENCE_PROFILE", str(path))
    monkeypatch.setenv("EXGENTIC_INFERENCE_PROFILE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return path


@pytest.mark.parametrize("role,temperature,limit", [("actor", 0.0, 2048), ("analysis", 0.7, 8192)])
def test_every_role_activates_medium_thinking_and_preserves_visible_limit(active, role, temperature, limit):
    request = apply_request(
        {"max_tokens": 20000, "temperature": 1.0, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        role,
    )
    assert request["temperature"] == temperature
    assert request["reasoning_effort"] == "medium"
    assert request["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert request["max_tokens"] == limit


def test_actual_usage_is_unchanged_and_only_visible_tokens_consume_turn_budget(active):
    limits = experiment_limits("bfcl")
    assert limits["version"] == "actor-turn-v2-nonreasoning"
    budget = ActorTurnBudget(limits)
    budget.begin(1)
    usage = {"prompt_tokens": 100, "completion_tokens": 9000, "completion_tokens_details": {"reasoning_tokens": 8000}}
    budget.record(usage)
    assert budget.output_tokens == 1000
    assert usage["completion_tokens"] == 9000
    assert budget.request_kwargs({"max_tokens": 2048})["max_tokens"] == 1048
    budget.record(
        {"prompt_tokens": 100, "completion_tokens": 1048, "completion_tokens_details": {"reasoning_tokens": 0}}
    )
    with pytest.raises(ExperimentBudgetExceededError):
        budget.request_kwargs({})


def test_missing_reasoning_measurement_is_not_silently_treated_as_zero(active):
    budget = ActorTurnBudget(experiment_limits("bfcl"))
    budget.begin(1)
    with pytest.raises(ExperimentBudgetExceededError, match="missing_reasoning_usage"):
        budget.record({"prompt_tokens": 1, "completion_tokens": 5})


def test_modified_profile_rejected_before_a_new_consumer_uses_it(active, monkeypatch):
    monkeypatch.setenv("EXGENTIC_INFERENCE_PROFILE_SHA256", "wrong")
    with pytest.raises(ValueError, match="bytes changed"):
        profile()


def test_reasoning_boundary_and_obvious_repetition_detection():
    assert split_tokens([1, 2, 99, 7, 8], 99, [5, 6]) == 3
    assert split_tokens([1, 2, 5, 6, 7], 99, [5, 6]) == 2
    assert split_tokens([1, 2], 99, [5, 6]) == 2
    assert repeated(list(range(32)) * 12)
    assert not repeated(list(range(384)))


def test_seed_release_delays_only_stagger_gepa_and_skillopt():
    for method in ("ace", "premise"):
        assert [release_delay(method, seed) for seed in (42, 43, 44)] == [0, 0, 0]
    for method in ("gepa", "skillopt"):
        assert [release_delay(method, seed) for seed in (42, 43, 44)] == [0, 900, 1800]
def test_legacy_health_probe_is_narrow_and_never_bypasses_profiled_actor_checks():
    from scripts.utils.serve_reasoning_bridge import is_health_probe

    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
    assert is_health_probe(body)
    assert not is_health_probe({**body, "max_tokens": 2048})
    assert not is_health_probe({**body, "messages": [{"role": "user", "content": "do work"}]})
    assert not is_health_probe({**body, "exgentic_reasoning_budget": 16384})
    assert not is_health_probe({**body, "tools": [{"type": "function"}]})
