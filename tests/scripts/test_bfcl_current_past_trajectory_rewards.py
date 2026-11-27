# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Check current/past weighting, completion priority, and exact baseline parity."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from scripts.premise.bfcl_current_past_trajectory_rewards import (
    current_past_weights,
    primary_count_gap,
    weighted_objective_reward_scores,
    weighted_reward_scores,
)
from scripts.premise.bfcl_trajectory_selection_rewards import (
    choice_probabilities,
    objective_reward_scores,
    reward_scores,
)


def arguments(n: int, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    candidate_shape = (3, 5, n)
    parent_shape = (3, n)
    return {
        "success": rng.integers(0, 2, candidate_shape),
        "turns": rng.integers(0, 25, candidate_shape),
        "progress": rng.random(candidate_shape),
        "material": rng.integers(0, 2, candidate_shape),
        "avoidable": rng.integers(0, 4, candidate_shape),
        "parent_success": rng.integers(0, 2, parent_shape),
        "parent_turns": rng.integers(0, 25, parent_shape),
        "parent_progress": rng.random(parent_shape),
        "parent_material": rng.integers(0, 2, parent_shape),
    }


@pytest.mark.parametrize("n", range(1, 11))
def test_task_mean_is_exactly_the_original_seven_recipes(n):
    inputs = arguments(n)
    original = reward_scores(**inputs)
    actual = weighted_reward_scores(**inputs, weighting="task_mean")
    for key, values in original.items():
        np.testing.assert_array_equal(actual[key], values, err_msg=key)


@pytest.mark.parametrize("weighting", ("task_mean", "period_mean"))
def test_current_only_uses_identical_scores_for_both_weightings(weighting):
    inputs = arguments(1)
    original = reward_scores(**inputs)
    actual = weighted_reward_scores(**inputs, weighting=weighting)
    for key, values in original.items():
        np.testing.assert_array_equal(actual[key], values, err_msg=key)


@pytest.mark.parametrize("n", range(2, 11))
def test_period_mean_has_equal_current_and_past_total_weight(n):
    weights = current_past_weights(n, "period_mean")
    assert weights[0] == .5
    assert weights[1:].sum() == pytest.approx(.5)
    np.testing.assert_array_equal(weights[1:], np.full(n - 1, .5 / (n - 1)))
    binary = np.array(list(itertools.product((0, 1), repeat=n)))
    possible_counts = np.unique(np.round(n * (binary * weights).sum(axis=-1), 12))
    assert np.diff(possible_counts).min() == pytest.approx(primary_count_gap(n, "period_mean"))


def test_weighting_changes_current_vs_past_preference_without_using_unobserved_tasks():
    inputs = arguments(4)
    inputs["success"][:] = 0
    inputs["success"][:, 0, 0] = 1
    inputs["success"][:, 1, 1:3] = 1
    task_mean = weighted_reward_scores(**inputs, weighting="task_mean")["success_count"]
    period_mean = weighted_reward_scores(**inputs, weighting="period_mean")["success_count"]
    np.testing.assert_array_equal(choice_probabilities(task_mean), [[0, 1, 0, 0, 0]] * 3)
    np.testing.assert_array_equal(choice_probabilities(period_mean), [[1, 0, 0, 0, 0]] * 3)


@pytest.mark.parametrize("n", range(2, 11))
def test_smallest_weighted_success_advantage_dominates_secondary_signals(n):
    inputs = arguments(n)
    inputs["success"][:] = 0
    inputs["success"][:, 1, -1] = 1
    inputs["turns"][:] = 0
    inputs["turns"][:, 1, :] = 20
    inputs["progress"][:] = 1
    inputs["progress"][:, 1, :] = 0
    inputs["material"][:] = 0
    inputs["avoidable"][:] = 0
    inputs["avoidable"][:, 1, :] = 100
    scores = weighted_reward_scores(**inputs, weighting="period_mean")
    for key in ("success_efficiency", "failure_progress", "certified_completion", "successful_recovery"):
        assert (scores[key][:, 1] > scores[key][:, 0]).all(), key
        np.testing.assert_array_equal(choice_probabilities(scores[key]), [[0, 1, 0, 0, 0]] * 3, err_msg=key)


def test_parent_regression_and_repair_coordinates_use_the_same_period_weights():
    inputs = arguments(4)
    inputs["parent_success"][:] = [1, 0, 0, 0]
    inputs["success"][:] = [[0, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
    scores = weighted_reward_scores(**inputs, weighting="period_mean")["parent_safe"]
    np.testing.assert_allclose(scores[0, :, :2], [[-2, 2], [0, 2 / 3], [0, 0], [0, 0], [0, 0]])
    np.testing.assert_array_equal(choice_probabilities(scores), [[0, 1, 0, 0, 0]] * 3)


@pytest.mark.parametrize("n", (1, 2, 6, 10))
@pytest.mark.parametrize("weighting", ("task_mean", "period_mean"))
def test_trace_only_and_full_api_agree_and_do_not_mutate_inputs(n, weighting):
    inputs = arguments(n)
    copies = {key: value.copy() for key, value in inputs.items()}
    objective_inputs = {key: inputs[key] for key in ("success", "turns", "avoidable")}
    objective = weighted_objective_reward_scores(**objective_inputs, weighting=weighting)
    full = weighted_reward_scores(**inputs, weighting=weighting)
    for key, values in objective.items():
        np.testing.assert_array_equal(values, full[key])
    if weighting == "task_mean":
        for key, values in objective_reward_scores(**objective_inputs).items():
            np.testing.assert_array_equal(objective[key], values)
    for key, value in inputs.items():
        np.testing.assert_array_equal(value, copies[key])


def test_permuting_only_past_tasks_preserves_all_period_scores():
    inputs = arguments(10)
    original = weighted_reward_scores(**inputs, weighting="period_mean")
    order = [0, 9, 3, 5, 6, 2, 1, 7, 8, 4]
    changed = {key: value[..., order] for key, value in inputs.items()}
    for key, values in weighted_reward_scores(**changed, weighting="period_mean").items():
        np.testing.assert_allclose(values, original[key], atol=1e-14)


@pytest.mark.parametrize("n,weighting", ((0, "task_mean"), (11, "period_mean"), (1.5, "task_mean"),
                                         (True, "task_mean"), (2, "invalid")))
def test_invalid_weight_specifications_are_rejected(n, weighting):
    with pytest.raises(ValueError):
        current_past_weights(n, weighting)
    with pytest.raises(ValueError):
        primary_count_gap(n, weighting)


def test_missing_observations_are_rejected_instead_of_receiving_zero_credit():
    inputs = arguments(3)
    inputs["progress"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        weighted_reward_scores(**inputs, weighting="period_mean")
