# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Check trajectory reward semantics, anti-short-failure gates, and n-task isolation."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.premise.bfcl_trajectory_selection_rewards import (
    LEXICOGRAPHIC_KEYS,
    REWARD_KEYS,
    choice_probabilities,
    objective_reward_scores,
    ordering_ranks,
    reward_scores,
)


def inputs(success: np.ndarray) -> dict:
    success = np.asarray(success, dtype=float)
    trials, _, count = success.shape
    return {
        "success": success,
        "turns": np.full(success.shape, 10.),
        "progress": np.zeros(success.shape),
        "material": np.zeros(success.shape),
        "avoidable": np.zeros(success.shape),
        "parent_success": np.zeros((trials, count)),
        "parent_turns": np.full((trials, count), 10.),
        "parent_progress": np.zeros((trials, count)),
        "parent_material": np.zeros((trials, count)),
    }


def test_short_failure_never_beats_long_success_under_completion_primary_rules():
    arguments = inputs(np.array([[[0], [1], [0], [0], [0]]]))
    arguments["turns"][0, :, 0] = [0, 20, 1, 1, 1]
    arguments["progress"][0, :, 0] = [1, 0, 1, 1, 1]
    scores = reward_scores(**arguments)
    for key, values in scores.items():
        np.testing.assert_array_equal(choice_probabilities(values), [[0, 1, 0, 0, 0]], err_msg=key)


@pytest.mark.parametrize("count", (1, 4, 10))
def test_one_more_success_dominates_all_bounded_secondary_effects(count):
    arguments = inputs(np.zeros((1, 5, count)))
    arguments["success"][0, 1, 0] = 1
    arguments["turns"][0, 1, :] = 20
    arguments["turns"][0, 0, :] = 0
    arguments["progress"][0, 0, :] = 1
    arguments["avoidable"][0, 1, :] = 100
    scores = reward_scores(**arguments)
    for key in ("success_efficiency", "failure_progress", "successful_recovery"):
        assert scores[key][0, 1] > scores[key][0, 0]
        np.testing.assert_array_equal(choice_probabilities(scores[key]), [[0, 1, 0, 0, 0]])


def test_evidence_progress_breaks_failed_binary_ties_without_rewarding_attempt_count():
    arguments = inputs(np.zeros((1, 5, 2)))
    arguments["progress"][0, :, :] = [[.5, .5], [0, 0], [1, 0], [0, 0], [0, 0]]
    arguments["turns"][0, 1, :] = 20
    scores = reward_scores(**arguments)
    np.testing.assert_array_equal(choice_probabilities(scores["success_count"]), [[.2] * 5])
    np.testing.assert_array_equal(choice_probabilities(scores["failure_progress"]), [[.5, 0, .5, 0, 0]])


def test_confirmed_material_violation_declassifies_only_the_affected_task():
    arguments = inputs(np.array([[[1, 1], [1, 0], [0, 0], [0, 0], [0, 0]]]))
    arguments["material"][0, 0, :] = 1
    arguments["progress"][0, 0, :] = 1
    scores = reward_scores(**arguments)
    assert scores["certified_completion"][0, 0] == 0
    np.testing.assert_array_equal(choice_probabilities(scores["certified_completion"]), [[0, 1, 0, 0, 0]])
    assert scores["success_count"][0, 0] == 2
    arguments["material"][0, 0, :] = 0
    np.testing.assert_array_equal(
        choice_probabilities(reward_scores(**arguments)["certified_completion"]), [[1, 0, 0, 0, 0]],
    )


def test_parent_safe_minimizes_regressions_before_maximizing_repairs():
    arguments = inputs(np.array([[[1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 0]]]))
    arguments["parent_success"][:] = [[1, 0, 0]]
    score = reward_scores(**arguments)["parent_safe"]
    np.testing.assert_array_equal(score[0, :, :2], [[0, 0], [-1, 2], [0, 1], [0, 1], [-1, 0]])
    np.testing.assert_array_equal(choice_probabilities(score), [[0, 0, .5, .5, 0]])


def test_parent_relative_efficiency_uses_only_certified_joint_successes():
    arguments = inputs(np.array([[[1, 0], [1, 0], [1, 0], [1, 0], [0, 1]]]))
    arguments["parent_success"][:] = [[1, 0]]
    arguments["turns"][0, :, :] = [[5, 0], [5, 20], [5, 0], [10, 0], [0, 1]]
    arguments["material"][0, 2, 0] = 1
    score = reward_scores(**arguments)["parent_safe"]
    np.testing.assert_allclose(score[0, :, 3], [.25, .25, 0, 0, 0])
    np.testing.assert_array_equal(choice_probabilities(score), [[.5, .5, 0, 0, 0]])


def test_parent_safe_progress_compares_only_joint_failures():
    arguments = inputs(np.array([[[1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]]))
    arguments["parent_success"][:] = [[1, 0]]
    arguments["parent_progress"][:] = [[0, .2]]
    arguments["progress"][0, :, :] = [[1, .4], [0, .4], [1, .2], [1, 0], [1, 0]]
    score = reward_scores(**arguments)["parent_safe"]
    np.testing.assert_allclose(score[0, :, 2], [.1, .1, 0, -.1, -.1])
    np.testing.assert_array_equal(choice_probabilities(score), [[.5, .5, 0, 0, 0]])


def test_successful_recovery_ignores_error_counts_of_failed_executions():
    arguments = inputs(np.array([[[1, 0], [1, 0], [0, 0], [0, 0], [0, 0]]]))
    arguments["avoidable"][0, 0, :] = [0, 20]
    arguments["avoidable"][0, 1, :] = [2, 0]
    scores = reward_scores(**arguments)
    assert scores["successful_recovery"][0, 0] == 1
    assert scores["successful_recovery"][0, 1] == pytest.approx(1 - .25 * .1 / 2)


def test_quality_gates_use_fixed_ten_bins_then_avoidable_errors_then_turns():
    arguments = inputs(np.ones((1, 5, 1)))
    arguments["progress"][0, :, 0] = [.049, .051, .051, .051, 1]
    arguments["avoidable"][0, :, 0] = [0, 1, 0, 0, 0]
    arguments["turns"][0, :, 0] = [0, 0, 10, 5, 0]
    arguments["material"][0, 4, 0] = 1
    score = reward_scores(**arguments)["quality_gated"]
    np.testing.assert_array_equal(score[0, :, 2], [0, 1, 1, 1, 10])
    np.testing.assert_array_equal(choice_probabilities(score), [[0, 0, 0, 1, 0]])


@pytest.mark.parametrize("count", range(1, 11))
def test_recipe_api_accepts_only_given_n_observations_and_expected_output_shapes(count):
    arguments = inputs(np.zeros((3, 5, count)))
    scores = reward_scores(**arguments)
    assert tuple(scores) == REWARD_KEYS
    for key, score in scores.items():
        expected = (3, 5, 4 if key == "parent_safe" else 5) if key in LEXICOGRAPHIC_KEYS else (3, 5)
        assert score.shape == expected
        np.testing.assert_array_equal(choice_probabilities(score), np.full((3, 5), .2))


def test_ordering_ranks_and_uniform_choices_use_first_differing_lex_coordinate():
    values = np.array([[[0, 100], [1, -100], [1, -100], [1, 1], [0, 1000]]], dtype=float)
    np.testing.assert_array_equal(ordering_ranks(values), [[1, 3.5, 3.5, 5, 2]])
    np.testing.assert_array_equal(choice_probabilities(values), [[0, 0, 0, 1, 0]])
    values[0, 1, 1] = 1
    values[0, 2, 1] = 1 + 1e-14
    np.testing.assert_array_equal(ordering_ranks(values), [[1, 4, 4, 4, 2]])
    np.testing.assert_allclose(choice_probabilities(values), [[0, 1 / 3, 1 / 3, 1 / 3, 0]])
    np.testing.assert_array_equal(ordering_ranks(np.array([[1, 1, 0, 2, 0]])), [[3.5, 3.5, 1.5, 5, 1.5]])


def test_task_candidate_permutations_preserve_semantics_without_mutating_inputs():
    rng = np.random.default_rng(42)
    arguments = inputs(rng.integers(0, 2, size=(4, 5, 7)))
    arguments["turns"] = rng.integers(0, 25, size=(4, 5, 7))
    arguments["progress"] = rng.random((4, 5, 7))
    arguments["avoidable"] = rng.integers(0, 3, size=(4, 5, 7))
    arguments["material"] = rng.integers(0, 2, size=(4, 5, 7))
    arguments["parent_success"] = rng.integers(0, 2, size=(4, 7))
    arguments["parent_progress"] = rng.random((4, 7))
    copies = {key: value.copy() for key, value in arguments.items()}
    original = reward_scores(**arguments)
    candidates = np.array([3, 1, 0, 4, 2])
    changed = {key: value if key.startswith("parent_") else value[:, candidates, :]
               for key, value in arguments.items()}
    for key, score in reward_scores(**changed).items():
        np.testing.assert_allclose(score, original[key][:, candidates, ...])
    tasks = np.array([3, 1, 0, 4, 2, 6, 5])
    changed = {key: value[..., tasks] for key, value in arguments.items()}
    for key, score in reward_scores(**changed).items():
        np.testing.assert_allclose(score, original[key], atol=1e-14)
    for key, value in arguments.items():
        np.testing.assert_array_equal(value, copies[key])


@pytest.mark.parametrize("invalid", ("success", "turns", "progress", "material", "avoidable",
                                     "parent_success", "parent_turns", "parent_progress", "parent_material"))
def test_nonfinite_inputs_are_rejected_instead_of_silently_becoming_zero(invalid):
    arguments = inputs(np.zeros((1, 5, 2)))
    arguments[invalid].flat[0] = np.nan
    with pytest.raises(ValueError):
        reward_scores(**arguments)


@pytest.mark.parametrize("invalid,value", (("success", .5), ("material", -.1), ("progress", 1.1),
                                          ("turns", -.1), ("turns", 1.5), ("avoidable", -.1)))
def test_invalid_ranges_and_noninteger_counts_are_rejected(invalid, value):
    arguments = inputs(np.zeros((1, 5, 2)))
    arguments[invalid].flat[0] = value
    with pytest.raises(ValueError):
        reward_scores(**arguments)


@pytest.mark.parametrize("count", (0, 11))
def test_n_outside_the_frozen_one_to_ten_budget_is_rejected(count):
    with pytest.raises(ValueError):
        reward_scores(**inputs(np.zeros((1, 5, count))))


@pytest.mark.parametrize("max_steps", (0, -1, 1.5, np.nan))
def test_invalid_step_limits_are_rejected(max_steps):
    with pytest.raises(ValueError):
        reward_scores(**inputs(np.zeros((1, 5, 2))), max_steps=max_steps)


def test_mismatched_parent_or_candidate_shapes_are_rejected():
    arguments = inputs(np.zeros((2, 5, 3)))
    arguments["parent_success"] = np.zeros((1, 3))
    with pytest.raises(ValueError):
        reward_scores(**arguments)
    arguments = inputs(np.zeros((2, 5, 3)))
    arguments["turns"] = np.zeros((2, 4, 3))
    with pytest.raises(ValueError):
        reward_scores(**arguments)


def test_trace_only_api_matches_full_recipes_without_judge_or_parent_placeholders():
    rng = np.random.default_rng(7)
    arguments = inputs(rng.integers(0, 2, size=(4, 5, 10)))
    arguments["turns"] = rng.integers(0, 25, size=(4, 5, 10))
    arguments["avoidable"] = rng.integers(0, 4, size=(4, 5, 10))
    arguments["progress"] = rng.random((4, 5, 10))
    arguments["material"] = rng.integers(0, 2, size=(4, 5, 10))
    arguments["parent_progress"] = rng.random((4, 10))
    full = reward_scores(**arguments)
    objective = objective_reward_scores(**{key: arguments[key] for key in ("success", "turns", "avoidable")})
    assert tuple(objective) == ("success_count", "success_efficiency", "successful_recovery")
    for key, value in objective.items():
        np.testing.assert_array_equal(value, full[key])


@pytest.mark.parametrize("values", (np.ones((5,)), np.ones((1, 4)), np.ones((1, 5, 0)),
                                    np.full((1, 5), np.nan)))
def test_invalid_ordering_arrays_are_rejected(values):
    with pytest.raises(ValueError):
        ordering_ranks(values)
