# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Check semantic reward identities, observation isolation, and small-sample degeneracies."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from scripts.premise.bfcl_full_pool_rewards import REWARD_KEYS, reward_metadata, reward_scores


def inputs(candidate: np.ndarray, parent: np.ndarray | None = None) -> dict:
    candidate = np.asarray(candidate, dtype=float)
    trials, _, count = candidate.shape
    return {
        "candidate": candidate,
        "parent": np.zeros((trials, count)) if parent is None else np.asarray(parent, dtype=float),
        "cluster_ids": np.broadcast_to(np.arange(count) % 5, (trials, count)).copy(),
        "observed_similarity": np.ones((trials, count, count)),
        "population_kernel_weights": np.full((trials, count), 1 / count),
        "cluster_proportions": np.array([.4, .25, .15, .1, .1]),
    }


def choices(scores: np.ndarray) -> np.ndarray:
    selected = np.isclose(scores, scores.max(axis=-1, keepdims=True), atol=1e-12, rtol=0)
    return selected / selected.sum(axis=-1, keepdims=True)


def test_every_single_observation_rule_preserves_the_exact_mean_tie_distribution():
    patterns = list(itertools.product((0, 1), repeat=5))
    combinations = list(itertools.product(patterns, (0, 1), range(5)))
    candidate = np.asarray([pattern for pattern, _, _ in combinations])[:, :, None]
    arguments = inputs(candidate, np.asarray([parent for _, parent, _ in combinations])[:, None])
    arguments["cluster_ids"] = np.asarray([group for _, _, group in combinations])[:, None]
    scores = reward_scores(**arguments)
    assert tuple(scores) == REWARD_KEYS
    for name, score in scores.items():
        np.testing.assert_array_equal(choices(score), choices(scores["mean"]), err_msg=name)


def test_parent_loss_aversion_is_exactly_parent_conditioned_linear_weighting():
    rng = np.random.default_rng(43)
    candidate = rng.integers(0, 2, size=(25, 5, 10))
    parent = rng.integers(0, 2, size=(25, 10))
    scores = reward_scores(**inputs(candidate, parent))
    expected = ((1 + parent[:, None, :]) * candidate).mean(axis=-1) - 2 * parent.mean(axis=-1)[:, None]
    np.testing.assert_allclose(scores["repair_aversion"], expected, atol=1e-14)
    metadata = {row["reward"]: row for row in reward_metadata()}
    assert "linear" in metadata["repair_aversion"]["selection_form"]
    assert {key for key, row in metadata.items() if row["requires_parent"]} == {
        "repair_aversion", "preserve_barrier", "paired_risk", "repair_retention_harmonic",
    }


def test_preservation_barrier_prefers_safe_candidates_then_uses_mean():
    candidate = np.array([[[0, 1, 1], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 0, 0]]])
    scores = reward_scores(**inputs(candidate, np.array([[1, 0, 0]])))
    np.testing.assert_array_equal(choices(scores["preserve_barrier"]), [[0, 0, .5, .5, 0]])
    unsafe = np.array([[[0, 1], [1, 0], [0, 0], [0, 1], [0, 0]]])
    scores = reward_scores(**inputs(unsafe, np.array([[1, 1]])))
    np.testing.assert_array_equal(choices(scores["preserve_barrier"]), choices(scores["mean"]))


@pytest.mark.parametrize("count", range(2, 11))
@pytest.mark.parametrize("parent_value", (0, 1))
def test_paired_risk_ties_zero_and_one_success_when_parent_labels_are_constant(count, parent_value):
    candidate = np.zeros((1, 5, count))
    for index, successes in enumerate((0, 1, 2, count - 1, count)):
        candidate[0, index, :successes] = 1
    score = reward_scores(**inputs(candidate, np.full((1, count), parent_value)))["paired_risk"]
    assert score[0, 0] == pytest.approx(score[0, 1], abs=1e-14)
    assert score[0, 2] > score[0, 1]


@pytest.mark.parametrize("parent_value", (0, 1))
def test_harmonic_uses_half_utility_for_an_unobserved_parent_stratum(parent_value):
    candidate = np.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0]]])
    score = reward_scores(**inputs(candidate, np.full((1, 3), parent_value)))["repair_retention_harmonic"]
    observed_utility = (candidate.sum(axis=-1) + .5) / 4
    expected = observed_utility / (observed_utility + .5)
    np.testing.assert_allclose(score, expected)


def test_cluster_rules_use_observed_half_count_posteriors_and_population_weights():
    candidate = np.array([[[1, 1, 0], [1, 0, 1], [0, 0, 0], [1, 1, 1], [0, 1, 1]]])
    arguments = inputs(candidate)
    arguments["cluster_ids"] = np.array([[0, 0, 1]])
    scores = reward_scores(**arguments)
    posterior = np.full((1, 5, 5), .5)
    posterior[:, :, 0] = (candidate[:, :, :2].sum(axis=-1) + .5) / 3
    posterior[:, :, 1] = (candidate[:, :, 2] + .5) / 2
    expected_population = (posterior * arguments["cluster_proportions"]).sum(-1)
    np.testing.assert_allclose(scores["cluster_population_mean"], expected_population)
    np.testing.assert_allclose(scores["cluster_macro_mean"], posterior.mean(-1))
    np.testing.assert_allclose(scores["cluster_geometric"], np.exp(np.log(posterior).mean(-1)))
    np.testing.assert_allclose(scores["cluster_soft_worst"], -.25 * np.log(np.exp(-posterior / .25).mean(-1)))


def test_one_observed_metadata_cluster_cannot_change_cluster_rule_rankings():
    rng = np.random.default_rng(44)
    arguments = inputs(rng.integers(0, 2, size=(50, 5, 10)))
    arguments["cluster_ids"][:] = 2
    scores = reward_scores(**arguments)
    for name in ("cluster_population_mean", "cluster_macro_mean", "cluster_geometric", "cluster_soft_worst"):
        np.testing.assert_array_equal(choices(scores[name]), choices(scores["mean"]), err_msg=name)


def test_rarity_values_a_single_rare_success_over_two_common_successes():
    candidate = np.array([[[1, 0, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]]])
    scores = reward_scores(**inputs(candidate))
    np.testing.assert_array_equal(choices(scores["committee_rarity"]), [[1, 0, 0, 0, 0]])
    assert scores["mean"][0, 0] < scores["mean"][0, 1]


def test_local_consistency_excludes_self_labels_and_falls_back_to_other_observations():
    candidate = np.array([[[1, 0, 0], [0, 1, 1], [0, 0, 0], [1, 1, 1], [1, 0, 1]]])
    arguments = inputs(candidate)
    arguments["observed_similarity"] = np.eye(3)[None, :, :] * 1000
    scores = reward_scores(**arguments)
    prediction = (candidate.sum(axis=-1, keepdims=True) - candidate) / 2
    expected = candidate.mean(axis=-1) - .25 * np.sqrt(((candidate - prediction) ** 2).mean(axis=-1))
    np.testing.assert_allclose(scores["local_consistency"], expected)
    assert scores["local_consistency"][0, 0] < scores["mean"][0, 0]


def test_local_consistency_uses_only_off_diagonal_metadata_neighbors():
    candidate = np.array([[[1, 0, 1], [0, 1, 0], [0, 0, 0], [1, 1, 1], [1, 0, 0]]])
    arguments = inputs(candidate)
    arguments["observed_similarity"] = np.array([[[900, 4, 0], [4, 900, 1], [0, 1, 900]]])
    scores = reward_scores(**arguments)
    prediction = np.stack((candidate[:, :, 1], (.8 * candidate[:, :, 0] + .2 * candidate[:, :, 2]),
                           candidate[:, :, 1]), axis=-1)
    expected = candidate.mean(axis=-1) - .25 * np.sqrt(((candidate - prediction) ** 2).mean(axis=-1))
    np.testing.assert_allclose(scores["local_consistency"], expected)


def test_population_kernel_mean_is_the_supplied_metadata_weighted_mean():
    candidate = np.array([[[1, 0], [0, 1], [0, 0], [0, 0], [0, 0]]])
    arguments = inputs(candidate)
    arguments["population_kernel_weights"] = np.array([[.9, .1]])
    scores = reward_scores(**arguments)
    np.testing.assert_allclose(scores["kernel_population_mean"], [[.9, .1, 0, 0, 0]])


def test_candidate_and_task_permutations_preserve_reward_semantics_and_inputs():
    rng = np.random.default_rng(42)
    arguments = inputs(rng.integers(0, 2, size=(5, 5, 7)), rng.integers(0, 2, size=(5, 7)))
    original = {key: value.copy() for key, value in arguments.items()}
    scores = reward_scores(**arguments)
    for key in arguments:
        np.testing.assert_array_equal(arguments[key], original[key])
    order = np.array([3, 1, 4, 0, 2])
    changed = {**arguments, "candidate": arguments["candidate"][:, order, :]}
    for key, score in reward_scores(**changed).items():
        np.testing.assert_allclose(score, scores[key][:, order])
    tasks = np.array([5, 0, 4, 3, 1, 6, 2])
    changed = {**arguments, "candidate": arguments["candidate"][:, :, tasks],
               "parent": arguments["parent"][:, tasks], "cluster_ids": arguments["cluster_ids"][:, tasks],
               "population_kernel_weights": arguments["population_kernel_weights"][:, tasks],
               "observed_similarity": arguments["observed_similarity"][:, tasks, :][:, :, tasks]}
    for key, score in reward_scores(**changed).items():
        np.testing.assert_allclose(score, scores[key])


@pytest.mark.parametrize("invalid", ("candidate", "parent", "cluster_ids", "observed_similarity",
                                     "population_kernel_weights", "cluster_proportions"))
def test_invalid_or_unmatched_observed_inputs_are_rejected(invalid):
    arguments = inputs(np.zeros((2, 5, 3)))
    if invalid == "cluster_ids":
        arguments[invalid][0, 0] = 5
    elif invalid == "cluster_proportions":
        arguments[invalid][0] = 0
    else:
        arguments[invalid].flat[0] = np.nan
    with pytest.raises(ValueError):
        reward_scores(**arguments)


def test_metadata_records_all_rules_in_exact_output_order_and_same_task_parent_cost():
    rows = reward_metadata()
    assert tuple(row["reward"] for row in rows) == REWARD_KEYS
    assert all(isinstance(row["requires_parent"], bool) for row in rows)
    assert len([row for row in rows if row["role"] == "control"]) == 4
