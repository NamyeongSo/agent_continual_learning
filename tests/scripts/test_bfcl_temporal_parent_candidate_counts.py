# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Verify exact candidate-subset probabilities against exhaustive enumeration."""

from itertools import combinations, product

import numpy as np
import pytest

from scripts.premise.analyze_bfcl_temporal_parent_candidate_counts import candidate_count_probabilities
from scripts.premise.analyze_bfcl_temporal_parent_preservation import selection_scores
from scripts.premise.bfcl_trajectory_selection_rewards import choice_probabilities, ordering_ranks


def brute_force(ranks):
    probabilities = np.zeros((len(ranks), 5, 5))
    for count in range(1, 6):
        subsets = list(combinations(range(5), count))
        for subset in subsets:
            selected = ranks[:, subset] == ranks[:, subset].max(axis=1, keepdims=True)
            for position, candidate in enumerate(subset):
                probabilities[:, candidate, count - 1] += selected[:, position] / selected.sum(axis=1) / len(subsets)
    return probabilities


def test_lookup_matches_brute_force_for_every_five_candidate_weak_ordering():
    scores = np.asarray(list(product(range(5), repeat=5)))
    ranks = np.unique(ordering_ranks(scores), axis=0)
    assert len(ranks) == 541
    probabilities = candidate_count_probabilities(ranks)
    np.testing.assert_allclose(probabilities, brute_force(ranks), rtol=0, atol=1e-14)


@pytest.mark.parametrize("lexicographic", [False, True])
def test_scalar_and_lexicographic_scores_match_exhaustive_subsets(lexicographic):
    generator = np.random.default_rng(123)
    shape = (100, 5, 3) if lexicographic else (100, 5)
    scores = generator.integers(0, 3, size=shape)
    probabilities = candidate_count_probabilities(ordering_ranks(scores))
    np.testing.assert_allclose(probabilities, brute_force(ordering_ranks(scores)), rtol=0, atol=1e-14)
    np.testing.assert_allclose(probabilities[:, :, 0], .2, rtol=0, atol=1e-14)
    np.testing.assert_allclose(probabilities[:, :, 4], choice_probabilities(scores), rtol=0, atol=1e-14)


@pytest.mark.parametrize("n", [1, 5, 10])
def test_parent_preservation_subset_endpoints(n):
    generator = np.random.default_rng(55)
    success = generator.integers(0, 2, size=(100, 6, n))
    turns = generator.integers(1, 21, size=success.shape).astype(float)
    turns[success == 0] = np.nan
    scores = selection_scores(success, turns, np.full(n, 1 / n))["parent_preservation"]
    probabilities = candidate_count_probabilities(ordering_ranks(scores))
    np.testing.assert_allclose(probabilities[:, :, 0], .2, rtol=0, atol=1e-14)
    np.testing.assert_allclose(probabilities[:, :, 4], choice_probabilities(scores), rtol=0, atol=1e-14)
    targets = generator.random(5)
    np.testing.assert_allclose(probabilities[:, :, 0] @ targets - targets.mean(), 0, rtol=0, atol=1e-14)


@pytest.mark.parametrize("ranks", [[[1, 1, 1, 1, 1]], [[0, 1, 2, 3, 4]], [[1, 2, 3, 4, 6]],
                                  [[1, 2, 3, 4]], [[1, 2, 3, 4, float("nan")]]])
def test_invalid_average_rank_partitions_are_rejected(ranks):
    with pytest.raises(ValueError):
        candidate_count_probabilities(np.asarray(ranks))
