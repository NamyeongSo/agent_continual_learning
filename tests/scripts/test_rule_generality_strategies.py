# SPDX-License-Identifier: Apache-2.0
"""Regression checks for fixed-reference context scoring and equal-context strategy averages."""

import numpy as np
import pytest

from scripts.ace.analyze_rule_generality import context_maxima
from scripts.ace.analyze_rule_generality_strategies import aggregate_selections, query_context_maxima


def test_external_query_uses_every_reference_context_with_equal_weight():
    targets = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [1, 0]], dtype=np.float32)
    queries = np.array([[1, 0]], dtype=np.float32)
    scores, nearest = query_context_maxima(queries, targets, [1, 3, 1], [-1])
    assert np.array_equal(scores, [[1, 0, 1]])
    assert np.isclose(scores.mean(), 2 / 3)
    assert np.array_equal(nearest, [[0, 1, 4]])


def test_member_query_excludes_whole_own_reference_context():
    vectors = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    expected, expected_nearest = context_maxima(vectors, [2, 1])
    actual, nearest = query_context_maxima(vectors, vectors, [2, 1], [0, 0, 1])
    assert np.allclose(actual, expected, equal_nan=True)
    assert np.array_equal(nearest, expected_nearest)


def test_external_and_member_queries_share_same_target_pool():
    targets = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    queries = np.array([[1, 0], [1, 0]], dtype=np.float32)
    actual, nearest = query_context_maxima(queries, targets, [2, 1], [-1, 0], block_size=1)
    assert np.array_equal(actual[0], [1, 0])
    assert np.isnan(actual[1, 0]) and actual[1, 1] == 0
    assert nearest[1, 0] == -1


def selected_rows(means):
    contexts, selections = [], []
    for seed, mean in zip((42, 43, 44), means, strict=True):
        for task in range(50):
            context_id = f"{seed}-{task}"
            contexts.append({"context_id": context_id, "model": "qwen38", "mode": "thinking",
                             "task_id": str(task), "rules": 100 if task % 2 else 1, "context_mean_cosine": mean})
            selections.append({"context_id": context_id, "model": "qwen38", "mode": "thinking",
                               "task_id": str(task), "strategy": "best", "seed": seed})
    return contexts, selections


def test_contexts_are_equal_weight_and_seed_sd_is_sample_sd():
    contexts, selections = selected_rows((.2, .5, .8))
    _, seeds, summary = aggregate_selections(contexts, selections)
    assert all(row["valid_contexts"] == 50 for row in seeds)
    assert np.isclose(summary[0]["mean_context_cosine"], .5)
    assert np.isclose(summary[0]["sample_std_across_selection_seeds"], .3)
    assert summary[0]["selected_context_instances"] == 150


def test_ruleless_contexts_are_missing_not_zero():
    contexts, selections = selected_rows((.8, .8, .8))
    contexts[0]["rules"] = 0
    contexts[0]["context_mean_cosine"] = None
    _, seeds, summary = aggregate_selections(contexts, selections)
    assert seeds[0]["valid_contexts"] == 49 and seeds[0]["empty_contexts"] == 1
    assert summary[0]["valid_context_instances"] == 149
    assert np.isclose(summary[0]["mean_context_cosine"], .8)


def test_partial_task_coverage_is_rejected_instead_of_silently_averaged():
    contexts, selections = selected_rows((.5, .5, .5))
    with pytest.raises(ValueError, match="all50"):
        aggregate_selections(contexts, selections[:-1])
