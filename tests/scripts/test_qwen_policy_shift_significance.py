# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import numpy as np
import pytest

from scripts.premise.analyze_qwen_policy_shift_significance import (
    analyze_graphs,
    distance_means,
    order_draws,
    shift_graphs,
    trend_statistics,
)


def synthetic_graphs():
    index = np.arange(4)
    shift = np.abs(index[:, None] - index[None, :]) / 10
    return np.stack([np.stack((shift, 2 * shift), axis=-1)] * 3)


def test_distance_means_and_trends_match_a_known_linear_policy_shift():
    means = distance_means(synthetic_graphs())
    assert means.shape == (3, 3, 2)
    assert means[0] == pytest.approx(np.array([[0.1, 0.2], [0.2, 0.4], [0.3, 0.6]]))
    statistics = trend_statistics(means, np.array([5, 10, 15]))
    assert statistics[:, :, 0] == pytest.approx(np.ones((3, 2)))
    assert statistics[0, :, 1] == pytest.approx([0.02, 0.04])


def test_whole_checkpoint_permutations_keep_initial_context_fixed_and_preserve_comparison_edges():
    draws = order_draws(np.random.default_rng(7), 30, 3, 4)
    assert (draws[..., 0] == 0).all()
    assert np.sort(draws, axis=-1) == pytest.approx(np.broadcast_to(np.arange(4), draws.shape))
    graph = synthetic_graphs()[0]
    left, right = np.triu_indices(4, k=1)
    permutation = draws[0, 0]
    moved = graph[permutation[:, None], permutation[None, :]]
    assert np.sort(moved[left, right, 0]) == pytest.approx(np.sort(graph[left, right, 0]))
    assert moved[..., 1] == pytest.approx(2 * moved[..., 0])


def test_graph_loader_rejects_missing_or_duplicate_pairs_and_preserves_symmetry():
    points, seeds = (0, 5, 10, 15), (42, 43, 44)
    rows = [{"seed": seed, "left": left, "right": right, "distance": right - left,
             "strict_accuracy": 1 - (right - left) / 50,
             "strict_f1": 1 - (right - left) / 25}
            for seed in seeds for index, left in enumerate(points) for right in points[index + 1:]]
    graph = shift_graphs(rows, points, seeds)
    assert graph == pytest.approx(synthetic_graphs())
    assert graph == pytest.approx(graph.transpose(0, 2, 1, 3))
    with pytest.raises(ValueError, match="complete, unique"):
        shift_graphs(rows[:-1], points, seeds)
    with pytest.raises(ValueError, match="complete, unique"):
        shift_graphs([*rows, rows[0]], points, seeds)


def test_permutation_tails_are_reproducible_and_move_both_action_metrics_together():
    graphs, distances = synthetic_graphs(), np.array([5, 10, 15])
    first = analyze_graphs(graphs, distances, trials=100, rng=np.random.default_rng(7))
    assert first == analyze_graphs(graphs, distances, trials=100, rng=np.random.default_rng(7))
    assert len(first) == 16
    assert all(0 < row["p_two_sided"] <= 1 for row in first)
    correlations = [row for row in first if row["statistic"] == "pearson_r"]
    for scope in ("42", "43", "44", "all"):
        pair = [row for row in correlations if row["seed"] == scope]
        assert pair[0]["p_two_sided"] == pair[1]["p_two_sided"]


def test_flat_policy_shift_has_no_invented_positive_trend():
    curves = np.ones((2, 3, 2)) / 2
    assert trend_statistics(curves, np.array([5, 10, 15])) == pytest.approx(np.zeros((2, 2, 2)))
