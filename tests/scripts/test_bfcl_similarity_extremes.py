# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import numpy as np
import pytest

from scripts.premise.analyze_bfcl_similarity_extremes import COUNTS, calculate, mixed_indices, ranked_draws


def test_similar_and_dissimilar_tasks_are_disjoint_and_have_fixed_cost():
    similarity = np.linspace(0, 1, 15)
    ranking = ranked_draws(similarity, 50, np.random.default_rng(2))
    assert np.all(ranking == np.arange(14, -1, -1))
    for near in COUNTS:
        indices = mixed_indices(ranking, near)
        assert indices.shape == (50, 9)
        assert all(len(set(row)) == 9 for row in indices)
        assert np.all(indices[:, :near] == ranking[:, :near])
        if near < 9:
            assert np.all(indices[:, near:] == ranking[:, -(9 - near):])


def test_exact_similarity_ties_are_randomized_without_reordering_unequal_values():
    similarity = np.array([1, 1, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0])
    ranking = ranked_draws(similarity, 4000, np.random.default_rng(8))
    assert np.all(np.diff(similarity[ranking], axis=1) <= 0)
    assert set(ranking[:, 0]) == {0, 1}
    assert 0.46 < (ranking[:, 0] == 0).mean() < 0.54
    assert ranking == pytest.approx(ranked_draws(similarity, 4000, np.random.default_rng(8)))


def test_nine_past_tasks_make_every_near_far_ratio_use_the_same_pool():
    ranking = ranked_draws(np.ones(9), 30, np.random.default_rng(12))
    for near in COUNTS:
        assert np.all(np.sort(mixed_indices(ranking, near), axis=1) == np.arange(9))


def test_random_complement_shares_similar_prefix_without_duplicates_and_has_matching_endpoint():
    rng = np.random.default_rng(22)
    ranking = ranked_draws(np.ones(15), 100, rng)
    priorities = rng.random(ranking.shape)
    for near in COUNTS:
        indices = mixed_indices(ranking, near, priorities)
        assert indices.shape == (100, 9)
        assert np.all(indices[:, :near] == ranking[:, :near])
        assert all(len(set(row)) == 9 for row in indices)
    assert np.array_equal(mixed_indices(ranking, 9, priorities), mixed_indices(ranking, 9))


def test_selectors_never_read_validation_rewards_or_future_task_metadata():
    rng = np.random.default_rng(3)
    rewards = rng.integers(2, size=(5, 50))
    order = [f"task{i}" for i in range(50)]
    point = {"key": "cp10", "seed": 42, "checkpoint": 10, "train_order": order,
             "train_rewards": rewards.tolist(), "parent_scores": {"full_train": 0.3},
             "scores": {"full_train": rewards.mean(axis=1).tolist(), "full_validation": [0, 0.1, 0.2, 0.3, 0.4]}}
    # Missing future metadata causes an immediate failure if the selector tries to inspect it.
    metadata = {task: {"first_request": f"Action {i} for saved document", "tool_classes": [f"Tools{i % 3}"]}
                for i, task in enumerate(order[:11])}
    model = {"label": "Synthetic", "points": [point]}
    before = calculate(model, metadata, {"cp10"}, 30, 7)[1]
    point["scores"]["full_validation"] = [1, 0.8, 0.4, 0, 0.2]
    after = calculate(model, metadata, {"cp10"}, 30, 7)[1]
    for left, right in zip(before, after, strict=True):
        assert (left["weight"], left["near"]) == (right["weight"], right["near"])
        assert left["current_tasks"] == 1 and left["tasks"] == 10
        for index in range(1, 6):
            assert left[f"candidate_{index}_probability"] == right[f"candidate_{index}_probability"]


def test_flat_candidates_have_zero_lift_and_undefined_correlations():
    point = {"key": "flat", "seed": 42, "checkpoint": 10, "train_order": [f"t{i}" for i in range(11)],
             "train_rewards": np.ones((5, 11)).tolist(), "parent_scores": {"full_train": 1},
             "scores": {"full_train": [1] * 5, "full_validation": [0.4] * 5}}
    metadata = {task: {"first_request": "Read saved document", "tool_classes": ["Tools"]}
                for task in point["train_order"]}
    aggregate, rows = calculate({"label": "Flat", "points": [point]}, metadata, {"flat"}, 30, 7)
    assert all(abs(row["validation_lift"]) < 1e-12 for row in rows)
    assert all(row["rho_full_validation"] is None for row in rows)
    assert all(row["rho_full_validation"]["defined_trials"] == 0 for row in aggregate)
    assert all(row["rho_full_validation_common_checkpoints"]["mean"] == 0 for row in aggregate)


def test_invalid_pools_are_rejected():
    with pytest.raises(ValueError, match="nine"):
        ranked_draws(np.ones(8), 10, np.random.default_rng(3))
    with pytest.raises(ValueError, match="supported"):
        mixed_indices(np.arange(10)[None, :], 2)
