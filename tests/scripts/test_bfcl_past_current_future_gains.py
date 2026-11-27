# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Target boundaries, query-only selection, and matched gain decomposition."""

import numpy as np
import pytest

from scripts.premise.analyze_bfcl_past_current_future_gains import (
    aggregate_rows,
    success_selection_scores,
    target_metrics,
    target_task_indices,
)
from scripts.premise.bfcl_trajectory_selection_rewards import choice_probabilities


def test_targets_partition_training_with_current_only_in_first_target():
    targets = target_task_indices(50, 20)
    assert np.array_equal(targets["past_current"], np.arange(21))
    assert np.array_equal(targets["future"], np.arange(21, 50))
    assert not set(targets["past_current"]) & set(targets["future"])
    assert len(np.concatenate(list(targets.values()))) == 50


@pytest.mark.parametrize("current", [-1, 49, 50, 1.5])
def test_empty_future_or_invalid_boundary_rejected(current):
    with pytest.raises(ValueError, match="nonempty"):
        target_task_indices(50, current)


def test_selector_cannot_observe_parent_or_unqueried_tasks():
    success = np.zeros((6, 50))
    success[1, 3] = 1
    queries = np.array([[3]])
    before = choice_probabilities(success_selection_scores(success[1:, queries].transpose(1, 0, 2), np.ones(1)))
    success[0, :] = 1
    success[2:, 4:] = 1
    after = choice_probabilities(success_selection_scores(success[1:, queries].transpose(1, 0, 2), np.ones(1)))
    assert np.array_equal(before, after)
    assert np.array_equal(after, [[1, 0, 0, 0, 0]])


def test_unqueried_gain_removes_only_target_queries_and_uses_matched_parent():
    rng = np.random.default_rng(42)
    success = rng.integers(0, 2, (6, 12))
    queries = np.array([[1, 7, 9], [2, 5, 11]])
    probabilities = np.array([[.5, .5, 0, 0, 0], [0, 0, 1 / 3, 1 / 3, 1 / 3]])
    target = np.arange(6)
    actual = target_metrics(success, queries, probabilities, target)
    for draw, queried in enumerate(queries):
        full_delta = (success[1:, target] - success[:1, target]).mean(axis=1)
        remaining = np.setdiff1d(target, queried)
        remaining_delta = (success[1:, remaining] - success[:1, remaining]).mean(axis=1)
        assert actual["selected_gain"][draw] == pytest.approx(probabilities[draw] @ full_delta)
        assert actual["proposal_gain"][draw] == pytest.approx(full_delta.mean())
        assert actual["selected_gain_unqueried"][draw] == pytest.approx(probabilities[draw] @ remaining_delta)
        assert actual["proposal_gain_unqueried"][draw] == pytest.approx(remaining_delta.mean())
    for suffix in ("", "_unqueried"):
        assert np.allclose(actual[f"selected_gain{suffix}"],
                           actual[f"proposal_gain{suffix}"] + actual[f"selection_lift{suffix}"])


def test_uniform_single_proposal_has_zero_lift_but_nonzero_parent_gain():
    success = np.ones((6, 12))
    success[0] = 0
    result = target_metrics(success, np.array([[0, 1], [1, 2]]), np.full((2, 5), .2), np.arange(6))
    assert np.allclose(result["selected_gain"], 1)
    assert np.allclose(result["selected_gain_unqueried"], 1)
    assert np.allclose(result["selection_lift"], 0)
    assert np.allclose(result["selection_lift_unqueried"], 0)


def test_query_disjoint_target_does_not_change_when_queries_removed():
    rng = np.random.default_rng(11)
    success = rng.integers(0, 2, (6, 12))
    queries = np.array([[0, 1], [1, 2]])
    result = target_metrics(success, queries, np.full((2, 5), .2), np.arange(6, 12))
    for name in ("selected_gain", "proposal_gain", "selection_lift"):
        assert np.allclose(result[name], result[f"{name}_unqueried"])


def test_size_weighted_split_gains_recover_full_training_gain():
    rng = np.random.default_rng(7)
    success = rng.integers(0, 2, (6, 12))
    queries = np.array([[1, 8], [3, 6]])
    probabilities = np.array([[.5, .5, 0, 0, 0], [0, 0, 1, 0, 0]])
    targets = target_task_indices(12, 4)
    result = {name: target_metrics(success, queries, probabilities, indices)
              for name, indices in targets.items()}
    reconstructed = sum(len(indices) * result[name]["selected_gain"] for name, indices in targets.items()) / 12
    direct = probabilities @ (success[1:] - success[:1]).mean(axis=1)
    assert np.allclose(reconstructed, direct)


@pytest.mark.parametrize("queries", [np.array([[1, 1]]), np.array([[1, 20]]), np.array([[1., 2.]])])
def test_invalid_or_repeated_query_rejected(queries):
    with pytest.raises(ValueError, match="query indices"):
        target_metrics(np.zeros((6, 12)), queries, np.full((1, 5), .2), np.arange(6))


def test_empty_query_excluded_target_is_rejected():
    with pytest.raises(ValueError, match="remain nonempty"):
        target_metrics(np.zeros((6, 12)), np.array([[1, 2]]), np.full((1, 5), .2), np.array([1, 2]))


def test_ties_uniform_and_weighting_applies_only_to_queried_tasks():
    sample = np.zeros((1, 5, 3))
    sample[0, 0] = [1, 0, 0]
    sample[0, 1] = [0, 1, 1]
    task_mean = success_selection_scores(sample, np.full(3, 1 / 3))
    period_mean = success_selection_scores(sample, np.array([.5, .25, .25]))
    assert np.array_equal(choice_probabilities(task_mean), [[0, 1, 0, 0, 0]])
    assert np.array_equal(choice_probabilities(period_mean), [[.5, .5, 0, 0, 0]])


def test_undefined_rho_excluded_from_aggregation_never_imputed_zero():
    row = {"model": "Qwen", "case": "past", "weight": "task_mean", "n": 1,
           "target": "future", "seed": 42, "selected_gain": .1, "proposal_gain": .05,
           "selection_lift": .05, "selected_gain_unqueried": .1,
           "proposal_gain_unqueried": .05, "selection_lift_unqueried": .05, "rho": None}
    rows = aggregate_rows([row, {**row, "rho": .4}])
    assert rows[0]["rho"]["mean"] == pytest.approx(.4)
    assert rows[0]["rho"]["defined_checkpoints"] == 1
