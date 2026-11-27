# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Semantic invariants for the LLM-free temporal parent-preservation ablation."""

import numpy as np
import pytest

from scripts.premise.analyze_bfcl_temporal_parent_preservation import (
    allocation,
    selected_indices,
    selection_scores,
    task_weights,
    temporal_prefixes,
)
from scripts.premise.bfcl_trajectory_selection_rewards import choice_probabilities


def test_current_is_always_exactly_one_task_and_mixtures_need_every_period():
    assert all(allocation("current", n) == (1, 0, 0) for n in range(1, 11))
    assert allocation("current_past", 1) is None
    assert allocation("past_future", 1) is None
    assert allocation("current_past_future", 2) is None
    assert allocation("current_past_future", 10) == (1, 5, 4)
    assert allocation("past_future", 9) == (0, 5, 4)


def test_period_weighting_changes_only_period_mass_not_within_period_weights():
    counts = (1, 5, 4)
    assert np.allclose(task_weights(counts, "task_mean"), .1)
    weights = task_weights(counts, "period_mean")
    assert np.allclose([weights[:1].sum(), weights[1:6].sum(), weights[6:].sum()], 1 / 3)
    assert np.allclose(task_weights((0, 5, 5), "task_mean"), task_weights((0, 5, 5), "period_mean"))


def test_uniform_prefixes_shared_across_cases_models_and_nested_sample_sizes():
    point = {"key": "s42_cp020", "checkpoint": 20, "train_order": list(range(50))}
    a = temporal_prefixes(point, 100, 71)
    b = temporal_prefixes({**point, "label": "another model"}, 100, 71)
    for period in a:
        assert np.array_equal(a[period], b[period])
    five = selected_indices(a, allocation("current_past", 5))
    six = selected_indices(a, allocation("current_past", 6))
    assert np.array_equal(five, six[:, :5])
    all_periods = selected_indices(a, allocation("current_past_future", 10))
    assert np.all(all_periods[:, 0] == 20)
    assert np.all(all_periods[:, 1:6] < 20)
    assert np.all(all_periods[:, 6:] > 20)
    assert all(len(set(row)) == 10 for row in all_periods)


def test_preservation_priority_can_reject_more_total_successes():
    # Parent passes first task. A preserves it and repairs one failure; B breaks it
    # but repairs three, so B wins success mean and A wins preservation.
    success = np.array([[[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 1],
                         [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=float)
    turns = np.full(success.shape, 10.)
    scores = selection_scores(success, turns, np.full(4, .25))
    assert choice_probabilities(scores["success_mean"])[0, 1] == 1
    assert choice_probabilities(scores["parent_preservation"])[0, 0] == 1


def test_repairs_take_priority_over_any_successful_step_reduction():
    success = np.array([[[1, 0], [1, 1], [1, 0], [0, 0], [0, 0], [0, 0]]], dtype=float)
    turns = np.full(success.shape, 10.)
    turns[0, 1] = 20  # More expensive but one actual repair.
    turns[0, 2] = 1
    scores = selection_scores(success, turns, np.full(2, .5))
    assert choice_probabilities(scores["parent_preservation"])[0, 0] == 1


def test_joint_success_efficiency_breaks_success_ties_without_short_failure_bonus():
    success = np.array([[[1], [1], [1], [0], [0], [0]]], dtype=float)
    turns = np.array([[[10], [10], [5], [np.nan], [0], [1]]], dtype=float)
    scores = selection_scores(success, turns, np.ones(1))
    assert np.allclose(choice_probabilities(scores["success_mean"]), [[.5, .5, 0, 0, 0]])
    assert np.allclose(choice_probabilities(scores["parent_preservation"]), [[0, 1, 0, 0, 0]])
    assert np.allclose(scores["efficiency"], [[0, .5, 0, 0, 0]])


def test_failed_step_values_cannot_affect_selection():
    success = np.array([[[0, 1], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0]]], dtype=float)
    turns = np.where(success, 10., np.nan)
    before = selection_scores(success, turns, np.array([.5, .5]))
    turns[success == 0] = -9999
    after = selection_scores(success, turns, np.array([.5, .5]))
    assert all(np.array_equal(before[k], after[k]) for k in before)


def test_missing_successful_steps_rejected_not_imputed():
    success = np.ones((1, 6, 1))
    turns = np.ones((1, 6, 1))
    turns[0, 2, 0] = np.nan
    with pytest.raises(ValueError, match="positive integer"):
        selection_scores(success, turns, np.ones(1))


def test_equal_orderings_use_uniform_ties_without_candidate_index_bias():
    success = np.zeros((3, 6, 2))
    turns = np.full(success.shape, np.nan)
    scores = selection_scores(success, turns, np.array([.5, .5]))
    assert np.allclose(choice_probabilities(scores["parent_preservation"]), .2)
