# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Check the additional, explicitly post hoc process reward diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.premise.analyze_bfcl_process_reward_ablations import process_ablation_scores
from scripts.premise.bfcl_trajectory_selection_rewards import choice_probabilities


def test_all_run_loop_penalty_distinguishes_failed_loops_without_overriding_success():
    success = np.zeros((1, 5, 2))
    success[0, 4, 0] = 1
    repeats = np.array([[[0, 0], [10, 0], [20, 20], [100, 100], [100, 100]]])
    scores = process_ablation_scores(success, np.full_like(success, 20), repeats,
                                     np.zeros((1, 2)), np.full((1, 2), 20))
    values = scores["all_trajectory_loop_penalty"]
    assert values[0, 0] > values[0, 1] > values[0, 2]
    assert values[0, 2] == values[0, 3]
    np.testing.assert_array_equal(choice_probabilities(values), [[0, 0, 0, 0, 1]])


def test_paired_cost_compares_only_the_same_jointly_successful_task():
    success = np.array([[[1, 0], [1, 0], [0, 1], [0, 0], [0, 0]]])
    turns = np.array([[[5, 0], [10, 0], [0, 0], [0, 0], [20, 20]]])
    scores = process_ablation_scores(success, turns, np.zeros_like(success),
                                     np.array([[1, 0]]), np.array([[10, 20]]))
    values = scores["paired_success_efficiency"]
    assert values[0, 0] > values[0, 1]
    assert values[0, 1] == values[0, 2] == 1
    assert values[0, 3] == values[0, 4] == 0


def test_one_extra_success_dominates_even_extreme_parent_cost_advantage():
    success = np.zeros((1, 5, 10))
    success[0, 0, :4] = 1
    success[0, 1, :5] = 1
    turns = np.full_like(success, 100)
    turns[0, 0] = 0
    scores = process_ablation_scores(success, turns, np.full_like(success, 100),
                                     np.ones((1, 10)), np.ones((1, 10)))
    assert scores["paired_success_efficiency"][0, 1] > scores["paired_success_efficiency"][0, 0]


def test_parent_shape_and_n_budget_cannot_silently_broadcast():
    for parent_shape in ((1, 1), (2, 3)):
        with pytest.raises(ValueError):
            process_ablation_scores(np.zeros((1, 5, 3)), np.zeros((1, 5, 3)), np.zeros((1, 5, 3)),
                                    np.zeros(parent_shape), np.zeros(parent_shape))
