# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import numpy as np
import pytest

from scripts.premise.analyze_bfcl_temporal_selection import (
    CASES,
    allocation,
    calculate,
    correlations,
    fixed_cohort,
    flat_scores,
    sample_prefix_sums,
    score_cases,
    selected_performance,
)


def test_task_budgets_keep_one_current_and_give_the_extra_odd_task_to_the_past():
    assert allocation("current_past_future", 10) == (1, 5, 4)
    assert allocation("current_past_future", 9) == (1, 4, 4)
    assert allocation("current_past", 10) == (1, 9, 0)
    assert allocation("current_future", 10) == (1, 0, 9)
    assert allocation("current", 10) == (1, 0, 0)
    assert allocation("past", 10) == (0, 10, 0)
    assert allocation("future", 10) == (0, 0, 10)
    assert allocation("current_past", 1) is None
    assert allocation("current_future", 1) is None
    assert allocation("current_past_future", 2) is None
    with pytest.raises(ValueError):
        allocation("unknown", 5)


def test_uniform_prefix_draws_are_without_replacement_and_share_subsets_across_candidates():
    matrix = np.broadcast_to(np.arange(12), (5, 12))
    sums = sample_prefix_sums(matrix, 500, 10, np.random.default_rng(7))
    draws = np.diff(sums, axis=1)
    assert all(len(set(row)) == 10 for row in draws[:, :, 0])
    assert draws == pytest.approx(np.broadcast_to(draws[:, :, :1], draws.shape))
    assert sums == pytest.approx(sample_prefix_sums(matrix, 500, 10, np.random.default_rng(7)))
    with pytest.raises(ValueError, match="Insufficient"):
        sample_prefix_sums(matrix[:, :9], 10, 10, np.random.default_rng(7))


def test_task_and_period_means_use_the_same_draws_but_distinct_intended_weights():
    current = np.array([1, 0, 1, 0, 0])
    past = np.broadcast_to(np.arange(11)[None, :, None] * 0.2, (2, 11, 5))
    future = np.broadcast_to(np.arange(11)[None, :, None] * 0.8, (2, 11, 5))
    scores = score_cases(current, past, future, 10)
    index = CASES.index("current_past_future")
    assert scores[0, index, 0] == pytest.approx((current + 5 * 0.2 + 4 * 0.8) / 10)
    assert scores[1, index, 0] == pytest.approx((current + 0.2 + 0.8) / 3)
    for case in ("current", "past", "future"):
        index = CASES.index(case)
        assert scores[0, index] == pytest.approx(scores[1, index])
    scores = score_cases(current, past, future, 3)
    index = CASES.index("current_past_future")
    assert scores[0, index] == pytest.approx(scores[1, index])
    scores = score_cases(current, past, future, 1)
    assert np.isnan(scores[:, CASES.index("current_past")]).all()


def test_uniform_ties_use_the_actual_tied_candidates_and_flat_scores_keep_proposal_mean():
    targets = np.array([0, 0.1, 0.2, 0.4, 0.8])
    scores = np.array([[0, 1 / 3, 1 / 3 + 1e-16, 0, 0], [0, 0, 0, 0, 0]])
    assert selected_performance(scores, targets) == pytest.approx([0.15, targets.mean()])
    assert np.isnan(correlations(scores, targets)[1])
    assert np.isnan(correlations(scores, np.zeros(5))).all()
    assert correlations(targets[None, :], targets)[0] == pytest.approx(1)


def test_fixed_cohort_requires_ten_past_and_ten_future_tasks_and_matched_model_identities():
    def point(key, t):
        order = list(range(50))
        return {"key": key, "checkpoint": t, "train_order": order, "past": order[:t], "future": order[t + 1:]}

    points = [point("early", 9), point("first", 10), point("last", 39), point("late", 40)]
    assert fixed_cohort([{"points": points}, {"points": points}], 10) == ["first", "last"]
    with pytest.raises(ValueError, match="identities"):
        fixed_cohort([{"points": points}, {"points": points[:-1]}], 10)


def test_aggregate_matches_known_selected_candidates_and_keeps_validation_parent_unmeasured():
    past = np.array([0, 0, 1, 1, 0])
    current = np.array([0, 1, 0, 1, 0])
    future = np.array([1, 0, 1, 0, 0])
    matrix = np.column_stack([past, past, past, current, future, future, future])
    point = {"key": "test", "seed": 42, "checkpoint": 3, "train_rewards": matrix.tolist(),
             "scores": {"full_train": matrix.mean(axis=1).tolist(), "full_validation": [0.1, 0.2, 0.3, 0.8, 0.6]},
             "parent_scores": {"full_train": 0.5, "full_validation": None}}
    model = {"label": "Synthetic", "points": [point], "original_errors": 0, "original_evaluations": 1}
    result, checkpoint_rows = calculate(model, ["test"], trials=30, seed=7, maximum_n=3)
    row = next(row for row in result["scores"] if row["case"] == "current_past_future" and row["n"] == 3
               and row["weight"] == "period_mean" and row["target"] == "full_validation")
    assert row["selected_performance"]["mean"] == pytest.approx(0.55)
    assert row["selected_performance"]["std"] == 0
    assert row["selection_lift"]["mean"] == pytest.approx(0.15)
    assert row["parent_gain"] is None
    assert checkpoint_rows
    flattened = flat_scores([result])
    assert len({tuple(row) for row in flattened}) == 1
    validation = next(row for row in flattened if row["target"] == "full_validation")
    assert validation["parent_gain_mean"] is None
