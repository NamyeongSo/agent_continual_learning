# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Pareto maximization must preserve ties, unsupported optima, and n feasibility."""

import numpy as np
import pytest

from scripts.premise.plot_bfcl_retention_pareto import panel_points, pareto_mask


def test_maximize_both_axes_and_keep_nondominated_tradeoffs():
    points = np.array([[1, 1], [2, 2], [3, 1], [1, 3], [0, 0]])
    assert np.array_equal(pareto_mask(points), [False, True, True, True, False])


def test_equal_coordinate_with_strict_other_improvement_is_dominated():
    assert np.array_equal(pareto_mask([[1, 1], [2, 1], [2, 2]]), [False, False, True])


def test_coincident_alternatives_both_remain_optimal():
    assert np.array_equal(pareto_mask([[2, 1], [2, 1], [0, 0]]), [True, True, False])


def test_nondominated_point_below_convex_envelope_is_not_discarded():
    assert pareto_mask([[0, 3], [1, 1], [3, 0]]).all()


def test_roundoff_does_not_break_true_ties():
    assert pareto_mask([[.1, .2], [.1 + 1e-15, .2]]).all()


@pytest.mark.parametrize("points", [[[1, np.nan]], [[np.inf, 1]], [1, 2], [[1, 2, 3]]])
def test_invalid_frontier_coordinates_rejected(points):
    with pytest.raises(ValueError, match="finite"):
        pareto_mask(points)


def test_n1_omits_infeasible_mixtures_and_unmeasured_val10():
    metric = {"mean": .01, "defined_checkpoints": 36}
    def row(case):
        return {"model": "Qwen", "weight": "task_mean", "scope": "all", "n": 1, "case": case,
                "checkpoints": 36, "gain_past": metric, "lift_validation40": metric}
    summary = {"checkpoints_per_model": 36, "aggregate": [row(c) for c in ("current", "past", "future")],
               "proposal_reference": [row("proposal")], "validation_reference": [row("val10")]}
    points = panel_points(summary, "Qwen", "task_mean", 1)
    assert {r["case"] for r in points} == {"current", "past", "future", "proposal"}
    summary["aggregate"].append(row("current_past"))
    with pytest.raises((ValueError, TypeError)):
        panel_points(summary, "Qwen", "task_mean", 1)
