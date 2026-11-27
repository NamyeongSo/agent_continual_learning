# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Protect cohort comparisons and measured reference coordinates in research plots."""

import numpy as np
import pytest

from scripts.premise.plot_bfcl_temporal_parent_preservation import (
    index_rows,
    index_uniform_reference,
    metric_value,
    reference_points,
    surface_grid,
)


def row(**changes):
    return {"model": "Qwen", "scope": "all", "weight": "task_mean", "case": "past", "n": 10,
            "method": "parent_preservation", "checkpoints": 36, "validation_lift": {"mean": .002}, **changes}


def test_missing_measurement_remains_a_gap():
    assert np.isnan(metric_value({}, "validation_lift"))
    assert np.isnan(metric_value({"validation_lift": {"mean": None}}, "validation_lift"))
    assert metric_value({"validation_lift": {"mean": 0}}, "validation_lift") == 0


def test_different_checkpoint_counts_cannot_share_comparison_figure():
    summary = {"aggregate": [row(), row(weight="period_mean", checkpoints=35)]}
    with pytest.raises(ValueError, match="fixed checkpoint cohort"):
        index_rows(summary)


def test_zero_coverage_cannot_report_zero_as_a_measured_gain():
    with pytest.raises(ValueError, match="Zero-checkpoint"):
        index_rows({"aggregate": [row(checkpoints=0, validation_lift={"mean": 0})]})


def test_minimum_sample_budget_for_three_periods_is_enforced():
    with pytest.raises(ValueError, match="three-period"):
        index_rows({"aggregate": [row(case="current_past_future", n=2)]})


def test_seed_summaries_are_not_mixed_into_overall_curve():
    indexed = index_rows({"aggregate": [row(), row(scope=42, checkpoints=12)]})
    assert len(indexed) == 1


def test_only_n10_reference_does_not_become_a_n1_to_10_curve():
    summary = {"validation_reference": [
        {"model": "Qwen", "target": "full_validation", "selected_lift": .01, "checkpoints": 36},
        {"model": "Gemma", "target": "full_validation", "selected_lift": .02, "checkpoints": 36},
    ]}
    x, y = reference_points(summary, "Qwen", "full_validation")
    np.testing.assert_array_equal(x, [10])
    np.testing.assert_allclose(y, [.01])


def test_duplicate_reference_coordinate_is_rejected():
    reference = {"model": "Qwen", "target": "full_validation", "n": 10, "selected_lift": .01}
    with pytest.raises(ValueError, match="one measured row"):
        reference_points({"validation_reference": [reference, reference]}, "Qwen", "full_validation")


def test_validation_reference_cannot_use_a_larger_checkpoint_cohort():
    reference = {"model": "Qwen", "target": "full_validation", "n": 10, "selected_lift": .01,
                 "checkpoints": 50}
    with pytest.raises(ValueError, match="checkpoint cohort"):
        reference_points({"validation_reference": [reference]}, "Qwen", "full_validation", 36)


def test_surface_keeps_unmeasured_cells_and_zero_single_candidate_lift():
    grid, counts = surface_grid([row(candidate_count=1, validation_lift={"mean": 0})],
                                "Qwen", "task_mean", "past")
    assert grid[0, 9] == 0 and counts[0, 9] == 36
    assert np.isnan(grid[4, 9]) and counts[4, 9] == 0


def test_single_candidate_surface_cannot_claim_selection_gain():
    with pytest.raises(ValueError, match="single candidate"):
        surface_grid([row(candidate_count=1)], "Qwen", "task_mean", "past")


def test_surface_cannot_change_coverage_when_candidate_count_changes():
    rows = [row(candidate_count=2), row(candidate_count=5, checkpoints=30)]
    with pytest.raises(ValueError, match="fixed checkpoint cohort"):
        surface_grid(rows, "Qwen", "task_mean", "past")


def test_surface_rejects_an_infeasible_period_budget():
    with pytest.raises(ValueError, match="infeasible"):
        surface_grid([row(candidate_count=2, case="current_past_future", n=2)],
                     "Qwen", "task_mean", "current_past_future")


def test_uniform_training_control_requires_the_same_checkpoint_identities():
    summary = {"fixed_checkpoint_keys": ["s42_cp010"], "source_sha256": "a", "llm_judgments_used": False}
    reference = {**summary, "fixed_checkpoint_keys": ["s42_cp011"]}
    with pytest.raises(ValueError, match="same LLM-free data"):
        index_uniform_reference(summary, reference)


def test_uniform_control_is_one_pool_without_an_invented_period_weighting():
    summary = {"fixed_checkpoint_keys": ["s42_cp010"], "source_sha256": "a", "llm_judgments_used": False}
    reference = {**summary, "aggregate": [row(weight="uniform", checkpoints=1)]}
    indexed = index_uniform_reference(summary, reference)
    assert indexed["Qwen", "parent_preservation", 10]["weight"] == "uniform"
