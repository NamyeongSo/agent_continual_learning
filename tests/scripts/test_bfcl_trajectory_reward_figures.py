# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Check scientific figure inputs without rendering artificial experiment data."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.premise.plot_bfcl_trajectory_rewards import metric_value, standardize_bivariate


def rows() -> list[dict]:
    result = []
    target = [0, .2, .8, .6, 1]
    for key in ("cp1", "cp2"):
        for recipe, score in (("success_count", [0, 1, 2, 3, 4]),
                              ("failure_progress", [0, 2, 1, 3, 4]),
                              ("certified_completion", [0, 1, 2, 3, 4])):
            result.append({"model": "Qwen", "key": key, "recipe": recipe, "cohort": "primary", "n": 10,
                           "candidate_scores": score, "full_validation_scores": target})
    return result


def test_missing_summary_values_never_become_plotted_zero():
    assert np.isnan(metric_value({}, "validation_lift"))
    assert np.isnan(metric_value({"validation_lift": {"mean": None}}, "validation_lift"))
    assert metric_value({"validation_lift": {"mean": 0}}, "validation_lift") == 0
    assert np.isnan(metric_value({"validation_lift": 1}, "validation_lift", "std"))


def test_bivariate_cloud_preserves_actual_pairs_and_mean_within_checkpoint_rho():
    source = rows()
    clouds, notes = standardize_bivariate(source, "full_validation")
    cloud = clouds["Qwen", "failure_progress"]
    assert cloud.shape == (10, 2)
    np.testing.assert_allclose(cloud[:5].mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(cloud[:5].std(axis=0), 1, atol=1e-14)
    expected = np.corrcoef(source[1]["candidate_scores"], source[1]["full_validation_scores"])[0, 1]
    note = next(row for row in notes if row["model"] == "Qwen" and row["recipe"] == "failure_progress")
    assert note["rho"] == pytest.approx(expected)
    assert note["candidate_pairs"] == 10
    assert note["common_defined_checkpoints"] == 2


def test_bivariate_constants_are_excluded_jointly_and_not_shown_as_zero_correlation():
    source = rows()
    source[1]["candidate_scores"] = [1] * 5
    clouds, notes = standardize_bivariate(source, "full_validation")
    assert all(clouds["Qwen", recipe].shape == (5, 2)
               for recipe in ("success_count", "failure_progress", "certified_completion"))
    assert all(row["checkpoint_keys"] == ["cp2"] for row in notes if row["model"] == "Qwen")
    assert all(row["rho"] is None for row in notes if row["model"] == "Gemma")


def test_bivariate_does_not_use_lex_encoding_or_mismatched_targets():
    with pytest.raises(ValueError):
        standardize_bivariate(rows(), "full_validation", ("parent_safe",))
    source = rows()
    source[1]["full_validation_scores"] = [0, 1, 0, 1, 0]
    with pytest.raises(ValueError):
        standardize_bivariate(source, "full_validation")


def test_bivariate_ignores_other_n_and_objective_cohorts_and_rejects_duplicates():
    source = rows()
    source[1]["n"] = 9
    clouds, _ = standardize_bivariate(source, "full_validation")
    assert clouds["Qwen", "success_count"].shape == (5, 2)
    source = rows()
    source[1]["cohort"] = "objective"
    clouds, _ = standardize_bivariate(source, "full_validation")
    assert clouds["Qwen", "success_count"].shape == (5, 2)
    with pytest.raises(ValueError):
        standardize_bivariate([*rows(), rows()[0]], "full_validation")


def test_objective_bivariate_selects_separate_cohort_without_reusing_semantic_rows():
    source = rows()
    for row in source:
        row["cohort"] = "objective"
        row["recipe"] = {"failure_progress": "success_efficiency",
                         "certified_completion": "successful_recovery"}.get(row["recipe"], row["recipe"])
    recipes = ("success_count", "success_efficiency", "successful_recovery")
    clouds, notes = standardize_bivariate(source, "full_validation", recipes, cohort="objective")
    assert clouds["Qwen", "success_count"].shape == (10, 2)
    assert all(row["cohort"] == "objective" for row in notes)
    default_clouds, _ = standardize_bivariate(source, "full_validation")
    assert not len(default_clouds["Qwen", "success_count"])
