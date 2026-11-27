# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import copy

import numpy as np
import pytest

from scripts.premise.analyze_bfcl_alignment import calculate
from scripts.premise.plot_bfcl_alignment_sample_sizes import (
    calculate_model,
    correlation_table,
    sampling_std,
    subset_draws,
)


def model():
    matrix = [[1] * 5 + [0] * 5, [0] * 5 + [1] * 5, [0] * 10, [0] * 10, [0] * 10]
    point = {"key": "s42_cp010", "seed": 42, "checkpoint": 10,
             "samples": {"train10": matrix, "val10": matrix},
             "scores": {"train10": [0.5, 0.5, 0, 0, 0], "val10": [0.5, 0.5, 0, 0, 0],
                        "full_train": [1, 0, 0, 0, 0], "full_validation": [0, 1, 0, 0, 0],
                        "train_remaining40": [1, 0, 0, 0, 0], "validation_remaining40": [0, 1, 0, 0, 0]},
             "parent_scores": {"full_train": 0.1, "full_validation": None,
                               "train_remaining40": 0.1, "validation_remaining40": None}}
    return {"label": "Synthetic", "points": [point], "reference": calculate([point])["aggregate"],
            "errors": 0, "evaluations": 550}


def test_one_task_selection_integrates_every_task_and_uniform_ties_exactly():
    result = calculate_model(model(), trials=200, seed=42)
    # Five tasks choose the perfect candidate; the other five choose a zero-reward candidate.
    train = result["sizes"][0]["targets"]["full_train"]
    for selector in ("train10", "val10"):
        assert train[selector]["mean_selected_performance"] == pytest.approx(0.5)
        assert train[selector]["mean_selected_gain"] == pytest.approx(0.4)
        assert train[selector]["mean_selection_gain"] == pytest.approx(0.3)
    # At n=10 the first two candidates tie, and their expected performance remains 0.5.
    assert result["sizes"][-1]["targets"]["full_train"]["train10"]["mean_selected_performance"] == 0.5
    assert result["n10_matches_existing_summary"]


def test_gain_is_exact_and_independent_of_monte_carlo_seed_and_trial_count():
    first = calculate_model(model(), trials=10, seed=1)
    second = calculate_model(model(), trials=150, seed=99)
    for a, b in zip(first["sizes"], second["sizes"], strict=True):
        for target in ("full_train", "full_validation"):
            for selector in ("train10", "val10"):
                assert a["targets"][target][selector]["mean_selected_performance"] == (
                    b["targets"][target][selector]["mean_selected_performance"]
                )


def test_undefined_correlations_do_not_remove_checkpoints_from_the_gain_average():
    data = model()
    constant = copy.deepcopy(data["points"][0])
    constant.update(key="s42_cp020", checkpoint=20)
    for target in ("full_train", "full_validation", "train_remaining40", "validation_remaining40"):
        constant["scores"][target] = [0.2] * 5
    constant["parent_scores"]["full_train"] = 0
    data["points"].append(constant)
    data["reference"] = calculate(data["points"])["aggregate"]
    result = calculate_model(data, trials=100, seed=42)
    train = result["sizes"][0]["targets"]["full_train"]
    assert train["paired_checkpoint_count_mean"] == 1
    assert train["train10"]["mean_selected_gain"] == pytest.approx(0.3)
    constant_rows = [row for row in result["checkpoint_scores"] if row["checkpoint"] == "s42_cp020"]
    assert all(row["defined_correlation_subsets"] == 0 for row in constant_rows)


def test_no_validation_parent_gain_is_invented_and_ten_task_shading_collapses():
    result = calculate_model(model(), trials=100, seed=42)
    for size in result["sizes"]:
        for selector in ("train10", "val10"):
            assert size["targets"]["full_validation"][selector]["mean_selected_gain"] is None
    last = result["sizes"][-1]["targets"]["full_train"]["train10"]
    assert last["selected_gain_sampling_range"] == pytest.approx([0.4, 0.4])
    assert last["pearson_sampling_range"][0] == pytest.approx(last["pearson_sampling_range"][1])
    assert last["pearson_std"] == 0
    assert last["selected_gain_std"] == 0
    assert last["selection_gain_std"] == 0


def test_sampling_standard_deviation_excludes_undefined_trials_and_collapses_without_sampling():
    assert sampling_std(np.array([0, 2, np.nan])) == pytest.approx(1)
    assert sampling_std(np.array([0.8, 0.8, 0.8])) == 0
    assert sampling_std(np.array([np.nan])) is None


def test_constant_arrays_have_undefined_correlation_despite_rounding_in_numpy_mean():
    values = correlation_table(np.array([[[0.8] * 5, [0, 0.2, 0.4, 0.6, 0.8]]]),
                               np.array([[0.8] * 5]))
    assert np.isnan(values).all()


def test_random_subset_draws_are_stable_per_checkpoint_and_match_across_models():
    first = subset_draws(["s42_cp010", "s43_cp020"], "train10", 4, 210, 100, 42)
    reordered = subset_draws(["s43_cp020", "s42_cp010"], "train10", 4, 210, 100, 42)
    assert np.array_equal(first, reordered[:, ::-1])
    validation = subset_draws(["s42_cp010", "s43_cp020"], "val10", 4, 210, 100, 42)
    assert not np.array_equal(first, validation)
    assert ((first >= 0) & (first < 210)).all()


def test_ten_task_validation_rejects_an_accidentally_changed_existing_target():
    data = model()
    data["reference"]["full_train"]["train10"]["mean_selected_gain"] += 0.01
    with pytest.raises(ValueError, match="n=10 differs"):
        calculate_model(data, trials=10, seed=42)
