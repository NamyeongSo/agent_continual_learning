# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import numpy as np
import pytest

from scripts.premise.analyze_bfcl_alignment_significance import (
    association_tests,
    correlation_permutations,
    gain_measurements,
    holm_adjust,
    paired_block_contributions,
    paired_points,
    sign_flip_test,
)


def point(key, checkpoint, seed=42):
    return {"key": key, "seed": seed, "checkpoint": checkpoint,
            "scores": {"train10": [0, 0, 0.1, 0.1, 1], "val10": [1, 1, 0.9, 0.9, 0],
                       "full_train": [0.2, 0.4, 0.6, 0.8, 1]}}


def test_correlation_permutations_preserve_the_two_evaluators_pairing_and_identity():
    table = correlation_permutations([point("cp", 10)], "full_train")
    assert table.shape == (1, 120, 2)
    expected = np.corrcoef([0, 0, 0.1, 0.1, 1], [0.2, 0.4, 0.6, 0.8, 1])[0, 1]
    assert table[0, 0, 0] == pytest.approx(expected)
    assert table[0, :, 0] == pytest.approx(-table[0, :, 1])
    assert table.mean(axis=1) == pytest.approx(np.zeros((1, 2)), abs=1e-14)


def test_jointly_undefined_checkpoint_is_excluded_and_never_zero_imputed():
    flat = point("flat", 11)
    flat["scores"]["val10"] = [0] * 5
    assert [row["key"] for row in paired_points([flat, point("ok", 10)], "full_train")] == ["ok"]
    with pytest.raises(ValueError, match="Duplicate"):
        paired_points([flat, flat], "full_train")


def test_blocks_use_original_checkpoint_order_and_do_not_compress_missing_correlations():
    points = [point(str(i), i) for i in range(5)]
    values, blocks = paired_block_contributions(points, [points[0], points[3], points[4]], np.array([1, 2, 3]), 2)
    assert values == pytest.approx([1 / 3, 2 / 3, 1])
    assert blocks[0]["all_checkpoint_keys"] == ["0", "1"]
    assert blocks[1]["all_checkpoint_keys"] == ["2", "3"]
    assert blocks[2]["all_checkpoint_keys"] == ["4"]


def test_seed_boundaries_are_preserved_when_forming_blocks():
    first, second = point("first", 10, 42), point("second", 10, 43)
    values, blocks = paired_block_contributions([first, second], [first, second], np.array([1, 3]), 1000)
    assert values == pytest.approx([0.5, 1.5])
    assert [block["seed"] for block in blocks] == [42, 43]


def test_exact_sign_flip_counts_include_both_tails_and_dependence_blocks():
    result = sign_flip_test(np.array([0.1, 0.2, 0.3]), 100, np.random.default_rng(0))
    assert result["exact"]
    assert result["permutations"] == 8
    assert result["p_two_sided"] == pytest.approx(0.25)
    assert sign_flip_test(np.array([0.6]), 100, np.random.default_rng(0))["p_two_sided"] == 1
    assert sign_flip_test(np.array([0.0, 0.0]), 100, np.random.default_rng(0))["p_two_sided"] == 1


def test_association_test_reports_within_seed_means_and_checkpoint_weighted_overall_mean():
    points = [point("a", 10), point("b", 20), point("c", 10, 43)]
    points[2]["scores"]["full_train"] = [1, 0.8, 0.6, 0.4, 0.2]
    rows = association_tests(points, "full_train", trials=100, seed=7)
    means = {row["seed"]: row["mean_rho"] for row in rows if row["selector"] == "train10"}
    assert means["42"] == pytest.approx(-means["43"])
    assert means["all"] == pytest.approx((2 * means["42"] + means["43"]) / 3)
    assert all(0 < row["p_two_sided"] <= 1 for row in rows)
    assert rows == association_tests(points, "full_train", trials=100, seed=7)


def test_holm_adjustment_preserves_original_order_and_controls_the_full_family():
    assert holm_adjust([0.04, 0.01, 0.03, 0.002]) == pytest.approx([0.06, 0.03, 0.06, 0.008])


def test_gains_include_flat_scores_integrate_ties_and_do_not_invent_validation_parent_scores():
    first = {"key": "a", "seed": 42, "checkpoint": 10,
             "scores": {"train10": [1, 1, 0, 0, 0], "val10": [0, 1, 0, 0, 0],
                        "full_train": [1, 0, 0, 0, 0], "full_validation": [0, 1, 0, 0, 0]},
             "parent_scores": {"full_train": 0.4, "full_validation": None}}
    second = {"key": "b", "seed": 43, "checkpoint": 20,
              "scores": {"train10": [0, 0, 0, 0, 0], "val10": [0, 0, 0, 0, 1],
                         "full_train": [0.4, 0.5, 0.6, 0.7, 0.8],
                         "full_validation": [0.8, 0.7, 0.6, 0.5, 0.4]},
              "parent_scores": {"full_train": 0.6, "full_validation": None}}
    reference = {"full_train": {"train10": {"mean_selected_performance": 0.55},
                                "val10": {"mean_selected_performance": 0.4}},
                 "full_validation": {"train10": {"mean_selected_performance": 0.55},
                                     "val10": {"mean_selected_performance": 0.7}}}
    rows = gain_measurements({"checkpoints": [first, second], "sizes": [{"n": 10, "targets": reference}]})
    assert len(rows) == 9
    data = {(row["target"], row["metric"], row["selector"]): row["values"] for row in rows}
    train = "full_train"
    assert data[(train, "selection_lift_vs_proposal_mean", "train10")] == pytest.approx([0.3, 0])
    assert data[(train, "selected_gain_vs_parent", "train10")] == pytest.approx([0.1, 0])
    assert data[("full_validation", "selection_lift_vs_proposal_mean", "val10")] == pytest.approx([0.8, -0.2])
    assert not any(row["target"] == "full_validation" and row["metric"].endswith("vs_parent") for row in rows)
    difference = data[(train, "aligned_minus_cross_selected_performance", "train10")]
    parent_difference = data[(train, "selected_gain_vs_parent", "train10")] - data[
        (train, "selected_gain_vs_parent", "val10")
    ]
    proposal_difference = data[(train, "selection_lift_vs_proposal_mean", "train10")] - data[
        (train, "selection_lift_vs_proposal_mean", "val10")
    ]
    assert difference == pytest.approx(parent_difference)
    assert difference == pytest.approx(proposal_difference)
