# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Checkpoint distribution signs must not mistake floating noise for improvement."""

import pytest

from scripts.premise.plot_bfcl_checkpoint_distributions import (
    distribution_stats,
    jitter,
    summarize,
)


def test_rate_to_pp_conversion_and_numerical_ties():
    result = distribution_stats([-.02, -.01, 1e-16, .01, .02], "gain_past")
    assert result["unit"] == "pp"
    assert result["minimum"] == -2
    assert result["maximum"] == 2
    assert result["positive_checkpoints"] == 2
    assert result["negative_checkpoints"] == 2
    assert result["zero_checkpoints"] == 1
    assert result["median"] == pytest.approx(0, abs=1e-12)


def test_perfect_retention_tolerates_only_roundoff():
    result = distribution_stats([.8, .99, 1, 1 + 1e-16], "retention_past")
    assert result["unit"] == "%"
    assert result["mean"] == pytest.approx(94.75)
    assert result["perfect_retention_checkpoints"] == 2
    assert result["regression_checkpoints"] == 2
    with pytest.raises(ValueError, match="fraction"):
        distribution_stats([1.00001], "retention_past")


def test_distribution_groups_remain_within_n_seed_and_method():
    def row(seed, case, n, gain):
        return {"model": "Qwen", "weight": "task_mean", "n": n, "actual_n": n, "case": case,
                "seed": seed, "gain_past": gain, "retention_past": 1, "lift_validation40": 0}
    rows = [row(42, "past", 1, -.1), row(43, "past", 1, .1), row(42, "future", 1, .7),
            row(42, "past", 2, .5)]
    summaries = {(r["scope"], r["case"], r["n"], r["metric"]): r for r in summarize(rows)}
    past = summaries["all", "past", 1, "gain_past"]
    assert past["checkpoints"] == 2
    assert past["mean"] == 0
    assert summaries[42, "past", 1, "gain_past"]["mean"] == -10
    assert summaries["all", "past", 2, "gain_past"]["mean"] == 50
    assert summaries["all", "future", 1, "gain_past"]["mean"] == 70


def test_jitter_does_not_change_with_method_or_sample_size():
    point = {"key": "s42_cp010", "seed": 42}
    assert jitter(point) == jitter({**point, "n": 10, "case": "future", "model": "Gemma"})
    assert -.16 < jitter(point) < -.08
