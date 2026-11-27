# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import pytest

from scripts.utils.measure_checkpoint_stability_gap import aggregate, gap


def test_task_gaps_expose_losses_even_when_repairs_preserve_overall_accuracy():
    before, after = [1, 1, 0], [0, 1, 1]
    assert sum(before) == sum(after)
    results = [gap(a, [b]) for a, b in zip(before, after, strict=True)]
    assert results == [(1, "measured"), (0, "measured"), (None, "zero_reference")]
    assert sum(value for value, _ in results if value is not None) / 2 == .5


def test_zero_reference_and_missing_are_undefined_and_negative_gaps_are_retained():
    assert gap(0, [0])[0] is None
    assert gap(None, [1]) == (None, "missing_reference")
    assert gap(1, [None]) == (None, "missing_window")
    assert gap(.4, [.8]) == pytest.approx((-1, "measured"))


def test_blocks_have_equal_weight_despite_growing_numbers_of_past_tasks():
    windows = [{"variant": "ace_aud", "benchmark": "bfcl", "seed": 42, "start_cp": start,
                "end_cp": start + 10, "past_tasks": start, "valid_pairs": count, "sg": value,
                "zero_reference": 0, "missing_reference": 0, "missing_window": 0}
               for start, count, value in [(10, 1, 1), (20, 9, 0)]]
    seeds, summary = aggregate(windows)
    assert seeds[0]["sg"] == .5
    assert seeds[0]["valid_pairs"] == 10
    row = next(r for r in summary if (r["variant"], r["benchmark"]) == ("ace_aud", "bfcl"))
    assert row["mean_sg"] == .5 and row["sample_sd"] is None
