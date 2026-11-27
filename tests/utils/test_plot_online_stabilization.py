# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Check causal smoothing and sustained plateau decisions without inventing missing observations."""

import pytest

from scripts.utils.plot_online_stabilization import assess_stability, smooth_scores


def test_ewma_pauses_at_missing_score_without_creating_a_valid_plateau_point():
    rows = smooth_scores([0, 100, None, 100], "ewma", window=2, alpha=0.2)
    assert [row["smoothed_score_percent"] for row in rows] == [0, 20, None, 36]
    assert [row["step"] for row in rows] == [1, 2, 3, 4]
    metrics, summary = assess_stability(rows, assessment_window=2, patience=1)
    assert not metrics[-1]["assessment_eligible"]
    assert summary["status"] == "insufficient_data"


def test_rolling_window_retains_missing_task_positions_and_excludes_incomplete_coverage():
    rows = smooth_scores([0, 100, None, 0], "rolling10", window=2)
    assert [row["smoothed_score_percent"] for row in rows] == [0, 50, 100, 0]
    assert [row["window_scored_tasks"] for row in rows] == [1, 2, 1, 1]
    assert not rows[-1]["eligible"]


@pytest.mark.parametrize("kind", ["ewma", "rolling10"])
def test_plateau_requires_full_warmup_assessment_window_and_confirmation(kind):
    rows = smooth_scores([50] * 25, kind)
    metrics, summary = assess_stability(rows)
    assert not metrics[17]["assessment_eligible"]
    assert metrics[18]["assessment_eligible"]
    assert summary["stable_onset_step"] == 19
    assert summary["stable_confirmation_step"] == 23
    assert summary["status"] == "stable_tail"
    assert summary["episodes"] == [{"onset_step": 19, "confirmation_step": 23, "last_step": 25}]


def test_later_shift_revokes_sustained_stability_and_keeps_temporary_episode():
    rows = smooth_scores([50] * 25 + [100] * 4, "ewma")
    _metrics, summary = assess_stability(rows)
    assert summary["status"] == "temporary_only"
    assert summary["stable_onset_step"] is None
    assert summary["first_plateau_onset_step"] == 19
    assert summary["episodes"][0]["last_step"] == 25


@pytest.mark.parametrize("scores", [[50, 60, 50, 60, 50, 50, 60, 50, 60, 50], [50 + 0.5 * k for k in range(10)]])
def test_both_small_slope_and_small_range_are_required(scores):
    rows = [{"step": k, "smoothed_score_percent": score, "eligible": True} for k, score in enumerate(scores, 1)]
    metrics, summary = assess_stability(rows, patience=1)
    assert metrics[-1]["assessment_eligible"]
    assert not metrics[-1]["criterion_passed"]
    assert summary["status"] == "not_stabilized"
