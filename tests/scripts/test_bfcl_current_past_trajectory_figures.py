# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Guard matched-cohort and missing-data semantics in current/past figures."""

from __future__ import annotations

import pytest

from scripts.premise.plot_bfcl_current_past_trajectory_rewards import index_rows


def row(**overrides) -> dict:
    value = {"model": "Qwen", "cohort": "primary", "recipe": "success_count", "n": 10,
             "weighting": "task_mean", "checkpoints": 2, "validation_lift": {"mean": .01}}
    return value | overrides


def test_weightings_and_reward_rules_must_use_matched_checkpoint_coverage():
    with pytest.raises(ValueError, match="matched checkpoint"):
        index_rows({"aggregate": [row(), row(weighting="period_mean", checkpoints=3)]})
    with pytest.raises(ValueError, match="matched checkpoint"):
        index_rows({"aggregate": [row(), row(recipe="failure_progress", checkpoints=3)]})
    result = index_rows({"aggregate": [row(), row(weighting="period_mean"), row(n=9, checkpoints=4)]})
    assert len(result) == 3


def test_zero_coverage_cannot_be_presented_as_zero_measured_gain():
    with pytest.raises(ValueError, match="Zero-checkpoint"):
        index_rows({"aggregate": [row(checkpoints=0, validation_lift={"mean": 0})]})
    result = index_rows({"aggregate": [row(checkpoints=0, validation_lift={"mean": None})]})
    assert len(result) == 1


def test_objective_cohort_cannot_silently_include_judgment_dependent_rewards():
    with pytest.raises(ValueError, match="semantic judgments"):
        index_rows({"aggregate": [row(cohort="objective", recipe="parent_safe")]})
    with pytest.raises(ValueError, match="Unsupported"):
        index_rows({"aggregate": [row(model="OSS")]})
    with pytest.raises(ValueError, match="Duplicate"):
        index_rows({"aggregate": [row(), row()]})
