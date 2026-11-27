# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Guard coverage and no-selection baseline semantics in candidate surfaces."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.premise.plot_bfcl_candidate_count_surfaces import surface_grid


def row(**overrides) -> dict:
    value = {"model": "Qwen", "cohort": "primary", "recipe": "success_count", "n": 10,
             "candidate_count": 1, "checkpoints": 5, "validation_lift": {"mean": 0}}
    return value | overrides


def test_grid_preserves_measured_pp_values_and_missing_cells():
    source = [row(), row(candidate_count=5, validation_lift={"mean": -.001})]
    values, counts = surface_grid(source, "Qwen", "primary", "success_count")
    assert values[0, 9] == 0
    assert values[4, 9] == pytest.approx(-.1)
    assert counts[4, 9] == 5
    assert np.isnan(values[1, 9])
    assert np.isnan(values[0, 0])


def test_one_candidate_cannot_claim_selection_gain():
    with pytest.raises(ValueError, match="zero lift"):
        surface_grid([row(validation_lift={"mean": .01})], "Qwen", "primary", "success_count")


def test_candidate_counts_must_share_checkpoint_coverage_and_cells_cannot_repeat():
    with pytest.raises(ValueError, match="matched checkpoint"):
        surface_grid([row(), row(candidate_count=2, checkpoints=6)], "Qwen", "primary", "success_count")
    with pytest.raises(ValueError, match="Duplicate"):
        surface_grid([row(), row()], "Qwen", "primary", "success_count")
    with pytest.raises(ValueError, match="positive checkpoint"):
        surface_grid([row(checkpoints=0)], "Qwen", "primary", "success_count")
