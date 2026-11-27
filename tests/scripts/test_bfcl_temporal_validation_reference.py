# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import copy

import numpy as np
import pytest

from scripts.premise.add_bfcl_temporal_validation_reference import validation_reference


def point(key, matrix):
    return {"key": key, "seed": 42, "validation_task_ids": [f"val_{i:02d}" for i in range(50)],
            "candidate_rewards": np.asarray(matrix).tolist()}


def test_validation_reference_uses_fixed_checkpoints_and_uniform_ties_with_same_proposal_baseline():
    perfect = np.zeros((5, 50))
    perfect[0] = 1
    flat = np.zeros((5, 50))
    flat[:, :25] = 1
    model = {"label": "Synthetic", "points": [point("a", perfect), point("b", flat), point("unused", perfect)]}
    rows, checkpoint_rows = validation_reference(model, ["a", "b"], trials=100, seed=7, maximum_n=10)
    assert len(rows) == 10 and len(checkpoint_rows) == 20
    for row in rows:
        assert row["checkpoints"] == 2
        assert row["validation_pool_size"] == 50
        assert row["proposal_mean"] == pytest.approx(0.35)
        assert row["selected_performance"]["mean"] == pytest.approx(0.75)
        assert row["selection_lift"]["mean"] == pytest.approx(0.4)
        assert row["selection_lift"]["std"] == 0
    assert {row["key"] for row in checkpoint_rows} == {"a", "b"}


def test_canonical_task_order_makes_frozen_val10_first_order_irrelevant():
    rng = np.random.default_rng(8)
    matrix = rng.integers(2, size=(5, 50))
    model = {"label": "Synthetic", "points": [point("a", matrix)]}
    reordered = copy.deepcopy(model)
    reordered["points"][0]["validation_task_ids"].reverse()
    reordered["points"][0]["candidate_rewards"] = matrix[:, ::-1].tolist()
    assert validation_reference(model, ["a"], 500, 7, 10) == validation_reference(reordered, ["a"], 500, 7, 10)


def test_missing_checkpoint_or_wrong_pool_size_cannot_change_the_reference_cohort_silently():
    model = {"label": "Synthetic", "points": [point("a", np.ones((5, 50)))]}
    with pytest.raises(ValueError, match="cohort"):
        validation_reference(model, ["a", "missing"], 10, 7, 10)
    model["points"][0]["candidate_rewards"] = np.ones((5, 10)).tolist()
    with pytest.raises(ValueError, match="50-task"):
        validation_reference(model, ["a"], 10, 7, 10)
