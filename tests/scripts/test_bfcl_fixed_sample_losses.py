# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import numpy as np
import pytest

from scripts.premise.analyze_bfcl_alignment_significance import rng_for
from scripts.premise.analyze_bfcl_fixed_sample_losses import (
    BETAS,
    LOSS_NAMES,
    calculate,
    fixed_groups,
    loss_metadata,
    loss_scores,
)
from scripts.premise.analyze_bfcl_similarity_extremes import mixed_indices, ranked_draws
from scripts.premise.analyze_bfcl_training_selection_hypotheses import selection_probabilities, similarities


def fixture_point():
    rng = np.random.default_rng(3)
    rewards = rng.integers(2, size=(5, 50))
    order = [f"task{i}" for i in range(50)]
    point = {"key": "s42_cp010", "seed": 42, "checkpoint": 10, "train_order": order,
             "train_rewards": rewards.tolist(), "parent_scores": {"full_train": 0.3},
             "scores": {"full_train": rewards.mean(axis=1).tolist(), "full_validation": [0, 0.1, 0.2, 0.3, 0.4]}}
    metadata = {task: {"first_request": f"Action {i} for saved document", "tool_classes": [f"Tools{i % 3}"]}
                for i, task in enumerate(order[:11])}
    return point, metadata


def test_task_mean_matches_all_ten_individual_rewards_and_negative_loss_argmin():
    rng = np.random.default_rng(12)
    rewards = rng.integers(2, size=(20, 10, 5))
    current, near, far = rewards[:, 0], rewards[:, 1:6].mean(axis=1), rewards[:, 6:].mean(axis=1)
    scores = loss_scores(current, near, far)
    assert set(scores) == set(LOSS_NAMES)
    assert scores["task_mean"] == pytest.approx(rewards.mean(axis=1))
    for score in scores.values():
        assert np.argmax(score, axis=1).tolist() == np.argmin(-score, axis=1).tolist()


def test_beta_endpoints_discard_only_the_opposite_group_and_record_effective_sample_counts():
    current = np.array([[0, 1, 0, 1, 0]])
    near = np.array([[1, 0.2, 0.6, 0.3, 0.4]])
    far = np.array([[0.1, 0.4, 0.6, 0.8, 0.2]])
    before = loss_scores(current, near, far)
    changed_near = loss_scores(current, 1 - near, far)
    changed_far = loss_scores(current, near, 1 - far)
    assert before["beta_000"] == pytest.approx(changed_near["beta_000"])
    assert before["beta_100"] == pytest.approx(changed_far["beta_100"])
    metadata = {row["loss"]: row for row in loss_metadata()}
    assert metadata["beta_000"]["fixed_nonzero_weight_tasks"] == 5
    assert metadata["beta_100"]["fixed_nonzero_weight_tasks"] == 6
    assert metadata["robust_replay"]["fixed_nonzero_weight_tasks"] is None
    for row in metadata.values():
        assert row["evaluated_tasks"] == 10
        if row["loss"] != "robust_replay":
            assert row["current_weight"] + row["near_group_weight"] + row["far_group_weight"] == pytest.approx(1)


def test_half_current_weight_cannot_reverse_current_success_except_extreme_ties():
    # A perfect replay-only candidate can tie the current-success candidate, but cannot beat it.
    scores = loss_scores(np.array([[1, 0]]), np.array([[0, 1]]), np.array([[0, 1]]))
    for name in ("period_mean", "robust_replay", *(f"beta_{int(100 * b):03d}" for b in BETAS)):
        assert scores[name][0, 0] == pytest.approx(scores[name][0, 1])
    assert scores["task_mean"][0, 0] < scores["task_mean"][0, 1]
    assert scores["equal_three"][0, 0] < scores["equal_three"][0, 1]
    scores = loss_scores(np.array([[1, 0]]), np.array([[0.1, 0.9]]), np.array([[0.1, 0.9]]))
    for name in ("period_mean", "robust_replay", *(f"beta_{int(100 * b):03d}" for b in BETAS)):
        assert scores[name][0, 0] > scores[name][0, 1]


def test_robust_replay_penalizes_one_sided_group_performance():
    scores = loss_scores(np.array([[0, 0]]), np.array([[1, 0.5]]), np.array([[0, 0.5]]))
    assert np.argmax(scores["beta_100"]) == 0
    assert np.argmax(scores["robust_replay"]) == 1


def test_fixed_sample_groups_reuse_original_near_far_draws_exactly():
    point, metadata = fixture_point()
    current, near, far, digest = fixed_groups(point, metadata, 50, 20261021)
    sim = similarities(point["train_order"][:11], metadata)
    ranking = ranked_draws(sim[-1, :-1], 50, rng_for(20261021, "near-far-ties:s42_cp010"))
    indices = mixed_indices(ranking, 5)
    matrix = np.asarray(point["train_rewards"])
    assert np.all(indices < point["checkpoint"])
    assert all(len(set(row)) == 9 for row in indices)
    assert current == pytest.approx(matrix[:, 10][None, :])
    assert near == pytest.approx(matrix[:, indices[:, :5]].mean(axis=-1).T)
    assert far == pytest.approx(matrix[:, indices[:, 5:]].mean(axis=-1).T)
    assert len(digest["past_index_draws_sha256"]) == 64


def test_selection_never_uses_validation_targets_or_future_rewards_and_metadata():
    point, metadata = fixture_point()
    model = {"label": "Synthetic", "points": [point]}
    before = calculate(model, metadata, {point["key"]}, 30, 7)
    point["scores"]["full_validation"] = [1, 0.8, 0.4, 0, 0.2]
    matrix = np.asarray(point["train_rewards"])
    matrix[:, 11:] = 1 - matrix[:, 11:]
    point["train_rewards"] = matrix.tolist()
    after = calculate(model, metadata, {point["key"]}, 30, 7)
    assert before[2] == after[2]
    for left, right in zip(before[1], after[1], strict=True):
        assert left["loss"] == right["loss"]
        for index in range(1, 6):
            assert left[f"candidate_{index}_probability"] == right[f"candidate_{index}_probability"]


def test_score_ties_select_uniformly_and_flat_targets_have_no_correlations():
    point, metadata = fixture_point()
    point["train_rewards"] = np.ones((5, 50)).tolist()
    point["scores"] = {"full_train": [1] * 5, "full_validation": [0.4] * 5}
    aggregate, rows, _ = calculate({"label": "Flat", "points": [point]}, metadata, {point["key"]}, 30, 7)
    assert selection_probabilities(np.ones((3, 5))) == pytest.approx(np.full((3, 5), 0.2))
    assert all(abs(row["validation_lift"]) < 1e-12 for row in rows)
    assert all(row["rho_full_validation"] is None for row in rows)
    assert all(row["rho_full_validation_common_checkpoints"]["mean"] == 0 for row in aggregate)
    assert all(row["rho_full_validation"]["defined_trials"] == 0 for row in aggregate)


def test_common_rho_coverage_excludes_cp_if_any_loss_is_constant():
    point, metadata = fixture_point()
    matrix = np.zeros((5, 50))
    # Equal near/far means cancel the current term under every half-current score. Task mean still has
    # a varying score, so this CP must be excluded from common coverage even for that valid loss.
    matrix[:, :10] = np.arange(5)[:, None] / 4
    matrix[:, 10] = 1 - np.arange(5) / 4
    point["train_rewards"] = matrix.tolist()
    point["scores"] = {"full_train": [0, 0.2, 0.4, 0.6, 0.8], "full_validation": [0, 0.1, 0.2, 0.3, 0.4]}
    aggregate, rows, _ = calculate({"label": "Coverage", "points": [point]}, metadata, {point["key"]}, 30, 7)
    task_mean = next(row for row in rows if row["loss"] == "task_mean")
    assert task_mean["rho_full_validation"] == pytest.approx(1)
    assert all(row["rho_full_validation_common_checkpoints"]["mean"] == 0 for row in aggregate)


def test_invalid_score_inputs_and_zero_draws_are_rejected():
    with pytest.raises(ValueError, match="draw-by-candidate"):
        loss_scores(np.array([1, 2]), np.array([1, 2]), np.array([1, 2]))
    with pytest.raises(ValueError, match="finite"):
        loss_scores(np.array([[np.nan]]), np.ones((1, 1)), np.ones((1, 1)))
    point, metadata = fixture_point()
    with pytest.raises(ValueError, match="positive"):
        fixed_groups(point, metadata, 0, 7)
