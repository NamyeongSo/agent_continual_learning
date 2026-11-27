# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Frozen trajectory-grounded selection rewards using only observed sampled tasks.

The caller supplies all five candidates and their shared parent on the same n tasks.
No full-set scores, trajectories, file paths, or sampling populations are accepted.
Progress is the verified lower bound on a shared public requirement inventory, in
[0, 1]. A material violation is binary and requires confirmed cited evidence.
Unconfirmed violations can be zero only if their unknown flags remain available
outside this module; judge/schema failures must be handled before calling it.

Missing turns cannot earn efficiency: the caller must exclude that observation or
provide a conservative budget-cap turn count where its stated analysis permits.
Avoidable counts exclude appropriate recovery from unexpected or transient errors.
"""

from __future__ import annotations

import numpy as np

REWARD_KEYS = (
    "success_count", "success_efficiency", "failure_progress", "certified_completion",
    "parent_safe", "successful_recovery", "quality_gated",
)
LEXICOGRAPHIC_KEYS = frozenset(("parent_safe", "quality_gated"))


def _array(value: np.ndarray, shape: tuple[int, ...], name: str, kind: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must have shape {shape} with finite values")
    if kind == "binary" and not np.isin(value, (0, 1)).all():
        raise ValueError(f"{name} must be binary")
    if kind == "progress" and not ((value >= 0) & (value <= 1)).all():
        raise ValueError(f"{name} must be in [0, 1]")
    if kind == "count" and not ((value >= 0) & (value == np.floor(value))).all():
        raise ValueError(f"{name} must contain nonnegative integer counts")
    return value


def objective_reward_scores(
    success: np.ndarray,
    turns: np.ndarray,
    avoidable: np.ndarray,
    max_steps: int = 20,
) -> dict[str, np.ndarray]:
    """The three trace-only rules, without fabricated judge or parent fields."""
    success = np.asarray(success, dtype=float)
    if success.ndim != 3 or success.shape[0] < 1 or success.shape[1] != 5 or not 1 <= success.shape[2] <= 10:
        raise ValueError("success must have shape (positive trials, 5 candidates, n=1..10)")
    if not np.isscalar(max_steps) or not np.isfinite(max_steps) or max_steps < 1 or int(max_steps) != max_steps:
        raise ValueError("max_steps must be a positive integer")
    shape = success.shape
    success = _array(success, shape, "success", "binary")
    turns = _array(turns, shape, "turns", "count")
    avoidable = _array(avoidable, shape, "avoidable", "count")
    count = success.sum(axis=-1)
    efficiency = np.maximum(0, 1 - np.minimum(turns, max_steps) / max_steps)
    avoidable_cost = np.minimum(avoidable, max_steps) / max_steps
    return {
        "success_count": count,
        "success_efficiency": count + .25 * (success * efficiency).sum(axis=-1) / np.maximum(1, count),
        "successful_recovery": count - .25 * (success * avoidable_cost).mean(axis=-1),
    }


def reward_scores(
    success: np.ndarray,
    turns: np.ndarray,
    progress: np.ndarray,
    material: np.ndarray,
    avoidable: np.ndarray,
    parent_success: np.ndarray,
    parent_turns: np.ndarray,
    parent_progress: np.ndarray,
    parent_material: np.ndarray,
    max_steps: int = 20,
) -> dict[str, np.ndarray]:
    """Return the fixed baseline and six recipes, maximizing every coordinate.

    Candidate inputs have shape (trials, 5, n), parent inputs (trials, n), and
    1 <= n <= 10. Scalar scores have shape (trials, 5). ``parent_safe`` has four
    lexicographic coordinates and ``quality_gated`` has five; earlier coordinates
    always dominate later ones. The function does not mutate supplied arrays.
    """
    success = np.asarray(success, dtype=float)
    if success.ndim != 3 or success.shape[0] < 1 or success.shape[1] != 5 or not 1 <= success.shape[2] <= 10:
        raise ValueError("success must have shape (positive trials, 5 candidates, n=1..10)")
    if not np.isscalar(max_steps) or not np.isfinite(max_steps) or max_steps < 1 or int(max_steps) != max_steps:
        raise ValueError("max_steps must be a positive integer")
    shape = success.shape
    parent_shape = (shape[0], shape[2])
    success = _array(success, shape, "success", "binary")
    turns = _array(turns, shape, "turns", "count")
    progress = _array(progress, shape, "progress", "progress")
    material = _array(material, shape, "material", "binary")
    avoidable = _array(avoidable, shape, "avoidable", "count")
    parent_success = _array(parent_success, parent_shape, "parent_success", "binary")[:, None, :]
    parent_turns = _array(parent_turns, parent_shape, "parent_turns", "count")[:, None, :]
    parent_progress = _array(parent_progress, parent_shape, "parent_progress", "progress")[:, None, :]
    parent_material = _array(parent_material, parent_shape, "parent_material", "binary")[:, None, :]

    objective = objective_reward_scores(success, turns, avoidable, max_steps)
    count = objective["success_count"]
    efficiency = np.maximum(0, 1 - np.minimum(turns, max_steps) / max_steps)
    regressions = ((parent_success == 1) & (success == 0)).sum(axis=-1)
    repairs = ((parent_success == 0) & (success == 1)).sum(axis=-1)
    joint_failure = (parent_success == 0) & (success == 0)
    certified_joint_success = (parent_success == 1) & (success == 1) & (parent_material == 0) & (material == 0)
    relative_efficiency = np.clip((parent_turns - turns) / np.maximum(1, parent_turns), -1, 1)

    scores = {
        "success_count": count,
        "success_efficiency": objective["success_efficiency"],
        "failure_progress": count + .25 * ((1 - success) * progress).mean(axis=-1),
        "certified_completion": (success * (1 - material)).sum(axis=-1)
        + .25 * ((1 - material) * ((1 - success) * progress + success * efficiency)).mean(axis=-1),
        "parent_safe": np.stack((
            -regressions, repairs,
            (joint_failure * (progress - parent_progress)).mean(axis=-1),
            (certified_joint_success * relative_efficiency).mean(axis=-1),
        ), axis=-1),
        "successful_recovery": objective["successful_recovery"],
        "quality_gated": np.stack((
            count, -material.sum(axis=-1), np.floor(10 * progress.mean(axis=-1) + .5),
            -(success * avoidable).sum(axis=-1), -(success * turns).sum(axis=-1),
        ), axis=-1),
    }
    assert tuple(scores) == REWARD_KEYS
    return scores


def ordering_ranks(values: np.ndarray) -> np.ndarray:
    """Return ascending average ranks 1..5 for scalar or lexicographic scores.

    Round coordinates to twelve decimals solely to stabilize arithmetic ties.
    Lexicographic scores never receive a scalar encoding with arbitrary spacing.
    """
    values = np.asarray(values, dtype=float)
    if (values.ndim not in (2, 3) or values.shape[0] < 1 or values.shape[1] != 5
            or (values.ndim == 3 and values.shape[2] < 1) or not np.isfinite(values).all()):
        raise ValueError("values must be finite scalar (trials, 5) or lexicographic (trials, 5, coordinates) scores")
    values = np.round(values, 12)
    if values.ndim == 2:
        values = values[..., None]
    differences = values[:, :, None, :] - values[:, None, :, :]
    unequal = differences != 0
    first_difference = np.take_along_axis(differences, unequal.argmax(axis=-1)[..., None], axis=-1)[..., 0]
    smaller_count = (first_difference > 0).sum(axis=-1)
    equal_count = (~unequal.any(axis=-1)).sum(axis=-1)
    return 1 + smaller_count + .5 * (equal_count - 1)


def choice_probabilities(values: np.ndarray) -> np.ndarray:
    """Uniform probability over candidates with the maximal scalar/lex score."""
    ranks = ordering_ranks(values)
    selected = ranks == ranks.max(axis=-1, keepdims=True)
    return selected / selected.sum(axis=-1, keepdims=True)
