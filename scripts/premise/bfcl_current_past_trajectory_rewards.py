# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Trajectory rewards for one current task followed by n-1 sampled past tasks.

``task_mean`` weights every task by 1/n and exactly reproduces the unweighted
recipes. ``period_mean`` weights current and past equally as periods: the current
task receives 1/2 and each past task receives 1/(2(n-1)). At n=1 both modes use
only the current task, with weight one.

Let <z> denote the selected weighted mean, x success, q verified progress, h a
confirmed material violation, t turns, u avoidable repeated errors, and T the
turn limit. Set K=n<x>, e=max(0, 1-min(t,T)/T), c=min(u,T)/T. The scalar recipes
are K, K+b<xe>/max(1/n,<x>), K+b<(1-x)q>,
n<x(1-h)>+b<(1-h)((1-x)q+xe)>, and K-b<xc>. The two lexicographic recipes
replace every task mean and sum in the original rules by <.> and n<.>.

For task_mean, b=1/4. For period_mean, b=(1/4)n/(2(n-1)) when n>1. This scales
the secondary term to one quarter of the smallest attainable nonzero difference
in weighted success count, preserving the original completion priority. Without
this correction, a secondary term could outweigh a higher weighted success
count. The score is fractional under period_mean and is not a literal count.

Only the supplied n observations enter a score; this module has no task pool or
full-set target access. Missing observations and judge failures must be handled
by the caller before scoring, as for the original trajectory reward module.
"""

from __future__ import annotations

import numpy as np

from scripts.premise.bfcl_trajectory_selection_rewards import (
    REWARD_KEYS,
    objective_reward_scores,
    reward_scores,
)

WEIGHTING_MODES = ("task_mean", "period_mean")


def _validate_weighting(n: int, weighting: str) -> None:
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or not 1 <= n <= 10:
        raise ValueError("n must be an integer from 1 to 10")
    if weighting not in WEIGHTING_MODES:
        raise ValueError(f"weighting must be one of {WEIGHTING_MODES}")


def current_past_weights(n: int, weighting: str = "task_mean") -> np.ndarray:
    """Return weights in current-first order; do not silently resample tasks."""
    _validate_weighting(n, weighting)
    if weighting == "task_mean" or n == 1:
        return np.full(n, 1 / n)
    weights = np.full(n, 1 / (2 * (n - 1)))
    weights[0] = .5
    return weights


def primary_count_gap(n: int, weighting: str = "task_mean") -> float:
    """Minimum attainable positive gap in n times weighted binary success."""
    _validate_weighting(n, weighting)
    return n / (2 * (n - 1)) if weighting == "period_mean" and n > 1 else 1.


def weighted_objective_reward_scores(
    success: np.ndarray,
    turns: np.ndarray,
    avoidable: np.ndarray,
    max_steps: int = 20,
    *,
    weighting: str = "task_mean",
) -> dict[str, np.ndarray]:
    """Weight the three observed-trace recipes without fabricating judge data."""
    original = objective_reward_scores(success, turns, avoidable, max_steps)
    success = np.asarray(success, dtype=float)
    n = success.shape[-1]
    weights = current_past_weights(n, weighting)
    if weighting == "task_mean" or n == 1:
        return original
    turns = np.asarray(turns, dtype=float)
    avoidable = np.asarray(avoidable, dtype=float)
    mean_success = (success * weights).sum(axis=-1)
    count = n * mean_success
    efficiency = np.maximum(0, 1 - np.minimum(turns, max_steps) / max_steps)
    avoidable_cost = np.minimum(avoidable, max_steps) / max_steps
    secondary = .25 * primary_count_gap(n, weighting)
    return {
        "success_count": count,
        "success_efficiency": count + secondary * (success * efficiency * weights).sum(axis=-1)
        / np.maximum(1 / n, mean_success),
        "successful_recovery": count - secondary * (success * avoidable_cost * weights).sum(axis=-1),
    }


def weighted_reward_scores(
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
    *,
    weighting: str = "task_mean",
) -> dict[str, np.ndarray]:
    """Return the existing seven rules using task or current/past period weights.

    Candidate arrays are (trials, 5, n), parent arrays (trials, n), n in 1..10.
    Entry zero is current; all remaining entries are past. Scalar and
    lexicographic output shapes and uniform tie handling are unchanged.
    """
    original = reward_scores(
        success, turns, progress, material, avoidable,
        parent_success, parent_turns, parent_progress, parent_material, max_steps,
    )
    success = np.asarray(success, dtype=float)
    n = success.shape[-1]
    weights = current_past_weights(n, weighting)
    if weighting == "task_mean" or n == 1:
        return original

    turns = np.asarray(turns, dtype=float)
    progress = np.asarray(progress, dtype=float)
    material = np.asarray(material, dtype=float)
    avoidable = np.asarray(avoidable, dtype=float)
    parent_success = np.asarray(parent_success, dtype=float)[:, None, :]
    parent_turns = np.asarray(parent_turns, dtype=float)[:, None, :]
    parent_progress = np.asarray(parent_progress, dtype=float)[:, None, :]
    parent_material = np.asarray(parent_material, dtype=float)[:, None, :]

    def mean(value: np.ndarray) -> np.ndarray:
        return (value * weights).sum(axis=-1)

    objective = weighted_objective_reward_scores(success, turns, avoidable, max_steps, weighting=weighting)
    count = objective["success_count"]
    efficiency = np.maximum(0, 1 - np.minimum(turns, max_steps) / max_steps)
    regressions = (parent_success == 1) & (success == 0)
    repairs = (parent_success == 0) & (success == 1)
    joint_failure = (parent_success == 0) & (success == 0)
    certified_joint_success = (parent_success == 1) & (success == 1) & (parent_material == 0) & (material == 0)
    relative_efficiency = np.clip((parent_turns - turns) / np.maximum(1, parent_turns), -1, 1)
    secondary = .25 * primary_count_gap(n, weighting)

    scores = {
        "success_count": count,
        "success_efficiency": objective["success_efficiency"],
        "failure_progress": count + secondary * mean((1 - success) * progress),
        "certified_completion": n * mean(success * (1 - material))
        + secondary * mean((1 - material) * ((1 - success) * progress + success * efficiency)),
        "parent_safe": np.stack((
            -n * mean(regressions), n * mean(repairs),
            mean(joint_failure * (progress - parent_progress)),
            mean(certified_joint_success * relative_efficiency),
        ), axis=-1),
        "successful_recovery": objective["successful_recovery"],
        "quality_gated": np.stack((
            count, -n * mean(material), np.floor(10 * mean(progress) + .5),
            -n * mean(success * avoidable), -n * mean(success * turns),
        ), axis=-1),
    }
    assert tuple(scores) == REWARD_KEYS
    return scores
