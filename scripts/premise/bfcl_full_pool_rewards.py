# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Selection rewards using only shared observed BFCL tasks and public metadata.

The caller supplies metadata-derived weights, never unobserved task outcomes.
All five candidates use exactly the same observed tasks. Parent-aware scores
add one parent evaluation per observed task unless those outcomes are cached.
"""

from __future__ import annotations

import numpy as np

REWARD_KEYS = (
    "mean", "kernel_population_mean", "cluster_population_mean", "cluster_macro_mean",
    "repair_aversion", "preserve_barrier", "paired_risk", "repair_retention_harmonic",
    "cluster_geometric", "cluster_soft_worst", "committee_rarity", "local_consistency",
)


def reward_metadata() -> list[dict]:
    """Document what each rule optimizes and when its apparent nonlinearity vanishes."""
    rows = [
        ("mean", "Task mean", "Each observed task has equal value", "mean(r)", False,
         "linear", "Reference rule"),
        ("kernel_population_mean", "Kernel population mean", "Represent all 50 public task descriptions",
         "sum(a_j*r_j), sum(a)=1", False, "metadata-conditioned linear",
         "The mean kernel prediction over the pool is exactly a weighted observed-task mean"),
        ("cluster_population_mean", "Cluster population mean", "Weight public capabilities by training prevalence",
         "sum(pi_g*q_g)", False, "metadata-conditioned affine",
         "The half-count prior is shared across candidates; this is linear in observed rewards plus a constant"),
        ("cluster_macro_mean", "Cluster macro mean", "Give each public capability the same value",
         "mean(q_g)", False, "metadata-conditioned affine",
         "Equal capability weights change the objective relative to empirical task prevalence"),
        ("repair_aversion", "Repair / regression", "A lost parent success costs twice a rescued failure",
         "mean(max(d,0)-2*max(-d,0)), d=r-p", True, "parent-conditioned linear for binary rewards",
         "Exactly mean((1+p)*r)-2*mean(p); not a new nonlinear ordering"),
        ("preserve_barrier", "Preservation barrier", "Prefer candidates breaking no observed parent success",
         "mean(r)-2*I(any(d<0))", True, "constraint barrier",
         "If all candidates violate the barrier, its constant penalty leaves the mean ordering unchanged"),
        ("paired_risk", "Paired-risk penalty", "Discount unstable parent-relative changes",
         "mean(d)-std(d,ddof=1)/sqrt(n); penalty=0 when n=1", True, "mean-minus-risk",
         "This is a risk penalty, not a certified confidence bound. With constant parent labels, "
         "zero and one observed successes tie for n>=2"),
        ("repair_retention_harmonic", "Repair-retention harmonic", "Balance repairing failures and retaining ability",
         "2*u*v/(u+v); u=(repairs+.5)/(failures+1); v=(retained+.5)/(successes+1)", True,
         "harmonic utility", "An unobserved parent stratum has utility .5. New orderings require both strata"),
        ("cluster_geometric", "Geometric capability utility", "Reward joint competence across public capabilities",
         "exp(mean(log(q_g)))", False, "geometric utility",
         "An unobserved cluster has q=.5; with only one observed cluster this is mean-rank equivalent"),
        ("cluster_soft_worst", "Soft worst capability", "Improve the weakest public capabilities smoothly",
         "-.25*log(mean(exp(-q_g/.25)))", False, "smooth worst-group utility",
         "An unobserved cluster has q=.5; with only one observed cluster this is mean-rank equivalent"),
        ("committee_rarity", "Committee-rarity reward", "Value success on tasks few proposals solve",
         "mean(r_j*(-log((k_j+.5)/6))), k_j=sum_i(r_ij)", False,
         "committee-conditioned linear", "Weights depend on the same five observed proposals. "
         "Rare successes can reflect transfer or stochastic fragility"),
        ("local_consistency", "Local consistency", "Prefer performance locally consistent in public metadata",
         "mean(r)-.25*sqrt(mean((r-LOO_kernel_prediction)**2)); penalty=0 when n=1", False,
         "metadata consistency penalty", "Exclude the diagonal; zero-neighbor rows use other-task means. "
         "All rules preserve mean selection at n=1"),
    ]
    return [{"reward": key, "label": label, "meaning": meaning, "formula": formula,
             "requires_parent": parent, "selection_form": form, "degeneracy": degeneracy,
             "role": "control" if index < 4 else "semantic reward"}
            for index, (key, label, meaning, formula, parent, form, degeneracy) in enumerate(rows)]


def _validated_inputs(candidate, parent, cluster_ids, observed_similarity, population_kernel_weights,
                      cluster_proportions) -> tuple:
    candidate = np.asarray(candidate, dtype=float)
    if candidate.ndim != 3 or candidate.shape[0] < 1 or candidate.shape[1] != 5 or candidate.shape[2] < 1:
        raise ValueError("Expected a nonempty draw-by-five-candidate-by-observed-task reward array")
    trials, _, count = candidate.shape
    parent = np.asarray(parent, dtype=float)
    clusters = np.asarray(cluster_ids)
    similarity = np.asarray(observed_similarity, dtype=float)
    weights = np.asarray(population_kernel_weights, dtype=float)
    proportions = np.asarray(cluster_proportions, dtype=float)
    if parent.shape != (trials, count) or not np.isin(candidate, (0, 1)).all() or not np.isin(parent, (0, 1)).all():
        raise ValueError("Candidate and same-task parent rewards must be finite binary outcomes with matching draws")
    if (clusters.shape != (trials, count) or not np.issubdtype(clusters.dtype, np.integer)
            or not ((clusters >= 0) & (clusters < 5)).all()):
        raise ValueError("Expected one integer public cluster ID in [0,4] per observed task")
    if (similarity.shape != (trials, count, count) or not np.isfinite(similarity).all()
            or (similarity < 0).any() or not np.allclose(similarity, similarity.swapaxes(-1, -2))):
        raise ValueError("Observed-task similarities must be finite, nonnegative, symmetric matrices")
    if (weights.shape != (trials, count) or not np.isfinite(weights).all() or (weights < 0).any()
            or not np.allclose(weights.sum(axis=-1), 1)):
        raise ValueError("Metadata-only population weights must be nonnegative and sum to one for each draw")
    if (proportions.shape != (5,) or not np.isfinite(proportions).all() or (proportions <= 0).any()
            or not np.isclose(proportions.sum(), 1)):
        raise ValueError("Expected five positive population cluster proportions summing to one")
    return candidate, parent, clusters, similarity, weights, proportions


def reward_scores(candidate: np.ndarray, parent: np.ndarray, cluster_ids: np.ndarray,
                  observed_similarity: np.ndarray, population_kernel_weights: np.ndarray,
                  cluster_proportions: np.ndarray) -> dict[str, np.ndarray]:
    """Return twelve draw-by-candidate scores without accessing any unobserved reward.

    Args:
        candidate: Binary observed outcomes with shape [draws, 5, n].
        parent: Binary parent outcomes on the same tasks, shape [draws, n].
        cluster_ids: Public metadata cluster IDs, shape [draws, n], values 0..4.
        observed_similarity: Public metadata similarities, shape [draws, n, n].
        population_kernel_weights: Metadata-only normalized weights, shape [draws, n].
        cluster_proportions: Public full-pool proportions of five clusters, shape [5].
    """
    candidate, parent, clusters, similarity, weights, proportions = _validated_inputs(
        candidate, parent, cluster_ids, observed_similarity, population_kernel_weights, cluster_proportions,
    )
    trials, _, count = candidate.shape
    mean = candidate.mean(axis=-1)
    delta = candidate - parent[:, None, :]
    parent_successes = parent.sum(axis=-1)[:, None]
    parent_failures = count - parent_successes
    repaired = np.maximum(delta, 0).sum(axis=-1)
    retained = (candidate * parent[:, None, :]).sum(axis=-1)
    repair_utility = (repaired + .5) / (parent_failures + 1)
    retention_utility = (retained + .5) / (parent_successes + 1)
    cluster_posterior = np.empty((trials, 5, 5), dtype=float)
    for group in range(5):
        mask = clusters == group
        successes = (candidate * mask[:, None, :]).sum(axis=-1)
        cluster_posterior[:, :, group] = (successes + .5) / (mask.sum(axis=-1)[:, None] + 1)
    penalty = np.zeros((trials, 5), dtype=float)
    consistency_penalty = np.zeros_like(penalty)
    if count > 1:
        penalty = delta.std(axis=-1, ddof=1) / np.sqrt(count)
        kernel = similarity.copy()
        diagonal = np.arange(count)
        kernel[:, diagonal, diagonal] = 0
        sums = kernel.sum(axis=-1, keepdims=True)
        fallback = np.broadcast_to((1 - np.eye(count)) / (count - 1), kernel.shape)
        normalized = np.array(fallback, copy=True)
        np.divide(kernel, sums, out=normalized, where=sums > 0)
        prediction = np.einsum("tjk,tik->tij", normalized, candidate)
        consistency_penalty = np.sqrt(np.mean((candidate - prediction) ** 2, axis=-1))
    rarity = -np.log((candidate.sum(axis=1) + .5) / 6)
    result = {
        "mean": mean,
        "kernel_population_mean": np.sum(candidate * weights[:, None, :], axis=-1),
        "cluster_population_mean": np.sum(cluster_posterior * proportions, axis=-1),
        "cluster_macro_mean": cluster_posterior.mean(axis=-1),
        "repair_aversion": (np.maximum(delta, 0) - 2 * np.maximum(-delta, 0)).mean(axis=-1),
        "preserve_barrier": mean - 2 * (delta < 0).any(axis=-1),
        "paired_risk": delta.mean(axis=-1) - penalty,
        "repair_retention_harmonic": 2 * repair_utility * retention_utility
        / (repair_utility + retention_utility),
        "cluster_geometric": np.exp(np.log(cluster_posterior).mean(axis=-1)),
        "cluster_soft_worst": -.25 * np.log(np.exp(-cluster_posterior / .25).mean(axis=-1)),
        "committee_rarity": np.mean(candidate * rarity[:, None, :], axis=-1),
        "local_consistency": mean - .25 * consistency_penalty,
    }
    if tuple(result) != REWARD_KEYS or any(value.shape != (trials, 5) or not np.isfinite(value).all()
                                          for value in result.values()):
        raise ValueError("Reward outputs must be finite draw-by-five matrices in the recorded rule order")
    return result
