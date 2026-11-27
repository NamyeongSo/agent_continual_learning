# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Budgeted BFCL task selection from an entire training pool, without validation labels.

Only the adaptive sampler reads rewards, and it does so after choosing each task.
The returned ordering is nested: stopping at n observes exactly its first n tasks.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

STRATEGIES = ("uniform", "representative", "cluster_stratified", "adaptive_disagreement")


def prepare_features(
    task_ids: list[str], metadata: dict, random_state: int = 20261021, n_clusters: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Fit text/tool features on the whole allowed training pool, never on task outcomes."""
    if not task_ids or len(set(task_ids)) != len(task_ids) or n_clusters < 1:
        raise ValueError("A nonempty unique training task pool and positive cluster count are required")
    if any(task not in metadata for task in task_ids):
        raise ValueError("Public metadata is missing for a training task")
    requests = [str(metadata[task]["first_request"]) for task in task_ids]
    classes = [set(metadata[task]["tool_classes"]) for task in task_ids]
    try:
        vectors = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), strip_accents="unicode").fit_transform(
            requests
        ).toarray()
    except ValueError as error:
        if "empty vocabulary" not in str(error):
            raise
        vectors = np.zeros((len(task_ids), 0))
    lexical = vectors @ vectors.T
    tool_similarity = np.asarray([
        [len(a & b) / len(a | b) if a | b else 1 for b in classes] for a in classes
    ])
    similarity = (lexical + tool_similarity) / 2
    np.fill_diagonal(similarity, 1)
    vocabulary = sorted(set().union(*classes))
    tools = np.asarray([[tool in row for tool in vocabulary] for row in classes], dtype=float)
    lengths = np.linalg.norm(tools, axis=1, keepdims=True)
    tools = np.divide(tools, lengths, out=np.zeros_like(tools), where=lengths > 0)
    features = np.concatenate((vectors, tools), axis=1) / np.sqrt(2)
    if not features.shape[1]:
        features = np.zeros((len(task_ids), 1))
    # Degenerate synthetic pools can contain fewer than five distinct public descriptions.
    effective_clusters = min(n_clusters, len(task_ids), len(np.unique(features, axis=0)))
    with threadpool_limits(limits=1):
        labels = KMeans(n_clusters=effective_clusters, n_init=10, random_state=random_state).fit_predict(features)
    return np.clip(similarity, 0, 1), labels


def _random_matrix(generators: list[np.random.Generator], width: int) -> np.ndarray:
    return np.vstack([generator.random(width) for generator in generators])


def _pick(criterion: np.ndarray, selected: np.ndarray, generators: list[np.random.Generator]) -> np.ndarray:
    available = np.where(selected, -np.inf, criterion)
    tied = np.isclose(available, available.max(axis=1, keepdims=True), atol=1e-12, rtol=0)
    return np.argmax(np.where(tied, _random_matrix(generators, criterion.shape[1]), -np.inf), axis=1)


def _coverage_gain(similarity: np.ndarray, covered: np.ndarray) -> np.ndarray:
    return np.maximum(similarity[None, :, :] - covered[:, :, None], 0).mean(axis=1)


def _adaptive_criterion(
    similarity: np.ndarray, indices: np.ndarray, observations: np.ndarray, covered: np.ndarray
) -> np.ndarray:
    fallback = _coverage_gain(similarity, covered)
    if not indices.shape[1]:
        return fallback
    # Predictions use similarities to already queried tasks and only their observed five-way rewards.
    weights = similarity[:, indices].transpose(1, 0, 2) ** 2
    total = weights.sum(axis=-1, keepdims=True)
    predicted = np.einsum("tnq,tqc->tnc", weights, observations)
    prior = observations.mean(axis=1)[:, None, :]
    predicted = np.divide(predicted, total, out=np.broadcast_to(prior, predicted.shape).copy(), where=total > 0)
    criterion = predicted.var(axis=-1) * np.maximum(1 - covered, 0)
    observed_signal = np.ptp(observations, axis=-1).max(axis=-1) > 1e-12
    informative = observed_signal & (criterion.max(axis=-1) > 1e-12)
    return np.where(informative[:, None], criterion, fallback)


def sample_prefixes(
    strategy: str,
    similarity: np.ndarray,
    clusters: np.ndarray,
    reward_matrix,
    trials: int,
    max_n: int,
    rng: np.random.Generator,
    *,
    audit: dict | None = None,
) -> np.ndarray:
    """Choose shared five-candidate task prefixes; adaptive queries only chosen columns.

    Nonadaptive strategies never access ``reward_matrix``, so callers can pass None.
    Adaptive reward reads are exactly five values per chosen task, per trial. Selection
    at step n uses at most the n-1 previous observations. Stopping earlier produces
    identical prefixes for every trial because each trial receives its own RNG stream.
    """
    similarity = np.asarray(similarity, dtype=float)
    clusters = np.asarray(clusters)
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown full-pool sampler: {strategy}")
    if (similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]
            or not np.isfinite(similarity).all() or np.any(similarity < -1e-12) or np.any(similarity > 1 + 1e-12)):
        raise ValueError("Similarity must be a finite square matrix in [0,1]")
    similarity = np.clip(similarity, 0, 1)
    size = len(similarity)
    if trials < 1 or max_n < 1 or max_n > size or clusters.shape != (size,):
        raise ValueError("Expected positive trials, a pool-sized cluster vector, and 1 <= max_n <= pool size")
    if strategy == "adaptive_disagreement" and getattr(reward_matrix, "shape", None) != (5, size):
        raise ValueError("Adaptive selection requires a five-candidate by pool-task reward source")
    trial_seeds = rng.integers(0, np.iinfo(np.int64).max, size=trials)
    generators = [np.random.default_rng(int(seed)) for seed in trial_seeds]
    indices = np.empty((trials, max_n), dtype=int)
    selected = np.zeros((trials, size), dtype=bool)
    covered = np.zeros((trials, size))
    observations = np.empty((trials, max_n, 5))
    row_indices = np.arange(trials)
    unique_clusters, cluster_codes = np.unique(clusters, return_inverse=True)
    cluster_priority = _random_matrix(generators, len(unique_clusters)) if strategy == "cluster_stratified" else None
    task_priority = _random_matrix(generators, size) if strategy in ("uniform", "cluster_stratified") else None
    cluster_counts = np.zeros((trials, len(unique_clusters)), dtype=int)
    for step in range(max_n):
        if strategy == "uniform":
            chosen = np.argmax(np.where(selected, -np.inf, task_priority), axis=1)
        elif strategy == "cluster_stratified":
            available = np.column_stack([(~selected[:, cluster_codes == code]).any(1)
                                         for code in range(len(unique_clusters))])
            least_used = np.where(available, cluster_counts, size + 1).min(axis=1, keepdims=True)
            eligible = available & (cluster_counts == least_used)
            cluster = np.argmax(np.where(eligible, cluster_priority, -np.inf), axis=1)
            eligible_task = (cluster_codes[None, :] == cluster[:, None]) & ~selected
            chosen = np.argmax(np.where(eligible_task, task_priority, -np.inf), axis=1)
            cluster_counts[row_indices, cluster] += 1
        else:
            criterion = (_adaptive_criterion(similarity, indices[:, :step], observations[:, :step], covered)
                         if strategy == "adaptive_disagreement" else _coverage_gain(similarity, covered))
            chosen = _pick(criterion, selected, generators)
        indices[:, step] = chosen
        selected[row_indices, chosen] = True
        covered = np.maximum(covered, similarity[:, chosen].T)
        if strategy == "adaptive_disagreement":
            # Do not coerce the full source into an array or inspect unchosen task outcomes.
            queried = np.asarray(reward_matrix[:, chosen], dtype=float)
            if queried.shape != (5, trials) or not np.isin(queried, [0, 1]).all():
                raise ValueError("Queried rewards must be five finite binary values per selected task")
            observations[:, step] = queried.T
    if audit is not None:
        adaptive = strategy == "adaptive_disagreement"
        audit.update({"strategy": strategy, "trials": trials, "max_n": max_n, "pool_size": size,
                      "selector_task_queries": trials * max_n if adaptive else 0,
                      "selector_candidate_reward_reads": 5 * trials * max_n if adaptive else 0,
                      "queries_per_trial": max_n if adaptive else 0,
                      "candidate_values_read_per_trial": 5 * max_n if adaptive else 0,
                      "observations_before_choice": list(range(max_n)) if adaptive else [0] * max_n,
                      "query_indices": indices.tolist() if adaptive else [],
                      "stopped_prefix_replays_same_choices": True})
    return indices
