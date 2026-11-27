# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import copy

import numpy as np
import pytest

from scripts.premise.bfcl_full_pool_selection import STRATEGIES, prepare_features, sample_prefixes


class NeverReadRewards:
    @property
    def shape(self):
        raise AssertionError("A metadata-only sampler inspected reward storage")

    def __array__(self, *args, **kwargs):
        raise AssertionError("A metadata-only sampler materialized task rewards")

    def __getitem__(self, index):
        raise AssertionError("A metadata-only sampler queried task rewards")


class QueriedRewards:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.queries = []

    def __array__(self, *args, **kwargs):
        raise AssertionError("Adaptive sampler materialized unqueried rewards")

    def __getitem__(self, index):
        candidates, tasks = index
        assert candidates == slice(None)
        self.queries.append(np.asarray(tasks).copy())
        return self.values[index]


def _pool(size=20):
    rng = np.random.default_rng(21)
    features = rng.random((size, 6))
    features /= np.linalg.norm(features, axis=-1, keepdims=True)
    return features @ features.T, np.arange(size) % 5, rng.integers(2, size=(5, size))


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_budget_has_unique_shared_tasks_and_reproduces_all_trials_when_stopped_early(strategy):
    similarity, clusters, rewards = _pool()
    source = rewards if strategy == "adaptive_disagreement" else NeverReadRewards()
    full = sample_prefixes(strategy, similarity, clusters, source, 23, 10, np.random.default_rng(7))
    assert full.shape == (23, 10)
    assert full.min() >= 0 and full.max() < len(similarity)
    assert all(len(set(row)) == 10 for row in full)
    for budget in (1, 3, 7):
        stopped = sample_prefixes(strategy, similarity, clusters, source, 23, budget, np.random.default_rng(7))
        assert np.array_equal(stopped, full[:, :budget])


@pytest.mark.parametrize("strategy", STRATEGIES[:-1])
def test_nonadaptive_strategies_observe_no_candidate_or_parent_rewards(strategy):
    similarity, clusters, _ = _pool()
    audit = {}
    sample_prefixes(strategy, similarity, clusters, NeverReadRewards(), 12, 8, np.random.default_rng(3), audit=audit)
    assert audit["selector_task_queries"] == audit["selector_candidate_reward_reads"] == 0
    assert audit["observations_before_choice"] == [0] * 8
    assert audit["query_indices"] == []


def test_representative_sampler_spends_its_second_query_covering_the_other_metadata_cluster():
    labels = np.array([0] * 6 + [1] * 4)
    similarity = (labels[:, None] == labels[None, :]).astype(float)
    indices = sample_prefixes("representative", similarity, labels, None, 40, 2, np.random.default_rng(9))
    assert (indices[:, 0] < 6).all()  # The first representative covers the larger group.
    assert (indices[:, 1] >= 6).all()  # The second covers the four still-unrepresented tasks.


def test_cluster_stratification_covers_each_cluster_before_revisiting_and_handles_exhaustion():
    labels = np.repeat(np.arange(5), 10)
    indices = sample_prefixes("cluster_stratified", np.eye(50), labels, None, 40, 10, np.random.default_rng(9))
    for row in labels[indices]:
        assert len(set(row[:5])) == 5
        assert np.bincount(row, minlength=5).tolist() == [2] * 5
    labels = np.array([0, 1, 1, 1, 2, 2, 2])
    full = sample_prefixes("cluster_stratified", np.eye(7), labels, None, 10, 7, np.random.default_rng(9))
    assert all(sorted(row) == list(range(7)) for row in full)


def test_adaptive_queries_only_the_chosen_task_after_each_choice_and_records_exact_cost():
    similarity, clusters, matrix = _pool()
    source, audit = QueriedRewards(matrix), {}
    indices = sample_prefixes("adaptive_disagreement", similarity, clusters, source, 17, 6,
                              np.random.default_rng(7), audit=audit)
    assert len(source.queries) == 6
    for step, queried_tasks in enumerate(source.queries):
        assert np.array_equal(queried_tasks, indices[:, step])
    assert audit["selector_task_queries"] == 17 * 6
    assert audit["selector_candidate_reward_reads"] == 17 * 6 * 5
    assert audit["queries_per_trial"] == 6
    assert audit["candidate_values_read_per_trial"] == 30
    assert audit["observations_before_choice"] == list(range(6))
    assert np.array_equal(audit["query_indices"], indices)


def test_changing_every_unqueried_outcome_cannot_change_the_adaptive_path():
    similarity, clusters, matrix = _pool()
    before = sample_prefixes("adaptive_disagreement", similarity, clusters, matrix, 1, 6, np.random.default_rng(11))
    hidden = sorted(set(range(len(similarity))) - set(before[0]))
    changed = matrix.copy()
    changed[:, hidden] = 1 - changed[:, hidden]
    after = sample_prefixes("adaptive_disagreement", similarity, clusters, changed, 1, 6, np.random.default_rng(11))
    assert np.array_equal(before, after)
    # Even invalid hidden values cannot be noticed without spending their query budget.
    hidden_invalid = matrix.astype(float)
    hidden_invalid[:, hidden] = np.nan
    untouched = sample_prefixes("adaptive_disagreement", similarity, clusters, hidden_invalid,
                                1, 6, np.random.default_rng(11))
    assert np.array_equal(before, untouched)


def test_outcome_at_step_n_cannot_influence_its_own_query_or_earlier_queries():
    similarity, clusters, matrix = _pool()
    original = sample_prefixes("adaptive_disagreement", similarity, clusters, matrix, 1, 10, np.random.default_rng(18))
    for step in (0, 2, 5):
        changed = matrix.copy()
        changed[:, original[0, step]] = 1 - changed[:, original[0, step]]
        updated = sample_prefixes("adaptive_disagreement", similarity, clusters, changed,
                                  1, 10, np.random.default_rng(18))
        assert np.array_equal(updated[:, :step + 1], original[:, :step + 1])


def test_adaptive_flat_committee_falls_back_to_metadata_representative_selection():
    similarity, clusters, _ = _pool()
    matrix = np.broadcast_to(np.arange(20) % 2, (5, 20)).copy()
    representative = sample_prefixes("representative", similarity, clusters, None, 30, 10, np.random.default_rng(6))
    adaptive = sample_prefixes("adaptive_disagreement", similarity, clusters, matrix, 30, 10, np.random.default_rng(6))
    assert np.array_equal(representative, adaptive)


def test_public_full_pool_features_are_deterministic_and_ignore_answers_or_unavailable_metadata():
    task_ids = [f"task{i}" for i in range(50)]
    metadata = {
        task: {"first_request": f"Schedule meeting document token{i}", "tool_classes": [f"Tools{i % 7}"],
               "answer": "Private solution", "initial_config": {"state": 42}}
        for i, task in enumerate(task_ids)
    }
    before_similarity, before_clusters = prepare_features(task_ids, metadata)
    changed = copy.deepcopy(metadata)
    for row in changed.values():
        row["answer"] = "A different private solution"
        row["initial_config"] = {"state": -1000}
        row["later_request"] = "This unavailable text must not affect feature fitting"
    after_similarity, after_clusters = prepare_features(task_ids, changed)
    assert np.array_equal(before_similarity, after_similarity)
    assert np.array_equal(before_clusters, after_clusters)
    assert len(set(before_clusters)) == 5
    assert before_similarity.shape == (50, 50)
    assert before_similarity == pytest.approx(before_similarity.T)
    assert np.diag(before_similarity) == pytest.approx(np.ones(50))


def test_public_feature_builder_handles_empty_lexical_and_tool_information_without_inventing_clusters():
    metadata = {str(i): {"first_request": "the and", "tool_classes": []} for i in range(6)}
    similarity, clusters = prepare_features(list(metadata), metadata)
    assert similarity.shape == (6, 6) and np.isfinite(similarity).all()
    assert len(set(clusters)) == 1


@pytest.mark.parametrize("budget", [0, 21])
def test_invalid_query_budgets_are_rejected(budget):
    similarity, clusters, _ = _pool()
    with pytest.raises(ValueError, match="max_n"):
        sample_prefixes("uniform", similarity, clusters, None, 1, budget, np.random.default_rng(1))


def test_adaptive_rejects_invalid_observed_reward_without_scanning_hidden_columns():
    similarity, clusters, matrix = _pool()
    first = sample_prefixes("adaptive_disagreement", similarity, clusters, matrix, 1, 1, np.random.default_rng(7))[0, 0]
    matrix = matrix.astype(float)
    matrix[:, first] = np.nan
    with pytest.raises(ValueError, match="Queried rewards"):
        sample_prefixes("adaptive_disagreement", similarity, clusters, matrix, 1, 1, np.random.default_rng(7))
