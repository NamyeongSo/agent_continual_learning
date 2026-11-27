# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Validate exact candidate subsets and prevent target-driven selection."""

from copy import deepcopy

import numpy as np
import pytest

from scripts.premise.analyze_bfcl_candidate_count_surface import (
    fixed_cohort_rows,
    replay_rows,
    subset_choice_probabilities,
    summarize,
)


def fixture_row(scores=None):
    scores = [0, 1, 2, 3, 4] if scores is None else scores
    probability = subset_choice_probabilities(np.asarray(scores), 5)
    validation = np.asarray([.1, .2, .3, .4, .5])
    return {"model": "Qwen", "cohort": "objective", "key": "s42_cp010", "seed": 42,
            "checkpoint": 10, "n": 3, "recipe": "success_count", "score_type": "scalar",
            "candidate_scores": scores, "choice_probabilities": probability.tolist(),
            "full_validation_scores": validation.tolist(), "full_train_scores": [.5] * 5,
            "proposal_mean_train_gain": .1, "validation_lift": float(probability @ validation - validation.mean())}


@pytest.mark.parametrize("candidate_count", range(1, 6))
def test_exact_strict_order_subset_selection_has_known_combinatorial_distribution(candidate_count):
    from math import comb

    actual = subset_choice_probabilities(np.arange(5), candidate_count)
    expected = [comb(rank, candidate_count - 1) / comb(5, candidate_count)
                if rank >= candidate_count - 1 else 0 for rank in range(5)]
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual.sum(), 1)


@pytest.mark.parametrize("candidate_count", range(1, 6))
def test_fully_tied_pool_is_uniform_for_every_subset_size(candidate_count):
    np.testing.assert_allclose(subset_choice_probabilities(np.ones(5), candidate_count), [.2] * 5)


def test_partial_ties_are_symmetric_and_lexicographic_order_has_no_scalar_encoding():
    scores = np.asarray([[0, 1000000], [1, -1000000], [1, -1000000], [1, 0], [2, -1000000]])
    probability = subset_choice_probabilities(scores, 2)
    assert probability[1] == probability[2]
    assert probability[0] == 0
    np.testing.assert_allclose(probability, [0, .15, .15, .3, .4])


def test_one_candidate_has_zero_lift_and_five_matches_archived_argmax():
    source = fixture_row()
    rows = replay_rows([source])
    assert len(rows) == 5
    assert rows[0]["validation_lift"] == 0
    np.testing.assert_allclose(rows[0]["choice_probabilities"], [.2] * 5)
    assert rows[-1]["validation_lift"] == pytest.approx(source["validation_lift"])
    assert rows[-1]["choice_probabilities"] == source["choice_probabilities"]


def test_validation_target_changes_evaluation_but_never_selection():
    source = fixture_row()
    before = replay_rows([source])
    changed = deepcopy(source)
    changed["full_validation_scores"].reverse()
    changed["validation_lift"] = -.2
    after = replay_rows([changed])
    for first, second in zip(before, after, strict=True):
        assert first["candidate_scores"] == second["candidate_scores"]
        assert first["choice_probabilities"] == second["choice_probabilities"]
    assert before[-1]["validation_lift"] != after[-1]["validation_lift"]


def test_summary_retains_empty_semantic_cells_and_equal_coverage_across_lambda():
    row = fixture_row()
    row["cohort"] = "primary"
    result = summarize([row])
    measured = [item for item in result["aggregate"] if item["model"] == "Qwen" and item["n"] == 3]
    assert [item["checkpoints"] for item in measured] == [1] * 5
    missing = next(item for item in result["aggregate"] if item["model"] == "Gemma")
    assert missing["checkpoints"] == 0
    assert missing["validation_lift"]["mean"] is None


def test_corrupted_original_selection_and_unmatched_recipe_cohorts_are_rejected():
    row = fixture_row()
    row["choice_probabilities"] = [.2] * 5
    with pytest.raises(ValueError, match="archived"):
        replay_rows([row])
    first = fixture_row()
    second = deepcopy(first)
    second["recipe"] = "success_efficiency"
    third = deepcopy(first)
    third["key"] = "s43_cp010"
    third["seed"] = 43
    with pytest.raises(ValueError, match="matched"):
        summarize([first, second, third])


def test_fixed_cohort_preserves_only_checkpoint_keys_available_at_every_n():
    complete, partial = [], []
    for n in range(1, 11):
        row = fixture_row()
        row["n"] = n
        complete.append(row)
        if n < 10:
            other = deepcopy(row)
            other["key"] = "s43_cp010"
            other["seed"] = 43
            partial.append(other)
    kept, counts = fixed_cohort_rows(complete + partial)
    assert len(kept) == 10
    assert {row["key"] for row in kept} == {"s42_cp010"}
    assert next(row for row in counts if row["model"] == "Qwen" and row["cohort"] == "objective")[
        "checkpoints"
    ] == 1
    assert not any(row["model"] == "Gemma" and row["checkpoints"] > 0 for row in counts)


@pytest.mark.parametrize("count", [0, 6, True, 1.5])
def test_invalid_candidate_count_is_rejected(count):
    with pytest.raises(ValueError, match="candidate_count"):
        subset_choice_probabilities(np.arange(5), count)
