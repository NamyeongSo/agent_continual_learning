# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Retention must expose regressions concealed by repairs in net gain."""

import numpy as np
import pytest

from scripts.premise.analyze_bfcl_retention_generalization import evaluate_choice, outcome_vectors


def targets():
    return {"full_validation": [.4, .5, .6, .7, .8], "validation_remaining40": [.3, .4, .5, .6, .7]}


def test_repairs_cannot_hide_lost_parent_successes_in_retention():
    success = np.zeros((6, 50))
    success[0, :2] = 1
    success[1, 1:4] = 1  # Lose one old success, repair two: net positive.
    vectors = outcome_vectors(success, 4, targets())
    assert vectors["gain_past"][0] == pytest.approx(.25)
    assert vectors["retention_past"][0] == pytest.approx(.5)
    assert vectors["repair_past"][0] == pytest.approx(.5)
    assert vectors["regression_past"][0] == pytest.approx(.25)


def test_current_is_not_silently_counted_as_previously_trained():
    success = np.zeros((6, 50))
    success[0, [0, 4]] = 1
    success[1, 0] = 1  # Retains all trained successes but fails current task.
    vectors = outcome_vectors(success, 4, targets())
    assert vectors["retention_past"][0] == 1
    assert vectors["retention_past_current"][0] == .5
    assert vectors["gain_past"][0] == 0
    assert vectors["gain_past_current"][0] == pytest.approx(-.2)


def test_zero_parent_successes_are_undefined_not_perfect_or_zero_retention():
    success = np.zeros((6, 50))
    vectors = outcome_vectors(success, 4, targets())
    assert np.isnan(vectors["retention_past"]).all()
    assert evaluate_choice(np.full(5, .2), vectors)["retention_past"] is None
    assert np.array_equal(vectors["regression_past"], np.zeros(5))


def test_uniform_proposal_has_zero_validation_lift_without_parent_score():
    vectors = outcome_vectors(np.ones((6, 50)), 4, targets())
    result = evaluate_choice(np.full(5, .2), vectors)
    assert result["lift_validation50"] == pytest.approx(0)
    assert result["lift_validation40"] == pytest.approx(0)
    assert result["score_validation50"] == pytest.approx(.6)
    assert result["score_validation40"] == pytest.approx(.5)


def test_validation_scores_do_not_change_frozen_retention_choices():
    success = np.ones((6, 50))
    success[1, 0] = 0
    probability = np.array([.5, .5, 0, 0, 0])
    before = evaluate_choice(probability, outcome_vectors(success, 4, targets()))
    after = evaluate_choice(probability, outcome_vectors(success, 4,
                                                        {key: [1, 0, 0, 0, 0] for key in targets()}))
    assert before["retention_past"] == after["retention_past"]
    assert before["gain_past"] == after["gain_past"]


@pytest.mark.parametrize("probabilities", [[1, 0, 0, 0], [0, 0, 0, 0, 0], [2, -1, 0, 0, 0]])
def test_invalid_choice_distribution_rejected(probabilities):
    with pytest.raises(ValueError, match="choice distribution"):
        evaluate_choice(probabilities, {"retention": np.ones(5)})


def test_gain_decomposition_holds_for_every_candidate_and_temporal_target():
    success = np.random.default_rng(7).integers(0, 2, (6, 50))
    vectors = outcome_vectors(success, 17, targets())
    for period in ("past", "past_current"):
        assert np.allclose(vectors[f"gain_{period}"],
                           vectors[f"repair_{period}"] - vectors[f"regression_{period}"])
