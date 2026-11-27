# SPDX-License-Identifier: Apache-2.0
"""The independent expectation must match the chosen PCT-ranked reflection exactly."""

import pytest

from scripts.ace.compare_reflection_accumulation import candidate_metrics, selected_metrics


def rows(passed, flags):
    return [{"evaluation_repeat": i, "passed_tests": p, "total_tests": 3, "pct": p / 3, "success": flag}
            for i, (p, flag) in enumerate(zip(passed, flags, strict=True))]


def test_expectation_keeps_selected_reflection_instead_of_reranking_by_tgc():
    bank = {("with_gt", "task", 1): rows([3, 3, 0], [True, True, False]),
            ("with_gt", "task", 3): rows([2, 2, 2], [False, False, False])}
    choice = {"candidate": {"task_id": "task", "source_repeat": 3,
                            "score_numerator": 2, "score_denominator": 3}}
    result = selected_metrics(choice, "with_gt", "task", bank)
    assert result["pct"] == 2 / 3
    assert result["tgc"] == 0
    assert candidate_metrics(bank[("with_gt", "task", 1)])["tgc"] == 2 / 3


def test_missing_tgc_outcome_is_not_a_measured_failure():
    result = candidate_metrics(rows([3, 3, 0], [True, None, False]))
    assert result["pct"] == 2 / 3
    assert result["tgc"] == 0.5
    assert result["tgc_outcomes"] == 2


def test_incomplete_or_duplicate_repeats_and_changed_score_are_rejected():
    source = rows([2, 2, 2], [False, False, False])
    with pytest.raises(ValueError):
        candidate_metrics(source[:2])
    with pytest.raises(ValueError):
        candidate_metrics([source[0], source[0], source[2]])
    choice = {"candidate": {"task_id": "task", "source_repeat": 0,
                            "score_numerator": 1, "score_denominator": 1}}
    with pytest.raises(ValueError, match="frozen"):
        selected_metrics(choice, "without_gt", "task", {("without_gt", "task", 0): source})
