# SPDX-License-Identifier: Apache-2.0
"""Guard exact score aggregation, paired units, and complete-bank analysis."""

import pytest

from scripts.ace import analyze_context_proposal_bottleneck as analysis


def bank():
    initial = [{"task_id": "task", "source_repeat": r, "passed_tests": 1, "total_tests": 2,
                "pct": 0.5, "success": False, "appworld_policy_sha256": "policy"} for r in range(5)]
    evaluated = [{"task_id": "task", "arm": arm, "source_repeat": r, "evaluation_repeat": e,
                  "passed_tests": 1 if arm == "without_gt" else 2, "total_tests": 2,
                  "pct": 0.5 if arm == "without_gt" else 1.0, "success": arm == "with_gt",
                  "appworld_policy_sha256": "policy"}
                 for arm in analysis.ARMS for r in range(5) for e in range(3)]
    return initial, evaluated


def test_task_and_context_means_preserve_matched_units():
    initial, evaluated = bank()
    tasks, candidates = analysis.build_tables(["task"], initial, evaluated, "policy")
    assert len(tasks) == 1 and len(candidates) == 10
    assert tasks[0]["gt_difference_pct"] == 0.5
    assert tasks[0]["without_gt_direction"] == "unchanged"
    assert tasks[0]["with_gt_direction"] == "improved"
    assert tasks[0]["with_gt_best_tie_count"] == 5
    assert all(r["score_denominator"] in {1, 2} for r in candidates)


@pytest.mark.parametrize("change", ["missing", "duplicate", "policy", "unapproved_task"])
def test_incomplete_or_mismatched_bank_is_rejected(change):
    initial, evaluated = bank()
    if change == "missing":
        evaluated.pop()
    elif change == "duplicate":
        evaluated.append(dict(evaluated[0]))
    elif change == "policy":
        evaluated[0]["appworld_policy_sha256"] = "other"
    else:
        evaluated[0]["task_id"] = "val_task"
    with pytest.raises(ValueError):
        analysis.build_tables(["task"], initial, evaluated, "policy")


def test_non_turn_budget_failure_is_included_as_a_measured_zero():
    initial, evaluated = bank()
    evaluated[0] = {**evaluated[0], "budget_failure": {"reason": "input_tokens"}, "pct": 0,
                    "success": False}
    evaluated[0].pop("passed_tests")
    evaluated[0].pop("total_tests")
    tasks, candidates = analysis.build_tables(["task"], initial, evaluated, "policy")
    selected = next(r for r in candidates if r["arm"] == "without_gt" and r["source_repeat"] == 0)
    assert selected["mean_pct"] == 1 / 3
    assert selected["other_budget_failures"] == 1
    assert tasks[0]["without_gt_direction"] == "degraded"


def test_bootstrap_of_a_constant_paired_effect_keeps_that_effect():
    assert analysis.bootstrap_interval([0.1] * 50, samples=100) == pytest.approx([0.1, 0.1])
