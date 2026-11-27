# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Guard sampled-task isolation, matched cohorts, and seed-local inference."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from scripts.premise.analyze_bfcl_trajectory_rewards import (
    CONTEXTS,
    PROCESS_FEATURES,
    _flatten,
    _retryable_infrastructure,
    analyze,
    statistical_tests,
)


def fixture_data() -> tuple[list[dict], dict, dict]:
    tasks = [f"task{index}" for index in range(50)]
    rewards = [[0] * 50 for _ in range(5)]
    groups, judgments = [], {}
    for index in range(10):
        traces = []
        for candidate, context in enumerate(CONTEXTS):
            features = dict.fromkeys(PROCESS_FEATURES, 0)
            features["environment_steps"] = 8 + candidate
            traces.append({"context": context, "official_success": 0,
                           "available": True, "features": features})
        group = {"id": f"group{index}", "model": "Qwen", "key": "s42_cp010", "seed": 42,
                 "checkpoint": 10, "task_id": tasks[index], "train_order_index": index,
                 "query_position": index, "public_task": {"user_turns": [{"turn_id": 0, "text": "Work"}]},
                 "traces": traces}
        groups.append(group)
        judgments[group["id"]] = {"ok": True, "result": {"ok": True, "judgments": {
            context: {"metrics": {"fulfillment": {"lower_bound": candidate / 5,
                                                  "shared_turn_count": 1, "unknown_turns": 0},
                                  "material_violation": {"value": 0}}}
            for candidate, context in enumerate(CONTEXTS)
        }}}
    target = {"train_order": tasks, "train_rewards": rewards,
              "scores": {"full_train": [.1, .2, .3, .4, .5], "full_validation": [.5, .4, .3, .2, .1]},
              "parent_scores": {"full_train": .2}}
    return groups, {("Qwen", "s42_cp010"): target}, judgments


def row_at(summary: dict, recipe: str, n: int, cohort: str = "primary") -> dict:
    return next(row for row in summary["checkpoint_scores"]
                if (row["recipe"], row["n"], row["cohort"]) == (recipe, n, cohort))


def test_unqueried_tasks_and_full_targets_never_change_selection_scores():
    groups, targets, judgments = fixture_data()
    before = analyze(groups, targets, judgments)
    changed_groups, changed_targets = deepcopy(groups), deepcopy(targets)
    # Task ten is outside n=3; even its sampled outcome cannot affect n=3.
    changed_groups[9]["traces"][1]["official_success"] = 1
    changed_targets[("Qwen", "s42_cp010")]["train_rewards"][0][9] = 1
    changed_targets[("Qwen", "s42_cp010")]["scores"] = {
        "full_train": [1, 0, 0, 0, 0], "full_validation": [0, 0, 0, 0, 1],
    }
    after = analyze(changed_groups, changed_targets, judgments)
    for recipe in ("success_count", "success_efficiency", "failure_progress", "certified_completion",
                   "parent_safe", "successful_recovery", "quality_gated"):
        for field in ("candidate_scores", "choice_probabilities"):
            assert row_at(before, recipe, 3)[field] == row_at(after, recipe, 3)[field]
    assert row_at(before, "failure_progress", 3)["validation_lift"] != row_at(
        after, "failure_progress", 3,
    )["validation_lift"]


def test_failed_judge_is_missing_not_zero_and_matched_prefix_cohorts_change_together():
    groups, targets, judgments = fixture_data()
    judgments["group3"] = {"ok": False, "status": "transport_error"}
    result = analyze(groups, targets, judgments)
    assert row_at(result, "failure_progress", 3)["choice_probabilities"] == [0, 0, 0, 0, 1]
    assert not any(row["cohort"] == "primary" and row["n"] >= 4 for row in result["checkpoint_scores"])
    assert row_at(result, "success_count", 10, "objective")["choice_probabilities"] == [.2] * 5
    coverage = next(row for row in result["checkpoint_coverage"] if row["n"] == 4)
    assert coverage["missing_or_invalid_judgment_groups"] == 1
    assert coverage["objective_complete"] and not coverage["primary_complete"]


def test_missing_trajectory_excludes_both_matched_cohorts_without_fabricating_process_zeros():
    groups, targets, judgments = fixture_data()
    groups[4]["traces"][2].update({"available": False, "features": None})
    result = analyze(groups, targets, judgments)
    assert all(row["n"] < 5 for row in result["checkpoint_scores"])
    assert len(result["checkpoint_coverage"]) == 10


def test_ties_use_expected_uniform_selection_and_lex_scores_have_no_pearson_encoding():
    groups, targets, judgments = fixture_data()
    result = analyze(groups, targets, judgments)
    baseline = row_at(result, "success_count", 10)
    assert baseline["validation_lift"] == pytest.approx(0)
    assert baseline["validation_score"] == pytest.approx(.3)
    assert baseline["rho_full_validation"] is None
    assert row_at(result, "parent_safe", 10)["rho_full_validation"] is None
    assert row_at(result, "quality_gated", 10)["spearman_full_validation"] == pytest.approx(-1)


def test_task_identity_join_rejects_wrong_training_order_or_observed_reward():
    groups, targets, judgments = fixture_data()
    groups[0]["task_id"] = "other"
    with pytest.raises(ValueError, match="identity"):
        analyze(groups, targets, judgments)
    groups, targets, judgments = fixture_data()
    groups[0]["traces"][1]["official_success"] = 1
    with pytest.raises(ValueError, match="reward"):
        analyze(groups, targets, judgments)


def test_statistical_blocks_are_seed_local_and_frozen_before_missing_checkpoint_exclusion():
    coverage, rows = [], []
    for seed in (42, 43):
        for checkpoint in range(4):
            key = f"s{seed}_cp{checkpoint:03d}"
            coverage.append({"model": "Qwen", "key": key, "seed": seed, "checkpoint": checkpoint, "n": 10})
            if checkpoint == 1:
                continue
            for recipe, lift in (("success_count", 0), ("success_efficiency", .02), ("successful_recovery", 0)):
                rows.append({"model": "Qwen", "cohort": "objective", "recipe": recipe,
                             "key": key, "seed": seed, "checkpoint": checkpoint, "n": 10,
                             "validation_lift": lift})
    tests = statistical_tests(rows, 42, coverage)
    assert len(tests) == 10
    assert all(row["family_size"] == 2 and row["cohort"] == "objective" for row in tests)
    selected = next(row for row in tests if row["recipe"] == "success_efficiency" and row["block_size"] == 2)
    assert selected["difference"] == pytest.approx(.02)
    assert selected["checkpoints"] == 6 and selected["seeds"] == [42, 43]
    assert selected["chronological_blocks"][0]["all_checkpoint_keys"] == ["s42_cp000", "s42_cp001"]
    assert selected["chronological_blocks"][0]["valid_checkpoint_keys"] == ["s42_cp000"]
    assert selected["chronological_blocks"][1]["all_checkpoint_keys"] == ["s42_cp002", "s42_cp003"]
    assert all(len({key.split("_")[0] for key in block["all_checkpoint_keys"]}) == 1
               for block in selected["chronological_blocks"])
    assert np.isfinite(selected["p_holm"])
    stream_test = next(row for row in tests if row["recipe"] == "success_efficiency"
                       and row["block_size"] == "whole_seed")
    assert stream_test["blocks"] == 2 and stream_test["permutations"] == 4
    assert stream_test["p_two_sided"] == .5
    assert stream_test["difference"] == pytest.approx(.02)


def test_csv_schema_union_preserves_missing_semantic_fields_as_empty_cells():
    rows = _flatten([{"cohort": "objective", "lift": {"mean": .2}},
                     {"cohort": "primary", "lift": {"mean": .4}, "progress": .7}])
    assert list(rows[0]) == list(rows[1])
    assert rows[0]["progress"] is None
    assert rows[1]["progress"] == .7


def test_retryable_queue_status_counts_actual_completions_not_refused_tokenization():
    record = {"ok": False, "status": "judge_transport_error",
              "attempts": [{"result": {"status": "judge_transport_error"}} for _ in range(3)]}
    assert _retryable_infrastructure(record)
    record["attempts"] = [{"result": {"chat_request_started": True}} for _ in range(3)]
    assert not _retryable_infrastructure(record)
    record["attempts"] = [{"result": {"prompt_tokens_counted": 12000}} for _ in range(2)]
    assert _retryable_infrastructure(record)
