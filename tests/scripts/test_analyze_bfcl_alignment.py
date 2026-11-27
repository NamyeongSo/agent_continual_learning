# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import json

import pytest

from scripts.premise import analyze_bfcl_alignment as analysis


def point(key, train10, val10):
    return {"key": key, "seed": 42, "checkpoint": 10,
            "parent_scores": {"full_train": 0.25, "train_remaining40": 0.25,
                              "full_validation": None, "validation_remaining40": None},
            "scores": {"train10": train10, "val10": val10,
                       "full_train": [0.1, 0.2, 0.3, 0.4, 0.5],
                       "full_validation": [0.5, 0.4, 0.3, 0.2, 0.1],
                       "train_remaining40": [0.1, 0.2, 0.3, 0.4, 0.5],
                       "validation_remaining40": [0.5, 0.4, 0.3, 0.2, 0.1]}}


def test_scores_can_align_with_opposite_target_distributions():
    result = analysis.calculate([point("one", [0, 0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1, 0])])
    train, val = result["aggregate"]["full_train"], result["aggregate"]["full_validation"]
    assert train["train10"]["mean_pearson"] == pytest.approx(1)
    assert train["val10"]["mean_pearson"] == pytest.approx(-1)
    assert val["train10"]["mean_pearson"] == pytest.approx(-1)
    assert val["val10"]["mean_pearson"] == pytest.approx(1)


def test_both_bars_use_identical_checkpoints_when_one_score_has_no_variation():
    result = analysis.calculate([
        point("both_defined", [0, 0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1, 0]),
        point("constant_val10", [0.4, 0.3, 0.2, 0.1, 0], [0.2] * 5),
    ])
    target = result["aggregate"]["full_train"]
    assert target["paired_checkpoint_keys"] == ["both_defined"]
    assert target["train10"]["mean_pearson"] == pytest.approx(1)
    assert target["train10"]["individually_valid_mean_pearson"] == pytest.approx(0)
    assert target["train10"]["individually_valid_checkpoints"] == 2


def test_subtracting_a_parent_baseline_does_not_change_correlation():
    score = [0, 0.1, 0.4, 0.3, 0.5]
    performance = [0.2, 0.4, 0.3, 0.7, 0.6]
    assert analysis.correlation(score, performance) == pytest.approx(
        analysis.correlation(score, [v - 0.45 for v in performance]))


def test_error_bounds_detect_a_corrected_constant_score_without_floating_point_variation():
    p = point("one", [0.7, 0.8, 0.8, 0.8, 0.8], [0.4, 0.3, 0.2, 0.1, 0])
    errors = [{"checkpoint_key": "one", "candidate_index": 0, "partition": "train",
               "score": 0, "in_score_sample": True}]
    result = analysis.error_sensitivity([p], errors)
    assert result["full_train"]["train10"]["paired_checkpoint_counts"] == [0, 1]
    assert p["scores"]["train10"][0] == 0.7


def test_proposal_and_selection_gains_sum_to_parent_relative_selected_gain():
    p = point("one", [0, 0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1, 0])
    train = analysis.calculate([p])["aggregate"]["full_train"]
    assert train["proposal_mean_gain"] == pytest.approx(0.05)
    for selector in analysis.SELECTORS:
        assert train[selector]["mean_selected_gain"] == pytest.approx(
            train["proposal_mean_gain"] + train[selector]["mean_selection_gain"])
    val = analysis.calculate([p])["aggregate"]["full_validation"]
    assert val["proposal_mean_gain"] is None
    assert all(val[s]["mean_selected_gain"] is None for s in analysis.SELECTORS)
    assert val["proposal_mean_performance"] == pytest.approx(0.3)


def test_constant_score_checkpoints_remain_in_the_proposal_gain_average():
    first = point("defined", [0, 0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1, 0])
    second = point("constant", [0.2] * 5, [0.2] * 5)
    second["parent_scores"]["full_train"] = 0.35
    train = analysis.calculate([first, second])["aggregate"]["full_train"]
    assert train["paired_checkpoints"] == 1 and train["gain_checkpoint_count"] == 2
    assert train["proposal_mean_gain"] == pytest.approx(0)


def test_parent_execution_error_changes_gain_bounds_without_changing_correlations():
    p = point("one", [0, 0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1, 0])
    errors = [{"checkpoint_key": "one", "candidate_index": None, "partition": "train",
               "score": 0, "in_score_sample": False}]
    bounds = analysis.error_sensitivity([p], errors)["full_train"]
    assert bounds["proposal_mean_gain_bounds"] == pytest.approx([0.03, 0.05])
    assert bounds["train10"]["mean_pearson_bounds"] == pytest.approx([1, 1])


def partial_results(root):
    checkpoint = {"key": "s42_cp010", "seed": 42, "checkpoint": 10,
                  "train_task_ids": [f"train-{i}" for i in range(50)],
                  "validation_full_task_ids": [f"val-{i}" for i in range(50)],
                  "past_task_ids": [f"train-{i}" for i in range(10)],
                  "validation_task_ids": [f"val-{i}" for i in range(10)]}
    (root / "launch_manifest.json").write_text(json.dumps({"max_steps": 20, "checkpoints": [checkpoint]}))
    for context in ("parent", *(f"candidate_{i:02d}" for i in range(1, 6))):
        for partition in (("train",) if context == "parent" else ("train", "val")):
            for index in (0, 10):
                row = {"checkpoint_key": checkpoint["key"], "context": context, "partition": partition,
                       "task_id": f"{partition}-{index}", "score": float(index == 0), "status": "completed",
                       "steps": 5, "context_sha256": context}
                path = root / "results" / checkpoint["key"] / context / partition / f"{index:03d}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(row))


def test_partial_means_use_observed_tasks_instead_of_treating_missing_rewards_as_zero(tmp_path):
    partial_results(tmp_path)
    points, errors = analysis.load_points([tmp_path], allow_partial=True)
    assert len(points) == 1 and errors == []
    p = points[0]
    assert p["scores"]["train10"] == [1.0] * 5
    assert p["scores"]["full_train"] == [0.5] * 5
    assert p["parent_scores"]["full_train"] == 0.5
    assert p["coverage"]["full_train"] == [2] * 5
    assert p["coverage"]["train10"] == [1] * 5
    assert not p["evaluation_complete"]
    assert analysis.calculate(points)["aggregate"]["full_train"]["proposal_mean_gain"] == 0


def test_default_loader_requires_all_fifty_tasks(tmp_path):
    partial_results(tmp_path)
    with pytest.raises(ValueError, match="Incomplete parent baseline"):
        analysis.load_points([tmp_path])


def test_partial_loader_omits_whole_checkpoint_if_one_candidate_has_no_score_observations(tmp_path):
    partial_results(tmp_path)
    (tmp_path / "results/s42_cp010/candidate_05/train/000.json").unlink()
    points, errors = analysis.load_points([tmp_path], allow_partial=True)
    assert points == [] and errors == []


def test_partial_loader_rejects_duplicate_rewards_for_the_same_candidate_task(tmp_path):
    partial_results(tmp_path)
    row = tmp_path / "results/s42_cp010/candidate_01/train/000.json"
    row.with_name("duplicate.json").write_bytes(row.read_bytes())
    with pytest.raises(ValueError, match="Duplicate task results"):
        analysis.load_points([tmp_path], allow_partial=True)
