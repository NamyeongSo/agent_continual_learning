# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Identity and missing-data checks for offline BFCL parent temporal metrics."""

import json

import pytest

from scripts.premise.cache_bfcl_parent_temporal_metrics import positive_integer_steps, read_context


@pytest.mark.parametrize("value", [None, False, True, 0, -1, 1.5, float("nan"), float("inf"), "3"])
def test_invalid_steps_are_missing_not_zero(value):
    assert positive_integer_steps(value) is None


@pytest.mark.parametrize("value", [1, 3.0, 20])
def test_positive_integral_steps_are_preserved(value):
    assert positive_integer_steps(value) == value


def write_result(tmp_path, file_index, task_id, score, steps, **overrides):
    folder = tmp_path / "results" / "s42_cp001" / "parent" / "train"
    folder.mkdir(parents=True, exist_ok=True)
    row = {"checkpoint_key": "s42_cp001", "context": "parent", "partition": "train", "task_id": task_id,
           "context_sha256": "frozen", "score": score, "steps": steps, "status": "completed", "max_steps": 20}
    row.update(overrides)
    path = folder / f"{file_index:03d}.json"
    path.write_text(json.dumps(row))
    return path


def read(tmp_path, rewards=(1, 0), errors=(False, False)):
    return read_context(tmp_path, "s42_cp001", "parent", ["task_b", "task_a"],
                        list(rewards), list(errors), "frozen")


def test_task_identity_controls_order_and_failed_missing_turns_are_allowed(tmp_path):
    write_result(tmp_path, 0, "task_a", 0, None)
    write_result(tmp_path, 1, "task_b", 1, 8)
    values = read(tmp_path)
    assert values["success"] == [1, 0]
    assert values["turns"] == [8, None]
    assert not values["missing_success_turns"]


def test_successful_missing_turns_are_explicitly_reported(tmp_path):
    write_result(tmp_path, 0, "task_a", 0, 0)
    write_result(tmp_path, 1, "task_b", 1, None)
    values = read(tmp_path)
    assert values["turns"] == [None, None]
    assert [row["task_id"] for row in values["missing_success_turns"]] == ["task_b"]


@pytest.mark.parametrize("field,value", [("checkpoint_key", "wrong"), ("context", "candidate_01"),
                                         ("partition", "validation"), ("context_sha256", "changed")])
def test_mismatched_frozen_identity_is_rejected(tmp_path, field, value):
    write_result(tmp_path, 0, "task_a", 0, 0, **{field: value})
    write_result(tmp_path, 1, "task_b", 1, 8)
    with pytest.raises(ValueError, match="mismatch"):
        read(tmp_path)


def test_duplicate_and_missing_task_results_are_rejected(tmp_path):
    write_result(tmp_path, 0, "task_b", 1, 8)
    write_result(tmp_path, 1, "task_b", 1, 8)
    with pytest.raises(ValueError, match="duplicate"):
        read(tmp_path)
    (tmp_path / "results/s42_cp001/parent/train/001.json").unlink()
    with pytest.raises(ValueError, match="Missing archived"):
        read(tmp_path)


def test_changed_rewards_or_error_flags_are_rejected(tmp_path):
    write_result(tmp_path, 0, "task_a", 0, None, status="error")
    write_result(tmp_path, 1, "task_b", 1, 8)
    with pytest.raises(ValueError, match="disagree"):
        read(tmp_path)
    values = read(tmp_path, errors=(False, True))
    assert values["infrastructure_flags"] == [False, True]
    with pytest.raises(ValueError, match="disagree"):
        read(tmp_path, rewards=(0, 0), errors=(False, True))
