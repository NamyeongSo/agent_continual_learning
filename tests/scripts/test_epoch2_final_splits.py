# SPDX-License-Identifier: Apache-2.0
"""Final evaluation must use held-fixed splits and never rerun final tests."""

import copy
from pathlib import Path

import pytest

from scripts.utils.checkpoint_queue import Queue
from scripts.utils.evaluate_epoch2_final_splits import eligible_tasks, final_snapshot, write


def job_fixture(tmp_path, method):
    train = tmp_path / "run/train"
    spec = {
        "model": "Qwen/Qwen3.6-35B-A3B",
        "method": method,
        "benchmark": "bfcl",
        "seed": 42,
        "temperature": 0.7,
        "benchmark_kwargs": {"subset": "multi_turn_base"},
        "splits": {"train": ["a", "b"], "val": ["v"], "test": ["t"]},
    }
    job = {
        "method": method,
        "benchmark": "bfcl",
        "train_dir": str(train),
        "output": str(tmp_path / "run"),
        "variant": method,
        "source_spec": spec,
        "source_snapshot": str(tmp_path / "frozen"),
        "train_count": 10,
        "epoch1_order": ["a", "b"],
        "epoch2_order": ["b", "a"],
    }
    return job, train


@pytest.mark.parametrize("method", ["ace", "premise", "gepa", "skillopt"])
def test_final_policy_is_not_registered_before_training_finishes(tmp_path, method):
    job, train = job_fixture(tmp_path, method)
    if method == "premise":
        write(train / "state.json", {"cursor": 5})
    assert final_snapshot(job, tmp_path) is None


@pytest.mark.parametrize("method", ["ace", "premise", "gepa", "skillopt"])
def test_final_snapshot_has_unique_train_membership_and_frozen_context(tmp_path, method):
    from scripts.utils.evaluate_epoch2_final_splits import digest

    job, train = job_fixture(tmp_path, method)
    if method == "ace":
        write(train / "training_complete.json", {"completed_task_count": 10})
        write(train / "playbook_ace_isolated_bfcl.json", {"playbook": "final context"})
    elif method == "premise":
        write(train / "state.json", {"cursor": 10, "states": {"bfcl": {"beam": [{"playbook": "final context"}]}}})
    else:
        write(train / "training_result.json", {})
        for index in range(10):
            write(train / f"online/{index}.json", {})
        write(train / "frozen_instruction.json", {"content": "final context", "sha256": digest("final context")})
    original = copy.deepcopy(job)
    spec, frozen = final_snapshot(job, tmp_path)
    assert frozen["content"] == "final context" and frozen["point"] == 10
    assert spec["splits"]["train"] == ["a", "b"]  # Not the repeated epoch stream.
    assert spec["splits"]["val"] == ["v"] and spec["splits"]["test"] == ["t"]
    assert job == original
    if method in ("ace", "premise"):
        write(Path(job["output"]) / "evaluation/frozen_checkpoint.json", {"playbook": "wrong policy"})
        with pytest.raises(ValueError, match="differs from the final-test"):
            final_snapshot(job, tmp_path)


def test_dispatch_does_not_evaluate_test_even_when_test_is_first_priority(tmp_path):
    queue = Queue(tmp_path / "queue", interval=10)
    spec = {
        "id": "stream",
        "train_total": 10,
        "method": "gepa",
        "benchmark": "bfcl",
        "seed": 42,
        "variant": "epoch2",
        "splits": {"train": ["a", "b"], "val": ["v"], "test": ["t"]},
    }
    queue.register(spec, {"point": 10, "content": "final context", "selection": "final"})
    assert queue.pending(1)[0]["split"] == "test"
    assert {row["split"] for row in eligible_tasks(queue, 50)} == {"train", "val"}
    assert len(eligible_tasks(queue, 50)) == 3
    queue.db.close()
