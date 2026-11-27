# SPDX-License-Identifier: Apache-2.0
"""Continuation safety checks; no server or benchmark calls."""

import hashlib
import json
import random

import pytest

from scripts.utils.continue_epoch2 import PrefixMiss, baseline_runtime, completed_job, second_order, write


def setup_runtime(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "train"
    config = {"output": str(destination), "workers": 2, "method": "gepa", "benchmark": "bfcl"}
    write(destination / "config.json", config)
    job = {"source_train_dir": str(source), "train_dir": str(destination), "method": "gepa", "epoch1_count": 2}
    return baseline_runtime(job, False), source


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def test_shuffle_reuses_actual_first_epoch_order():
    first = ["c", "a", "e", "b", "d"]
    expected = first.copy()
    random.Random(42).shuffle(expected)
    assert second_order(first, 42) == expected
    assert first == ["c", "a", "e", "b", "d"]
    assert set(expected) == set(first)


def test_prefix_cache_miss_fails_without_actor_retry(tmp_path, monkeypatch):
    runtime, _ = setup_runtime(tmp_path)
    monkeypatch.setattr("scripts.baselines.runtime.Runtime.task", lambda *args: pytest.fail("No new prefix calls"))
    with pytest.raises(PrefixMiss):
        runtime.task({"task_id": "task", "split": "val"}, "initial", "val")


def test_prefix_cache_read_is_immutable(tmp_path):
    runtime, source = setup_runtime(tmp_path)
    item = {"task_id": "task", "split": "val"}
    key = digest(json.dumps({"item": item, "prompt_sha256": digest("initial"), "label": "val"}, sort_keys=True))
    cache = source / "rollouts" / key / "result.json"
    write(cache, {"hard": 0, "soft": 0.0})
    before = cache.read_bytes()
    assert runtime.task(item, "initial", "val") == {"hard": 0, "soft": 0.0}
    assert cache.read_bytes() == before
    assert not (runtime.output / "rollouts").exists()


def test_epoch_two_encounter_does_not_overwrite_epoch_one(tmp_path):
    runtime, _ = setup_runtime(tmp_path)
    old = {"task_id": "task", "split": "train"}
    new = {**old, "epoch": 2}
    runtime.online([old], [{"hard": 0}], "initial", 0)
    runtime.online([new], [{"hard": 1}], "learned", 1)
    records = [json.loads(p.read_text()) for p in (runtime.output / "online").glob("*.json")]
    assert len(records) == 2
    assert sorted(row["hard"] for row in records) == [0, 1]
    with pytest.raises(RuntimeError, match="diverged"):
        runtime.online([new], [{"hard": 0}], "different", 1)


def test_epoch_two_final_test_has_distinct_cache_namespace(tmp_path, monkeypatch):
    runtime, _ = setup_runtime(tmp_path)
    runtime.activate()
    labels = []
    monkeypatch.setattr(
        "scripts.baselines.runtime.Runtime.task", lambda self, item, learned, label: labels.append(label)
    )
    runtime.task({"task_id": "task", "split": "test"}, "unchanged", "final_test")
    assert labels == ["epoch2_final_test"]


def test_partial_final_marker_never_releases_next_benchmark(tmp_path):
    job = {
        "output": str(tmp_path),
        "train_dir": str(tmp_path / "train"),
        "method": "ace",
        "id": "job",
        "train_count": 100,
        "test_ids": ["a", "b"],
    }
    write(tmp_path / "job_complete.json", {"complete": True})
    assert not completed_job(job)
    marker = {"complete": True, "job_id": "job", "tasks": 2, "test_ids": ["a", "b"], "infrastructure_errors": 0}
    write(tmp_path / "job_complete.json", marker)
    write(tmp_path / "train/training_complete.json", {"completed_task_count": 50})
    assert not completed_job(job)
    write(tmp_path / "train/training_complete.json", {"completed_task_count": 100})
    assert completed_job(job)


@pytest.mark.parametrize("epoch,resume_from", [(1, 11), (2, 11), (2, 21)])
def test_buffer_restore_ignores_completed_and_fresh_epochs(tmp_path, epoch, resume_from):
    from scripts.utils.continue_epoch2 import restore_skillopt_buffer

    # Previous epochs can contain legitimate no-patch steps without digests.
    history = [{"epoch": 1, "step": 2, "action": "skip_no_patches"}]
    assert restore_skillopt_buffer(tmp_path, history, epoch, 10, resume_from) == []


@pytest.mark.parametrize("skip_action", ["skip_no_patches", "skip_no_rewrite"])
def test_mid_epoch_buffer_restore_preserves_entries_and_native_skips(tmp_path, skip_action):
    from scripts.utils.continue_epoch2 import restore_skillopt_buffer

    history = [
        {"epoch": 1, "step": 2, "action": "reject"},  # Old epoch is never loaded.
        {"epoch": 2, "step": 11, "action": "reject"},
        {"epoch": 2, "step": 12, "action": skip_action},
        {"epoch": 2, "step": 13, "action": "accept_new_best"},
    ]
    first = {"step": 11, "action": "reject", "rejected_edits": ["keep this diagnostic"]}
    last = {"step": 13, "action": "accept_new_best"}
    write(tmp_path / "steps/step_0011/trajectory_digest.json", first)
    write(tmp_path / "steps/step_0013/trajectory_digest.json", last)
    assert restore_skillopt_buffer(tmp_path, history, 2, 10, 14) == [first, last]


def test_mid_epoch_buffer_restore_does_not_hide_missing_real_state(tmp_path):
    from scripts.utils.continue_epoch2 import restore_skillopt_buffer

    history = [{"epoch": 2, "step": 11, "action": "reject"}]
    with pytest.raises(RuntimeError, match="Missing saved SkillOpt step buffer"):
        restore_skillopt_buffer(tmp_path, history, 2, 10, 12)
    with pytest.raises(RuntimeError, match="Missing saved SkillOpt history"):
        restore_skillopt_buffer(tmp_path, [], 2, 10, 12)


def test_controller_adopts_existing_worker_and_detects_pid_reuse(tmp_path, monkeypatch):
    import scripts.utils.continue_epoch2 as continuation

    identity = {
        "pid": 123,
        "start_ticks": "456",
        "state": "S",
        "args": ["python", "old_frozen_runner.py", "worker", "--output", str(tmp_path), "--job", "job"],
    }
    monkeypatch.setattr(continuation, "process_identity", lambda pid: identity)
    monkeypatch.setattr(continuation, "completed_job", lambda job: False)
    worker = continuation.existing_worker({"id": "job"}, {"status": "running", "pid": 123}, tmp_path)
    assert worker.pid == 123 and worker.poll() is None
    identity["start_ticks"] = "789"
    assert worker.poll() == 1


def test_controller_refuses_unrelated_pid(tmp_path, monkeypatch):
    import scripts.utils.continue_epoch2 as continuation

    identity = {"pid": 123, "start_ticks": "456", "state": "S", "args": ["other-user-service"]}
    monkeypatch.setattr(continuation, "process_identity", lambda pid: identity)
    with pytest.raises(RuntimeError, match="different process"):
        continuation.existing_worker({"id": "job"}, {"status": "running", "pid": 123}, tmp_path)
