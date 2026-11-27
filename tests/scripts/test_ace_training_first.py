# SPDX-License-Identifier: Apache-2.0
"""Validate the cross-mode training barrier, parallel learners and saved-result reuse."""

from concurrent.futures import Future
from pathlib import Path

import pytest

from scripts.ace import run_training_first as study


def mark_completed(job):
    directory = Path(job["path"]) / "conditions" / job["condition"]["id"]
    study.write(directory / "training_completed.json", {})
    study.write(directory / "training_status.json", {"completed": 1})
    study.write(directory / "checkpoints/001.json", {
        "session_count": 1, "condition": job["condition"], "playbook": "saved context",
    })
    (directory / "checkpoints/001.txt").write_text("saved context")


def test_evaluation_is_not_built_or_started_until_every_context_is_complete(tmp_path, monkeypatch):
    job = {"path": str(tmp_path), "condition": {"id": "unfinished"}, "steps": 1}
    seen = []
    monkeypatch.setattr(study, "train_all", lambda *args: seen.append("training_returned"))
    monkeypatch.setattr(study, "build_queue", lambda *args: seen.append("queue_built"))
    monkeypatch.setattr(study, "evaluate_all", lambda *args: seen.append("evaluation_started"))
    with pytest.raises(RuntimeError, match="Evaluation barrier"):
        study.execute(tmp_path, tmp_path, [job], 100)
    assert seen == ["training_returned"]


def test_six_saved_contexts_are_reused_and_54_learners_share_one_pool(tmp_path, monkeypatch):
    jobs = []
    for kind, count in (("full", 36), ("advantage", 24)):
        for index in range(count):
            mode = "high" if index < count // 2 else "low"
            directory = tmp_path / mode / "accumulation" if kind == "full" else tmp_path / "advantage" / mode
            jobs.append({"path": str(directory), "condition": {"id": f"{kind}_{index}"},
                         "steps": 1, "kind": kind, "sampler": mode})
    for job in jobs[:6]:
        mark_completed(job)
    original = {j["condition"]["id"]: study.digest(
        Path(j["path"]) / "conditions" / j["condition"]["id"] / "checkpoints/001.json"
    ) for j in jobs[:6]}
    calls, pools = [], []

    class Pool:
        def __init__(self, max_workers, **kwargs):
            pools.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, job):
            calls.append(job["condition"]["id"])
            mark_completed(job)
            future = Future()
            future.set_result({"ok": True})
            return future

    monkeypatch.setattr(study, "ProcessPoolExecutor", Pool)
    control = tmp_path / "control"
    control.mkdir()
    study.train_all(tmp_path, control, jobs)
    assert pools == [54]
    assert len(calls) == len(set(calls)) == 54
    assert not set(calls).intersection(original)
    assert read_status(control)["completed_contexts"] == 60
    for job in jobs[:6]:
        assert study.digest(Path(job["path"]) / "conditions" / job["condition"]["id"] / "checkpoints/001.json") == (
            original[job["condition"]["id"]]
        )


def read_status(control):
    return study.read(control / "status.json")


def test_existing_official_result_is_validated_without_a_new_actor_call(tmp_path, monkeypatch):
    item = {"root": str(tmp_path), "job": {"id": "saved"}, "source": "unused"}
    row = {"pct": 0.0, "success": False, "official_zero": True}
    study.write(study.result_path(item), row)
    original = study.digest(study.result_path(item))
    monkeypatch.setattr(study, "validate_result", lambda _: study.read(study.result_path(item)))
    monkeypatch.setattr(study, "frozen", lambda _: pytest.fail("Must not load actor worker for a saved result"))
    assert study.evaluate_one(item) == row
    assert study.digest(study.result_path(item)) == original


def test_shared_queue_keeps_phase_seed_and_context_groups_together():
    queue = [{"sampler": mode, "ordinal": ordinal, "job": {
        "phase": phase, "seed": seed, "condition_id": context,
    }} for mode in reversed(study.MODES) for seed in (44, 42, 43)
        for context in ("worst", "best") for phase in reversed(study.PHASES) for ordinal in (1, 0)]
    ordered = sorted(queue, key=study.queue_key)
    assert [row["job"]["phase"] for row in ordered[:24]] == ["train_final"] * 24
    assert all(row["job"]["seed"] == 42 for row in ordered[:8])
    assert [row["ordinal"] for row in ordered[:2]] == [0, 1]
    assert all(row["job"]["condition_id"] == "best" for row in ordered[:2])
