# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import hashlib
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ace import evaluate_test_checkpoints as evaluation
from scripts.ace.checkpoint_recovery import completed_prefix, recover_checkpoint
from scripts.ace.run_experiment import task_order_from_train_manifest
from scripts.ace.run_parallel_bfcl_appworld_terminalbench2 import prepare_sequential_bfcl_prefix

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json"


def create_session(output, mode, slug, task, index, checkpoint=True, status="success"):
    session = output / "run" / "sessions" / f"session-{index}"
    (session / "agent").mkdir(parents=True)
    (session / "config.json").write_text(json.dumps({"benchmark": slug, "task_id": task}))
    (session / "results.json").write_text(json.dumps({"task_id": task, "status": status}))
    if checkpoint:
        (session / "agent/playbook_checkpoint.json").write_text(json.dumps({
            "store_id": f"ace_{mode}_global", "playbook": f"after {index}", "session_count": index,
        }))
    return session


def test_plan_uses_all_three_test_sets_for_each_seed_and_stage(tmp_path):
    jobs = []
    for mode in ("sequential", "interleaved"):
        for seed in (42, 43, 44):
            jobs.append({"mode": mode, "seed": seed, "output_dir": str(tmp_path / f"{mode}_s{seed}"),
                         "task_order": task_order_from_train_manifest(
                             MANIFEST, list(evaluation.BENCHMARKS), seed, None, mode)})
    launch = {"jobs": jobs, "source_manifest": str(MANIFEST), "source_manifest_sha256": evaluation.digest(MANIFEST),
              "model": "openai/test", "max_tokens": 8192, "terminalbench2_max_turns": 100}
    plan = evaluation.build_plan(launch)
    splits = json.loads(MANIFEST.read_text())
    assert len(plan) == 54
    assert sum(len(job["task_ids"]) for job in plan) == 2322
    for job in plan:
        assert job["task_ids"] == splits[f"{job['benchmark']}_{job['seed']}"]["test"]
        assert job["stage"] in (50, 100, 130)
        assert job["max_steps"] == (100 if job["benchmark"] == "terminalbench2" else 50)
    launch["source_manifest_sha256"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        evaluation.build_plan(launch)


@pytest.mark.parametrize("mode", ["sequential", "interleaved"])
def test_snapshots_use_exact_task_ordinal_and_ignore_final_playbook(tmp_path, mode):
    order = [["bfcl", f"task-{i}"] for i in range(1, 131)]
    output = tmp_path / "train"
    for index, (slug, task) in enumerate(order, 1):
        create_session(output, mode, slug, task, index)
    run = {"mode": mode, "seed": 42, "output_dir": str(output), "task_order": order}
    paths = evaluation.archive_snapshots(run, tmp_path / "test")
    for stage, path in paths.items():
        payload = json.loads(path.read_text())
        assert payload["playbook"] == f"after {stage}"
        assert payload["training_task_ordinal"] == stage
    assert evaluation.archive_snapshots(run, tmp_path / "test") == paths


def test_recovery_waits_for_reflection_checkpoint_and_preserves_completed_prefix(tmp_path):
    order = [["bfcl", "first"], ["bfcl", "second"], ["appworld", "third"]]
    create_session(tmp_path, "sequential", "bfcl", "first", 1)
    create_session(tmp_path, "sequential", "bfcl", "second", 2, checkpoint=False)
    create_session(tmp_path, "sequential", "appworld", "third", 3)
    assert len(completed_prefix(tmp_path, order)) == 1
    recovered = recover_checkpoint(tmp_path, order, "sequential")
    assert len(recovered) == 1
    payload = json.loads((tmp_path / "playbook_ace_sequential_global.json").read_text())
    assert payload["playbook"] == "after 1"
    assert payload["completed_task_count"] == 1


@pytest.mark.parametrize("with_snapshot", [True, False])
def test_reused_bfcl_prefix_starts_sequential_training_at_appworld(tmp_path, with_snapshot):
    old_base = tmp_path / "old"
    new_base = tmp_path / "new"
    source = old_base / "sequential_s42"
    order = [["bfcl", f"bfcl-{index}"] for index in range(50)] + [["appworld", "appworld-0"]]
    (source / "experiment_config.json").parent.mkdir(parents=True)
    (source / "experiment_config.json").write_text("{}")
    (source / "online_metrics.jsonl").write_text("".join(
        json.dumps({"benchmark_slug": slug, "task_id": task, "session_index": index,
                    "score": 1.0, "status": "success"}) + "\n"
        for index, (slug, task) in enumerate(order[:50])
    ))
    session = create_session(source, "sequential", "bfcl", "bfcl-49", 50)
    checkpoint = session / "agent/playbook_checkpoint.json"
    snapshot = old_base / "test_evaluation/snapshots/sequential_s42/checkpoint_050.json"
    if with_snapshot:
        snapshot.parent.mkdir(parents=True)
        snapshot.write_text(json.dumps({
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }))
    (old_base / "launch_manifest.json").write_text(json.dumps({
        "source_manifest_sha256": "frozen-manifest",
        "jobs": [{"mode": "sequential", "seed": 42, "task_order": order, "output_dir": str(source)}],
    }))
    output = new_base / "sequential_s42"
    job = {"mode": "sequential", "seed": 42, "task_order": [tuple(item) for item in order],
           "output_dir": str(output), "command": ["run_experiment.py"]}

    reused = prepare_sequential_bfcl_prefix(old_base, new_base, [job], "frozen-manifest")
    prefix = new_base / "reused_bfcl_prefix/sequential_s42"
    shutil.copy2(prefix / "online_metrics.jsonl", output / "online_metrics.jsonl")

    assert reused["sequential_s42"]["reused_train_tasks"] == 50
    assert job["command"][-2:] == ["--continue-from", str(prefix)]
    assert len(completed_prefix(output, job["task_order"])) == 50
    assert completed_prefix(output, job["task_order"])[-1]["checkpoint"]["session_count"] == 50


def test_error_at_boundary_uses_previous_checkpoint_without_future_learning(tmp_path):
    order = [["bfcl", f"task-{i}"] for i in range(1, 131)]
    for index, (slug, task) in enumerate(order, 1):
        create_session(tmp_path / "train", "sequential", slug, task, index,
                       checkpoint=index != 50, status="error" if index == 50 else "success")
    run = {"mode": "sequential", "seed": 42, "output_dir": str(tmp_path / "train"), "task_order": order}
    snapshots = evaluation.archive_snapshots(run, tmp_path / "test")
    assert json.loads(snapshots[50].read_text())["playbook"] == "after 49"


def test_resumed_prefix_without_old_session_folders_uses_verified_metrics(tmp_path):
    order = [["bfcl", f"task-{i}"] for i in range(1, 131)]
    output = tmp_path / "train"
    rows = []
    for index, (slug, task) in enumerate(order, 1):
        session = create_session(output, "sequential", slug, task, index)
        if index <= 17:
            shutil.rmtree(session)
        rows.append({"benchmark_slug": slug, "task_id": task, "session_index": index - 1,
                     "score": 1.0, "status": "success"})
    (output / "online_metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    prefix = completed_prefix(output, order)
    assert len(prefix) == 130
    assert prefix[0]["checkpoint"] is None
    run = {"mode": "sequential", "seed": 42, "output_dir": str(output), "task_order": order}
    paths = evaluation.archive_snapshots(run, tmp_path / "test")
    for stage in (50, 100, 130):
        assert json.loads(paths[stage].read_text())["playbook"] == f"after {stage}"
    rows[0]["task_id"] = "different-task"
    (output / "online_metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="planned task prefix"):
        completed_prefix(output, order)


@pytest.mark.parametrize("actor_context", ["legacy", "shared-empty-v1"])
def test_test_evaluation_is_frozen_and_terminalbench_uses_terminus2(tmp_path, monkeypatch, actor_context):
    job = {"run": "sequential_s42", "mode": "sequential", "seed": 42, "stage": 50,
           "benchmark": "terminalbench2", "task_ids": ["a", "b"], "max_tokens": 8192,
           "model": "openai/test", "max_steps": 15, "manifest_sha256": "manifest"}
    if actor_context != "legacy":
        from exgentic.actor_context import actor_context_metadata

        job.update(actor_context=actor_context,
                   actor_context_metadata=actor_context_metadata(actor_context, ["terminalbench2"]))
    snapshot = tmp_path / "snapshots/sequential_s42/checkpoint_050.json"
    evaluation.save_json(snapshot, {"playbook": "frozen"})
    captured = {}
    monkeypatch.setattr(evaluation, "load_agent", lambda _name: lambda **kwargs: captured.setdefault("agent", kwargs))
    monkeypatch.setattr(evaluation, "load_benchmark",
                        lambda _name: lambda **kwargs: captured.setdefault("benchmark", kwargs))

    def fake_evaluate(**kwargs):
        captured["evaluation"] = kwargs
        return SimpleNamespace(benchmark_score=0.5, session_results=[
            SimpleNamespace(task_id=task, session_id=task, score=1.0 if task == "a" else 0.0,
                            status="success" if task == "a" else "error", steps=1)
            for task in job["task_ids"]
        ])

    monkeypatch.setattr(evaluation, "evaluate", fake_evaluate)
    summary = evaluation.run_job(job, tmp_path, max_workers=4)
    assert captured["agent"]["evaluation_mode"] is True
    assert captured["agent"]["training_time"] is False
    assert captured["agent"]["initial_playbook"] == "frozen"
    assert captured["agent"]["actor_context"] == actor_context
    assert captured["benchmark"]["actor"] == "terminus2"
    assert captured["benchmark"]["max_interactions"] == 15
    assert captured["evaluation"]["max_workers"] == 1
    assert summary["error_count"] == 1
    monkeypatch.setattr(evaluation, "evaluate", lambda **_kwargs: pytest.fail("completed test rerun"))
    assert evaluation.run_job(job, tmp_path, max_workers=4) == summary


def test_bfcl_checkpoint_can_be_archived_before_later_benchmarks_finish(tmp_path):
    order = [["bfcl", f"task-{i}"] for i in range(1, 51)] + [["appworld", "later"]]
    for index, (slug, task) in enumerate(order, 1):
        create_session(tmp_path / "train", "sequential", slug, task, index)
    run = {"mode": "sequential", "seed": 42, "output_dir": str(tmp_path / "train"), "task_order": order}
    snapshots = evaluation.archive_snapshots(run, tmp_path / "test", stages=(50,))
    assert set(snapshots) == {50}
    assert json.loads(snapshots[50].read_text())["playbook"] == "after 50"
    with pytest.raises(ValueError, match="Incomplete training"):
        evaluation.archive_snapshots(run, tmp_path / "test")


def test_overlapping_controllers_reuse_one_checkpoint_evaluation(tmp_path, monkeypatch):
    job = {"run": "sequential_s42", "mode": "sequential", "seed": 42, "stage": 50,
           "benchmark": "bfcl", "task_ids": ["task"], "max_tokens": 8192,
           "model": "openai/test", "max_steps": 50, "manifest_sha256": "manifest"}
    evaluation.save_json(tmp_path / "snapshots/sequential_s42/checkpoint_050.json", {"playbook": "frozen"})
    monkeypatch.setattr(evaluation, "load_agent", lambda _name: lambda **kwargs: kwargs)
    monkeypatch.setattr(evaluation, "load_benchmark", lambda _name: lambda **kwargs: kwargs)
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def fake_evaluate(**_kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(benchmark_score=1.0, session_results=[
            SimpleNamespace(task_id="task", session_id="task", score=1.0, status="success", steps=1),
        ])

    monkeypatch.setattr(evaluation, "evaluate", fake_evaluate)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(evaluation.run_job, job, tmp_path, 4)
        assert entered.wait(5)
        second = pool.submit(evaluation.run_job, job, tmp_path, 4)
        time.sleep(0.1)
        assert not second.done()
        release.set()
        assert first.result() == second.result()
    assert len(calls) == 1


def test_training_recovery_discards_error_checkpoint_and_every_later_update(tmp_path):
    order = [["bfcl", "first"], ["bfcl", "broken"], ["bfcl", "later"]]
    create_session(tmp_path, "sequential", "bfcl", "first", 1)
    create_session(tmp_path, "sequential", "bfcl", "broken", 2, status="error")
    create_session(tmp_path, "sequential", "bfcl", "later", 3)
    prefix = recover_checkpoint(tmp_path, order, "sequential")
    assert [item["task_id"] for item in prefix] == ["first"]
    checkpoint = json.loads((tmp_path / "playbook_ace_sequential_global.json").read_text())
    assert checkpoint["playbook"] == "after 1"
    assert checkpoint["completed_task_count"] == 1
    assert len(completed_prefix(tmp_path, order)) == 3  # Historical archive still sees all exposures.


def test_training_stops_before_next_task_after_infrastructure_failure(monkeypatch):
    from scripts.ace import run_experiment as training

    calls = []

    def evaluate(**kwargs):
        task = kwargs["task_ids"][0]
        calls.append(task)
        # An official budget failure is a measured zero, so training advances.
        result = SimpleNamespace(status="limit_reached", score=0, details={}) if task == "budget" else (
            SimpleNamespace(status="error", score=None, details={"error": "verifier transport"})
        )
        return SimpleNamespace(session_results=[result])

    monkeypatch.setattr(training, "evaluate", evaluate)
    iterator = training.evaluate_training_tasks(["budget", "broken", "later"], run_id="train")
    assert next(iterator)[0] == "budget"
    with pytest.raises(RuntimeError, match=r"broken.*no valid outcome"):
        next(iterator)
    assert calls == ["budget", "broken"]
