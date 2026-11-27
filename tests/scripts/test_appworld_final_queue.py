# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Check final-only dispatch scope, bounded concurrency and result attribution."""

from __future__ import annotations

import hashlib
from concurrent.futures import Future
from pathlib import Path

import pytest

from scripts.utils import evaluate_appworld_final_queue as final
from scripts.utils.checkpoint_queue import Queue


def test_one_queue_interleaves_contexts_and_bounds_all_evaluations_at_fifty(tmp_path):
    queue = Queue(tmp_path)
    for seed in (42, 43, 44):
        spec = {"id": str(seed), "method": "ace", "benchmark": "appworld", "seed": seed,
                "variant": "fixture", "model": "model", "temperature": 0.7, "train_total": 50,
                "splits": {"test": [f"task{i}" for i in range(50)]}}
        queue.register(spec, {"point": 50, "content": f"context{seed}", "selection": "final"})
    futures = []

    class Pool:
        def submit(self, function, task):
            future = Future()
            futures.append(future)
            return future

    active = {}
    final.bounded_dispatch(queue, Pool(), active)
    assert len(active) == 50
    assert {task["cp"] for task in active.values()} == {f"{seed}_cp050" for seed in (42, 43, 44)}
    assert {task["split"] for task in active.values()} == {"test"}
    final.bounded_dispatch(queue, Pool(), active)
    assert len(active) == len(futures) == 50
    queue.db.close()


def test_results_return_to_the_original_context_and_zero_remains_measured(tmp_path):
    frozen = tmp_path / "context.json"
    final.write(frozen, {"playbook": "frozen context"})
    original = {"task_id": "task", "index": 3, "frozen": str(frozen), "frozen_sha256": final.sha(frozen)}
    context = {"frozen": str(frozen), "frozen_sha256": final.sha(frozen), "evaluations": [original],
               "source_output": str(tmp_path / "source"), "variant": "add_only",
               "policy_sha256": hashlib.sha256(b"frozen context").hexdigest()}
    queue_output = tmp_path / "queue"
    final.publish_attempt(queue_output, context, {"task_id": "task", "attempt": 1},
                          {"status": "completed", "score": 0, "partial_score": 0.9}, True)
    final.summaries(queue_output, [context], "running")
    record = final.read(tmp_path / "source/evaluation/records/003.json")
    assert record["score"] == 0 and record["partial_score"] == 0.9
    assert record["frozen_sha256"] == final.sha(frozen) and record["final_test_queue"] == str(queue_output)
    summary = final.read(tmp_path / "source/evaluation/summary.json")
    assert summary["score"] == 0
    final.write(frozen, {"playbook": "changed"})
    with pytest.raises(ValueError, match="context changed"):
        final.publish_attempt(queue_output, context, {"task_id": "task", "attempt": 2}, {}, True)


def test_infrastructure_retry_is_not_published_as_an_official_zero(tmp_path):
    frozen = tmp_path / "frozen.json"
    final.write(frozen, {"playbook": "context"})
    context = {"frozen": str(frozen), "frozen_sha256": final.sha(frozen),
               "evaluations": [{"task_id": "task", "index": 0}], "source_output": str(tmp_path / "source")}
    final.publish_attempt(tmp_path / "queue", context, {"task_id": "task", "attempt": 1},
                          {"status": "error", "score": None, "error": "transport"}, False)
    assert not list((Path(context["source_output"]) / "evaluation/records").glob("*.json"))
    attempt = final.read(tmp_path / "source/evaluation/attempts/000_1.json")
    assert attempt["score"] is None


def test_training_completion_requires_both_full_cursor_and_commit_marker(tmp_path):
    job = {"method": "premise", "train_dir": str(tmp_path), "train_count": 50}
    final.write(tmp_path / "state.json", {"cursor": 49})
    assert not final.training_finished(job)
    final.write(tmp_path / "training_complete.json", {"completed_task_count": 50})
    assert not final.training_finished(job)
    final.write(tmp_path / "state.json", {"cursor": 50})
    assert final.training_finished(job)


def test_completed_ace_queue_keeps_watching_until_all_peace_contexts_register(tmp_path):
    contexts = [{"variant": variant, "seed": seed} for variant in ("add_only", "add_update_delete")
                for seed in (42, 43, 44)]
    plan = {"contexts": contexts, "workers": 50, "tasks": 300}
    final.write(tmp_path / "final_test_queue_manifest.json", plan)
    updated = final.enable_peace(tmp_path)
    assert updated["contexts"] == contexts and updated["tasks"] == 300 and updated["workers"] == 50
    assert updated["expected_tasks"] == 600 and updated["registration_interval_seconds"] == 5
    assert not final.all_registered(updated)
    updated["contexts"] += [{"method": "premise", "variant": variant, "seed": seed}
                            for variant in ("add_only", "add_update_delete") for seed in (42, 43, 44)]
    assert final.all_registered(updated)


def test_completed_peace_context_enrolls_once_and_never_adds_train_or_val(tmp_path, monkeypatch):
    from scripts.utils import run_qwen_isolated_when_free as runner

    queue_output = tmp_path / "queue"
    queue = Queue(queue_output)
    source = tmp_path / "source"
    frozen = source / "evaluation/frozen_checkpoint.json"
    final.write(frozen, {"playbook": "final rank1"})
    policy_sha = "unchanged-environment-policy"
    tasks = [f"task{i}" for i in range(50)]
    job = {"method": "premise", "benchmark": "appworld", "seed": 42, "train_count": 50,
           "test_ids": tasks, "output": str(source)}
    child = {"root": str(tmp_path), "journal_id": "experiment"}
    spec = {"id": "premise42", "method": "premise", "benchmark": "appworld", "seed": 42,
            "model": "model", "temperature": 0.7, "variant": "cohort", "train_total": 50,
            "splits": {"test": tasks}, "appworld_policy_sha256": policy_sha, "playbook_edit_mode": "add_only"}
    evaluations = [{"frozen": str(frozen), "frozen_sha256": final.sha(frozen), "task_id": task, "index": i}
                   for i, task in enumerate(tasks)]
    monkeypatch.setattr(runner, "frozen_evaluation_jobs", lambda job: evaluations)
    monkeypatch.setattr(final, "completed_sources", lambda plan: [] if plan["contexts"] else [
        (child, job, spec, policy_sha)])
    plan = {"contexts": [], "tasks": 0}
    assert len(final.collect_finished(queue, plan, queue_output)) == 1
    assert final.collect_finished(queue, plan, queue_output) == []
    assert queue.db.execute("SELECT count(*) FROM tasks WHERE split='test'").fetchone()[0] == 50
    assert queue.db.execute("SELECT count(*) FROM tasks WHERE split!='test'").fetchone()[0] == 0
    assert final.read(queue_output / "final_test_queue_manifest.json")["tasks"] == 50
    assert "premise" in (queue_output / "context_task_tuples.csv").read_text()
    queue.db.close()


def test_new_peace_tasks_share_the_existing_fifty_task_cap(tmp_path):
    queue = Queue(tmp_path)
    for method in ("ace", "premise"):
        spec = {"id": method, "method": method, "benchmark": "appworld", "seed": 42,
                "variant": "fixture", "model": "model", "temperature": 0.7, "train_total": 50,
                "splits": {"test": [f"task{i}" for i in range(50)]}}
        queue.register(spec, {"point": 50, "content": method, "selection": "final"})

    class Pool:
        def submit(self, function, task):
            return Future()

    active = {}
    final.bounded_dispatch(queue, Pool(), active)
    assert len(active) == 50 and {task["cp"] for task in active.values()} == {"ace_cp050", "premise_cp050"}
    final.bounded_dispatch(queue, Pool(), active)
    assert len(active) == 50
    queue.db.close()
