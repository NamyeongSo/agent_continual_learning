# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""The isolated plan uses each seed's frozen split and one matching test set."""

import hashlib
import json

from scripts.ace import evaluate_isolated
from scripts.ace.checkpoint_recovery import recover_checkpoint
from scripts.ace.run_experiment import task_order_from_train_manifest
from scripts.utils import run_isolated_baselines as isolated


def test_isolated_plan_has_27_training_and_27_same_benchmark_evaluations(tmp_path):
    rows = isolated.plan_jobs(tmp_path, isolated.MANIFEST)
    assert len(rows) == 27
    assert [(row["method"], row["benchmark"]) for row in rows] == [
        (method, benchmark)
        for method in isolated.METHODS for benchmark in isolated.BENCHMARKS for _seed in isolated.SEEDS
    ]
    assert {(row["method"], row["seed"], row["benchmark"]) for row in rows} == {
        (method, seed, benchmark)
        for method in isolated.METHODS for seed in isolated.SEEDS for benchmark in isolated.BENCHMARKS
    }
    assert {(row["benchmark"], row["train_count"], row["test_count"]) for row in rows} == {
        ("bfcl", 50, 50), ("appworld", 50, 50), ("terminalbench2", 30, 29),
    }
    for row in rows:
        command = isolated.base_command(row, isolated.MANIFEST)
        assert command[command.index("--benchmarks") + 1] == row["benchmark"]
        assert "--task-split-manifest" in command
        assert "--all-train-tasks" in command


def test_ace_isolated_evaluation_uses_only_matching_seed_test_ids(tmp_path):
    manifest = isolated.MANIFEST
    slug = "terminalbench2"
    seed = 43
    order = task_order_from_train_manifest(manifest, [slug], seed, None, "isolated")
    train = tmp_path / "train"
    train.mkdir()
    (train / "experiment_config.json").write_text(json.dumps({
        "mode": "isolated", "seed": seed, "benchmarks": [slug], "task_order": order,
        "task_split_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "training_time": True, "model": "openai/test", "max_tokens": 8192,
        "benchmark_max_turns": {slug: 100},
    }))
    (train / "training_complete.json").write_text(json.dumps({"completed_task_count": 30}))
    (train / f"playbook_ace_isolated_{slug}.json").write_text(json.dumps({
        "store_id": f"ace_isolated_{slug}", "playbook": "trained",
    }))
    job = evaluate_isolated.prepare_job(train, tmp_path / "evaluation", manifest, slug, seed)
    splits = json.loads(manifest.read_text())
    assert job["task_ids"] == splits[f"{slug}_{seed}"]["test"]
    assert job["stage"] == 30
    assert job["benchmark"] == slug
    assert job["max_steps"] == 100
    snapshot = tmp_path / "evaluation/snapshots/isolated_terminalbench2_s43/checkpoint_030.json"
    assert json.loads(snapshot.read_text())["playbook"] == "trained"


def test_ace_isolated_recovery_stops_before_infrastructure_error(tmp_path):
    order = [["bfcl", "first"], ["bfcl", "second"]]
    for index, (_slug, task) in enumerate(order, 1):
        session = tmp_path / "run/sessions" / f"session-{index}"
        (session / "agent").mkdir(parents=True)
        (session / "config.json").write_text(json.dumps({"benchmark": "bfcl", "task_id": task}))
        (session / "results.json").write_text(json.dumps({
            "task_id": task, "status": "success" if index == 1 else "error",
        }))
        if index == 1:
            (session / "agent/playbook_checkpoint.json").write_text(json.dumps({
                "store_id": "ace_isolated_bfcl", "playbook": "after first",
                "next_global_id": 2, "session_count": 1,
            }))
    assert len(recover_checkpoint(tmp_path, order, "isolated")) == 1
    state = json.loads((tmp_path / "playbook_ace_isolated_bfcl.json").read_text())
    assert state["playbook"] == "after first"
    assert state["completed_task_count"] == 1
