# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.baselines.run_experiment import make_config
from scripts.utils import checkpoint_eval_worker, checkpoint_sources
from scripts.utils import run_qwen_isolated_when_free as runner
from scripts.utils.checkpoint_queue import Queue


def test_native_config_applies_policy_without_changing_splits_or_budgets(tmp_path):
    manifest = Path(__file__).resolve().parents[2] / (
        "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json"
    )
    args = SimpleNamespace(manifest=manifest, benchmark="appworld", seed=42, pilot=False, method="gepa",
                           model=runner.MODEL, base_url=runner.BASE_URL, workers=50, output=tmp_path)
    original = make_config(args)
    policy = tmp_path / "policy.txt"
    policy.write_text("Team policy for {allowed}; finish with {}")
    args.appworld_policy = policy
    updated = make_config(args)
    for key in ("train", "validation", "test", "workers", "experiment_limits", "temperature",
                "total_training_limit", "analysis_input_limit", "protocol_sha256"):
        assert updated[key] == original[key]
    assert updated["benchmark_kwargs"]["policy_override"] == policy.read_text()
    assert updated["appworld_policy_sha256"] == hashlib.sha256(policy.read_bytes()).hexdigest()


@pytest.mark.parametrize("method", ["ace", "premise", "gepa", "skillopt"])
def test_stream_freezes_the_actual_environment_policy_and_detects_changed_bytes(tmp_path, method):
    policy = "Team policy for {allowed}; finish with {}"
    sha = checkpoint_sources.digest(policy)
    path = tmp_path / "policy.txt"
    path.write_text(policy)
    manifest = tmp_path / "split.json"
    runner.write(manifest, {"appworld_42": {"train": ["a"], "val": ["b"], "test": ["c"]}})
    output = tmp_path / "run"
    job = {"id": "one", "method": method, "benchmark": "appworld", "seed": 42,
           "output": str(output), "train_dir": str(output),
           "command": ["python", "--task-split-manifest", str(manifest), "--appworld-policy", str(path)]}
    runner.write(output / "config.json", {"model": runner.MODEL, "split_manifest_sha256": "split",
                                           "train": [{"task_id": "a"}], "validation": [{"task_id": "b"}],
                                           "test": [{"task_id": "c"}],
                                           "benchmark_kwargs": {"policy_override": policy}})
    runner.write(tmp_path / "launch_manifest.json", {"jobs": [job], "appworld_policy_sha256": sha})
    spec = checkpoint_sources.streams([tmp_path], checkpoint_sources.Reader())[0]
    assert spec["appworld_policy_override"] == policy and spec["appworld_policy_sha256"] == sha
    if method in {"ace", "premise"}:
        path.write_text("changed")
    else:
        config = runner.read(output / "config.json")
        config["benchmark_kwargs"]["policy_override"] = "changed"
        runner.write(output / "config.json", config)
    with pytest.raises(ValueError, match="policy changed"):
        checkpoint_sources.streams([tmp_path], checkpoint_sources.Reader())


def test_queue_refuses_policy_changes_even_when_the_learned_playbook_is_unchanged(tmp_path):
    queue = Queue(tmp_path)
    spec = {"id": "one", "method": "premise", "benchmark": "appworld", "seed": 42,
            "train_total": 10, "splits": {"train": ["a"], "val": ["b"], "test": ["c"]},
            "appworld_policy_sha256": "original"}
    queue.register(spec, {"point": 5, "content": "same playbook"})
    with pytest.raises(ValueError, match="appworld_policy_sha256"):
        queue.register({**spec, "appworld_policy_sha256": "changed"}, {"point": 10, "content": "same playbook"})


@pytest.mark.parametrize("method", ["ace", "premise"])
def test_checkpoint_worker_passes_policy_to_the_frozen_actor(tmp_path, monkeypatch, method):
    worker = tmp_path / "workers" / f"{os.getpid()}.json"
    runner.write(worker, {"pid": os.getpid()})
    monkeypatch.setattr(checkpoint_eval_worker, "_OUTPUT", tmp_path)
    seen = []

    def evaluate(job):
        seen.append(job)
        return {"status": "completed", "score": 0, "appworld_policy_sha256": job["appworld_policy_sha256"]}

    monkeypatch.setattr(runner, "evaluate_one", evaluate)
    policy = "team policy"
    frozen = tmp_path / "cp.json"
    runner.write(frozen, {"content": "playbook", "stream": {
        "method": method, "benchmark": "appworld", "seed": 42,
        "appworld_policy_override": policy, "appworld_policy_sha256": checkpoint_sources.digest(policy)}})
    task = {"id": 1, "attempt": 1, "task_id": "a", "split": "train", "checkpoint_file": str(frozen),
            "content_sha256": checkpoint_sources.digest("playbook")}
    result = checkpoint_eval_worker.evaluate(task)
    assert result["status"] == "completed"
    assert seen[0]["appworld_policy_override"] == policy
    assert result["appworld_policy_sha256"] == checkpoint_sources.digest(policy)


def test_final_evaluation_binds_the_training_policy(tmp_path):
    policy = tmp_path / "policy.txt"
    policy.write_text("team policy")
    sha = runner.digest(policy)
    train = tmp_path / "train"
    runner.write(train / "training_complete.json", {"completed_task_count": 5})
    runner.write(train / "state.json", {"cursor": 5, "config": {
        "temperature_all_roles": 0.7, "seed": 42, "mode": "isolated", "model": runner.MODEL,
        "appworld_policy_sha256": sha}, "states": {"appworld": {"beam": [{"playbook": "rank1"}]}}})
    job = {"train_dir": str(train), "output": str(tmp_path), "method": "premise", "benchmark": "appworld",
           "seed": 42, "train_count": 5, "test_ids": ["test"], "appworld_policy": str(policy),
           "appworld_policy_sha256": sha}
    jobs = runner.frozen_evaluation_jobs(job)
    assert jobs[0]["appworld_policy_override"] == policy.read_text()
    assert runner.read(jobs[0]["frozen"])["appworld_policy_sha256"] == sha
