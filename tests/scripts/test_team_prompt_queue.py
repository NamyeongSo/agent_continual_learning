# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import fcntl
import json
import statistics
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

from scripts.utils import evaluate_team_prompt_queue as runner


@pytest.fixture
def predecessor(tmp_path):
    upstream = tmp_path / "predecessor"
    upstream.mkdir()
    split = {f"appworld_{seed}": {
        "train": [f"train-{seed}-{index}" for index in range(50)],
        "val": [f"val-{seed}-{index}" for index in range(50)],
        "test": [f"test-{seed}-{index}" for index in range(50)],
    } for seed in (42, 43, 44)}
    raw = json.dumps(split).encode()
    (upstream / "task_split_manifest.json").write_bytes(raw)
    runner.shared.write_json(upstream / "launch_manifest.json", {
        "manifest_sha256": runner.shared.digest(raw), "stages": ["appworld", "bfcl"],
        "groups": [{"model": "openai/test-model"}],
    })
    (upstream / "controller.lock").touch()
    return upstream


def test_next_queue_requires_both_stages_and_predecessor_workers_to_exit(predecessor):
    expected = runner.shared.digest((predecessor / "launch_manifest.json").read_bytes())
    runner.shared.write_json(predecessor / "status.json", {"phase": "complete", "processed_tasks": 12000})
    completion = {"all_tasks_terminal": True, "tasks": 6000, "groups": 120, "execution_errors": 1}
    runner.shared.write_json(predecessor / "appworld_complete.json", completion)
    assert not runner.predecessor_ready(predecessor, expected)[0]
    runner.shared.write_json(predecessor / "bfcl_complete.json", completion)
    with (predecessor / "controller.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ready, reason, _ = runner.predecessor_ready(predecessor, expected)
        assert not ready
        assert "worker processes" in reason
    assert runner.predecessor_ready(predecessor, expected)[0]


def test_incomplete_stage_marker_cannot_start_next_queue(predecessor):
    expected = runner.shared.digest((predecessor / "launch_manifest.json").read_bytes())
    runner.shared.write_json(predecessor / "appworld_complete.json", {
        "all_tasks_terminal": False, "tasks": 5999, "groups": 120,
    })
    with pytest.raises(ValueError, match="completion marker"):
        runner.predecessor_ready(predecessor, expected)


def checkpoint(source, method, seed, point, prompt):
    path = source / f"{method}_interleaved_seed{seed}" / f"cp{point:03d}_shared"
    path.mkdir(parents=True)
    raw = json.dumps({"best_candidate": {"system_prompt": prompt}}).encode() if method == "gepa" else prompt.encode()
    filename = "candidate.json" if method == "gepa" else "skill.md"
    (path / filename).write_bytes(raw)
    if method == "gepa":
        (path / "prompt.md").write_text(prompt + "\n")
    runner.shared.write_json(path / "provenance.json", {
        "method": method, "mode": "interleaved", "seed": seed, "checkpoint": point,
        "sha256": runner.shared.digest(raw),
    })
    return path / filename


def test_collection_preserves_identical_distinct_checkpoints_and_deduplicates_file_alias(tmp_path, predecessor):
    source, output = tmp_path / "source", tmp_path / "output"
    original = checkpoint(source, "gepa", 42, 50, "Exact GEPA prompt")
    checkpoint(source, "gepa", 42, 100, "Exact GEPA prompt")
    checkpoint(source, "skillopt", 43, 130, "Exact SkillOpt prompt")
    alias = original.parent.parent / "cp050_candidate.json"
    alias.write_bytes(original.read_bytes())
    launch = runner.prepare(source, output, predecessor, 100, "http://localhost/v1")
    assert launch["checkpoint_count"] == 3
    assert launch["total_tasks"] == 150
    assert len(launch["aliases"]) == 1
    assert len({group["group"] for group in launch["groups"]}) == 3
    for group in launch["groups"]:
        assert all(task.startswith(f"test-{group['seed']}-") for task in group["task_ids"])
        frozen = group["checkpoint_path"]
        assert runner.shared.digest(Path(frozen).read_bytes()) == group["checkpoint_sha256"]


@pytest.mark.parametrize("method", ["gepa", "skillopt"])
def test_task_uses_exact_frozen_prompt_without_learning(tmp_path, method):
    prompt = "Checkpoint instruction\nKeep its complete text."
    path = checkpoint(tmp_path, method, 42, 50, prompt)
    job = {"group": f"config-test/{method}", "checkpoint_path": str(path), "method": method,
           "checkpoint_sha256": runner.shared.digest(path.read_bytes()),
           "prompt_sha256": runner.shared.digest(prompt.encode()), "model": "openai/test-model",
           "temperature": 0.0, "task_id": "test-task"}
    runner._PROMPTS.clear()
    config = runner.task_config(job, tmp_path / "sessions", "test-run")
    assert config.agent == "premise"
    assert config.agent_kwargs["system_prompt_override"] == prompt
    assert config.agent_kwargs["playbook"] == ""
    assert "context_policy_override" not in config.agent_kwargs
    assert config.agent_kwargs["model_settings"].max_tokens == 2048
    assert config.agent_kwargs["model_settings"].num_retries == 0
    assert config.max_workers == 1
    assert config.task_ids == ["test-task"]


def test_seed_aggregate_reports_sample_std_and_separates_execution_errors(tmp_path):
    groups = [{"group": f"gepa/{seed}", "method": "gepa", "mode": "sequential", "checkpoint": 50,
               "seed": seed} for seed in (42, 44)]
    records = {"gepa/42": [{"status": "completed", "score": 1, "partial_score": 1}] * 50,
               "gepa/44": [{"status": "completed", "score": 0, "partial_score": 0}] * 50}
    runner.save_aggregates(tmp_path, groups, records)
    row = runner.shared.read_json(tmp_path / "scores.json")["aggregates"][0]
    assert row["seed_count"] == 2
    assert row["mean"] == 0.5
    assert row["std"] == pytest.approx(statistics.stdev([1, 0]))
    records["gepa/44"][0] = {"status": "error", "score": 0, "partial_score": 0}
    runner.save_aggregates(tmp_path, groups, records)
    row = runner.shared.read_json(tmp_path / "scores.json")["aggregates"][0]
    assert row["mean"] is None
    assert row["mean_with_errors_zero"] == 0.5
    assert row["execution_errors"] == 1


def test_priority_bypasses_predecessor_wait_only_when_explicit(tmp_path, monkeypatch):
    group = {"group": "priority", "method": "gepa", "mode": "sequential", "checkpoint": 130,
             "seed": 42, "checkpoint_sha256": "checkpoint", "prompt_sha256": "prompt",
             "task_ids": [f"task-{index}" for index in range(50)]}
    for index, task_id in enumerate(group["task_ids"]):
        runner.shared.write_json(tmp_path / "results/priority" / f"{index:03d}.json", {
            "task_id": task_id, "checkpoint_sha256": "checkpoint", "prompt_sha256": "prompt",
            "status": "completed", "score": 1.0, "partial_score": 1.0,
        })
    launch = {"upstream": "/unfinished-predecessor", "upstream_launch_sha256": "hash", "workers": 150,
              "base_url": "http://localhost/v1", "total_tasks": 50, "checkpoint_count": 1, "groups": [group]}

    def wait_for_unfinished(*args):
        raise RuntimeError("Predecessor still running")

    monkeypatch.setattr(runner, "predecessor_ready", wait_for_unfinished)
    monkeypatch.setattr(runner, "ProcessPoolExecutor", lambda **kwargs: nullcontext())
    with pytest.raises(RuntimeError, match="Predecessor still running"):
        runner.run(tmp_path, launch)
    assert runner.run(tmp_path, launch, wait_for_predecessor=False) == 0
    assert runner.shared.read_json(tmp_path / "status.json")["phase"] == "complete"


def test_remaining_team_queue_waits_only_for_priority_completion_and_controller_exit(tmp_path):
    runner.shared.write_json(tmp_path / "launch_manifest.json", {"total_tasks": 600, "checkpoint_count": 12})
    (tmp_path / "controller.lock").touch()
    runner.shared.write_json(tmp_path / "status.json", {"phase": "running"})
    dependency = {"output": str(tmp_path), "total_tasks": 600, "checkpoint_count": 12,
                  "launch_sha256": runner.shared.digest((tmp_path / "launch_manifest.json").read_bytes())}
    assert not runner.priority_ready(tmp_path, dependency)[0]
    runner.shared.write_json(tmp_path / "appworld_test_complete.json", {
        "all_tasks_terminal": True, "processed_tasks": 600, "checkpoint_count": 12, "execution_errors": 5,
    })
    runner.shared.write_json(tmp_path / "status.json", {"phase": "complete"})
    with (tmp_path / "controller.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not runner.priority_ready(tmp_path, dependency)[0]
    assert runner.priority_ready(tmp_path, dependency)[0]


def test_explicit_hold_prevents_team_cli_from_launching_workers(tmp_path, monkeypatch):
    output = tmp_path / "paused"
    runner.shared.write_json(output / "automatic_resume_disabled.json", {"reason": "user requested stop"})
    runner.shared.write_json(output / "status.json", {"phase": "paused", "processed_tasks": 1360})

    def unexpected(*args, **kwargs):
        raise AssertionError("Paused evaluation must not prepare or launch tasks")

    monkeypatch.setattr(runner, "prepare", unexpected)
    monkeypatch.setattr(runner, "run", unexpected)
    monkeypatch.setattr(sys, "argv", ["evaluation", "--source", str(tmp_path / "source"),
                                      "--output", str(output), "--after", str(tmp_path / "after")])
    assert runner.main() == 0
    state = runner.shared.read_json(output / "status.json")
    assert state["phase"] == "paused"
    assert state["processed_tasks"] == 1360
    assert state["active_tasks"] == 0
    assert state["automatic_resume_disabled"]
