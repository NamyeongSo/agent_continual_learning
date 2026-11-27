# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import json
from pathlib import Path

import pytest

from scripts.utils import retry_interleaved_queue_errors as recovery


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "original"
    root.mkdir()
    checkpoint = root / "cp110.json"
    checkpoint.write_text(json.dumps({
        "config": {"mode": "interleaved", "seed": 44}, "cursor": 110,
        "states": {"global": {"beam": [{"playbook": "frozen rank one policy"}]}},
    }))
    split = root / "task_split_manifest.json"
    split.write_text("{}")
    groups = []
    for benchmark, count in [("bfcl", 50), ("appworld", 50), ("terminalbench2", 29)]:
        group = {
            "group": f"premise/interleaved/s44/cp110/test/{benchmark}", "method": "premise",
            "mode": "interleaved", "seed": 44, "checkpoint": 110, "split": "test", "benchmark": benchmark,
            "task_ids": [f"{benchmark}-{index}" for index in range(count)],
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": recovery.shared.digest(checkpoint.read_bytes()),
            "playbook_sha256": recovery.shared.digest(b"frozen rank one policy"),
            "model": "openai/Qwen/Qwen3.6-35B-A3B", "temperature": 0.0, "model_num_retries": 5,
            "max_tokens": None if benchmark == "terminalbench2" else 8192,
            "max_steps": 100 if benchmark == "terminalbench2" else 50, "subset": "2.0",
        }
        groups.append(group)
        for index, task_id in enumerate(group["task_ids"]):
            failed = benchmark == "terminalbench2" and index == 2
            row = {key: value for key, value in group.items() if key != "task_ids"}
            row.update(index=index, task_id=task_id, status="error" if failed else "completed",
                       score=0.0 if failed else 1.0, partial_score=0.0 if failed else 1.0)
            recovery.shared.write_json(root / "results" / group["group"] / f"{index:03d}.json", row)
    recovery.shared.write_json(root / "launch_manifest.json", {
        "seed": 44, "mode": "interleaved", "split": "test", "groups": groups,
        "manifest_sha256": recovery.shared.digest(split.read_bytes()), "terminalbench2_enforce_session_timeouts": True,
    })
    recovery.shared.write_json(root / "status.json", {
        "phase": "complete", "processed_tasks": 129, "total_tasks": 129, "active_tasks": 0, "workers": 150,
    })
    return root


def test_recovery_only_retries_errors_with_unchanged_model_policy_and_official_deadlines(source, tmp_path):
    sources = recovery.sources_for([source])
    output = tmp_path / "recovery"
    jobs = recovery.discover_errors(sources, output)
    assert len(jobs) == 1
    job = next(iter(jobs.values()))
    assert job["task_id"] == "terminalbench2-2"
    assert job["index"] == 2
    original = Path(job["original_result_path"])
    assert Path(job["archived_original_result"]).read_bytes() == original.read_bytes()
    assert recovery.shared.digest(original.read_bytes()) == job["original_result_sha256"]
    config = recovery.runner.task_config(job, output, "recovery-test")
    settings = config.agent_kwargs["model_settings"].model_dump(exclude_none=True)
    assert settings["temperature"] == 0
    assert settings["num_retries"] == 5
    assert settings["max_tokens"] == 2048
    assert config.agent_kwargs["playbook"] == "frozen rank one policy"
    assert config.benchmark_kwargs["enforce_session_timeouts"]


def test_recovered_score_preserves_originals_and_the_full_129_task_denominator(source, tmp_path):
    sources = recovery.sources_for([source])
    output = tmp_path / "recovery"
    job = next(iter(recovery.discover_errors(sources, output).values()))
    original = Path(job["original_result_path"]).read_bytes()
    retry = {**job, "status": "completed", "score": 0.0, "partial_score": 0.0}
    recovery.shared.write_json(output / "results" / job["group"] / "002.json", retry)
    groups, rows = recovery.combined_records(sources, output)
    recovery.runner.save_scores(output, groups, rows)
    score = recovery.shared.read_json(output / "scores.json")["scores"][0]
    assert (score["processed"], score["total"], score["execution_errors"]) == (129, 129, 0)
    assert score["hard"] == 128 / 129
    assert score["by_benchmark"]["terminalbench2"]["total"] == 29
    assert Path(job["original_result_path"]).read_bytes() == original
    assert recovery.discover_errors(sources, output) == {f"{job['group']}/002": job}
    recovery.validate_retry(retry, job)
    retry["temperature"] = 1.0
    with pytest.raises(ValueError, match="frozen retry job"):
        recovery.validate_retry(retry, job)


def test_recovery_rejects_changed_original_settings(source, tmp_path):
    sources = recovery.sources_for([source])
    group = sources[0]["launch"]["groups"][-1]
    path = source / "results" / group["group"] / "002.json"
    row = recovery.shared.read_json(path)
    row["temperature"] = 1.0
    recovery.shared.write_json(path, row)
    with pytest.raises(ValueError, match="temperature"):
        recovery.discover_errors(sources, tmp_path / "recovery")


def test_recovery_reserves_concurrency_for_existing_evaluations(source):
    sources = recovery.sources_for([source])
    recovery.shared.write_json(source / "status.json", {
        "phase": "running", "processed_tasks": 126, "total_tasks": 129,
        "active_tasks": 3, "queued_tasks": 0, "workers": 150,
    })
    assert recovery.source_capacity(sources, 150) == (147, False)
    status = recovery.shared.read_json(source / "status.json")
    status.update(queued_tasks=4)
    recovery.shared.write_json(source / "status.json", status)
    assert recovery.source_capacity(sources, 150) == (0, False)
