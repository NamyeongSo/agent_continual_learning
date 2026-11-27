# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""The retrieval ablation must use one fresh epoch, twelve distinct runs and real replay."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.utils import launch_peace_proposal_retrieval as runner
from scripts.utils import launch_peace_span_policy_shift as shared
from scripts.utils.evaluate_epoch2_final_splits import final_snapshot


@pytest.mark.parametrize("shift", [False, True])
def test_prepare_twelve_matched_fresh_single_epoch_jobs(tmp_path, monkeypatch, shift):
    source, output = tmp_path / "baseline", tmp_path / "new"
    shared.write(source / "task_splits.json", {"algorithm_seed": 42})
    baseline = {
        "model": "Qwen/Qwen3.6-35B-A3B",
        "base_url": "http://localhost/v1",
        "environment": {},
        "manifest": str(source / "task_splits.json"),
        "manifest_sha256": shared.sha(source / "task_splits.json"),
        "jobs": [
            {
                "id": f"peace_bfcl_s{seed}",
                "variant": "peace_no_alignment_aud",
                "method": "premise",
                "benchmark": "bfcl",
                "seed": seed,
                "source_snapshot": str(source / "code"),
                "source_plan": str(source / "old.json"),
                "epoch1_order": [f"t{i}" for i in range(50)],
                "epoch2_order": [f"t{i}" for i in reversed(range(50))],
                "test_ids": ["test"],
                "source_spec": {
                    "model": "openai/test",
                    "method": "premise",
                    "benchmark": "bfcl",
                    "seed": seed,
                    "splits": {"train": [f"t{i}" for i in range(50)], "val": ["val"], "test": ["test"]},
                },
            }
            for seed in [42, 43, 44]
        ],
    }
    shared.write(source / "launch_manifest.json", baseline)

    def snapshot(_source, target):
        for name in ["src/exgentic/agents/premise", "scripts/premise", "scripts/utils"]:
            (target / name).mkdir(parents=True, exist_ok=True)
        relative = "src/exgentic/agents/premise/llm.py"
        path = Path(_source) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((runner.REPO / relative).read_text())
        shared.write(target.parent / "baseline_actor_audit.json", {
            relative: {"baseline_sha256": shared.sha(path), "run_sha256": shared.sha(path)},
        })

    monkeypatch.setattr(shared, "prepare_snapshot", snapshot)
    kwargs = {}
    if shift:
        previous = {"manifest_sha256": baseline["manifest_sha256"], "jobs": [
            {**job, "id": f"{mode}_{job['seed']}", "proposal_retrieval": mode, "epochs": 1,
             "train_ids": job["epoch1_order"], "temperature_all_roles": 0,
             "output": str(tmp_path / "previous" / mode / str(job["seed"])),
             "train_dir": str(tmp_path / "previous" / mode / str(job["seed"]) / "train"), "train_count": 50}
            for mode in runner.MODES for job in baseline["jobs"]]}
        shared.write(tmp_path / "previous/launch_manifest.json", previous)
        kwargs = {"candidate_evaluation": "span_policy_shift", "past": 9, "after_output": tmp_path / "previous"}
    plan = runner.prepare(output, source, "http://localhost:18886", **kwargs)
    assert len(plan["jobs"]) == plan["max_parallel"] == 12
    assert plan["epochs"] == 1
    assert {job["proposal_retrieval"] for job in plan["jobs"]} == {"a", "b", "ab", "random"}
    assert len({job["output"] for job in plan["jobs"]}) == 12
    for job in plan["jobs"]:
        assert job["train_count"] == len(job["train_ids"]) == len(set(job["train_ids"])) == 50
        assert job["train_ids"] == job["epoch1_order"] and job["epoch2_order"] == []
        assert job["past_count"] == (9 if shift else 3) and job["replay_workers"] == job["llm_calls"] == 28
        assert job["candidate_evaluation"] == ("span_policy_shift" if shift else "environment_replay")
        assert not job["alignment_scoring"]
        assert job["predecessor"] is None and job["from_scratch"] and job["restored_tasks"] == 0
        if shift:
            assert job["external_predecessor"]["id"] == f"{job['proposal_retrieval']}_{job['seed']}"
    assert runner.prepare(output, source, "http://localhost:18886", **kwargs) == plan
    with pytest.raises(ValueError, match="configuration differs"):
        runner.prepare(output, source, "http://localhost:18886", past=6)


def test_external_predecessors_release_individually_with_full_train_and_test_evidence(tmp_path):
    from scripts.utils.continue_epoch2 import completed_job

    jobs = []
    for seed in (42, 43):
        old = {"id": f"old{seed}", "output": str(tmp_path / str(seed)), "method": "premise",
               "train_dir": str(tmp_path / str(seed) / "train"), "test_ids": ["t"], "train_count": 50}
        shared.write(Path(old["train_dir"]) / "state.json", {"cursor": 50})
        shared.write(Path(old["output"]) / "job_complete.json", {
            "complete": True, "job_id": old["id"], "test_ids": ["t"], "tasks": 1,
            "infrastructure_errors": 0 if seed == 42 else 1})
        jobs.append({"id": f"new{seed}", "predecessor": None, "external_predecessor": old})
    progress = {job["id"]: {"status": "pending"} for job in jobs}
    assert shared.eligible_jobs({"jobs": jobs}, progress) == []
    eligible = shared.eligible_jobs({"jobs": jobs}, progress, completed=completed_job)
    assert [job["id"] for job in eligible] == ["new42"]
    shared.write(tmp_path / "42/train/state.json", {"cursor": 49})
    assert shared.eligible_jobs({"jobs": jobs}, progress, completed=completed_job) == []


def test_single_epoch_final_evaluation_registers_the_same_unique_training_split(tmp_path):
    train = tmp_path / "run/train"
    job = {
        "method": "premise",
        "benchmark": "bfcl",
        "epochs": 1,
        "proposal_retrieval": "ab",
        "seed": 42,
        "variant": "peace_proposal_ab_aud",
        "train_count": 2,
        "train_dir": str(train),
        "output": str(train.parent),
        "source_snapshot": str(tmp_path / "code"),
        "epoch1_order": ["one", "two"],
        "epoch2_order": [],
        "source_spec": {"model": "openai/test", "splits": {"train": ["one", "two"], "val": ["v"], "test": ["t"]}},
    }
    shared.write(train / "state.json", {"cursor": 2, "states": {"bfcl": {"beam": [{"playbook": "selected"}]}}})
    spec, checkpoint = final_snapshot(job, tmp_path)
    assert spec["splits"]["train"] == ["one", "two"]
    assert checkpoint["point"] == 2
    assert "real-replay official-score Pareto" in checkpoint["selection"]
    assert not (Path(job["output"]) / "sessions").exists()
