# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Matched cohorts and rolling seed lanes must retain fresh state and valid final tests."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

from scripts.utils import launch_peace_span_policy_shift as runner
from scripts.utils.continue_epoch2 import completed_job


def baseline_fixture(tmp_path):
    return {
        "jobs": [
            {
                "id": f"peace_no_alignment_aud_bfcl_s{seed}",
                "seed": seed,
                "method": "premise",
                "variant": "peace_no_alignment_aud",
                "benchmark": "bfcl",
                "source_plan": str(tmp_path / "old.json"),
                "epoch1_order": [f"task{i}" for i in range(50)],
                "epoch2_order": [f"task{i}" for i in reversed(range(50))],
                "test_ids": ["test1", "test2"],
                "playbook_edit_mode": "add_update_delete",
                "source_spec": {
                    "splits": {"train": [f"task{i}" for i in range(50)], "val": ["val"], "test": ["test1", "test2"]}
                },
            }
            for seed in runner.SEEDS
        ]
    }


def test_nine_runs_match_baseline_orders_and_only_change_past_count(tmp_path):
    baseline = baseline_fixture(tmp_path)
    original = copy.deepcopy(baseline)
    jobs = runner.build_jobs(baseline, tmp_path / "output", tmp_path / "frozen", tmp_path / "runner.py")
    assert len(jobs) == 9
    assert {job["past_count"] for job in jobs} == {3, 6, 9}
    for seed in runner.SEEDS:
        lane = [job for job in jobs if job["seed"] == seed]
        assert [job["past_count"] for job in lane] == [9, 3, 6]
        assert [job["predecessor"] for job in lane] == [None, lane[0]["id"], lane[1]["id"]]
        assert len({tuple(job["train_ids"]) for job in lane}) == 1
        assert all(job["train_ids"] == lane[0]["epoch1_order"] + lane[0]["epoch2_order"] for job in lane)
    assert all(job["current_count"] == 1 and job["replay_workers"] == job["llm_calls"] == 28 for job in jobs)
    assert all(job["from_scratch"] and job["restored_tasks"] == 0 for job in jobs)
    assert all(job["temperature_all_roles"] == job["policy_shift_temperature"] == 0 for job in jobs)
    assert len({job["output"] for job in jobs}) == 9
    assert baseline == original


def test_snapshot_keeps_baseline_actors_and_fixed_order_seed_behavior(tmp_path):
    source = tmp_path / "baseline"
    for relative in runner.ACTOR_FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Baseline bytes: {relative}\n")
    baseline_pipeline = source / "src/exgentic/agents/premise/pipeline.py"
    baseline_pipeline.write_text("def manifest_order(*args):\n    return 'baseline fixed order'\n")
    for name in (".venv", ".exgentic"):
        (source / name).mkdir()
    target = tmp_path / "frozen"
    audit = runner.prepare_snapshot(source, target)
    assert all(row["baseline_sha256"] == row["run_sha256"] for row in audit.values())
    code = (target / "src/exgentic/agents/premise/pipeline.py").read_text()
    function = next(
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.FunctionDef) and node.name == "manifest_order"
    )
    assert ast.get_source_segment(code, function) == baseline_pipeline.read_text().strip()
    assert 'self.algorithm_seed = int(json.loads(self.manifest.read_text()).get("algorithm_seed", seed))' in code
    assert '"algorithm_seed": self.algorithm_seed' in code
    assert "span_policy_shift" in code


def test_seed_lane_rolls_immediately_after_own_full_test_without_waiting_for_other_seeds(tmp_path, monkeypatch):
    output = tmp_path / "output"
    jobs = runner.build_jobs(baseline_fixture(tmp_path), output, tmp_path / "frozen", tmp_path / "runner.py")
    plan = {"jobs": jobs, "max_parallel": 3, "environment": {}}
    for job in jobs:
        job["base_url"] = "http://localhost/v1"
        Path(job["train_dir"]).mkdir(parents=True)
    runner.write(output / "progress.json", {"jobs": {job["id"]: {"status": "pending"} for job in jobs}})
    events = []

    class Process:
        def __init__(self, command, **_kwargs):
            self.job = next(job for job in jobs if job["id"] == command[command.index("--job") + 1])
            self.pid = 1000 + len(events)
            self.polls = 0
            predecessor = self.job["predecessor"]
            if predecessor:
                assert completed_job(next(job for job in jobs if job["id"] == predecessor))
            events.append(("start", self.job["past_count"], self.job["seed"]))

        def poll(self):
            self.polls += 1
            job = self.job
            runner.write(Path(job["train_dir"]) / "state.json", {"cursor": 100})
            threshold = {42: 2, 43: 4, 44: 6}[job["seed"]] if job["past_count"] == 9 else 1
            if self.polls < threshold:
                runner.write(
                    Path(job["output"]) / "job_complete.json",
                    {
                        "complete": False,
                        "job_id": job["id"],
                        "test_ids": job["test_ids"],
                        "tasks": 1,
                        "infrastructure_errors": 0,
                    },
                )
                return None
            runner.write(
                Path(job["output"]) / "job_complete.json",
                {
                    "complete": True,
                    "job_id": job["id"],
                    "test_ids": job["test_ids"],
                    "tasks": 2,
                    "infrastructure_errors": 0,
                },
            )
            events.append(("finish", job["past_count"], job["seed"]))
            return 0

    helper = SimpleNamespace(
        completed_job=completed_job,
        existing_worker=lambda *_a: None,
        process_identity=lambda _pid: {"start_ticks": "1"},
    )
    monkeypatch.setattr(runner, "module_file", lambda *_a: helper)
    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    runner.run(output, plan)
    assert events[:3] == [("start", 9, seed) for seed in runner.SEEDS]
    assert events.index(("start", 3, 42)) < events.index(("finish", 9, 43))
    assert events.index(("start", 6, 42)) < events.index(("finish", 9, 44))
    for seed in runner.SEEDS:
        assert events.index(("finish", 9, seed)) < events.index(("start", 3, seed))
        assert events.index(("finish", 3, seed)) < events.index(("start", 6, seed))
    assert runner.read(output / "status.json")["phase"] == "completed"


def test_failed_predecessor_cannot_release_next_setting(tmp_path):
    jobs = runner.build_jobs(baseline_fixture(tmp_path), tmp_path / "out", tmp_path / "code", tmp_path / "runner")
    progress = {job["id"]: {"status": "pending"} for job in jobs}
    first = next(job for job in jobs if job["seed"] == 42 and job["past_count"] == 9)
    progress[first["id"]]["status"] = "error"
    assert {job["seed"] for job in runner.eligible_jobs({"jobs": jobs}, progress)} == {43, 44}
