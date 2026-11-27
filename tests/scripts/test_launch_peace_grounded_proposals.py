# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""The grounded queue may occupy only the three slots released by completed matched baselines."""

import copy
from pathlib import Path

import pytest

from scripts.utils import launch_peace_grounded_proposals as runner
from scripts.utils.continue_epoch2 import completed_job
from scripts.utils.launch_peace_span_policy_shift import eligible_jobs, write


def baseline(tmp_path):
    jobs = []
    for seed in (42, 43, 44):
        base = {"id": f"base{seed}", "variant": "peace_no_alignment_aud_g0_analysis07", "seed": seed,
                "method": "premise", "benchmark": "bfcl", "epochs": 2, "train_count": 100,
                "output": str(tmp_path / str(seed)), "train_dir": str(tmp_path / str(seed) / "train"),
                "test_ids": ["test"], "source_spec": {}, "predecessor": None}
        jobs.append(base)
        for past, prior in ((9, None), (3, 9), (6, 3)):
            jobs.append({**base, "id": f"shift{past}_{seed}", "variant": f"shift{past}", "past_count": past,
                         "predecessor": f"shift{prior}_{seed}" if prior else None})
    return {"jobs": jobs, "max_parallel": 6}


def test_three_fresh_runs_wait_for_all_three_complete_tests_not_just_own_seed(tmp_path):
    plan = baseline(tmp_path)
    jobs = runner.build_jobs(plan, tmp_path / "new", tmp_path / "code", tmp_path / "runner.py", "http://embed")
    assert len(jobs) == 3 and {job["seed"] for job in jobs} == {42, 43, 44}
    progress = {job["id"]: {"status": "pending"} for job in jobs}
    old = runner.baseline_jobs(plan)
    for index, source in enumerate(old):
        write(Path(source["train_dir"]) / "state.json", {"cursor": 100})
        write(Path(source["output"]) / "job_complete.json", {
            "complete": True, "job_id": source["id"], "test_ids": ["test"], "tasks": 1,
            "infrastructure_errors": 0})
        eligible = eligible_jobs({"jobs": jobs}, progress, completed=completed_job)
        assert len(eligible) == (3 if index == 2 else 0)
    write(Path(old[-1]["train_dir"]) / "state.json", {"cursor": 99})
    assert eligible_jobs({"jobs": jobs}, progress, completed=completed_job) == []
    assert all(job["grounded_proposals"] and job["grounded_retrieval"] == "ab" for job in jobs)


def test_rejects_a_source_queue_that_could_exceed_six_live_experiments(tmp_path):
    plan = baseline(tmp_path)
    bad = copy.deepcopy(plan)
    next(job for job in bad["jobs"] if job["id"] == "shift3_42")["predecessor"] = None
    with pytest.raises(ValueError, match="concurrency bound"):
        runner.baseline_jobs(bad)
    bad = copy.deepcopy(plan)
    next(job for job in bad["jobs"] if job["id"] == "shift6_42")["predecessor"] = "shift9_42"
    with pytest.raises(ValueError, match="not serial"):
        runner.baseline_jobs(bad)
