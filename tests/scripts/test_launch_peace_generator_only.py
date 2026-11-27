# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""The reduced rerun queue contains exactly baseline and policy-shift methods."""

import copy

from scripts.utils import launch_peace_generator_only as runner
from scripts.utils.launch_peace_span_policy_shift import write


def test_twelve_fresh_jobs_keep_orders_and_per_seed_past_lanes(tmp_path):
    baseline = {"jobs": [{
        "id": f"peace_no_alignment_aud_bfcl_s{seed}", "variant": "peace_no_alignment_aud", "seed": seed,
        "method": "premise", "benchmark": "bfcl", "source_plan": str(tmp_path / "original.json"),
        "epoch1_order": [f"t{i}" for i in range(50)], "epoch2_order": [f"t{i}" for i in reversed(range(50))],
        "source_spec": {"splits": {}}, "source_train_dir": "must_not_resume", "temperature_all_roles": 0.0,
    } for seed in (42, 43, 44)]}
    shifts = {"jobs": []}
    for past in (9, 3, 6):
        for source in baseline["jobs"]:
            job = copy.deepcopy(source)
            job.update(id=f"shift_p{past}_s{source['seed']}", variant=f"peace_span_policy_shift_aud_p{past}",
                       past_count=past, candidate_evaluation="span_policy_shift",
                       predecessor=None if past == 9 else f"shift_p{9 if past == 3 else 3}_s{source['seed']}")
            shifts["jobs"].append(job)
    jobs = runner.build_jobs(baseline, shifts, tmp_path / "output", tmp_path / "snapshot", tmp_path / "runner.py")
    assert len(jobs) == len({job["id"] for job in jobs}) == 12
    assert sum(job["candidate_evaluation"] == "environment_replay" for job in jobs) == 3
    assert sum(job["candidate_evaluation"] == "span_policy_shift" for job in jobs) == 9
    for job in jobs:
        assert job["proposal_retrieval"] == "none" and "source_train_dir" not in job
        assert "temperature_all_roles" not in job and job["role_temperatures"] == runner.ROLES
        assert job["train_count"] == len(job["train_ids"]) == 100
        assert job["from_scratch"] and job["restored_tasks"] == 0
        if job["predecessor"]:
            assert job["predecessor"] in {item["id"] for item in jobs}


def test_training_arguments_remove_the_global_override_and_set_all_three_roles(tmp_path):
    write(tmp_path / "original.json", {"jobs": [{"id": "template", "train_command": [
        "python", "-u", "run.py", "--temperature", "0.0", "--resume-sessions",
        "--output-dir", "old", "--task-split-manifest", "old", "--base-url", "old",
        "--past", "3", "--replay-workers", "16", "--llm-calls", "32"]}]})
    job = {"template_source_plan": str(tmp_path / "original.json"), "template_job_id": "template",
           "train_dir": "new", "base_url": "http://localhost", "past_count": 9,
           "candidate_evaluation": "span_policy_shift"}
    argv = runner.train_arguments({"manifest": "newsplit"}, job)
    assert "--temperature" not in argv and "--resume-sessions" not in argv
    assert argv[argv.index("--actor-temperature") + 1] == "0.0"
    assert argv[argv.index("--reflection-temperature") + 1] == "0.7"
    assert argv[argv.index("--mutation-temperature") + 1] == "0.7"
    assert argv[argv.index("--past") + 1] == "9"
