# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Queue three fresh evidence-reviewed A+B runs after the three matched PEACE baselines."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
if (Path(__file__).parent / "generation_helpers.py").exists():
    spec = importlib.util.spec_from_file_location(
        "grounded_generation", Path(__file__).parent / "generation_helpers.py")
    generation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generation)
else:
    from scripts.utils import launch_peace_generator_only as generation

shared = generation.shared


def baseline_jobs(plan):
    jobs = [job for job in plan["jobs"] if job["variant"] == "peace_no_alignment_aud_g0_analysis07"]
    if len(jobs) != 3 or {job["seed"] for job in jobs} != {42, 43, 44}:
        raise ValueError("Expected the three matched two-epoch role-temperature PEACE baselines")
    # Once these finish, only three serial policy-shift lanes can remain in that queue.
    lanes = [job for job in plan["jobs"] if job not in jobs and not job.get("predecessor")]
    if len(lanes) != 3 or plan.get("max_parallel") != 6:
        raise ValueError("Cannot prove the combined six-experiment concurrency bound")
    others = [job for job in plan["jobs"] if job not in jobs]
    for seed in (42, 43, 44):
        lane = {job.get("past_count"): job for job in others if job["seed"] == seed}
        if set(lane) != {9, 3, 6} or len(others) != 9:
            raise ValueError("Expected exactly three serial policy-shift lanes")
        if (lane[9].get("predecessor") is not None or lane[3].get("predecessor") != lane[9]["id"]
                or lane[6].get("predecessor") != lane[3]["id"]):
            raise ValueError("Policy-shift lanes are not serial")
    return sorted(jobs, key=lambda job: job["seed"])


def build_jobs(plan, output, snapshot, runner, embedding_url):
    originals = baseline_jobs(plan)
    predecessors = [{key: job[key] for key in ("id", "method", "output", "train_dir", "train_count", "test_ids")}
                    for job in originals]
    jobs = []
    for index, original in enumerate(originals):
        job = copy.deepcopy(original)
        variant = "peace_grounded_ab_aud_g0_analysis07"
        directory = output / "runs" / variant / "bfcl" / f"s{job['seed']}"
        job.update(id=f"{variant}_bfcl_s{job['seed']}", variant=variant, output=str(directory),
                   train_dir=str(directory / "train"), source_snapshot=str(snapshot),
                   source_plan=str(output / "launch_manifest.json"), grounded_proposals=True,
                   grounded_retrieval="ab", proposal_embedding_url=embedding_url, predecessor=None,
                   external_predecessors=predecessors, status="pending", port_block_start=22000 + index * 500)
        for key in ("source_train_dir", "external_predecessor", "pid", "started_at", "start_ticks"):
            job.pop(key, None)
        job["source_spec"].update(id=hashlib.sha256(str(directory).encode()).hexdigest()[:20], root=str(output),
                                  output=str(directory), source_output=str(directory), train_dir=job["train_dir"],
                                  variant=output.name, run_variant=variant, snapshot=str(snapshot),
                                  grounded_proposals=True)
        job["train_command"] = [str(snapshot / ".venv/bin/python"), "-u", str(runner),
                                "train", "--output", str(output), "--job", job["id"]]
        job["command"] = [str(REPO / ".venv/bin/python"), "-u", str(runner),
                          "worker", "--output", str(output), "--job", job["id"]]
        jobs.append(job)
    return jobs


def prepare(output, source, embedding_url):
    output, source = output.resolve(), source.resolve()
    if (output / "launch_manifest.json").exists():
        saved = shared.read(output / "launch_manifest.json")
        if saved["comparison_source"] != str(source) or saved["proposal_embedding_url"] != embedding_url:
            raise ValueError("Existing grounded experiment configuration differs")
        return saved
    baseline = shared.read(source / "launch_manifest.json")
    originals = baseline_jobs(baseline)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "code_snapshot"
    shared.prepare_snapshot(Path(originals[0]["source_snapshot"]), snapshot)
    for relative in ("src/exgentic/agents/premise/grounded_proposals.py", "src/exgentic/agents/premise/llm.py",
                     "scripts/premise/serve_proposal_embeddings.py"):
        shutil.copy2(REPO / relative, snapshot / relative)
    audit = shared.read(output / "baseline_actor_audit.json")
    relative = "src/exgentic/agents/premise/llm.py"
    audit[relative].update(run_sha256=shared.sha(snapshot / relative),
                           change="Explicit system prompt for independent evidence/retrieval/proposal review stages")
    shared.write(output / "baseline_actor_audit.json", audit)
    runner = output / "grounded_runner.py"
    shutil.copy2(__file__, runner)
    shutil.copy2(REPO / "scripts/utils/launch_peace_generator_only.py", output / "generation_helpers.py")
    shutil.copy2(REPO / "scripts/utils/launch_peace_span_policy_shift.py", output / "runtime_helpers.py")
    manifest = output / "task_splits.json"
    shutil.copy2(baseline["manifest"], manifest)
    jobs = build_jobs(baseline, output, snapshot, runner, embedding_url)
    for job in jobs:
        folder = Path(job["output"])
        (folder / "train").mkdir(parents=True, exist_ok=False)
        shared.write(folder / "fresh_start_provenance.json", {
            "from_scratch": True, "restored_tasks": 0, "copied_training_artifacts": [],
            "role_temperatures": job["role_temperatures"], "epochs": 2, "train_ids": job["train_ids"],
            "grounded_proposals": True, "retrieval": "ab", "candidate_evaluation": "environment_replay",
            "past_count": 3, "external_predecessors": job["external_predecessors"],
        })
    plan = {key: baseline[key] for key in ("model", "mode", "epochs", "from_scratch", "role_temperatures",
                                         "temperature", "playbook_edit_mode", "base_url", "environment")}
    plan.update(created_at=shared.now(), comparison_source=str(source), after_output=str(source),
                max_parallel=3, global_experiment_limit=6, proposal_embedding_url=embedding_url,
                manifest=str(manifest), manifest_sha256=shared.sha(manifest),
                completion_gate="All three matched baselines finish full train and final Test; shift lanes <=3",
                final_evaluation_splits=["train", "val", "test"], jobs=jobs)
    shared.write(output / "launch_manifest.json", plan)
    shared.write(output / "protocol_public.json", {key: value for key, value in plan.items() if key != "environment"})
    shared.write(output / "progress.json", {"jobs": {job["id"]: {"status": "pending"} for job in jobs}})
    shared.write(output / "status.json", {"phase": "planned", "at": shared.now()})
    shared.write(output / "code_hashes.json", {
        "snapshot": {str(path.relative_to(snapshot)): shared.sha(path)
                     for name in ("src", "scripts") for path in (snapshot / name).rglob("*.py")},
        "runner_sha256": shared.sha(runner), "runtime_helpers_sha256": shared.sha(output / "runtime_helpers.py"),
        "generation_helpers_sha256": shared.sha(output / "generation_helpers.py"),
    })
    return plan


def train(plan, job, check=False):
    arguments = generation.train_arguments

    def grounded_arguments(plan, job):
        return [*arguments(plan, job), "--grounded-proposals", "--grounded-retrieval", job["grounded_retrieval"],
                "--proposal-embedding-url", job["proposal_embedding_url"]]

    generation.train_arguments = grounded_arguments
    generation.train(plan, job, check=check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "worker", "train", "check"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--embedding-url", default="http://127.0.0.1:18886")
    parser.add_argument("--job")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "prepare":
        plan = prepare(output, args.source, args.embedding_url)
        print(json.dumps({"jobs": len(plan["jobs"]), "max_parallel": 3, "global_limit": 6}))
        return
    plan = shared.read(output / "launch_manifest.json")
    if args.action == "run":
        shared.REPO = REPO
        shared.run(output, plan)
    else:
        job = next(job for job in plan["jobs"] if job["id"] == args.job)
        if args.action == "worker":
            os.environ.update(shared.environment(plan, job))
            shared.worker(plan, job)
        else:
            train(plan, job, check=args.action == "check")


if __name__ == "__main__":
    main()
