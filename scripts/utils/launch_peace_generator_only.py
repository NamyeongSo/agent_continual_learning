# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Fresh two-epoch BFCL PEACE and policy-shift runs with independent role temperatures."""

from __future__ import annotations

import argparse
import ast
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
if (Path(__file__).parent / "runtime_helpers.py").exists():
    spec = importlib.util.spec_from_file_location(
        "generator_only_helpers", Path(__file__).parent / "runtime_helpers.py")
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
else:
    from scripts.utils import launch_peace_span_policy_shift as shared

ROLES = {"actor": 0.0, "reflection": 0.7, "mutation": 0.7}


def build_jobs(baseline, shifts, output, snapshot, runner):
    originals = [job for job in baseline["jobs"] if job.get("variant") == "peace_no_alignment_aud"]
    originals += shifts["jobs"]
    if len(originals) != 12 or {job["seed"] for job in originals} != {42, 43, 44}:
        raise ValueError("Expected three baseline and nine policy-shift jobs")
    mapping = {job["id"]: job["id"] + "_g0_analysis07" for job in originals}
    jobs = []
    for old in originals:
        job = copy.deepcopy(old)
        variant = old["variant"] + "_g0_analysis07"
        directory = output / "runs" / variant / "bfcl" / f"s{old['seed']}"
        predecessor = old.get("predecessor")
        for key in ("source_train_dir", "source_upstream", "temperature_all_roles", "external_predecessor"):
            job.pop(key, None)
        job.update(
            id=mapping[old["id"]], variant=variant, output=str(directory), train_dir=str(directory / "train"),
            source_snapshot=str(snapshot), source_plan=str(output / "launch_manifest.json"),
            template_source_plan=old.get("template_source_plan", old["source_plan"]),
            template_job_id=old.get("template_job_id", old["id"]),
            role_temperatures=ROLES, actor_temperature=0.0, temperature=0.0,
            candidate_evaluation=old.get("candidate_evaluation", "environment_replay"),
            past_count=old.get("past_count", 3), current_count=1, policy_shift_temperature=0.0,
            proposal_retrieval="none", epochs=2, epoch1_count=50, train_count=100,
            train_ids=old["epoch1_order"] + old["epoch2_order"],
            predecessor=mapping[predecessor] if predecessor else None,
            from_scratch=True, restored_tasks=0, status="pending", replay_workers=28, llm_calls=28,
            test_workers=28, alignment_scoring=False, release_delay_seconds=0,
            port_block_start=14000 + 500 * len(jobs),
        )
        if len(job["train_ids"]) != 100 or len(set(job["train_ids"])) != 50:
            raise ValueError("Expected two complete epochs over the same fifty tasks")
        job["source_spec"].update(
            id=hashlib.sha256(str(directory).encode()).hexdigest()[:20], root=str(output),
            output=str(directory), source_output=str(directory), train_dir=job["train_dir"],
            variant=output.name, run_variant=variant, snapshot=str(snapshot), temperature=0.0,
            train_total=100, role_temperatures=ROLES, candidate_evaluation=job["candidate_evaluation"],
        )
        job["train_command"] = [str(snapshot / ".venv/bin/python"), "-u", str(runner),
                                "train", "--output", str(output), "--job", job["id"]]
        job["command"] = [str(REPO / ".venv/bin/python"), "-u", str(runner),
                          "worker", "--output", str(output), "--job", job["id"]]
        jobs.append(job)
    return jobs


def prepare(output, baseline_root, shift_root):
    output, baseline_root, shift_root = output.resolve(), baseline_root.resolve(), shift_root.resolve()
    if (output / "launch_manifest.json").exists():
        plan = shared.read(output / "launch_manifest.json")
        if plan["role_temperatures"] != ROLES or plan["max_parallel"] != 6:
            raise ValueError("Existing rerun configuration differs")
        return plan
    baseline = shared.read(baseline_root / "launch_manifest.json")
    shifts = shared.read(shift_root / "launch_manifest.json")
    if baseline["manifest_sha256"] != shifts["manifest_sha256"]:
        raise ValueError("Comparison splits differ")
    output.mkdir(parents=True, exist_ok=True)
    original = next(job for job in baseline["jobs"] if job.get("variant") == "peace_no_alignment_aud")
    snapshot = output / "code_snapshot"
    shared.prepare_snapshot(Path(original["source_snapshot"]), snapshot)
    llm_relative = "src/exgentic/agents/premise/llm.py"
    source_llm = (Path(original["source_snapshot"]) / llm_relative).read_text()
    shutil.copy2(REPO / llm_relative, snapshot / llm_relative)

    def reflect_source(code):
        node = next(node for node in ast.walk(ast.parse(code))
                    if isinstance(node, ast.FunctionDef) and node.name == "reflect")
        return ast.get_source_segment(code, node)

    if reflect_source(source_llm) != reflect_source((snapshot / llm_relative).read_text()):
        raise ValueError("Reflection prompt changed")
    audit = shared.read(output / "baseline_actor_audit.json")
    audit[llm_relative].update(run_sha256=shared.sha(snapshot / llm_relative),
                               change="Independent analysis temperatures; unchanged reflection prompt")
    audit[llm_relative]["reflect_function_identical"] = True
    # Extend only final-checkpoint provenance validation in the frozen baseline helper.
    relative = "scripts/utils/run_qwen_isolated_when_free.py"
    path = snapshot / relative
    path.write_text(shared.replace_function(
        path.read_text(), (REPO / relative).read_text(), "frozen_evaluation_jobs"))
    audit[relative].update(run_sha256=shared.sha(path),
                           change="Allow actor-only temperature provenance in final freeze")
    shared.write(output / "baseline_actor_audit.json", audit)
    runner = output / "generator_only_runner.py"
    shutil.copy2(__file__, runner)
    shutil.copy2(REPO / "scripts/utils/launch_peace_span_policy_shift.py", output / "runtime_helpers.py")
    manifest = output / "task_splits.json"
    shutil.copy2(baseline["manifest"], manifest)
    jobs = build_jobs(baseline, shifts, output, snapshot, runner)
    for job in jobs:
        directory = Path(job["output"])
        (directory / "train").mkdir(parents=True, exist_ok=False)
        shared.write(directory / "fresh_start_provenance.json", {
            "from_scratch": True, "restored_tasks": 0, "copied_training_artifacts": [],
            "role_temperatures": ROLES, "epochs": 2, "train_ids": job["train_ids"],
            "past_count": job["past_count"], "candidate_evaluation": job["candidate_evaluation"],
            "predecessor": job["predecessor"],
        })
    plan = {
        "created_at": shared.now(), "model": baseline["model"], "mode": "isolated", "epochs": 2,
        "from_scratch": True, "role_temperatures": ROLES, "temperature": 0.0,
        "playbook_edit_mode": "add_update_delete", "max_parallel": 6,
        "manifest": str(manifest), "manifest_sha256": shared.sha(manifest),
        "comparison_sources": [str(baseline_root), str(shift_root)],
        "base_url": baseline["base_url"], "environment": baseline["environment"],
        "final_evaluation_splits": ["train", "val", "test"], "jobs": jobs,
    }
    shared.write(output / "launch_manifest.json", plan)
    shared.write(output / "protocol_public.json", {key: value for key, value in plan.items() if key != "environment"})
    shared.write(output / "progress.json", {"jobs": {job["id"]: {"status": "pending"} for job in jobs}})
    shared.write(output / "status.json", {"phase": "planned", "at": shared.now()})
    shared.write(output / "code_hashes.json", {
        "snapshot": {str(path.relative_to(snapshot)): shared.sha(path)
                     for name in ("src", "scripts") for path in (snapshot / name).rglob("*.py")},
        "runner_sha256": shared.sha(runner), "runtime_helpers_sha256": shared.sha(output / "runtime_helpers.py"),
    })
    return plan


def train_arguments(plan, job):
    template = next(row for row in shared.read(job["template_source_plan"])["jobs"]
                    if row["id"] == job["template_job_id"])
    argv = [arg for arg in template["train_command"][3:] if arg != "--resume-sessions"]
    if "--temperature" in argv:
        index = argv.index("--temperature")
        del argv[index:index + 2]
    for flag, value in (
        ("--output-dir", job["train_dir"]), ("--task-split-manifest", plan["manifest"]),
        ("--base-url", job["base_url"]), ("--past", str(job["past_count"])),
        ("--replay-workers", "28"), ("--llm-calls", "28"),
    ):
        argv[argv.index(flag) + 1] = value
    argv.extend(["--actor-temperature", "0.0", "--reflection-temperature", "0.7", "--mutation-temperature", "0.7",
                 "--candidate-evaluation", job["candidate_evaluation"], "--policy-shift-temperature", "0.0"])
    return argv


def train(plan, job, check=False):
    snapshot = Path(job["source_snapshot"])
    sys.path[:0] = [str(snapshot / "src"), str(snapshot)]
    from exgentic.agents.premise.pipeline import PremisePipeline

    restore = PremisePipeline._restore_or_initialize

    def restore_order(self):
        self.order = [("bfcl", task) for task in job["train_ids"]]
        return restore(self)

    PremisePipeline._restore_or_initialize = restore_order
    path = snapshot / "scripts/premise/run_experiment.py"
    sys.argv = [str(path), *train_arguments(plan, job), *(["--dry-run"] if check else [])]
    shared.module_file("generator_only_frozen_peace", path).main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "worker", "train", "check"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--shifts", type=Path)
    parser.add_argument("--job")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "prepare":
        plan = prepare(output, args.baseline, args.shifts)
        print(json.dumps({"jobs": len(plan["jobs"]), "max_parallel": plan["max_parallel"], "roles": ROLES}))
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
