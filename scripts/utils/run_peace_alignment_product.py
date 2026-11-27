# SPDX-License-Identifier: Apache-2.0
"""Frozen one-parent/four-child PEACE cohort with independent train/Test slots."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import psutil

from scripts.utils import run_shared_reflection_matrix as shared

ROOT = Path(__file__).resolve().parents[2]
VARIANT = "peace_alignment_product_aud"


def prepare(source: Path, output: Path, *, appworld_pct_comparison: bool = False,
            auxiliary_comparison: bool = False):
    if appworld_pct_comparison and auxiliary_comparison:
        raise ValueError("Select one explicitly separated metric cohort")
    if output.exists():
        raise ValueError("Output already exists; use run to resume its controller")
    previous = shared.read(source / "launch_manifest.json")
    if shared.sha(previous["inference_profile"]) != previous["inference_profile_sha256"]:
        raise ValueError("Source inference profile differs from its pinned hash")
    output.mkdir(parents=True)
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for name in ("src", "scripts"):
        shutil.copytree(ROOT / name, snapshot / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "outputs", "installation"))
    for installation in (ROOT / "src/exgentic/benchmarks").glob("*/installation"):
        (snapshot / installation.relative_to(ROOT)).symlink_to(installation.resolve(), target_is_directory=True)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, snapshot / name)
    for name in (".venv", ".exgentic"):
        (snapshot / name).symlink_to((ROOT / name).resolve(), target_is_directory=True)
    for name in ("task_splits.json", "inference_profile.json", "appworld_original_policy.txt",
                 "private_environment.json"):
        shutil.copy2(source / name, output / name)
    os.chmod(output / "private_environment.json", 0o600)
    if shared.sha(snapshot / "src/exgentic/reflection.py") != previous["profile"]["reflection_template_sha256"]:
        raise ValueError("Shared reflector changed; cannot claim matched settings")
    plan = copy.deepcopy(previous)
    for name in ("output", "snapshot", "manifest", "inference_profile", "appworld_policy"):
        plan[name] = str(plan[name]).replace(str(source), str(output))
    plan.update(version="peace-alignment-product-v1", created_at=shared.now(), max_parallel=3,
                max_evaluation_parallel=3, source_cohort=str(source), seed_stagger_seconds=0,
                scheduling="rolling train slots=3; freeze/enqueue Test immediately; refill train independently",
                port_blocks=[24000, 25000, 26000, 27000, 28000, 29000],
                candidate_selection="alignment_product", parent_count=1, child_count=4,
                selection_aggregation="mean_over_current_and_each_past_replay_then_product",
                parent_in_selection_pool=False, mode="isolated")
    if appworld_pct_comparison:
        plan.update(version="peace-appworld-pct-comparison-v1", benchmark_order=["appworld"],
                    candidate_selection="pareto_vs_alignment_product", appworld_candidate_metric="pct",
                    appworld_reporting_metric="appworld_tgc_v1",
                    port_blocks=[30000, 31000, 32000, 33000, 34000, 35000],
                    search_topologies={"pareto": {"parents": 2, "children_per_parent": 2, "survivors": 2},
                                       "alignment_product": {"parents": 1, "children_per_parent": 4, "survivors": 1}})
        plan.pop("parent_count")
    if auxiliary_comparison:
        plan.update(version="peace-auxiliary-comparison-v1", benchmark_order=["bfcl", "terminalbench2"],
                    candidate_selection="pareto_vs_alignment_product", benchmark_candidate_metric="auxiliary",
                    reporting_metric="official_binary_and_auxiliary", parent_in_selection_pool=False,
                    auxiliary_metrics={"bfcl": "bfcl_turn_pass_rate_v1",
                                       "terminalbench2": "terminalbench_test_pass_rate_v1"},
                    port_blocks=[24000, 25000, 26000, 27000, 28000, 29000],
                    search_topologies={"pareto": {"parents": 2, "children_per_parent": 2, "survivors": 2},
                                       "alignment_product": {"parents": 1, "children_per_parent": 4, "survivors": 1}})
        plan.pop("parent_count")
    jobs = []
    for old in previous["jobs"]:
        if old["method"] != "premise":
            continue
        if appworld_pct_comparison and old["benchmark"] != "appworld":
            continue
        if auxiliary_comparison and old["benchmark"] not in {"bfcl", "terminalbench2"}:
            continue
        selections = ("pareto", "alignment_product") if (
            appworld_pct_comparison or auxiliary_comparison) else ("alignment_product",)
        for selection in selections:
            variant = (f"peace_aux_{selection}_aud" if auxiliary_comparison else
                       f"peace_pct_{selection}_aud" if appworld_pct_comparison else VARIANT)
            job = json.loads(json.dumps(old).replace(str(source), str(output)))
            job = json.loads(json.dumps(job).replace(old["variant"], variant))
            command = [arg for arg in job["train_command"] if arg != "--disable-alignment"]
            command += ["--candidate-selection", selection]
            if appworld_pct_comparison:
                command += ["--appworld-candidate-metric", "pct"]
                job["appworld_candidate_metric"] = "pct"
                job["appworld_reporting_metric"] = "appworld_tgc_v1"
            if auxiliary_comparison:
                command += ["--benchmark-candidate-metric", "auxiliary"]
                job["benchmark_candidate_metric"] = "auxiliary"
                job["candidate_score_metric"] = plan["auxiliary_metrics"][old["benchmark"]]
            job.update(train_command=command, command=command, release_delay_seconds=0,
                       candidate_selection=selection, alignment_scoring=True, status="pending")
            jobs.append(job)
    expected = [(benchmark, seed) for benchmark in shared.BENCHMARKS for seed in (42, 43, 44)]
    if appworld_pct_comparison:
        expected = [("appworld", seed) for seed in (42, 43, 44) for _ in range(2)]
    if auxiliary_comparison:
        expected = [(benchmark, seed) for benchmark in ("bfcl", "terminalbench2")
                    for seed in (42, 43, 44) for _ in range(2)]
    if [(job["benchmark"], job["seed"]) for job in jobs] != expected:
        raise ValueError("Expected BFCL/AppWorld/Terminal-Bench, seeds 42/43/44, in that order")
    plan["jobs"] = jobs
    shared.write(output / "launch_manifest.json", plan)
    shared.write(output / "source_sha256.json", {
        str(path.relative_to(snapshot)): shared.sha(path)
        for path in snapshot.rglob("*.py") if "installation" not in path.parts
    })
    return plan


def training_complete(job, plan):
    marker = Path(job["output"]) / "training_finished.json"
    if not marker.exists():
        return False
    value = shared.read(marker)
    if (value.get("job_id") != job["id"] or value.get("tasks") != job["train_count"]
            or value.get("inference_profile_sha256") != plan["inference_profile_sha256"]):
        raise ValueError("Training marker differs from this experiment")
    frozen = shared.read(Path(job["output"]) / "evaluation/frozen_checkpoint.json")
    if value.get("policy_sha256") != frozen["sha256"]:
        raise ValueError("Frozen final policy differs from the training marker")
    return True


def work(output, identity, stage, port):
    plan = shared.read(output / "launch_manifest.json")
    job = {**next(job for job in plan["jobs"] if job["id"] == identity), "port_block_start": port}
    plan = {**plan, "snapshot": job.get("snapshot", plan["snapshot"])}
    if port not in plan["port_blocks"]:
        raise ValueError("Port block outside this cohort's allocation")
    env = shared.environment(plan, job)
    os.environ.update(env)
    folder = Path(job["output"])
    folder.mkdir(parents=True, exist_ok=True)
    if stage == "train":
        if training_complete(job, plan):
            return
        shared.write(folder / "job_status.json", {"phase": "training", "at": shared.now(), "pid": os.getpid()})
        with (folder / "training.log").open("a") as log:
            subprocess.run(job["train_command"], cwd=plan["snapshot"], env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        _, _, frozen = shared.freeze_native(job)
        shared.write(folder / "training_finished.json", {
            "job_id": identity, "tasks": job["train_count"], "at": shared.now(),
            "inference_profile_sha256": plan["inference_profile_sha256"], "policy_sha256": frozen["sha256"],
        })
        shared.write(folder / "job_status.json", {"phase": "test_pending", "at": shared.now()})
    else:
        if not training_complete(job, plan):
            raise ValueError("Cannot evaluate before training freezes its final policy")
        if shared.complete(job, plan):
            return
        shared.write(folder / "job_status.json", {"phase": "test", "at": shared.now(), "pid": os.getpid()})
        summary = shared.finish_native(job)
        shared.write(folder / "job_complete.json", {
            **summary, "job_id": identity, "at": shared.now(),
            "inference_profile_sha256": plan["inference_profile_sha256"],
        })
        shared.write(folder / "job_status.json", {"phase": "completed", "at": shared.now(), "score": summary["score"]})


def alive(state):
    try:
        process = psutil.Process(state["pid"])
        return process.create_time() == state["created_at"] and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.Error, KeyError):
        return False


def tick(plan, progress):
    active = {"train": [], "test": []}
    for job in plan["jobs"]:
        state = progress["jobs"].setdefault(job["id"], {"status": "pending"})
        if state.get("status") in {"running", "evaluating"} and alive(state):
            active[state["stage"]].append(state)
            continue
        if shared.complete(job, plan):
            state.update(status="completed", ended_at=shared.now())
        elif training_complete(job, plan):
            if state.get("status") == "evaluating":
                state.update(status="evaluation_error", ended_at=shared.now())
            elif state.get("status") != "evaluation_error":
                state.update(status="evaluation_pending", training_ended_at=shared.now())
        elif state.get("status") == "running":
            state.update(status="training_error", ended_at=shared.now())
    used = {item["port_block_start"] for items in active.values() for item in items}
    # Fill training first: final Test execution never occupies a training slot.
    for stage, limit, pending in (("train", plan["max_parallel"], "pending"),
                                   ("test", plan["max_evaluation_parallel"], "evaluation_pending")):
        for job in plan["jobs"]:
            if len(active[stage]) >= limit:
                break
            state = progress["jobs"][job["id"]]
            if state["status"] != pending:
                continue
            port = next(port for port in plan["port_blocks"] if port not in used)
            folder = Path(job["output"])
            folder.mkdir(parents=True, exist_ok=True)
            runtime = {**plan, "snapshot": job.get("snapshot", plan["snapshot"])}
            command = [str(Path(runtime["snapshot"]) / ".venv/bin/python"), "-u", "-m",
                       "scripts.utils.run_peace_alignment_product", "work", "--output", plan["output"],
                       "--job", job["id"], "--stage", stage, "--port-block-start", str(port)]
            with (folder / f"{stage}_worker.log").open("a") as log:
                process = subprocess.Popen(command, cwd=runtime["snapshot"],
                                           env=shared.environment(runtime, {**job, "port_block_start": port}),
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status="running" if stage == "train" else "evaluating", stage=stage,
                         pid=process.pid, created_at=psutil.Process(process.pid).create_time(),
                         port_block_start=port, started_at=shared.now(), worker_command=command)
            active[stage].append(state)
            used.add(port)
            shared.write(Path(plan["output"]) / "progress.json", progress)
    return {stage: len(items) for stage, items in active.items()}


def run(output):
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = shared.read(output / "launch_manifest.json")
        for path, key in (("inference_profile", "inference_profile_sha256"), ("manifest", "manifest_sha256"),
                          ("appworld_policy", "appworld_policy_sha256")):
            if shared.sha(plan[path]) != plan[key]:
                raise ValueError(f"Pinned {path} changed")
        shared.write(output / "controller_process.json", {
            "pid": os.getpid(), "created_at": psutil.Process().create_time(), "output": str(output),
        })
        progress_path = output / "progress.json"
        progress = shared.read(progress_path) if progress_path.exists() else {"jobs": {}}
        while True:
            counts = tick(plan, progress)
            shared.write(progress_path, progress)
            statuses = [item["status"] for item in progress["jobs"].values()]
            done = all(status == "completed" for status in statuses)
            errors = [key for key, item in progress["jobs"].items() if item["status"].endswith("_error")]
            shared.write(output / "status.json", {
                "phase": "completed" if done else "running", "at": shared.now(),
                "active_training": counts["train"], "active_evaluations": counts["test"],
                "completed": statuses.count("completed"), "errors": errors,
            })
            if done:
                return
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "prepare-appworld-pct", "prepare-auxiliary", "run", "work"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--job")
    parser.add_argument("--stage", choices=("train", "test"))
    parser.add_argument("--port-block-start", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action in ("prepare", "prepare-appworld-pct", "prepare-auxiliary"):
        plan = prepare(args.source.resolve(), output, appworld_pct_comparison=args.action == "prepare-appworld-pct",
                       auxiliary_comparison=args.action == "prepare-auxiliary")
        print(json.dumps({"output": str(output), "jobs": len(plan["jobs"]), "max_parallel": plan["max_parallel"]}))
    elif args.action == "run":
        run(output)
    else:
        work(output, args.job, args.stage, args.port_block_start)


if __name__ == "__main__":
    main()
