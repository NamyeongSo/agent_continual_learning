# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Wait for six ACE runs, then evaluate frozen checkpoints on all three test sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint_recovery import completed_prefix  # noqa: E402
from run_experiment import BENCHMARK_REGISTRY  # noqa: E402

from exgentic.actor_context import require_matching_actor_context  # noqa: E402
from exgentic.agents.ace.playbook_store import PlaybookStore  # noqa: E402
from exgentic.core.types import ModelSettings  # noqa: E402
from exgentic.interfaces.lib.api import evaluate  # noqa: E402
from exgentic.interfaces.registry import load_agent, load_benchmark  # noqa: E402
from exgentic.utils.task_splits import select_task_split_sections, validate_task_split  # noqa: E402

BENCHMARKS = ("bfcl", "appworld", "terminalbench2")
STAGES = (50, 100, 130)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def build_plan(launch: dict) -> list[dict]:
    source = Path(launch["source_manifest"])
    if digest(source) != launch["source_manifest_sha256"]:
        raise ValueError("Task split manifest changed after training launch")
    manifest = json.loads(source.read_text())
    runs = {(item["mode"], item["seed"]) for item in launch["jobs"]}
    if runs != {(mode, seed) for mode in ("sequential", "interleaved") for seed in (42, 43, 44)}:
        raise ValueError("Expected exactly the six sequential/interleaved seed-42-44 runs")
    plan = []
    for run in launch["jobs"]:
        sections = select_task_split_sections(manifest, list(BENCHMARKS), run["seed"])
        splits = {slug: validate_task_split(sections[slug], slug) for slug in BENCHMARKS}
        for slug in BENCHMARKS:
            train = [str(task) for name, task in run["task_order"] if name == slug]
            expected = 30 if slug == "terminalbench2" else 50
            test_count = 29 if slug == "terminalbench2" else 50
            if len(train) != expected or set(train) != set(splits[slug]["train"]):
                raise ValueError(f"Training IDs differ from split for {slug}, seed={run['seed']}")
            if len(splits[slug]["test"]) != test_count:
                raise ValueError(f"Unexpected test count for {slug}")
        for stage in STAGES:
            for slug in BENCHMARKS:
                plan.append({
                    "run": f"{run['mode']}_s{run['seed']}", "mode": run["mode"], "seed": run["seed"],
                    "stage": stage, "benchmark": slug, "train_dir": run["output_dir"],
                    "task_ids": splits[slug]["test"], "model": launch["model"],
                    "max_tokens": launch["max_tokens"],
                    "max_steps": launch["terminalbench2_max_turns"] if slug == "terminalbench2" else 50,
                    "manifest_sha256": launch["source_manifest_sha256"],
                })
    if len(plan) != 54:
        raise ValueError("Expected 54 checkpoint/benchmark evaluations")
    return plan


def archive_snapshots(run: dict, output: Path, stages: tuple[int, ...] = STAGES) -> dict[int, Path]:
    if not stages or any(stage not in STAGES for stage in stages):
        raise ValueError("Expected checkpoint stages 50, 100 or 130")
    prefix = completed_prefix(Path(run["output_dir"]), run["task_order"])
    if len(prefix) < max(stages):
        raise ValueError(f"Incomplete training sessions for {run['mode']} s{run['seed']}: {len(prefix)}/{max(stages)}")
    paths = {}
    for stage in stages:
        item = next((entry for entry in reversed(prefix[:stage]) if entry["checkpoint"]), None)
        if item is None:
            raise ValueError(f"No playbook checkpoint at or before task {stage}")
        if item["checkpoint"]["store_id"] != f"ace_{run['mode']}_global":
            raise ValueError("Wrong checkpoint playbook store")
        path = output / "snapshots" / f"{run['mode']}_s{run['seed']}" / f"checkpoint_{stage:03d}.json"
        payload = {**item["checkpoint"], "training_task_ordinal": stage,
                   "source_checkpoint": item["checkpoint_path"],
                   "source_checkpoint_sha256": digest(Path(item["checkpoint_path"]))}
        if path.exists() and json.loads(path.read_text()) != payload:
            raise ValueError(f"Frozen checkpoint changed: {path}")
        if not path.exists():
            save_json(path, payload)
        paths[stage] = path
    return paths


def run_job(job: dict, output_root: Path, max_workers: int) -> dict:
    output = output_root / job["run"] / f"checkpoint_{job['stage']:03d}" / job["benchmark"]
    output.mkdir(parents=True, exist_ok=True)
    # Early and post-training controllers share the same immutable result.
    with FileLock(str(output / ".evaluation.lock")):
        return _run_job(job, output_root, max_workers)


def _run_job(job: dict, output_root: Path, max_workers: int) -> dict:
    context_path = Path(job["train_dir"]) / "experiment_config.json" if job.get("train_dir") else None
    context_config = json.loads(context_path.read_text()) if context_path and context_path.exists() else job
    actor_context = context_config.get("actor_context", "legacy")
    if job.get("actor_context", actor_context) != actor_context:
        raise ValueError("Evaluation actor context differs from training")
    if actor_context != "legacy":
        require_matching_actor_context(
            context_config, actor_context, context_config.get("benchmarks", [job["benchmark"]]),
        )
    output = output_root / job["run"] / f"checkpoint_{job['stage']:03d}" / job["benchmark"]
    summary_path = output / "summary.json"
    snapshot_path = output_root / "snapshots" / job["run"] / f"checkpoint_{job['stage']:03d}.json"
    snapshot = json.loads(snapshot_path.read_text())
    checkpoint_hash = digest(snapshot_path)
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("actor_context", "legacy") != actor_context:
            raise ValueError("Existing evaluation actor context differs")
        if (summary.get("checkpoint_sha256") != checkpoint_hash
                or summary.get("manifest_sha256") != job["manifest_sha256"]
                or summary.get("task_ids") != job["task_ids"]
                or len(summary.get("sessions", [])) != len(job["task_ids"])):
            raise ValueError(f"Existing test summary differs: {summary_path}")
        return summary
    slug = job["benchmark"]
    workers = 1 if slug == "terminalbench2" else max_workers
    print(f"START {job['run']} checkpoint={job['stage']} {slug} tasks={len(job['task_ids'])}", flush=True)
    PlaybookStore.reset_all()
    agent = load_agent("ace")(
        model=job["model"], curator_model=job["model"], shuffle_mode=job["mode"],
        benchmark_id=slug, runner="direct", model_settings=ModelSettings(max_tokens=job["max_tokens"]),
        enable_thinking=False, training_time=False, evaluation_mode=True,
        actor_context=actor_context,
        initial_playbook=snapshot["playbook"], **BENCHMARK_REGISTRY[slug]["agent_kwargs"],
    )
    kwargs = dict(context_config.get("benchmark_configs", {}).get(slug) or
                  job.get("benchmark_kwargs") or BENCHMARK_REGISTRY[slug]["bm_kwargs"])
    if actor_context != "legacy":
        kwargs["use_cache"] = False
    expected_policy = (
        context_config.get("appworld_policy_sha256") if slug == "appworld" and actor_context != "legacy" else None
    )
    if expected_policy and hashlib.sha256(kwargs.get("policy_override", "").encode()).hexdigest() != expected_policy:
        raise ValueError("Frozen evaluation AppWorld policy differs from training")
    if slug == "terminalbench2":
        kwargs.update(actor="terminus2", max_interactions=job["max_steps"])
    run_id = f"ace_test_{job['run']}_c{job['stage']}_{slug}"
    results = evaluate(
        benchmark=load_benchmark(slug)(**kwargs), agent=agent, task_ids=job["task_ids"],
        max_workers=workers, max_steps=job["max_steps"], output_dir=str(output), run_id=run_id,
    )
    for store in PlaybookStore.list_stores().values():
        if store.playbook != snapshot["playbook"] or store.session_count != 0:
            raise RuntimeError("Test evaluation modified the frozen playbook")
    sessions = []
    for result in results.session_results:
        skip_path = output / run_id / "sessions" / result.session_id / "agent/task_skip.json"
        sessions.append({
            "task_id": str(result.task_id), "session_id": result.session_id,
            "score": result.score,
            "status": result.status.value if hasattr(result.status, "value") else str(result.status),
            "steps": result.steps,
            "skip": json.loads(skip_path.read_text()) if skip_path.exists() else None,
        })
    if len(sessions) != len(job["task_ids"]) or {s["task_id"] for s in sessions} != set(job["task_ids"]):
        raise RuntimeError("Test evaluation returned incomplete or duplicate task IDs")
    summary = {
        **job, "checkpoint_sha256": checkpoint_hash, "checkpoint_path": str(snapshot_path),
        "actor_context": actor_context,
        "actor_context_metadata": context_config.get("actor_context_metadata"),
        "evaluation_mode": True, "training_time": False, "max_workers": workers,
        "sessions": sessions, "benchmark_score": results.benchmark_score,
        "mean_task_score": sum(float(s["score"] or 0) for s in sessions) / len(sessions),
        "error_count": sum(s["status"] == "error" for s in sessions),
        "context_skipped_count": sum(s["skip"] is not None for s in sessions),
        "completed_at": datetime.now(UTC).isoformat(),
    }
    save_json(summary_path, summary)
    print(f"DONE {job['run']} c{job['stage']} {slug} score={summary['mean_task_score']:.3f}", flush=True)
    return summary


def start_controller(launch_path: Path, env: dict | None = None) -> int:
    output = launch_path.parent / "test_evaluation"
    output.mkdir(exist_ok=True)
    metadata = output / "controller.json"
    if metadata.exists():
        old = json.loads(metadata.read_text())
        if alive(old["pid"]):
            return old["pid"]
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--launch-manifest", str(launch_path)]
    with (output / "controller.log").open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save_json(metadata, {"pid": process.pid, "command": command, "log": str(output / "controller.log")})
    return process.pid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--worker-run")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--port-base", type=int, default=42000)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 12 or not 1024 <= args.port_base <= 59535 or args.poll_seconds <= 0:
        parser.error("Invalid worker count, port base or poll interval")
    launch = json.loads(args.launch_manifest.read_text())
    plan = build_plan(launch)
    output = args.launch_manifest.parent / "test_evaluation"
    if args.dry_run:
        print(json.dumps({"evaluations": len(plan), "test_tasks": sum(len(j['task_ids']) for j in plan),
                          "stages": STAGES, "runs": sorted({j['run'] for j in plan})}))
        return
    output.mkdir(exist_ok=True)
    os.environ["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:8006/v1")
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["EXGENTIC_LLM_TRACE_FORMAT"] = "dedup_v1"
    if args.worker_run:
        jobs = [job for job in plan if job["run"] == args.worker_run]
        if len(jobs) != 9:
            parser.error("Worker run must identify one of the six experiments")
        failed = []
        for job in jobs:
            try:
                run_job(job, output, args.max_workers)
            except Exception as exc:
                failed.append({"stage": job["stage"], "benchmark": job["benchmark"], "error": str(exc)})
                print(f"FAILED c{job['stage']} {job['benchmark']}: {exc}", flush=True)
        save_json(output / args.worker_run / "worker_status.json", {"failed_jobs": failed, "complete": not failed})
        if failed:
            raise SystemExit(1)
        return
    with FileLock(str(output / "controller.lock"), timeout=0):
        save_json(output / "evaluation_plan.json", {"jobs": plan})
        print("Waiting for six training runs; 54 frozen test evaluations scheduled", flush=True)
        while True:
            progress = []
            for run in launch["jobs"]:
                train = Path(run["output_dir"])
                complete = (train / "training_complete.json").exists()
                progress.append({"run": f"{run['mode']}_s{run['seed']}", "complete": complete,
                                 "completed_tasks": len(completed_prefix(train, run["task_order"]))})
                if not complete and not alive(run["pid"]):
                    save_json(output / "status.json", {"phase": "training_failed", "runs": progress,
                                                       "planned_evaluations": 54})
                    raise RuntimeError(f"Training exited before completion: {progress[-1]['run']}")
            save_json(output / "status.json", {"phase": "waiting_for_training", "runs": progress,
                                               "planned_evaluations": 54, "completed_evaluations": 0})
            if all(run["complete"] for run in progress):
                break
            time.sleep(min(args.poll_seconds, 60))
        try:
            for run in launch["jobs"]:
                archive_snapshots(run, output)
        except Exception as exc:
            save_json(output / "status.json", {"phase": "checkpoint_error", "error": str(exc),
                                               "planned_evaluations": 54, "completed_evaluations": 0})
            raise
        workers = []
        for index, run in enumerate(launch["jobs"]):
            name = f"{run['mode']}_s{run['seed']}"
            env = os.environ.copy() | {"EXGENTIC_PORT_BLOCK_START": str(args.port_base + index * 1000)}
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--launch-manifest",
                       str(args.launch_manifest), "--worker-run", name, "--max-workers", str(args.max_workers)]
            with (output / f"{name}.log").open("a") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            workers.append((name, process))
        while True:
            done = len(list(output.glob("*_s*/checkpoint_*/*/summary.json")))
            running = [name for name, process in workers if process.poll() is None]
            phase = "evaluating" if running else ("complete" if done == 54 else "evaluation_failed")
            save_json(output / "status.json", {"phase": phase, "planned_evaluations": 54,
                                               "completed_evaluations": done, "running_workers": running,
                                               "workers": [{"run": n, "pid": p.pid} for n, p in workers]})
            if not running:
                print(f"Finished test evaluations: {done}/54", flush=True)
                if done != 54:
                    raise SystemExit(1)
                break
            time.sleep(min(args.poll_seconds, 60))


if __name__ == "__main__":
    main()
