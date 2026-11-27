# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
# ruff: noqa: E402

"""Restart isolated PREMiSE training and cp030 tests after reflection input repairs."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from exgentic.agents.premise.pipeline import manifest_order, write_json
from scripts.utils.run_isolated_baselines import ExistingWorker, base_command, job_complete, process_identity


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source: Path, output: Path, seeds: list[int], enforce_timeouts: bool) -> tuple[list[dict], dict]:
    states = {seed: json.loads((source / f"s{seed}/run/state.json").read_text()) for seed in seeds}
    manifest_path = Path(states[seeds[0]]["config"]["manifest"])
    original = json.loads(manifest_path.read_text())
    manifest = copy.deepcopy(original)
    jobs = []
    for seed, state in states.items():
        config = state["config"]
        if config["manifest_sha256"] != digest(manifest_path) or config["manifest"] != str(manifest_path):
            raise ValueError("Source seeds must share the original frozen task split")
        if (config["mode"], config["benchmarks"], config["past_count"], config["replay_retries"]) != (
            "isolated", ["terminalbench2"], 3, 2,
        ):
            raise ValueError("Expected isolated Terminal-Bench PREMiSE with three past tasks and two replay retries")
        if len(config["order"]) != 30 or state["cursor"] != 30:
            raise ValueError("Expected completed 30-task source training")
        manifest[f"terminalbench2_{seed}"].setdefault("benchmark_kwargs", {})[
            "enforce_session_timeouts"
        ] = enforce_timeouts
        jobs.append({
            "name": f"premise_terminalbench2_s{seed}", "method": "premise", "benchmark": "terminalbench2",
            "seed": seed, "train_count": 30, "test_count": 29, "model": config["model"],
            "output": str(output / f"s{seed}"), "original_config": config,
            "source_state_sha256": digest(source / f"s{seed}/run/state.json"),
        })
    split_path = output / "task_split_manifest.json"
    if split_path.exists() and json.loads(split_path.read_text()) != manifest:
        raise ValueError("Existing restart task split differs")
    write_json(split_path, manifest)
    for row in jobs:
        order, _ = manifest_order(output / "task_split_manifest.json", ["terminalbench2"],
                                  row["seed"], None, "isolated")
        if json.loads(json.dumps(order)) != row["original_config"]["order"]:
            raise ValueError("Reflection restart changed the training task order")
    files = [
        Path(__file__), ROOT / "scripts/premise/run_experiment.py",
        ROOT / "src/exgentic/agents/premise/llm.py", ROOT / "src/exgentic/agents/premise/pipeline.py",
        ROOT / "src/exgentic/agents/premise/instance.py", ROOT / "src/exgentic/agents/premise/selection.py",
        ROOT / "src/exgentic/benchmarks/terminalbench2/terminalbench2_eval.py",
        ROOT / "src/exgentic/benchmarks/terminalbench2/terminus_replay.py",
    ]
    plan = {
        "source_root": str(source), "source_manifest_sha256": digest(manifest_path),
        "manifest_sha256": digest(output / "task_split_manifest.json"), "seeds": seeds,
        "enforce_session_timeouts": enforce_timeouts, "enable_summarize": False,
        "training_tasks": 30 * len(seeds), "test_tasks": 29 * len(seeds), "jobs": jobs,
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in files},
        "protocol": "Fresh initial beam and RNG for each original seed. Same task order, model, 100 native turns, "
                    "parent/candidate/replay selection, mutation and judge. Reflection uses native evidence, "
                    "deduplication, token-aware field omission and bounded JSON spans. The timeout policy is recorded "
                    "separately; disabling it follows the prior user request and changes the official time budget.",
    }
    path = output / "launch_manifest.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Existing restart configuration or source code differs; use a new output root")
    write_json(path, plan)
    for path in files:
        destination = output / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return jobs, plan


def status(output: Path, jobs: list[dict], progress: dict) -> dict:
    rows = []
    for job in jobs:
        root = Path(job["output"]) / "run"
        state_path = root / "state.json"
        cursor = json.loads(state_path.read_text())["cursor"] if state_path.exists() else 0
        test_path = root / "test_evaluation/cp030_terminalbench2.json"
        test = json.loads(test_path.read_text()) if test_path.exists() else {}
        rows.append({"seed": job["seed"], **progress[job["name"]], "training_cursor": cursor,
                     "training_tasks": 30, "test_tasks_completed": sum(r is not None for r in test.get("results", [])),
                     "test_tasks": 29, "test_complete": test.get("complete", False),
                     "test_mean_score": test.get("mean_official_score"),
                     "test_execution_errors": test.get("execution_errors")})
    payload = {
        "pid": os.getpid(), "updated_at": datetime.now(UTC).isoformat(),
        "complete": all(job_complete(job) for job in jobs), "jobs": rows,
    }
    write_json(output / "status.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=ROOT / "outputs/isolated_baselines_20260930/terminalbench2/premise")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--base-url", default="http://127.0.0.1:18006/v1")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--port-block-base", type=int, default=61000)
    parser.add_argument("--enforce-session-timeouts", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct")
    if not 1024 <= args.port_block_base <= 64535 - 1000 * (len(args.seeds) - 1):
        parser.error("Port blocks exceed the allowed range")
    if args.env_file:
        os.environ.update({k: str(v) for k, v in json.loads(args.env_file.read_text()).items()})
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "launcher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs, plan = prepare(args.source_root.resolve(), output, args.seeds, args.enforce_session_timeouts)
    progress_path = output / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    for job in jobs:
        progress.setdefault(job["name"], {"status": "pending", "pid": None})
    if args.prepare_only:
        status(output, jobs, progress)
        print(json.dumps({k: plan[k] for k in ("training_tasks", "test_tasks", "seeds")}), flush=True)
        return 0
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + env["PATH"]
    running = {}
    for index, job in enumerate(jobs):
        record = progress[job["name"]]
        if job_complete(job):
            record["status"] = "completed"
            continue
        identity = process_identity(record["pid"]) if record.get("pid") else None
        run_output = str(Path(job["output"]) / "run")
        if (identity and "--output-dir" in identity[0]
                and identity[0][identity[0].index("--output-dir") + 1] == run_output):
            running[job["name"]] = ExistingWorker(record["pid"], identity, job)
        else:
            command = base_command(job, output / "task_split_manifest.json")
            command.extend(("--base-url", args.base_url, "--model", job["model"]))
            worker_env = {**env, "EXGENTIC_PORT_BLOCK_START": str(args.port_block_base + 1000 * index)}
            logs = output / "logs"
            logs.mkdir(exist_ok=True)
            with (logs / f"s{job['seed']}.log").open("ab") as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=worker_env, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            running[job["name"]] = process
            record.update(status="running", pid=process.pid, command=command)
            print(f"START seed={job['seed']} pid={process.pid}", flush=True)
        write_json(progress_path, progress)
    while running:
        for job in jobs:
            process = running.get(job["name"])
            if process is None:
                continue
            code = process.poll()
            if code is not None:
                progress[job["name"]].update(status="completed" if job_complete(job) else "failed",
                                             returncode=code)
                running.pop(job["name"])
                write_json(progress_path, progress)
        status(output, jobs, progress)
        if running:
            time.sleep(20)
    for job in jobs:
        path = args.source_root / f"s{job['seed']}/run/state.json"
        if digest(path) != job["source_state_sha256"]:
            raise RuntimeError("Source training state changed during the restart")
    return 0 if status(output, jobs, progress)["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
