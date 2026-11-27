# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Evaluate exported sequential CP100 prompts on AppWorld with its built-in policy."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import evaluate_appworld_base_prompts as base

PROMPT_ROOT = base.ROOT / "team_prompt"
BASE_EVALUATION = base.ROOT / "outputs/appworld_base_prompt_policy_seed42_43_20261006"
DEFAULT_OUTPUT = base.ROOT / "outputs/team_prompt_sequential_cp100_appworld_existing_policy_20261007"
PAIRS = (("s42_gepa", "s42_skillopt"), ("s44_gepa", "s44_skillopt"))
NAMES = {name for pair in PAIRS for name in pair}


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan(output: Path, workers: int) -> dict:
    source_path = BASE_EVALUATION / "launch_manifest.json"
    source = json.loads(source_path.read_text())
    manifest = Path(source["manifest"])
    if digest(manifest) != source["manifest_sha256"]:
        raise ValueError("AppWorld task split changed after the base-prompt evaluation")
    if source["model"] != base.MODEL or source["prompts"] != base.PROMPTS:
        raise ValueError("Base-prompt evaluation settings changed")
    split = json.loads(manifest.read_text())
    jobs = {}
    for seed in (42, 44):
        section = base.select_task_split_sections(split, ["appworld"], seed)["appworld"]
        task_ids = base.validate_task_split(section, "appworld")["test"]
        if len(task_ids) != 50:
            raise ValueError(f"Expected 50 AppWorld test tasks for seed {seed}")
        for method in ("gepa", "skillopt"):
            directory = PROMPT_ROOT / f"{method}_sequential_seed{seed}" / "cp100_appworld"
            prompt_path = directory / ("prompt.md" if method == "gepa" else "skill.md")
            provenance = json.loads((directory / "provenance.json").read_text())
            if (provenance["checkpoint"] != 100 or provenance["stage"] != "appworld"
                    or provenance["training_policy"] != "legacy" or provenance["seed"] != seed
                    or provenance["method"] != method or provenance["mode"] != "sequential"):
                raise ValueError(f"Unexpected prompt provenance: {directory}")
            source_artifact = directory / "candidate.json" if method == "gepa" else prompt_path
            if digest(source_artifact) != provenance["sha256"]:
                raise ValueError(f"Prompt source hash differs from provenance: {directory}")
            prompt = prompt_path.read_text()
            if method == "gepa":
                candidate = json.loads(source_artifact.read_text())["best_candidate"]["system_prompt"]
                if prompt != candidate:
                    raise ValueError(f"Exported GEPA prompt differs from candidate: {directory}")
            jobs[f"s{seed}_{method}"] = {
                "seed": seed, "prompt": method, "policy": "existing", "checkpoint": 100,
                "task_ids": task_ids, "prompt_path": str(prompt_path), "prompt_sha256": digest(prompt_path),
                "provenance_path": str(directory / "provenance.json"),
            }
    if jobs.keys() != NAMES:
        raise ValueError("Expected four sequential CP100 jobs")
    identity = {
        "jobs": jobs,
        "pairs": [list(pair) for pair in PAIRS],
        "source_launch_manifest_sha256": digest(source_path),
        "task_split_sha256": digest(manifest),
        "model": source["model"],
        "base_url": source["base_url"],
        "policy": "built-in AppWorld context (existing policy; no override)",
        "workers_per_process": workers,
        "simultaneous_processes": 2,
        "total_concurrency": workers * 2,
        "initial_max_tokens": 2048,
        "truncation_retry_max_tokens": 8192,
        "temperature": 0.0,
        "max_steps": 50,
    }
    path = output / "launch_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("Existing CP100 evaluation configuration differs")
    else:
        base.write_json(path, identity)
    return identity


def worker(output: Path, name: str, workers: int) -> int:
    identity = plan(output, workers)
    job = dict(identity["jobs"][name])
    job["system_prompt"] = Path(job["prompt_path"]).read_text()
    job["run_id_prefix"] = "team_prompt_cp100"
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.configure_llm_slots(workers)
        os.environ["OPENAI_API_BASE"] = identity["base_url"]
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        path = directory / "evaluation.json"
        rows = json.loads(path.read_text())["results"] if path.exists() else [None] * 50
        if len(rows) != 50 or any(
            row is not None and row["task_id"] != task_id
            for row, task_id in zip(rows, job["task_ids"], strict=True)
        ):
            raise ValueError(f"Existing results do not match the task split: {name}")

        def save() -> None:
            base.write_json(path, {"job": name, "results": rows, **base.summarize(rows)})

        def evaluate(index: int, task_id: str) -> dict:
            row = base.evaluate_one(job, index, task_id, directory, "", max_tokens=2048)
            if row["status"] == "error" and "truncated" in row.get("error", "").lower():
                initial_error = row["error"]
                row = base.evaluate_one(job, index, task_id, directory, "", max_tokens=8192)
                row["initial_error"] = initial_error
                row["truncation_retry_max_tokens"] = 8192
            return row

        save()
        print(f"START {name}: {sum(row is None for row in rows)} remaining tasks; workers={workers}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {
                pool.submit(evaluate, index, task_id): index
                for index, task_id in enumerate(job["task_ids"]) if rows[index] is None
            }
            for future in as_completed(pending):
                index = pending[future]
                rows[index] = future.result()
                save()
                print(f"DONE {name} {index:02d}: {rows[index]['status']} tgc={rows[index]['tgc']}", flush=True)
        summary = base.summarize(rows)
        print(f"FINISH {name}: {json.dumps(summary)}", flush=True)
        return 0 if summary["complete"] and summary["errors"] == 0 else 1


def previous_evaluation_running() -> bool:
    path = BASE_EVALUATION / "seed44_status.json"
    if not path.exists():
        return False
    status = json.loads(path.read_text())
    if status["phase"] != "running":
        return False
    pids = [status["controller_pid"]]
    for pair in status["pairs"]:
        if "exit_codes" not in pair:
            pids.extend(pair["pids"].values())
    return any(Path(f"/proc/{pid}").exists() for pid in pids)


def controller(output: Path, workers: int) -> int:
    plan(output, workers)
    status_path = output / "status.json"
    status = {"controller_pid": os.getpid(), "phase": "queued", "queued_behind": str(BASE_EVALUATION),
              "workers_per_process": workers, "total_concurrency": workers * 2, "pairs": []}
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.write_json(status_path, {**status, "updated_at": timestamp()})
        while previous_evaluation_running():
            time.sleep(30)
        status["phase"] = "running"
        base.write_json(status_path, {**status, "updated_at": timestamp()})
        for pair in PAIRS:
            processes = []
            for name in pair:
                with (output / f"{name}.log").open("a") as log:
                    process = subprocess.Popen(
                        [sys.executable, "-u", str(Path(__file__).resolve()), "--output", str(output),
                         "--workers", str(workers), "--worker", name],
                        stdout=log, stderr=subprocess.STDOUT,
                    )
                processes.append((name, process))
            pair_status = {"jobs": list(pair), "pids": {name: process.pid for name, process in processes},
                           "started_at": timestamp()}
            status["pairs"].append(pair_status)
            base.write_json(status_path, {**status, "updated_at": timestamp()})
            print(f"START PAIR {pair}: {pair_status['pids']}", flush=True)
            pair_status["exit_codes"] = {name: process.wait() for name, process in processes}
            pair_status["finished_at"] = timestamp()
            base.write_json(status_path, {**status, "updated_at": timestamp()})
            print(f"FINISH PAIR {pair}: {pair_status['exit_codes']}", flush=True)
        summaries = {}
        for name in NAMES:
            path = output / name / "evaluation.json"
            summaries[name] = base.summarize(json.loads(path.read_text())["results"]) if path.exists() else None
        clean = all(row and row["complete"] and row["errors"] == 0 for row in summaries.values())
        status.update({"phase": "complete" if clean else "failed", "jobs": summaries, "finished_at": timestamp()})
        base.write_json(status_path, {**status, "updated_at": timestamp()})
        return 0 if clean else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--worker", choices=sorted(NAMES))
    args = parser.parse_args()
    if args.workers != 50:
        parser.error("workers must be 50 to match the base-prompt evaluation")
    output = args.output.resolve()
    return worker(output, args.worker, args.workers) if args.worker else controller(output, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
