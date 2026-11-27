# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Run the seed 44 AppWorld prompt/policy matrix in pairs of 50-worker processes."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import evaluate_appworld_base_prompts as base

PAIRS = (
    ("s44_gepa_existing", "s44_gepa_teammate"),
    ("s44_skillopt_existing", "s44_skillopt_teammate"),
)
JOBS = {name for pair in PAIRS for name in pair}
DEFAULT_OUTPUT = base.ROOT / "outputs/appworld_base_prompt_policy_seed42_43_20261006"


def now() -> str:
    return datetime.now(UTC).isoformat()


def prepare(output: Path, workers: int) -> tuple[dict, dict[str, dict], str]:
    source_path = output / "launch_manifest.json"
    source = json.loads(source_path.read_text())
    manifest = Path(source["manifest"])
    policy = (output / "teammate_policy.txt").read_text()
    if base.sha256(manifest.read_bytes()) != source["manifest_sha256"]:
        raise ValueError("Task split manifest differs from the seed 42/43 evaluation")
    if base.sha256(policy.encode()) != source["team_policy_sha256"]:
        raise ValueError("Teammate policy differs from the seed 42/43 evaluation")
    if source["prompts"] != base.PROMPTS or source["model"] != base.MODEL:
        raise ValueError("Prompt or model differs from the seed 42/43 evaluation")
    if source["workers"] != workers:
        raise ValueError("Worker count differs from the seed 42/43 evaluation")

    section = base.select_task_split_sections(json.loads(manifest.read_text()), ["appworld"], 44)["appworld"]
    task_ids = base.validate_task_split(section, "appworld")["test"]
    if len(task_ids) != 50:
        raise ValueError(f"Expected 50 seed 44 AppWorld test tasks, got {len(task_ids)}")
    jobs = {
        f"s44_{prompt}_{policy_name}": {
            "seed": 44, "prompt": prompt, "policy": policy_name, "task_ids": task_ids,
        }
        for prompt in base.PROMPTS
        for policy_name in ("existing", "teammate")
    }
    if jobs.keys() != JOBS:
        raise ValueError("Job matrix is inconsistent")
    identity = {
        "source_launch_manifest_sha256": base.sha256(source_path.read_bytes()),
        "manifest_sha256": source["manifest_sha256"],
        "team_policy_sha256": source["team_policy_sha256"],
        "prompts": source["prompts"],
        "model": source["model"],
        "base_url": source["base_url"],
        "workers_per_process": workers,
        "simultaneous_processes": 2,
        "total_concurrency": workers * 2,
        "pairs": [list(pair) for pair in PAIRS],
        "jobs": jobs,
    }
    identity_path = output / "seed44_launch_manifest.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Existing seed 44 evaluation configuration differs")
    else:
        base.write_json(identity_path, identity)
    return identity, jobs, policy


def run_worker(output: Path, name: str, workers: int) -> int:
    identity, jobs, team_policy = prepare(output, workers)
    job = jobs[name]
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.configure_llm_slots(workers)
        os.environ["OPENAI_API_BASE"] = identity["base_url"]
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        result_path = directory / "evaluation.json"
        rows = json.loads(result_path.read_text())["results"] if result_path.exists() else [None] * 50
        if len(rows) != 50 or any(
            row is not None and row["task_id"] != task_id
            for row, task_id in zip(rows, job["task_ids"], strict=True)
        ):
            raise ValueError(f"Existing results do not match task split: {name}")

        def save() -> None:
            base.write_json(result_path, {"job": name, "results": rows, **base.summarize(rows)})

        save()
        print(f"START {name}: {sum(row is None for row in rows)} remaining tasks; workers={workers}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {
                pool.submit(base.evaluate_one, job, index, task_id, directory, team_policy): index
                for index, task_id in enumerate(job["task_ids"])
                if rows[index] is None
            }
            for future in as_completed(pending):
                index = pending[future]
                rows[index] = future.result()
                save()
                print(f"DONE {name} {index:02d}: {rows[index]['status']} tgc={rows[index]['tgc']}", flush=True)
        summary = base.summarize(rows)
        print(f"FINISH {name}: {json.dumps(summary)}", flush=True)
        return 0 if summary["complete"] and summary["errors"] == 0 else 1


def run_controller(output: Path, workers: int) -> int:
    prepare(output, workers)
    status_path = output / "seed44_status.json"
    status = {
        "controller_pid": os.getpid(),
        "phase": "running",
        "workers_per_process": workers,
        "total_concurrency": workers * 2,
        "pairs": [],
    }
    with (output / "seed44_controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.write_json(status_path, {**status, "updated_at": now()})
        for pair in PAIRS:
            processes = []
            for name in pair:
                log = (output / f"{name}.log").open("a")
                try:
                    process = subprocess.Popen(
                        [sys.executable, "-u", str(Path(__file__).resolve()),
                         "--output", str(output), "--workers", str(workers), "--worker", name],
                        stdout=log, stderr=subprocess.STDOUT,
                    )
                finally:
                    log.close()
                processes.append((name, process))
            pair_status = {"jobs": list(pair), "pids": {name: process.pid for name, process in processes},
                           "started_at": now()}
            status["pairs"].append(pair_status)
            base.write_json(status_path, {**status, "updated_at": now()})
            print(f"START PAIR {pair}: pids={pair_status['pids']}", flush=True)
            pair_status["exit_codes"] = {name: process.wait() for name, process in processes}
            pair_status["finished_at"] = now()
            base.write_json(status_path, {**status, "updated_at": now()})
            print(f"FINISH PAIR {pair}: {pair_status['exit_codes']}", flush=True)

        summaries = {}
        for name in JOBS:
            result_path = output / name / "evaluation.json"
            summaries[name] = (
                base.summarize(json.loads(result_path.read_text())["results"])
                if result_path.exists() else None
            )
        clean = all(summary and summary["complete"] and summary["errors"] == 0 for summary in summaries.values())
        status.update({"phase": "complete" if clean else "failed", "finished_at": now(), "jobs": summaries})
        base.write_json(status_path, {**status, "updated_at": now()})
        return 0 if clean else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--worker", choices=sorted(JOBS))
    args = parser.parse_args()
    if args.workers != 50:
        parser.error("workers must be 50 to match the seed 42/43 evaluation")
    output = args.output.resolve()
    return run_worker(output, args.worker, args.workers) if args.worker else run_controller(output, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
