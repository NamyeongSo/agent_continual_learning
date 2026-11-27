# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Retry failed AppWorld teammate-policy tasks using agent-side policy injection."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.premise import evaluate_appworld_base_prompts as base


def run(output: Path, workers: int) -> int:
    base.configure_llm_slots(workers)
    launch = json.loads((output / "launch_manifest.json").read_text())
    team_policy = (output / "teammate_policy.txt").read_text()
    if base.sha256(team_policy.encode()) != launch["team_policy_sha256"]:
        raise ValueError("Teammate policy no longer matches the original evaluation")
    all_jobs = base.plan_jobs(Path(launch["manifest"]))
    jobs = [job for job in all_jobs if job["policy"] == "teammate"]
    if len(jobs) != 4 or workers != launch["workers"]:
        raise ValueError("Expected four teammate-policy jobs at the original concurrency")
    if os.environ["OPENAI_API_BASE"] != launch["base_url"]:
        raise ValueError("Model endpoint differs from the original evaluation")
    base.write_json(output / "teammate_retry_manifest.json", {
        "reason": "The isolated AppWorld venv rejected benchmark-side policy_override before task execution",
        "policy_delivery": "PremiseAgent.context_policy_override",
        "original_launch_manifest": str(output / "launch_manifest.json"),
        "team_policy_sha256": launch["team_policy_sha256"],
        "model": launch["model"],
        "base_url": launch["base_url"],
        "workers": workers,
        "jobs": jobs,
    })
    status_path = output / "status.json"
    statuses = json.loads(status_path.read_text())["jobs"]
    order = [f"s{job['seed']}_{job['prompt']}_{job['policy']}" for job in all_jobs]
    for job in jobs:
        name = f"s{job['seed']}_{job['prompt']}_teammate"
        directory = output / name
        result_path = directory / "evaluation.json"
        backup_path = directory / "evaluation.before_teammate_retry.json"
        if not backup_path.exists():
            shutil.copy2(result_path, backup_path)
        rows = json.loads(result_path.read_text())["results"]
        if len(rows) != 50 or any(
            row is not None and row["task_id"] != task_id
            for row, task_id in zip(rows, job["task_ids"], strict=True)
        ):
            raise ValueError(f"Existing results do not match the task split: {name}")
        rows = [None if row is not None and row["status"] == "error" else row for row in rows]

        def save(rows=rows, result_path=result_path, name=name) -> None:
            summary = base.summarize(rows)
            base.write_json(result_path, {"job": name, "results": rows, **summary})
            statuses[name] = summary
            updated = {
                "pid": os.getpid(),
                "updated_at": datetime.now(UTC).isoformat(),
                "workers": workers,
                "job_order": order,
                "jobs": statuses,
                "teammate_retry": True,
            }
            base.write_json(status_path, updated)
            base.write_json(output / "teammate_retry_status.json", updated)

        save()
        print(f"START teammate retry {name}: {sum(row is None for row in rows)} tasks, workers={workers}", flush=True)
        pending_indexes = [i for i, row in enumerate(rows) if row is None]
        if pending_indexes:
            first = pending_indexes.pop(0)
            rows[first] = base.evaluate_one(job, first, job["task_ids"][first], directory, team_policy)
            save()
            print(f"SMOKE teammate retry {name} {first:02d}: {rows[first]['status']}", flush=True)
            if rows[first]["status"] == "error":
                raise RuntimeError(f"First corrected teammate-policy task failed: {rows[first]['error']}")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {
                pool.submit(base.evaluate_one, job, i, task_id, directory, team_policy): i
                for i, task_id in enumerate(job["task_ids"]) if i in pending_indexes
            }
            for future in as_completed(pending):
                index = pending[future]
                rows[index] = future.result()
                save()
                print(f"DONE teammate retry {name} {index:02d}: {rows[index]['status']} "
                      f"tgc={rows[index]['tgc']}", flush=True)
        print(f"FINISH teammate retry {name}: {json.dumps(base.summarize(rows))}", flush=True)
    clean = all(statuses[f"s{job['seed']}_{job['prompt']}_teammate"]["errors"] == 0 for job in jobs)
    base.write_json(output / "teammate_retry_complete.json", {
        "success": clean,
        "completed_at": datetime.now(UTC).isoformat(),
        "jobs": {name: row for name, row in statuses.items() if name.endswith("_teammate")},
    })
    base.write_json(output / "complete.json", {
        "success": all(row["errors"] == 0 and row["complete"] for row in statuses.values()),
        "completed_at": datetime.now(UTC).isoformat(),
        "jobs": statuses,
    })
    return 0 if clean else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=50)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    output = args.output.resolve()
    launch = json.loads((output / "launch_manifest.json").read_text())
    os.environ["OPENAI_API_BASE"] = launch["base_url"]
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return run(output, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
