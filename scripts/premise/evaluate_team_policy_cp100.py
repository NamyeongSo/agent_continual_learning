# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Evaluate a frozen PREMiSE CP100 AppWorld checkpoint with the teammate policy."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from exgentic.utils.task_splits import select_task_split_sections, validate_task_split  # noqa: E402
from scripts.premise import evaluate_appworld_policy as evaluation  # noqa: E402


def plan(run_dir: Path) -> dict:
    checkpoint = run_dir / "checkpoints/checkpoint_100.json"
    snapshot = json.loads(checkpoint.read_text())
    config = snapshot["config"]
    if snapshot["cursor"] != 100 or config["mode"] not in ("sequential", "interleaved"):
        raise ValueError("Expected PREMiSE sequential or interleaved checkpoint 100")
    manifest = Path(config["manifest"])
    if evaluation.sha256(manifest) != config["manifest_sha256"]:
        raise ValueError("Frozen task split manifest changed")
    sections = select_task_split_sections(json.loads(manifest.read_text()), config["benchmarks"], config["seed"])
    task_ids = validate_task_split(sections["appworld"], "appworld")["test"]
    if len(task_ids) != 50:
        raise ValueError("Expected 50 AppWorld test tasks")
    return {
        "seed": config["seed"],
        "mode": config["mode"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": evaluation.sha256(checkpoint),
        "manifest_sha256": config["manifest_sha256"],
        "model": config["model"],
        "benchmark_kwargs": config["benchmark_configs"]["appworld"],
        "max_steps": config["max_steps"]["appworld"],
        "task_ids": task_ids,
        "playbook": snapshot["states"]["global"]["beam"][0],
    }


def run(run_dir: Path, policy_path: Path, workers: int) -> int:
    job = plan(run_dir)
    policy = policy_path.read_text()
    policy_sha = hashlib.sha256(policy.encode()).hexdigest()
    output = run_dir / "team_policy_test_cp100"
    output.mkdir(exist_ok=True)
    identity = {
        "checkpoint_sha256": job["checkpoint_sha256"],
        "manifest_sha256": job["manifest_sha256"],
        "policy_sha256": policy_sha,
        "model": job["model"],
        "task_ids": job["task_ids"],
        "workers": workers,
        "max_steps": job["max_steps"],
        "temperature": 0.0,
        "initial_max_tokens": 2048,
        "truncation_retry_max_tokens": 8192,
        "base_url": os.environ["OPENAI_API_BASE"],
    }
    identity_path = output / "launch_manifest.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Existing CP100 evaluation configuration differs")
    evaluation.write_json(identity_path, identity)
    (output / "policy.txt").write_text(policy)
    result_path = output / "evaluation.json"
    rows = json.loads(result_path.read_text())["results"] if result_path.exists() else [None] * 50
    if len(rows) != 50 or any(
        row is not None and row["task_id"] != task_id
        for row, task_id in zip(rows, job["task_ids"], strict=True)
    ):
        raise ValueError("Existing result task order differs")

    def save():
        summary = evaluation.summarize(rows)
        evaluation.write_json(result_path, {
            "identity": identity, "results": rows, **summary,
            "updated_at": datetime.now(UTC).isoformat(),
        })

    def one(index: int, task_id: str) -> dict:
        row = evaluation.evaluate_one(job, index, task_id, policy, output)
        if row["status"] == "error" and "truncated" in row.get("error", ""):
            row = evaluation.evaluate_one(job, index, task_id, policy, output, max_tokens=8192)
            row["truncation_retry_max_tokens"] = 8192
        return row

    evaluation.configure_llm_slots(workers)
    save()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(one, index, task_id): index
            for index, task_id in enumerate(job["task_ids"]) if rows[index] is None
        }
        for future in as_completed(pending):
            index = pending[future]
            rows[index] = future.result()
            save()
            print(f"DONE {job['mode']} seed={job['seed']} index={index:02d} "
                  f"status={rows[index]['status']} tgc={rows[index]['tgc']}", flush=True)
    summary = evaluation.summarize(rows)
    evaluation.write_json(output / "complete.json", {
        "success": summary["complete"] and summary["errors"] == 0,
        "completed_at": datetime.now(UTC).isoformat(),
        **summary,
    })
    print(f"FINISH {job['mode']} seed={job['seed']} {json.dumps(summary)}", flush=True)
    return 0 if summary["complete"] and summary["errors"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--base-url", default="http://127.0.0.1:18006/v1")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    os.environ["OPENAI_API_BASE"] = args.base_url
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    run_dir = args.run_dir.resolve()
    output = run_dir / "team_policy_test_cp100"
    output.mkdir(exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(run_dir, args.policy.resolve(), args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
