# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Retry truncated PREMiSE sequential CP100 AppWorld evaluations with the same teammate policy."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.premise import evaluate_appworld_policy as evaluation  # noqa: E402

MAX_TOKENS = 8192


def run(output: Path, workers: int) -> int:
    launch = json.loads((output / "launch_manifest.json").read_text())
    policy = (output / "policy.txt").read_text()
    if hashlib.sha256(policy.encode()).hexdigest() != launch["policy_sha256"]:
        raise ValueError("Teammate policy differs from the original evaluation")
    plan = evaluation.build_plan()
    if [(job["seed"], job["checkpoint_sha256"]) for job in plan] != [
        (job["seed"], job["checkpoint_sha256"]) for job in launch["plan"]
    ]:
        raise ValueError("Checkpoint plan differs from the original evaluation")
    evaluation.configure_llm_slots(workers)
    retry_jobs = []
    statuses = {}
    for job in plan:
        seed = job["seed"]
        directory = output / f"s{seed}_sequential_cp100"
        result_path = directory / "evaluation.json"
        backup_path = directory / "evaluation.before_truncation_retry.json"
        if not backup_path.exists():
            shutil.copy2(result_path, backup_path)
        rows = json.loads(result_path.read_text())["results"]
        if len(rows) != len(job["task_ids"]) or any(
            row is not None and row["task_id"] != task_id
            for row, task_id in zip(rows, job["task_ids"], strict=True)
        ):
            raise ValueError(f"Task identities differ for seed {seed}")
        retry_indices = [i for i, row in enumerate(rows) if row is not None and row["status"] == "error"]
        if any("truncated" not in rows[i].get("error", "") for i in retry_indices):
            raise ValueError(f"Seed {seed} has an error unrelated to output truncation")
        retry_jobs.append({"seed": seed, "indices": retry_indices, "task_ids": [job["task_ids"][i]
                                                                                for i in retry_indices]})
    evaluation.write_json(output / "truncation_retry_manifest.json", {
        "reason": "Original 2048-token completion limit exhausted after five retries",
        "policy_sha256": launch["policy_sha256"],
        "original_max_tokens": launch["max_tokens"],
        "retry_max_tokens": MAX_TOKENS,
        "model": plan[0]["model"],
        "workers": workers,
        "jobs": retry_jobs,
    })

    def save_status():
        evaluation.write_json(output / "truncation_retry_status.json", {
            "pid": os.getpid(),
            "updated_at": datetime.now(UTC).isoformat(),
            "workers": workers,
            "seeds": statuses,
        })

    for job in plan:
        seed = job["seed"]
        directory = output / f"s{seed}_sequential_cp100"
        result_path = directory / "evaluation.json"
        rows = json.loads(result_path.read_text())["results"]
        retry_indices = [i for i, row in enumerate(rows) if row is not None and row["status"] == "error"]
        rows = [None if i in retry_indices else row for i, row in enumerate(rows)]

        def save(rows=rows, result_path=result_path, seed=seed, job=job):
            summary = evaluation.summarize(rows)
            evaluation.write_json(result_path, {
                "seed": seed,
                "checkpoint_sha256": job["checkpoint_sha256"],
                "policy_sha256": launch["policy_sha256"],
                "retry_max_tokens": MAX_TOKENS,
                "results": rows,
                **summary,
            })
            statuses[str(seed)] = summary
            save_status()

        save()
        print(f"START seed={seed}: retry={len(retry_indices)} workers={workers}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {
                pool.submit(evaluation.evaluate_one, job, i, job["task_ids"][i], policy, directory, MAX_TOKENS): i
                for i in retry_indices
            }
            for future in as_completed(pending):
                index = pending[future]
                row = future.result()
                row["retry_max_tokens"] = MAX_TOKENS
                rows[index] = row
                save()
                print(f"DONE seed={seed} index={index}: {row['status']} tgc={row['tgc']}", flush=True)
        print(f"FINISH seed={seed}: {json.dumps(evaluation.summarize(rows))}", flush=True)
    clean = all(row["errors"] == 0 and row["complete"] for row in statuses.values())
    evaluation.write_json(output / "truncation_retry_complete.json", {
        "success": clean, "completed_at": datetime.now(UTC).isoformat(), "seeds": statuses,
    })
    evaluation.write_json(output / "complete.json", {
        "success": clean, "completed_at": datetime.now(UTC).isoformat(), "seeds": statuses,
    })
    return 0 if clean else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=13)
    parser.add_argument("--base-url", default="http://127.0.0.1:18006/v1")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    os.environ["OPENAI_API_BASE"] = args.base_url
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    output = args.output.resolve()
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(output, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
