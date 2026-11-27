# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Retry only execution errors with frozen original settings and preserve both attempts."""

from __future__ import annotations

import argparse
import fcntl
import multiprocessing
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.utils import evaluate_interleaved_validation_queue as runner  # noqa: E402

shared = runner.shared


def sources_for(sources: list[Path]) -> list[dict]:
    launches = []
    for source in sources:
        path = source / "launch_manifest.json"
        launch = shared.read_json(path)
        if (launch["seed"], launch["mode"], launch["split"]) != (44, "interleaved", "test"):
            raise ValueError("Recovery is restricted to interleaved seed-44 test evaluations")
        if not launch["terminalbench2_enforce_session_timeouts"]:
            raise ValueError("Original evaluation does not enforce official Terminal-Bench deadlines")
        if shared.digest((source / "task_split_manifest.json").read_bytes()) != launch["manifest_sha256"]:
            raise ValueError("Original task split manifest changed")
        launches.append({"output": str(source), "launch_sha256": shared.digest(path.read_bytes()), "launch": launch})
    if len({entry["launch"]["manifest_sha256"] for entry in launches}) != 1:
        raise ValueError("Source evaluations use different task splits")
    keys = [group["group"] for entry in launches for group in entry["launch"]["groups"]]
    if len(set(keys)) != len(keys):
        raise ValueError("Source evaluation groups overlap")
    return launches


def discover_errors(sources: list[dict], output: Path) -> dict[str, dict]:
    """Archive original errors and create one unchanged evaluation job per failed tuple."""
    jobs = {}
    for entry in sources:
        source = Path(entry["output"])
        if shared.digest((source / "launch_manifest.json").read_bytes()) != entry["launch_sha256"]:
            raise ValueError("Source launch manifest changed")
        for group in entry["launch"]["groups"]:
            for index, task_id in enumerate(group["task_ids"]):
                path = source / "results" / group["group"] / f"{index:03d}.json"
                if not path.is_file():
                    continue
                raw = path.read_bytes()
                row = shared.read_json(path)
                if row["status"] != "error":
                    continue
                fields = {key: value for key, value in group.items() if key != "task_ids"}
                for key, value in fields.items():
                    if row.get(key) != value:
                        raise ValueError(f"Failed task differs from its frozen group: {key}")
                if (row["index"], row["task_id"]) != (index, task_id):
                    raise ValueError("Failed task identity differs from its original task split")
                checkpoint_raw = Path(group["checkpoint_path"]).read_bytes()
                if shared.digest(checkpoint_raw) != group["checkpoint_sha256"]:
                    raise ValueError("Original frozen checkpoint changed")
                frozen = output / "checkpoints" / group["method"] / f"cp{group['checkpoint']:03d}.json"
                if frozen.exists() and frozen.read_bytes() != checkpoint_raw:
                    raise ValueError("Recovery checkpoint changed")
                frozen.parent.mkdir(parents=True, exist_ok=True)
                frozen.write_bytes(checkpoint_raw)
                original = output / "original_errors" / group["group"] / f"{index:03d}.json"
                if original.exists() and original.read_bytes() != raw:
                    raise ValueError("Previously archived original error changed")
                original.parent.mkdir(parents=True, exist_ok=True)
                original.write_bytes(raw)
                key = f"{group['group']}/{index:03d}"
                jobs[key] = {
                    **fields, "checkpoint_path": str(frozen), "index": index, "task_id": task_id,
                    "evaluation_label": f"{group.get('evaluation_label', 'evaluation')}_error_repair",
                    "original_result_path": str(path), "original_result_sha256": shared.digest(raw),
                    "archived_original_result": str(original),
                }
    return jobs


def validate_retry(row: dict, job: dict) -> None:
    if row["status"] not in ("completed", "error"):
        raise ValueError("Unexpected recovery status")
    if any(row.get(key) != value for key, value in job.items()):
        raise ValueError("Existing recovery result differs from its frozen retry job")


def combined_records(sources: list[dict], output: Path) -> tuple[list[dict], dict[str, list]]:
    """Overlay retries on errors; retain every original success and the full task denominator."""
    groups = []
    records = {}
    for entry in sources:
        source = Path(entry["output"])
        for group in entry["launch"]["groups"]:
            key = group["group"]
            groups.append(group)
            rows = []
            for index, task_id in enumerate(group["task_ids"]):
                original_path = source / "results" / key / f"{index:03d}.json"
                original = shared.read_json(original_path) if original_path.is_file() else None
                retry_path = output / "results" / key / f"{index:03d}.json"
                if retry_path.is_file():
                    retry = shared.read_json(retry_path)
                    if original is None or original["status"] != "error":
                        raise ValueError("Recovery would replace an unfinished or successful original evaluation")
                    if retry["original_result_sha256"] != shared.digest(original_path.read_bytes()):
                        raise ValueError("Original error changed after recovery")
                    if (retry["task_id"], retry["checkpoint_sha256"]) != (task_id, group["checkpoint_sha256"]):
                        raise ValueError("Recovery result identity differs from its original task")
                    original = retry
                rows.append(original)
            records[key] = rows
    return groups, records


def source_capacity(sources: list[dict], workers: int) -> tuple[int, bool]:
    reserved = 0
    finished = True
    for entry in sources:
        status = shared.read_json(Path(entry["output"]) / "status.json")
        if status["phase"] == "complete":
            if status["processed_tasks"] != status["total_tasks"] or status["active_tasks"]:
                raise ValueError("Source completion counts are inconsistent")
        elif status["phase"] == "running":
            finished = False
            reserved += status["workers"] if status["queued_tasks"] else status["active_tasks"]
        else:
            raise ValueError("Recovery source is neither running nor complete")
    return max(0, workers - reserved), finished


def run(sources: list[dict], output: Path, workers: int, base_url: str) -> None:
    pending = {}
    completed = set()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=context, initializer=runner.worker_init,
        initargs=(str(output), base_url),
    ) as pool:
        while True:
            jobs = discover_errors(sources, output)
            shared.write_json(output / "recovery_jobs.json", {"updated_at": shared.timestamp(), "jobs": jobs})
            for key, job in jobs.items():
                path = output / "results" / job["group"] / f"{job['index']:03d}.json"
                if path.is_file():
                    validate_retry(shared.read_json(path), job)
                    completed.add(key)
            capacity, sources_finished = source_capacity(sources, workers)
            active = set(pending.values())
            queued = [key for key in jobs if key not in completed and key not in active]
            while queued and len(pending) < capacity:
                key = queued.pop(0)
                pending[pool.submit(runner.evaluate_task, jobs[key])] = key
            groups, records = combined_records(sources, output)
            runner.save_scores(output, groups, records)
            done = sources_finished and not pending and not queued
            remaining_errors = sum(
                row is not None and row["status"] == "error" for rows in records.values() for row in rows
            )
            shared.write_json(output / "status.json", {
                "phase": "complete" if done else "running", "controller_pid": os.getpid(), "workers": workers,
                "processed_tasks": len(completed), "total_tasks": len(jobs),
                "active_tasks": len(pending), "queued_tasks": len(queued),
                "execution_errors": remaining_errors, "sources_finished": sources_finished,
                "combined_processed_tasks": sum(row is not None for rows in records.values() for row in rows),
                "combined_total_tasks": sum(len(rows) for rows in records.values()), "updated_at": shared.timestamp(),
                "one_full_session_retry_per_error": True, "official_terminalbench_deadlines_enforced": True,
            })
            if done:
                return
            if pending:
                ready, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for future in ready:
                    key = pending.pop(future)
                    result = future.result()
                    validate_retry(result, jobs[key])
                    completed.add(key)
                    print(f"RETRY_DONE {key} {result['status']} score={result['score']}", flush=True)
            else:
                # Sources can still produce another error while their final
                # long-running tasks drain. Poll without creating extra tasks.
                time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=150)
    parser.add_argument("--base-url", default="http://127.0.0.1:18006/v1")
    args = parser.parse_args()
    if not 1 <= args.workers <= 150:
        parser.error("Recovery concurrency must be between 1 and 150")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = sources_for([path.resolve() for path in args.sources])
    if output in [Path(entry["output"]) for entry in sources]:
        raise ValueError("Recovery output must differ from original evaluation outputs")
    identity = {"sources": sources, "workers": args.workers, "base_url": args.base_url,
                "full_session_additional_retries": 1, "original_results_preserved": True}
    launch_path = output / "launch_manifest.json"
    with (output / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if launch_path.exists():
            if shared.read_json(launch_path) != identity:
                raise ValueError("Recovery launch inputs changed")
        else:
            shared.write_json(launch_path, identity)
        run(sources, output, args.workers, args.base_url)


if __name__ == "__main__":
    main()
