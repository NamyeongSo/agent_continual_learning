# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Drain an existing BFCL queue and share FIFO/LIFO work across two model servers."""

from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.premise import run_bfcl_alignment as alignment  # noqa: E402
from scripts.utils.evaluate_isolated_checkpoint_queue import digest, read_json, timestamp, write_json  # noqa: E402


def process_info(pid: int) -> dict | None:
    root = Path("/proc") / str(pid)
    try:
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        argv = [item.decode() for item in (root / "cmdline").read_bytes().split(b"\0") if item]
        return {"pid": pid, "state": fields[0], "ppid": int(fields[1]), "start_ticks": fields[19], "argv": argv}
    except (OSError, ValueError):
        return None


def same_process(saved: dict) -> bool:
    current = process_info(saved["pid"])
    return bool(current and current["start_ticks"] == saved["start_ticks"] and current["state"] not in ("Z", "X"))


def signal_saved(saved: dict, value: int) -> None:
    if same_process(saved):
        os.kill(saved["pid"], value)


def validate_record(row: dict, job: dict) -> None:
    if any(row.get(key) != value for key, value in job.items()) or row.get("status") not in ("completed", "error"):
        raise ValueError(f"Result differs from frozen job: {job['key']}")


def pop_job(queued: deque, records: dict, reverse: bool, reserve_prefix: int = 0):
    """The single coordinator owns both ends, including in-flight ownership."""
    while queued:
        index, job = queued[-1] if reverse else queued[0]
        if job["key"] in records:
            queued.pop() if reverse else queued.popleft()
            continue
        if reverse and index < reserve_prefix:
            return None
        queued.pop() if reverse else queued.popleft()
        return index, job
    return None


def publish(output: Path, job: dict, row: dict, endpoint: dict, index: int) -> dict:
    validate_record(row, job)
    path = output / "results" / f"{job['key']}.json"
    if path.exists():
        existing = read_json(path)
        validate_record(existing, job)
        raise RuntimeError(f"Duplicate job completion preserved separately: {job['key']}")
    row.update(runtime_endpoint=endpoint["base_url"], runtime_upstream=endpoint["upstream"],
               contributor_output=endpoint["output"], queue_index=index, queue_direction=endpoint["direction"])
    write_json(path, row)
    return row


def worker_init(output: str, settings: dict) -> None:
    path = Path(output) / "worker_logs" / f"{os.getpid()}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
    alignment.worker_init(output, settings)


def adopt_front(output: Path, jobs: list[dict], plan: dict) -> dict:
    path = output / "dual_endpoint_handover.json"
    if path.exists():
        saved = read_json(path)
        if saved["stage"] == "drained":
            return saved
        if not same_process(saved["runner"]):
            raise RuntimeError("Original controller disappeared during handover; inspect workers before resuming")
        return saved
    status, pipeline = read_json(output / "status.json"), read_json(output / "analysis_pipeline_status.json")
    runner, wrapper = process_info(status["controller_pid"]), process_info(pipeline["wrapper_pid"])
    if (not runner or not wrapper or str(ROOT / "scripts/premise/run_bfcl_alignment.py") not in runner["argv"]
            or "--output" not in runner["argv"]
            or runner["argv"][runner["argv"].index("--output") + 1] != str(output)
            or str(output / "launch_experiment.py") not in wrapper["argv"]
            or status["phase"] != "evaluating" or status["workers"] != plan["workers"]):
        raise ValueError("Expected the live original FIFO alignment controller")
    children = []
    for candidate in Path("/proc").iterdir():
        if candidate.name.isdecimal():
            info = process_info(int(candidate.name))
            if info and info["ppid"] == runner["pid"]:
                children.append(info)
    signal_saved(wrapper, signal.SIGSTOP)
    signal_saved(runner, signal.SIGSTOP)
    # Freeze dispatch only. Every already running session keeps evaluating.
    paused = read_json(output / "status.json")
    reserve = min(len(jobs), paused["processed_tasks"] + 2 * plan["workers"])
    lookup = {job["key"]: index for index, job in enumerate(jobs)}
    active = []
    for child in children:
        worker = output / "active_workers" / f"{child['pid']}.json"
        if worker.exists():
            row = read_json(worker)
            if row["phase"] == "running":
                active.append(row["key"])
                reserve = max(reserve, lookup[row["key"]] + 1)
    saved = {"stage": "draining", "runner": runner, "wrapper": wrapper, "children": children,
             "reserved_prefix": reserve, "active_at_handover": active, "started_at": timestamp(),
             "original_status": paused, "original_pipeline": pipeline}
    write_json(path, saved)
    return saved


def original_running(output: Path, handover: dict) -> list[str]:
    active = []
    worker_output = Path(handover.get("worker_output", output))
    for child in handover["children"]:
        if not same_process(child):
            continue
        path = worker_output / "active_workers" / f"{child['pid']}.json"
        if path.exists():
            row = read_json(path)
            if row["phase"] == "running":
                active.append(row["key"])
    return active


def complete_handover(output: Path, handover: dict) -> None:
    if original_running(output, handover):
        raise RuntimeError("Cannot retire original controllers while sessions are running")
    # Wrapper is paused, so it cannot write a spurious failed-pipeline status.
    for info in [handover["wrapper"], handover["runner"]]:
        signal_saved(info, signal.SIGTERM)
        signal_saved(info, signal.SIGCONT)
    for child in handover["children"]:
        signal_saved(child, signal.SIGTERM)
    handover.update(stage="drained", drained_at=timestamp())
    write_json(output / "dual_endpoint_handover.json", handover)


def make_pool(stack: ExitStack, endpoint: dict, plan: dict):
    settings = {key: plan[key] for key in ("max_tokens", "temperature", "request_timeout",
                                         "transport_additional_retries")}
    settings["base_url"] = endpoint["base_url"]
    if "reasoning_effort" in plan:
        settings["reasoning_effort"] = plan["reasoning_effort"]
    return stack.enter_context(ProcessPoolExecutor(
        max_workers=endpoint["workers"], mp_context=multiprocessing.get_context("spawn"),
        initializer=worker_init, initargs=(endpoint["output"], settings),
    ))


def save_runtime_plan(output: Path, allocation: dict) -> None:
    """Keep the first allocation as provenance while recording current server routing."""
    original_path = output / "dual_endpoint_plan.json"
    if original_path.exists():
        original = read_json(original_path)
        for key in ("queue_sha256", "settings", "model", "single_coordinator_ownership", "source_inputs_preserved"):
            if original[key] != allocation[key]:
                raise ValueError(f"Frozen evaluation configuration changed: {key}")
    else:
        alignment.save_immutable(original_path, allocation)
    signature = digest(json.dumps(allocation, sort_keys=True).encode())
    alignment.save_immutable(output / "runtime_allocations" / f"{signature}.json", allocation)
    write_json(output / "dual_endpoint_runtime_plan.json", allocation)


def run(output: Path, tail_url: str, tail_upstream: str, tail_workers: int, front_only: bool = False,
        front_base_url: str | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    front_only = front_only or (output / "endpoint_8006_stop.json").exists()
    with ExitStack() as stack:
        # The stopped predecessor retains its normal locks while front sessions finish.
        initial_lock = "front_only_endpoint.lock" if front_only else "dual_endpoint.lock"
        dual_lock = stack.enter_context((output / initial_lock).open("a"))
        fcntl.flock(dual_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read_json(output / "launch_manifest.json")
        jobs = read_json(output / "evaluation_queue.json")["jobs"]
        if any(job["model"] != "openai/google/gemma-4-26B-A4B-it" or job["max_steps"] != 20 for job in jobs):
            raise ValueError("Expected frozen Gemma BFCL evaluations at twenty steps")
        front_url = front_base_url or plan["base_url"]
        front_name = f"front_{urlparse(front_url).port}"
        front = {"name": front_name, "base_url": front_url, "upstream": front_url,
                 "workers": plan["workers"], "direction": "FIFO", "output": str(output / "contributors" / front_name)}
        tail = {"name": "tail_8006", "base_url": tail_url, "upstream": tail_upstream,
                "workers": tail_workers, "direction": "LIFO",
                "output": str(output / "contributors" / f"tail_8006_workers{tail_workers}")}
        if front_only:
            tail.update(workers=0, enabled=False, phase="stopped_by_user")
        save_runtime_plan(output, {
            "endpoints": [front, tail], "queue_sha256": digest((output / "evaluation_queue.json").read_bytes()),
            "settings": {key: plan[key] for key in ("max_tokens", "temperature", "request_timeout",
                                                   "transport_additional_retries", "max_steps")},
            "model": "openai/google/gemma-4-26B-A4B-it", "single_coordinator_ownership": True,
            "source_inputs_preserved": True,
        })
        handover = adopt_front(output, jobs, plan)
        records = {}
        for job in jobs:
            path = output / "results" / f"{job['key']}.json"
            if path.exists():
                row = read_json(path)
                validate_record(row, job)
                records[job["key"]] = row
        queued = deque((index, job) for index, job in enumerate(jobs) if job["key"] not in records)
        pending, pools = {}, {} if front_only else {tail["name"]: make_pool(stack, tail, plan)}
        locks_acquired, idle_since, peak = False, None, 0
        endpoints = [tail, front]
        by_key = {job["key"]: job for job in jobs}
        pipeline = {**handover["original_pipeline"], "phase": "proposal_and_evaluation", "wrapper_pid": os.getpid(),
                    "runtime_endpoints": [front, tail], "runtime_workers": front["workers"] + tail["workers"],
                    "updated_at": timestamp()}
        pipeline.pop("error", None)
        write_json(output / "analysis_pipeline_status.json", pipeline)

        def status(active_old):
            nonlocal peak
            count = len(pending) + len(active_old)
            peak = max(peak, count)
            endpoint_counts = {endpoint["name"]: sum(value[0]["name"] == endpoint["name"]
                                                    for value in pending.values()) for endpoint in endpoints}
            endpoint_counts[front["name"]] += len(active_old)
            write_json(output / "status.json", {
                "phase": "evaluating", "controller_pid": os.getpid(), "workers": front["workers"] + tail["workers"],
                "completed_candidates": 250, "candidate_count": 250, "total_evaluations": len(jobs),
                "processed_tasks": len(records), "active_tasks": count,
                "queued_tasks": len(jobs) - len(records) - count, "peak_active_tasks": peak,
                "execution_errors": sum(row["status"] == "error" for row in records.values()),
                "max_steps": 20, "started_at": handover["original_status"]["started_at"], "updated_at": timestamp(),
                "endpoints": [{**endpoint, "active": endpoint_counts[endpoint["name"]],
                               "completed_new": sum(row.get("runtime_endpoint") == endpoint["base_url"]
                                                    for row in records.values())} for endpoint in endpoints],
                "original_front_draining": handover["stage"] != "drained",
            })

        while queued or pending or handover["stage"] != "drained":
            active_old = original_running(output, handover) if handover["stage"] != "drained" else []
            for key in handover["active_at_handover"]:
                path = output / "results" / f"{key}.json"
                source = Path(handover["worker_output"]) / "results" / f"{key}.json" if (
                    "worker_output" in handover
                ) else None
                if key not in records and not path.exists() and source is not None and source.exists():
                    index = next(index for index, job in enumerate(jobs) if job["key"] == key)
                    publish(output, by_key[key], read_json(source), front, index)
                if key not in records and path.exists():
                    row = read_json(path)
                    validate_record(row, by_key[key])
                    records[key] = row
            if handover["stage"] != "drained":
                if not active_old:
                    idle_since = time.monotonic() if idle_since is None else idle_since
                    if time.monotonic() - idle_since >= 10:
                        complete_handover(output, handover)
                else:
                    idle_since = None
            if handover["stage"] == "drained" and not locks_acquired:
                lock_names = ("controller.lock", "pipeline.lock")
                if initial_lock != "dual_endpoint.lock":
                    lock_names = ("dual_endpoint.lock", *lock_names)
                for name in lock_names:
                    handle = stack.enter_context((output / name).open("a"))
                    fcntl.flock(handle, fcntl.LOCK_EX)
                locks_acquired = True
                # Reconcile any dispatch/result writes between the final status snapshot and SIGSTOP.
                for job in jobs:
                    path = output / "results" / f"{job['key']}.json"
                    if job["key"] not in records and path.exists():
                        row = read_json(path)
                        validate_record(row, job)
                        records[job["key"]] = row
                pools[front["name"]] = make_pool(stack, front, plan)
                print(f"HANDOVER complete: original sessions preserved; FIFO{front['workers']} "
                      f"+ LIFO{tail['workers']}", flush=True)
            for endpoint in endpoints:
                if endpoint["name"] not in pools:
                    continue
                active = sum(value[0]["name"] == endpoint["name"] for value in pending.values())
                while active < endpoint["workers"]:
                    item = pop_job(queued, records, endpoint["direction"] == "LIFO",
                                   handover["reserved_prefix"] if handover["stage"] != "drained" else 0)
                    if item is None:
                        break
                    index, job = item
                    future = pools[endpoint["name"]].submit(alignment.evaluate_pair, job)
                    pending[future] = endpoint, index, job
                    active += 1
            status(active_old)
            if not pending:
                time.sleep(5)
                continue
            done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                endpoint, index, job = pending.pop(future)
                row = publish(output, job, future.result(), endpoint, index)
                records[job["key"]] = row
                print(f"DONE {len(records)}/{len(jobs)} {endpoint['name']} {job['key']} {row['status']}", flush=True)
        if len(records) != len(jobs):
            raise RuntimeError("Queue drained without all frozen results")
        summary = alignment.analyze(output, plan, records)
        status([])
        final = read_json(output / "status.json")
        final.update(phase="complete_with_errors" if summary["execution_errors"] else "complete", active_tasks=0,
                     queued_tasks=0, updated_at=timestamp())
        write_json(output / "status.json", final)
        pipeline.update(phase="analyzing", runner_exit_code=0, updated_at=timestamp())
        write_json(output / "analysis_pipeline_status.json", pipeline)
        subprocess.run([sys.executable, str(ROOT / "scripts/premise/analyze_bfcl_alignment.py"),
                        "--input", str(output), "--output", str(output / "analysis")], check=True, cwd=ROOT)
        pipeline.update(phase="complete", analysis_exit_code=0, finished_at=timestamp(), updated_at=timestamp())
        write_json(output / "analysis_pipeline_status.json", pipeline)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tail-base-url", required=True)
    parser.add_argument("--tail-upstream", default="http://127.0.0.1:8006/v1")
    parser.add_argument("--tail-workers", type=int, default=500)
    parser.add_argument("--front-base-url", help="Override FIFO routing while preserving frozen inputs and settings")
    parser.add_argument("--front-only", action="store_true",
                        help="Keep the tail endpoint stopped; use FIFO workers only")
    args = parser.parse_args()
    if args.tail_workers < 1:
        parser.error("Tail concurrency must be positive")
    os.environ["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT), os.environ.get("PYTHONPATH", "")))
    try:
        run(args.output.resolve(), args.tail_base_url, args.tail_upstream, args.tail_workers, args.front_only,
            args.front_base_url)
    except Exception as exc:
        write_json(args.output / "dual_endpoint_error.json", {"error": str(exc), "updated_at": timestamp()})
        raise


if __name__ == "__main__":
    main()
