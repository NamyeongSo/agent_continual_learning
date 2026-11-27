# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""One-shot checkpoint pause of explicitly owned experiments and their GPU0 vLLM."""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils.schedule_qwen_pause import (  # noqa: E402
    ACTIVE,
    identify,
    journal_log,
    preserve_interrupted,
    read,
    send,
    stop_processes,
    tree,
    write,
)


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def deadline(value):
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError("Scheduled stop must include an explicit UTC offset")
    return parsed.timestamp()


def roots_and_cohort(plan):
    cohort = read(Path(plan["cohort"]) / "seed_schedule/schedule.json")
    roots = [Path(root) for root in plan["original_roots"]]
    roots.extend(Path(group) for b in cohort["batches"] for group in b["groups"].values())
    return list(dict.fromkeys(roots)), cohort


def controller(record, scope):
    if not record or not record.get("pid"):
        return None
    try:
        return identify(record["pid"], scope)
    except (ValueError, psutil.AccessDenied, psutil.ZombieProcess):
        # A completed controller's recorded PID can have been reused by another job.
        return None


def pause_jobs(progress, output, cutoff):
    for job in progress.get("jobs", []):
        if job.get("status") in ACTIVE | {"pending"}:
            preserve_interrupted(job, output, cutoff)
            job["pause_previous_status"] = job["status"]
            job["status"] = "paused"
            job.pop("pid", None)
    return {**progress, "at": now(), "running_training": 0, "pending_evaluations": 0}


def pause(plan):
    output, queue = Path(plan["output"]), Path(plan["queue"])
    roots, cohort = roots_and_cohort(plan)
    controls = {}
    scheduler = read(Path(plan["cohort"]) / "seed_schedule/launched_scheduler.json", {})
    record = controller(scheduler, Path(plan["cohort"]))
    if record:
        controls[record["pid"]] = record
    for root in [*roots, queue]:
        record = controller(read(root / "controller_process.json", {}), root)
        if record:
            controls[record["pid"]] = record
    # Prevent new waves and new CP work before stopping any actor.
    previous_marker = read(queue / "pause_dispatch.json", None)
    write(output / "prior_cp_pause_marker.json", previous_marker)
    write(queue / "pause_dispatch.json", {"owner": str(output), "reason": "User-scheduled 17:30 KST stop"})
    for record in controls.values():
        send(record, signal.SIGSTOP)
    cutoff, workers = time.time(), {}
    write(output / "state.json", {"phase": "pausing", "at": now(), "pause_at": plan["pause_at"]})
    for record in controls.values():
        workers.update({r["pid"]: r for r in tree(record) if r["pid"] not in controls})
    for root in roots:
        progress = read(root / "progress.json", {"jobs": []})
        key = hashlib.sha256(str(root).encode()).hexdigest()[:16]
        write(output / "snapshots" / f"{key}.json", {
            "root": str(root), "progress": progress, "status": read(root / "status.json", {}),
        })
        for job in progress["jobs"]:
            if job.get("status") in ACTIVE and job.get("pid"):
                record = controller({"pid": job["pid"]}, Path(job["output"]))
                if record:
                    workers.update({r["pid"]: r for r in tree(record) if r["pid"] not in controls})
    write(output / "processes.json", {"controllers": list(controls.values()),
                                     "workers": list(workers.values()), "cutoff": cutoff})
    for record in controls.values():
        send(record, signal.SIGKILL)
    stop_processes(list(workers.values()), grace=2)
    marker = {"phase": "paused", "at": now(), "pause_at": plan["pause_at"],
              "resume_at": None, "schedule": str(output)}
    for root in roots:
        key = hashlib.sha256(str(root).encode()).hexdigest()[:16]
        progress = pause_jobs(read(root / "progress.json", {"jobs": []}), output / "interrupted" / key, cutoff)
        write(root / "progress.json", progress)
        completed = bool(progress["jobs"]) and all(j["status"] == "completed" for j in progress["jobs"])
        if not completed:
            write(root / "status.json", marker)
            write(root / "control_schedule.json", marker)
    for batch in cohort["batches"]:
        root = Path(batch["root"])
        jobs = [j for g in batch["groups"].values() for j in read(Path(g) / "progress.json")["jobs"]]
        write(root / "progress.json", {"at": now(), "jobs": jobs, "running_training": 0})
        if not all(j["status"] == "completed" for j in jobs):
            write(root / "status.json", marker)
            write(root / "control_schedule.json", marker)
    # Resolve the current explicitly recorded service, without touching other GPUs or services.
    service = read(Path(plan["service_owner"]) / "server_process.json")
    record = identify(service["pid"])
    if record:
        from scripts.utils.process_identity import matches_identity

        if not matches_identity(service, record["created"]) or record["command"] != service["command"]:
            raise RuntimeError("Recorded vLLM PID changed ownership; refusing to stop an unrelated service")
        if psutil.Process(record["pid"]).environ().get("CUDA_VISIBLE_DEVICES") != "0":
            raise RuntimeError("The recorded service is not the explicitly owned GPU0 server")
        write(output / "stopped_server.json", record)
        stop_processes(tree(record), grace=5)
    write(output / "state.json", {**marker, "completed_at": now(), "server_was_running": record is not None})
    journal_log(plan, f"예약한 {plan['pause_at']} 일시정지 완료. 완료 결과/학습 체크포인트 보존, "
                "학습·test·공용 CP 평가와 seed 자동 실행기 중지, GPU0 vLLM 종료. 자동 재시작 예약 없음.")


def run(plan):
    output = Path(plan["output"])
    with FileLock(str(output / "schedule.lock"), timeout=0):
        if read(output / "state.json", {}).get("phase") == "paused":
            return
        write(output / "process.json", {**identify(os.getpid()), "started_at": now()})
        target = deadline(plan["pause_at"])
        while time.time() < target:
            write(output / "heartbeat.json", {"at": now(), "phase": "scheduled", "next_action_at": plan["pause_at"]})
            time.sleep(min(5, max(0.05, target - time.time())))
        try:
            pause(plan)
        except Exception as exc:
            write(output / "state.json", {"phase": "error", "at": now(), "error": f"{type(exc).__name__}: {exc}"})
            journal_log(plan, f"예약 일시정지 오류: {type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    run(read(p.parse_args().plan.resolve()))
