# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Checkpoint-stop and resume the explicitly owned Qwen experiment controllers."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils.schedule_qwen_pause import (  # noqa: E402
    ACTIVE,
    identify,
    preserve_interrupted,
    read,
    resumed_jobs,
    send,
    stop_processes,
    tree,
    write,
)


def now():
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def launch(command, cwd, env, log):
    with log.open("ab") as stream:
        p = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                             stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    return {"pid": p.pid, "command": command, "started_at": now()}


def pause(output):
    plan = read(output / "maintenance_plan.json")
    if (output / "paused.json").exists():
        raise ValueError("This maintenance already paused its recorded processes")
    cohort = read(Path(plan["cohort"]) / "seed_schedule/schedule.json")
    original = ROOT / "outputs/qwen36_isolated_ace_premise_curator_add_update_delete_temp07_20261021_045725"
    roots = [original, *[Path(b["groups"]["42"]) for b in cohort["batches"]]]
    scheduler = read(Path(plan["cohort"]) / "seed_schedule/launched_scheduler.json")
    controls = []
    for root in roots:
        meta = read(root / "controller_process.json")
        record = identify(meta["pid"], root)
        if record is None:
            raise ValueError(f"Expected live owned controller: {root}")
        controls.append(record)
    queue = Path(plan["queue"])
    meta = read(queue / "controller_process.json")
    controls.append(identify(meta["pid"], queue))
    controls.append(identify(scheduler["pid"], Path(plan["cohort"])))
    if any(r is None for r in controls):
        raise ValueError("A controller exited before maintenance; inspect before pausing")
    for record in controls:
        send(record, signal.SIGSTOP)
    cutoff, workers = time.time(), {}
    for record in controls:
        workers.update({r["pid"]: r for r in tree(record) if r["pid"] != record["pid"]})
    for root in roots:
        progress = read(root / "progress.json")
        write(output / "backup" / root.parent.name / root.name / "progress.json", progress)
        write(output / "backup" / root.parent.name / root.name / "status.json", read(root / "status.json"))
        for job in progress["jobs"]:
            if job["status"] in ACTIVE and job.get("pid"):
                identity = identify(job["pid"], Path(job["output"]))
                if identity:
                    workers.update({r["pid"]: r for r in tree(identity)})
    write(output / "pause_processes.json", {"controllers": controls, "workers": list(workers.values()),
                                           "roots": [str(r) for r in roots], "cutoff": cutoff})
    for record in controls:
        send(record, signal.SIGKILL)
    stop_processes(list(workers.values()), grace=2)
    for root in roots:
        progress = read(root / "progress.json")
        for job in progress["jobs"]:
            if job["status"] in ACTIVE:
                preserve_interrupted(job, output, cutoff)
                job["pause_previous_status"] = job["status"]
                job["status"] = "paused"
                job.pop("pid", None)
        write(root / "progress.json", {**progress, "at": now(), "running_training": 0,
                                       "pending_evaluations": 0})
        write(root / "status.json", {"phase": "paused", "at": now(), "maintenance": str(output)})
    for batch in cohort["batches"]:
        parent = Path(batch["root"])
        all_jobs = [j for g in batch["groups"].values() for j in read(Path(g) / "progress.json")["jobs"]]
        write(parent / "progress.json", {"at": now(), "jobs": all_jobs, "running_training": 0})
        write(parent / "status.json", {"phase": "paused", "at": now(), "maintenance": str(output)})
    server = identify(plan["server"]["pid"])
    if server is None or server["command"] != plan["server"]["command"]:
        raise ValueError("Owned vLLM service changed identity; refusing to stop another service")
    stop_processes(tree(server), grace=5)
    write(output / "paused.json", {"at": now(), "roots": [str(r) for r in roots], "cutoff": cutoff})
    print("Owned actor/CP workers checkpoint-stopped; vLLM GPU0 stopped; bridge retained", flush=True)


def resume_legacy(output, root):
    from scripts.utils import run_qwen_isolated_when_free as runner
    from scripts.utils.run_qwen_followup import reuse_services

    plan = read(output / "maintenance_plan.json")
    with FileLock(str(root / "controller.lock"), timeout=0):
        write(root / "controller_process.json", {"pid": os.getpid(), "started_at": now()})
        server, bridge = reuse_services(Path(plan["service_owner"]), root)
        jobs = runner.run_training_and_evaluation(read(root / "launch_manifest.json"), root,
                                                  server, bridge, resume=True)
        failed = [j["id"] for j in jobs if j["status"] != "completed"]
        write(root / "status.json", {"phase": "blocked" if failed else "completed", "at": now(),
                                     "failed_jobs": failed})
        runner.journal({}, root, "blocked" if failed else "completed", "서빙 튜닝 후 재개한 학습·최종 test 종료.")


def resume(output):
    if (output / "resumed.json").exists():
        raise ValueError("Resume was already issued; inspect recorded process identities before retrying")
    plan, paused = read(output / "maintenance_plan.json"), read(output / "paused.json")
    owner = Path(plan["service_owner"])
    server, bridge = read(owner / "server_process.json"), read(owner / "bridge_process.json")
    if identify(server["pid"]) is None or identify(bridge["pid"]) is None:
        raise ValueError("Validated serving processes must be running before resuming experiments")
    for name in ("server_process.json", "bridge_process.json"):
        metadata = read(owner / name)
        roots = {Path(plan["queue"]), *[Path(r) for r in paused["roots"]]}
        cohort = read(Path(plan["cohort"]) / "seed_schedule/schedule.json")
        roots.update(Path(b["root"]) for b in cohort["batches"])
        roots.update(Path(g) for b in cohort["batches"] for g in b["groups"].values())
        for root in roots:
            write(root / name, metadata)
    # Keep the pre-existing 43/44 pause policy. Only the interrupted active wave is resumed.
    roots = [Path(r) for r in paused["roots"]]
    old = roots[0]
    write(old / "progress.json", resumed_jobs(read(old / "progress.json")))
    env = read(output / "private_checkpoint_environment.json")
    python = str(ROOT / ".venv/bin/python")
    old_command = [python, str(Path(__file__).resolve()), "--output", str(output), "--resume-legacy", str(old)]
    queue = Path(plan["queue"])
    marker = read(queue / "pause_dispatch.json")
    if marker.get("owner") != str(output):
        raise ValueError("Queue pause owner changed; do not remove another operation's marker")
    (queue / "pause_dispatch.json").unlink()
    original = next(r for r in read(output / "pause_processes.json")["controllers"]
                    if any(arg == str(queue) for arg in r["command"]))
    heartbeat_cutoff = time.time()
    cp_process = launch(original["command"], ROOT, env, output / "checkpoint_resume.log")
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        heartbeat = queue / "heartbeat.json"
        controller = read(queue / "controller_process.json", {})
        if (controller.get("pid") == cp_process["pid"] and heartbeat.exists()
                and heartbeat.stat().st_mtime >= heartbeat_cutoff):
            break
        time.sleep(1)
    else:
        raise RuntimeError("Checkpoint queue heartbeat must recover before final-test consumers resume")
    old_process = launch(old_command, ROOT, env, output / "old_experiments_resume.log")
    scheduler = read(Path(plan["cohort"]) / "seed_schedule/launched_scheduler.json")
    seed_process = launch(scheduler["command"], ROOT, env, output / "seed_scheduler_resume.log")
    write(Path(plan["cohort"]) / "seed_schedule/launched_scheduler.json", seed_process)
    write(output / "resumed.json", {"at": now(), "old": old_process, "checkpoint": cp_process,
                                    "seed_scheduler": seed_process, "server": server, "bridge": bridge})
    print("Resumed original experiments, seed42 wave, automatic seed43/44 schedule and 100-slot CP queue", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pause", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-legacy", type=Path)
    args = p.parse_args()
    if sum([args.pause, args.resume, args.resume_legacy is not None]) != 1:
        p.error("Choose exactly one operation")
    if args.resume_legacy:
        resume_legacy(args.output.resolve(), args.resume_legacy.resolve())
    else:
        (pause if args.pause else resume)(args.output.resolve())
