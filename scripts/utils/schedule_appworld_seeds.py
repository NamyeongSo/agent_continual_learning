# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Run the teammate-policy seed 42 cohort before resuming seeds 43 and 44."""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.baselines.run_queue import complete as native_complete  # noqa: E402
from scripts.utils.schedule_qwen_pause import (  # noqa: E402
    identify,
    journal_log,
    preserve_interrupted,
    read,
    resumed_jobs,
    send,
    stop_processes,
    tree,
    write,
)


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def partition(jobs):
    if len(jobs) != 6 or Counter(j["seed"] for j in jobs) != {42: 2, 43: 2, 44: 2}:
        raise ValueError("Expected two methods and three matched seeds in each batch")
    return {seed: [copy.deepcopy(j) for j in jobs if j["seed"] == seed] for seed in (42, 43, 44)}


def completed(job, kind):
    if job.get("status") != "completed":
        return False
    if kind == "native":
        return native_complete(job)
    summary = read(Path(job["output"]) / "evaluation/summary.json", {})
    records = list((Path(job["output"]) / "evaluation/records").glob("*.json"))
    return (
        (Path(job["train_dir"]) / "training_complete.json").exists()
        and summary.get("complete") is True
        and summary.get("execution_errors") == 0
        and summary.get("tasks") == len(job["test_ids"])
        and len(records) == len(job["test_ids"])
    )


def first_seed_finished(schedule):
    for batch in schedule["batches"]:
        jobs = read(Path(batch["groups"]["42"]) / "progress.json")["jobs"]
        if len(jobs) != 2 or not all(completed(j, batch["kind"]) for j in jobs):
            return False
    return True


def spawned_identity(process, command):
    """Wait for exec to publish the child command before checking its ownership."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        record = identify(process.pid)
        if record is not None and record["command"] == command:
            return record
        if process.poll() is not None:
            raise RuntimeError("New group controller exited during startup")
        time.sleep(0.02)
    raise RuntimeError("New group controller did not publish its expected command")


def prepare(cohort_root):
    """Freeze dispatch, retain seed 42 actors, and checkpoint-stop only seeds 43/44."""
    output = cohort_root / "seed_schedule"
    if (output / "schedule.json").exists():
        raise ValueError("Seed schedule already exists; use --run to continue it")
    cohort = read(cohort_root / "cohort.json")
    if cohort.get("experiment_count") != 18 or len(cohort["children"]) != 3:
        raise ValueError("This scheduler requires the eighteen-run AppWorld teammate cohort")
    batches, controllers = [], []
    for child in cohort["children"]:
        root = Path(child["root"])
        if not root.is_relative_to(cohort_root):
            raise ValueError("Refusing to control a batch outside the requested cohort")
        plan, progress = read(root / "launch_manifest.json"), read(root / "progress.json")
        partition(progress["jobs"])
        record = identify(read(root / "controller_process.json")["pid"], root)
        if record is None:
            raise ValueError("Original controller is no longer alive; inspect before scheduling")
        controllers.append(record)
        batches.append({**child, "plan": plan})
    output.mkdir(parents=True, exist_ok=True)
    write(output / "original_controllers.json", controllers)
    for record in controllers:
        send(record, signal.SIGSTOP)
    cutoff = time.time()
    workers = {}
    try:
        for batch in batches:
            root = Path(batch["root"])
            progress = read(root / "progress.json")
            write(output / "backup" / root.name / "progress.json", progress)
            write(output / "backup" / root.name / "status.json", read(root / "status.json"))
            for job in progress["jobs"]:
                if job["seed"] in (43, 44) and job.get("pid"):
                    record = identify(job["pid"], Path(job["output"]))
                    if record is not None:
                        workers.update({r["pid"]: r for r in tree(record)})
        write(output / "interrupted_processes.json", list(workers.values()))
    except BaseException:
        for record in controllers:
            send(record, signal.SIGCONT)
        raise
    for record in controllers:
        send(record, signal.SIGKILL)
    # Preserve checkpoints instead of SIGSTOP-ing actors across HTTP/task deadlines.
    stop_processes(list(workers.values()), grace=2)
    for batch in batches:
        root, plan = Path(batch["root"]), batch.pop("plan")
        progress = read(root / "progress.json")
        for job in progress["jobs"]:
            if job["seed"] in (43, 44) and job["status"] != "completed":
                preserve_interrupted(job, output, cutoff)
                job["pause_previous_status"] = job["status"]
                job["status"] = "paused"
                job["paused_pid"] = job.pop("pid", None)
                job["pause_reason"] = "Waiting for all six seed 42 experiments to finish training and test"
        groups = {}
        for seed, jobs in partition(progress["jobs"]).items():
            group = root / "seed_schedule" / f"seed{seed}"
            group.mkdir(parents=True)
            (group / "code_snapshot").symlink_to(Path(cohort["snapshot"]), target_is_directory=True)
            subplan = {**plan, "jobs": [j for j in plan["jobs"] if j["seed"] == seed], "journal_id": None}
            write(group / "launch_manifest.json", subplan)
            write(group / "progress.json", {"at": now(), "jobs": jobs})
            write(group / "status.json", {"phase": "pending" if seed == 42 else "paused", "at": now()})
            groups[str(seed)] = str(group)
        batch["groups"] = groups
        write(root / "progress.json", {**progress, "at": now()})
        write(root / "control_schedule.json", {
            "phase": "seed42_first", "schedule": str(output), "active_seeds": [42],
            "waiting_seeds": [43, 44], "at": now(),
        })
    schedule = {
        "cohort": str(cohort_root), "output": str(output), "created_at": now(), "cutoff": cutoff,
        "batches": batches, "snapshot": cohort["snapshot"], "services_from": cohort["service_owner"],
        "release": "all_six_seed42_training_and_final_test_complete", "then_seeds": [43, 44],
        "journal_ids": [b["journal_id"] for b in batches],
        "controller_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write(output / "schedule.json", schedule)
    write(output / "state.json", {"phase": "seed42_first", "at": now()})
    journal_log(schedule, "사용자 요청으로 새 팀원 폴리시 실험은 seed42 여섯 개를 먼저 진행. "
                "seed43·44 열두 개는 완료 결과·학습 체크포인트를 보존해 일시중지. "
                "seed42 여섯 개의 학습과 최종 test가 모두 완료되면 seed43·44 자동 재개. "
                "중단한 미완료 rollout은 재개 시 다시 수행하며 실패/재시도 예산으로 차감하지 않음. "
                "기존 일곱 실험 및 GPU/vLLM 서비스와 공용 CP 큐는 계속 실행.")
    return schedule


def start_group(schedule, batch, seed):
    group = Path(batch["groups"][str(seed)])
    old = read(group / "controller_process.json", {})
    if old.get("pid"):
        record = identify(old["pid"], group)
        if record is not None:
            return record
    progress = read(group / "progress.json")
    if all(completed(j, batch["kind"]) for j in progress["jobs"]):
        return None
    resumed_jobs(progress)
    write(group / "progress.json", progress)
    snapshot = Path(schedule["snapshot"])
    if batch["kind"] == "legacy":
        command = [str(snapshot / ".venv/bin/python"),
                   str(snapshot / "scripts/utils/run_appworld_team_policy.py"), "--legacy-root", str(group)]
    else:
        command = [str(snapshot / ".venv/bin/python"), str(snapshot / "scripts/baselines/run_queue.py"),
                   "--run", "--output", str(group), "--poll-seconds", "10"]
    environment = read(Path(schedule["cohort"]) / "private_environment.json")
    with (group / "controller.log").open("ab") as log:
        process = subprocess.Popen(command, cwd=snapshot, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = spawned_identity(process, command)
    write(group / "controller_process.json", {**record, "started_at": now()})
    write(group / "launched_controller.json", record)
    return record


def relay(schedule, released):
    for batch in schedule["batches"]:
        root = Path(batch["root"])
        jobs = [j for group in batch["groups"].values() for j in read(Path(group) / "progress.json")["jobs"]]
        jobs.sort(key=lambda j: (j["method"], j["seed"]))
        counts = Counter(j["status"] for j in jobs)
        active = sum(counts[k] for k in ("training", "running", "evaluating"))
        write(root / "progress.json", {"at": now(), "jobs": jobs, "running_training": active,
                                      "seed_schedule": str(Path(schedule["output"]) / "schedule.json")})
        phase = "completed" if all(completed(j, batch["kind"]) for j in jobs) else "running"
        write(root / "status.json", {"phase": phase, "at": now(), "counts": dict(counts), "running": active})
        write(root / "control_schedule.json", {
            "phase": "completed" if phase == "completed" else ("seeds43_44" if released else "seed42_first"),
            "schedule": schedule["output"], "active_seeds": [43, 44] if released else [42],
            "waiting_seeds": [] if released else [43, 44], "at": now(),
        })
        # The monitoring owner can be any living group controller beneath this batch.
        for seed in ([43, 44, 42] if released else [42]):
            owner = read(Path(batch["groups"][str(seed)]) / "controller_process.json", {})
            if owner.get("pid") and (record := identify(owner["pid"], Path(batch["groups"][str(seed)]))):
                write(root / "controller_process.json", record)
                break


def run(cohort_root):
    output = cohort_root / "seed_schedule"
    schedule = read(output / "schedule.json")
    with FileLock(str(output / "scheduler.lock"), timeout=0):
        write(output / "scheduler_process.json", {**identify(os.getpid(), cohort_root), "started_at": now()})
        write(output / "runtime_source.json", {
            "path": str(Path(__file__).resolve()), "at": now(),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        })
        released = read(output / "state.json", {}).get("phase") in {"seeds43_44", "completed"}
        locks = [FileLock(str(Path(b["root"]) / "controller.lock"), timeout=5) for b in schedule["batches"]]
        try:
            for lock in locks:
                lock.acquire()
            for batch in schedule["batches"]:
                for seed in ([42, 43, 44] if released else [42]):
                    start_group(schedule, batch, seed)
            while True:
                if not released and first_seed_finished(schedule):
                    for batch in schedule["batches"]:
                        for seed in (43, 44):
                            start_group(schedule, batch, seed)
                    released = True
                    write(output / "state.json", {"phase": "seeds43_44", "at": now()})
                    journal_log(schedule, "seed42 여섯 실험의 학습·최종 test 완료 확인. "
                                "예약한 seed43·44 열두 실험 자동 재개.")
                relay(schedule, released)
                all_done = all(
                    completed(j, b["kind"]) for b in schedule["batches"]
                    for group in b["groups"].values() for j in read(Path(group) / "progress.json")["jobs"]
                )
                write(output / "heartbeat.json", {"at": now(), "phase": "completed" if all_done else (
                    "seeds43_44" if released else "seed42_first"), "pid": os.getpid()})
                if all_done:
                    write(output / "state.json", {"phase": "completed", "at": now()})
                    from scripts.utils.run_qwen_isolated_when_free import journal

                    for batch in schedule["batches"]:
                        root = Path(batch["root"])
                        write(root / "journal_record.json", {"id": batch["journal_id"]})
                        journal(read(root / "launch_manifest.json"), root, "completed",
                                "팀원 AppWorld 폴리시 학습·최종 test 완료. seed42 이후 seed43·44 재개 일정 종료.")
                    return
                time.sleep(10)
        finally:
            for lock in reversed(locks):
                if lock.is_locked:
                    lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.run:
        parser.error("Choose exactly one of --prepare or --run")
    (prepare if args.prepare else run)(args.cohort.resolve())


if __name__ == "__main__":
    main()
