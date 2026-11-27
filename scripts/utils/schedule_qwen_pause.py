# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Pause owned Qwen experiments at a wall-clock time and restart from retained results."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import psutil
from filelock import FileLock, Timeout

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.utils.process_identity import matches_identity, stable_identity  # noqa: E402

KST = ZoneInfo("Asia/Seoul")
ACTIVE = {"training", "running", "evaluating"}


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def identify(pid, target=None):
    try:
        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return None
        command = process.cmdline()
        if target and not any(arg == str(target) or arg.startswith(str(target) + "/") for arg in command):
            raise ValueError(f"PID {pid} no longer belongs to the recorded experiment")
        return {"pid": pid, "created": process.create_time(), "command": command, **stable_identity(pid)}
    except psutil.NoSuchProcess:
        return None


def owned(record):
    try:
        process = psutil.Process(record["pid"])
        same = matches_identity(record, process.create_time()) and process.status() != psutil.STATUS_ZOMBIE
        return process if same else None
    except psutil.NoSuchProcess:
        return None


def send(record, sig):
    process = owned(record)
    if process:
        process.send_signal(sig)


def stop_processes(records, grace=10):
    """Signal only retained process identities, never a broad user or GPU process group."""
    for sig, duration in ((signal.SIGINT, grace), (signal.SIGTERM, 3), (signal.SIGKILL, 2)):
        for record in records:
            send(record, sig)
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline and any(owned(record) for record in records):
            time.sleep(0.2)
        if not any(owned(record) for record in records):
            return
    remaining = [record["pid"] for record in records if owned(record)]
    if remaining:
        raise RuntimeError(f"Owned processes did not stop: {remaining}")


def tree(record):
    process = owned(record)
    if not process:
        return []
    result = [record]
    for child in process.children(recursive=True):
        identity = identify(child.pid)
        if identity:
            # Persistent SSH masters may be shared with unrelated experiments.
            command = identity["command"]
            if command and Path(command[0]).name == "ssh" and ("-M" in command or "-N" in command):
                continue
            result.append(identity)
    return result


def archive(path, output, label):
    if path.exists():
        target = output / "interrupted_artifacts" / label / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target = target.with_name(f"{path.parent.name}_{path.name}")
        shutil.move(str(path), target)


def trim_partial_jsonl(path, output, label):
    if not path.exists():
        return
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            json.loads(line)
        except (ValueError, UnicodeError):
            if index != len(lines) - 1:
                raise ValueError(f"A pre-existing JSONL record is corrupt: {path}") from None
            target = output / "interrupted_artifacts" / label / (path.name + ".partial")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(line)
            path.write_bytes(b"".join(lines[:index]))
            return
    if raw and not raw.endswith(b"\n"):
        path.write_bytes(raw + b"\n")


def preserve_interrupted(job, output, cutoff):
    """Do not convert scheduler cancellation into task failure or an optimizer decision."""
    folder = Path(job["output"])
    if job["method"] == "ace":
        training = Path(job["train_dir"])
        for result in training.glob("*/sessions/*/results.json"):
            checkpoint = result.parent / "agent/playbook_checkpoint.json"
            if not checkpoint.exists() and result.stat().st_mtime >= cutoff:
                archive(result, output, f"{job['id']}/{result.parent.name}")
        trim_partial_jsonl(training / "online_metrics.jsonl", output, job["id"])
    if job["method"] in {"gepa", "skillopt"}:
        for name in ("attempts.jsonl", "analysis_calls.jsonl", "analysis_errors.jsonl", "skipped_updates.jsonl"):
            trim_partial_jsonl(folder / name, output, job["id"])
        for attempt in folder.glob("rollouts/*/attempt_*.json"):
            if attempt.stat().st_mtime < cutoff:
                continue
            data = read(attempt, {})
            cached = attempt.parent / "result.json"
            if "hard" in data and data["hard"] in (0, 1) and not cached.exists():
                write(cached, data)  # The worker finished just before its parent was interrupted.
            elif "error" in data:
                archive(attempt, output, f"{job['id']}/{attempt.parent.name}")


def resumed_jobs(progress):
    """Preserve every finished result and leave bounded failure counters unchanged by pausing."""
    for job in progress["jobs"]:
        if job.get("status") != "paused":
            continue
        previous = job.pop("pause_previous_status")
        job["status"] = "evaluating" if previous == "evaluating" else "pending"
        job.pop("pid", None)
        # These counters count genuine launch failures; a scheduled suspension is not one.
        if previous in {"training", "running"}:
            for key in ("attempts", "launches"):
                if key in job:
                    job[key] = max(0, job[key] - 1)
    return progress


def docker_owned(roots):
    """Match exact Harbor project labels against session IDs under the requested run roots."""
    ids = set()
    for root in roots:
        for job in read(root / "progress.json", {}).get("jobs", []):
            folder = Path(job.get("train_dir", job["output"]))
            patterns = ("*/sessions/*/session.json", "sessions/*/sessions/*/session.json",
                        "rollouts/*/sessions/*/sessions/*/session.json")
            for pattern in patterns:
                ids.update(path.parent.name for path in folder.glob(pattern))
            ids.update(path.parent.name for path in
                       (Path(job["output"]) / "evaluation").glob("sessions/*/sessions/*/session.json"))
    docker = str(ROOT / ".venv/bin/docker")
    try:
        result = subprocess.run([docker, "ps", "--filter", "label=com.docker.compose.project", "--format",
                                 '{{.ID}} {{.Label "com.docker.compose.project"}}'],
                                capture_output=True, text=True, timeout=30, check=True)
        selected = []
        for line in result.stdout.splitlines():
            container, project = line.split(maxsplit=1)
            match = re.fullmatch(r"exgentic-([a-f0-9]+)-[a-f0-9]{12}", project)
            if match and match[1] in ids:
                selected.append({"id": container, "project": project})
        return selected
    except (OSError, subprocess.SubprocessError):
        return []


def journal_log(plan, message):
    try:
        base = "http://127.0.0.1:8878"
        info = json.load(urllib.request.urlopen(base + "/api/info", timeout=10))
        for experiment_id in plan.get("journal_ids", []):
            request = urllib.request.Request(base + f"/api/experiments/{experiment_id}/notes", method="POST",
                      data=json.dumps({"text": message}).encode(),
                      headers={"Content-Type": "application/json", "X-Journal-Token": info["csrf_token"]})
            with urllib.request.urlopen(request, timeout=10) as response:
                response.read()
    except Exception as exc:
        write(Path(plan["output"]) / "journal_pending.json", {"at": now(), "text": message,
                                                           "error_type": type(exc).__name__})


def pause(plan):
    output = Path(plan["output"])
    roots = [Path(path) for path in plan["roots"]]
    controllers = []
    for root in roots:
        record = read(root / "controller_process.json", {})
        identity = identify(record["pid"], root) if record.get("pid") else None
        if identity:
            controllers.append(identity)
    # Freeze scheduling before taking a snapshot or interrupting any worker.
    for record in controllers:
        send(record, signal.SIGSTOP)
    cutoff = time.time()
    write(output / "state.json", {"phase": "pausing", "at": now(), "cutoff": cutoff})
    workers = {}
    for controller in controllers:
        workers.update({r["pid"]: r for r in tree(controller) if r["pid"] != controller["pid"]})
    for root in roots:
        progress = read(root / "progress.json", {})
        write(output / "pause_snapshot" / root.name / "progress.json", progress)
        write(output / "pause_snapshot" / root.name / "status.json", read(root / "status.json", {}))
        for job in progress.get("jobs", []):
            if job.get("status") in ACTIVE and job.get("pid"):
                identity = identify(job["pid"], Path(job["output"]))
                if identity:
                    workers.update({r["pid"]: r for r in tree(identity)})
    containers = docker_owned(roots)
    write(output / "stopped_processes.json", {"controllers": controllers, "workers": list(workers.values()),
                                              "containers": containers})
    for record in controllers:
        send(record, signal.SIGKILL)  # Avoid error/retry handlers while deliberately stopping the queue.
    stop_processes(list(workers.values()))
    base = Path(plan["base_output"])
    services = []
    for name in ("bridge_process.json", "server_process.json"):
        entry = read(base / name, {})
        identity = identify(entry["pid"]) if entry.get("pid") else None
        if identity:
            expected = plan["services"][name]
            if identity["pid"] != expected["pid"] or not matches_identity(expected, identity["created"]):
                raise RuntimeError("Owned service identity changed; refusing to stop a different process")
            services.extend(tree(identity))
    stop_processes(services, grace=5)
    if containers:
        try:
            result = subprocess.run([str(ROOT / ".venv/bin/docker"), "stop", "--time", "5",
                                     *[c["id"] for c in containers]], capture_output=True, timeout=60, check=False)
            cleanup = {"returncode": result.returncode, "containers": containers}
        except (OSError, subprocess.SubprocessError) as exc:
            cleanup = {"returncode": None, "containers": containers, "error_type": type(exc).__name__}
        write(output / "container_stop.json", cleanup)
    for root in roots:
        progress = read(root / "progress.json", {})
        for job in progress.get("jobs", []):
            if job.get("status") in ACTIVE:
                preserve_interrupted(job, output, cutoff)
                job["pause_previous_status"] = job["status"]
                job["status"] = "paused"
        progress.update(at=now(), running_training=0, pending_evaluations=0)
        write(root / "progress.json", progress)
        marker = {"phase": "paused", "at": now(), "resume_at": plan["resume_at"], "schedule": str(output)}
        write(root / "control_schedule.json", marker)
        write(root / "status.json", marker)
    write(output / "state.json", {"phase": "paused", "at": now(), "resume_at": plan["resume_at"]})
    journal_log(plan, "예약한 18:00 KST 일시중지 수행. 실험 프로세스·Qwen vLLM/bridge 종료. 완료 결과 보존, "
                "미완료 작업은 다음 재개 시 다시 수행. 재개 예약: " + plan["resume_at"])


def load_runner():
    sys.path.insert(0, str(ROOT))
    from scripts.utils import run_qwen_isolated_when_free

    return run_qwen_isolated_when_free


def resume_followup(output):
    """Resume a frozen curator queue, including partial work if it had started before the pause."""
    code = output / "code_snapshot"
    sys.path.insert(0, str(code))
    sys.path.insert(0, str(code / "src"))
    from scripts.utils import run_qwen_followup as followup
    from scripts.utils import run_qwen_isolated_when_free as runner

    plan = read(output / "launch_manifest.json")
    with FileLock(str(output / "controller.lock"), timeout=0):
        write(output / "controller_process.json", {"pid": os.getpid(), "started_at": now()})
        followup.wait_for_predecessor(Path(plan["after_output"]), output, 30)
        lease = followup.acquire_gpu_lease(output, 30)
        try:
            server, bridge = followup.reuse_services(Path(plan["after_output"]), output)
            runner.preflight_model(output)
            jobs = runner.run_training_and_evaluation(plan, output, server, bridge, resume=True)
            errors = [j["id"] for j in jobs if j["status"] != "completed"]
            write(output / "status.json", {"phase": "blocked" if errors else "completed", "failed_jobs": errors})
        finally:
            lease.release()


def start_services(plan):
    output, base = Path(plan["output"]), Path(plan["base_output"])
    runner = load_runner()
    base_plan = read(base / "launch_manifest.json")
    lease = FileLock(str(Path.home() / ".cache/exgentic/qwen_gpu1_experiment.lock"), timeout=0)
    while True:
        try:
            lease.acquire()
            break
        except Timeout:
            write(output / "state.json", {"phase": "waiting_gpu_lease", "at": now()})
            time.sleep(30)
    try:
        while True:
            try:
                gpu = runner.gpu_state(base_plan["gpu"])
                if base_plan.get("gpu_uuid") and gpu["uuid"] != base_plan["gpu_uuid"]:
                    raise RuntimeError("The requested GPU identity changed")
                if runner.gpu_is_free(gpu) and runner.port_free(8006) and runner.port_free(18006):
                    break
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            write(output / "state.json", {"phase": "waiting_gpu", "at": now(), "resume_at": plan["resume_at"]})
            time.sleep(30)
        write(output / "state.json", {"phase": "starting_services", "at": now()})
        old_log = base / "output_nohup_vllm_server_qwen_gpu.log"
        if old_log.exists():
            shutil.move(str(old_log), base / f"output_nohup_vllm_server_qwen_gpu_before_{plan['id']}.log")
        server = runner.launch_server(base_plan, base)
        bridge = runner.launch_bridge(base)
        runner.preflight_model(output)
        return runner, server, bridge
    finally:
        lease.release()


def resume(plan):
    output, base = Path(plan["output"]), Path(plan["base_output"])
    _runner, server, bridge = start_services(plan)
    roots = [Path(path) for path in plan["roots"]]
    for root in roots:
        for filename in ("server_process.json", "bridge_process.json"):
            write(root / filename, read(base / filename))
        write(root / "progress.json", resumed_jobs(read(root / "progress.json")))
    commands = []
    for root in roots:
        job_plan = read(root / "launch_manifest.json")
        if all(j["status"] == "completed" for j in read(root / "progress.json")["jobs"]):
            write(root / "status.json", {"phase": "completed", "at": now()})
            continue
        snapshot = root / "code_snapshot"
        python = str(snapshot / ".venv/bin/python")
        if root == base:
            command = [python, str(snapshot / "scripts/utils/run_qwen_isolated_when_free.py"),
                       "--output-root", str(root), "--resume-existing"]
        elif job_plan.get("join_output"):
            command = [python, str(snapshot / "scripts/baselines/run_queue.py"), "--run", "--output", str(root),
                       "--poll-seconds", "10"]
        else:
            command = [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()),
                       "--resume-followup", str(root)]
        with (root / "scheduled_resume.log").open("ab") as log:
            process = subprocess.Popen(command, cwd=snapshot, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        commands.append({"root": str(root), "pid": process.pid, "command": command})
        write(root / "control_schedule.json", {"phase": "resumed", "at": now(), "schedule": str(output)})
    write(output / "resumed_processes.json", {"at": now(), "server_pid": server.pid,
                                              "bridge_pid": bridge.pid, "controllers": commands})
    write(output / "state.json", {"phase": "resumed", "at": now()})
    journal_log(plan, "예약한 10:00 KST 재개: 동일 Qwen 설정으로 vLLM·parser bridge를 다시 시작하고 "
                "체크포인트/완료 결과를 보존하여 실행기를 재시작했다. 미완료 태스크 재수행, 기존 병렬성 유지.")


def run(plan):
    output = Path(plan["output"])
    with FileLock(str(output / "schedule.lock"), timeout=0):
        write(output / "process.json", {"pid": os.getpid(), "started_at": now()})
        for phase, target, operation in (("scheduled", plan["pause_at"], pause),
                                         ("paused", plan["resume_at"], resume)):
            deadline = datetime.fromisoformat(target).timestamp()
            state = read(output / "state.json", {})
            if operation is pause and state.get("phase") in {"paused", "waiting_gpu", "resumed"}:
                continue
            if operation is resume and state.get("phase") == "resumed":
                continue
            while time.time() < deadline:
                write(output / "heartbeat.json", {"at": now(), "phase": phase, "next_action_at": target})
                time.sleep(min(10, max(0.1, deadline - time.time())))
            try:
                operation(plan)
            except Exception as exc:
                write(output / "state.json", {"phase": "error", "operation": operation.__name__, "at": now(),
                                               "error": f"{type(exc).__name__}: {exc}"})
                journal_log(plan, f"예약 {operation.__name__} 중단: {type(exc).__name__}: {exc}")
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--resume-followup", type=Path)
    args = parser.parse_args()
    if args.resume_followup:
        resume_followup(args.resume_followup.resolve())
    elif args.plan:
        run(read(args.plan))
    else:
        parser.error("--plan or --resume-followup is required")


if __name__ == "__main__":
    main()
