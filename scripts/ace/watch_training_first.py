# SPDX-License-Identifier: Apache-2.0
"""Audit one ACE training-first run hourly and recover retained infrastructure state."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ace.run_training_first import digest, read, retryable, write  # noqa: E402


def process(pid):
    try:
        directory = Path("/proc") / str(pid)
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in {"Z", "X"}:
            return None
        return {"pid": pid, "parent": int(fields[1]), "group": int(fields[2]), "start": fields[19],
                "args": (directory / "cmdline").read_bytes().decode().rstrip("\0").split("\0")}
    except (OSError, IndexError, UnicodeDecodeError):
        return None


def endpoint(output):
    protocol = read(output / "high/accumulation/protocol.json")
    model = protocol["model"].removeprefix("openai/")
    with urllib.request.urlopen(protocol["base_url"] + "/models", timeout=15) as stream:
        models = json.load(stream)
    if model not in {m["id"] for m in models["data"]}:
        raise ValueError("Serving endpoint model differs from the pinned experiment")
    running = None
    try:
        with urllib.request.urlopen(protocol["base_url"].removesuffix("/v1") + "/metrics", timeout=10) as stream:
            for line in stream.read().decode().splitlines():
                if line.startswith("vllm:num_requests_running{"):
                    running = float(line.rsplit(" ", 1)[1])
    except OSError:
        pass
    return {"healthy": True, "model": model, "running_requests": running}


def stop_owned_controller(pid, output, control):
    current = process(pid)
    if current and not (any(Path(a).name == "run_training_first.py" for a in current["args"])
                        and str(output) in current["args"] and str(control) in current["args"]):
        raise ValueError("Controller PID belongs to another process; refusing to stop it")
    if current:
        os.kill(pid, signal.SIGSTOP)
    records = {int(p.name): record for p in Path("/proc").glob("[0-9]*")
               if (record := process(int(p.name))) is not None}
    owned = {p for p, r in records.items() if r["group"] == pid}
    if current:
        owned.add(pid)
    while True:
        more = {p for p, r in records.items() if r["parent"] in owned} - owned
        if not more:
            break
        owned.update(more)
    for p in owned:
        if p == os.getpid() or any("vllm" in a for a in records[p]["args"]):
            raise ValueError("Unexpected server or watcher in controller process tree")
    for p in sorted(owned, reverse=True):
        try:
            os.kill(p, signal.SIGTERM)
            os.kill(p, signal.SIGCONT)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(process(p) for p in owned):
        time.sleep(0.1)
    for p in owned:
        live = process(p)
        if live and live["start"] == records[p]["start"]:
            os.kill(p, signal.SIGKILL)
    return sorted(owned)


def start_controller(output, control):
    command = read(control / "launcher.json")["command"]
    if str(output) not in command or str(control) not in command:
        raise ValueError("Saved command differs from this run")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
    with (control / "controller.log").open("a") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    launcher = {"pid": child.pid, "command": command, "started_at": datetime.now(UTC).isoformat(),
                "output": str(output), "control": str(control), "restarted_by_watchdog": True}
    write(control / "launcher.json", launcher)
    write(output / "scheduler_process.json", launcher)
    return child.pid


def restart_missing_server(output, watch):
    plan = read(output / "plan.json")
    serving = output / "serving"
    for name, expected in plan["serving_sha256"].items():
        if digest(serving / name) != expected:
            raise ValueError(f"Pinned serving input changed: {name}")
    for proc in Path("/proc").glob("[0-9]*"):
        row = process(int(proc.name))
        if row and any("vllm" in a for a in row["args"]) and "--port" in row["args"]:
            args = row["args"]
            if args[args.index("--port") + 1] == "8006":
                raise RuntimeError("Port 8006 still has a server process; retaining it for diagnosis")
    with (watch / "server_recovery.log").open("a") as log:
        child = subprocess.Popen(["bash", str(serving / "serve.sh")], cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(watch / "server_recovery_process.json", {"pid": child.pid, "started_at": datetime.now(UTC).isoformat()})
    return child.pid


def infrastructure_evidence(control):
    status = read(control / "status.json")
    evidence = [str(status.get("error", ""))]
    for job in read(control / "training_plan.json"):
        directory = Path(job["path"]) / "conditions" / job["condition"]["id"]
        marker = directory / "training_error.json"
        if marker.exists() and not (directory / "training_completed.json").exists():
            evidence.append(read(marker).get("traceback", ""))
    queue = control / "evaluation_queue.json"
    if queue.exists():
        for item in read(queue):
            directory = Path(item["root"]) / "stage3" / item["job"]["id"]
            if (directory / "result.json").exists():
                continue
            attempts = sorted(directory.glob("attempt_*"), key=lambda p: int(p.name.removeprefix("attempt_")))
            if attempts and (attempts[-1] / "error.json").exists():
                evidence.append(read(attempts[-1] / "error.json").get("error", ""))
    return evidence


def audit(output, control, watch, *, repair=True):
    current = datetime.now(UTC)
    status = read(control / "status.json")
    previous = read(watch / "latest.json") if (watch / "latest.json").exists() else {}
    launcher = read(control / "launcher.json")
    owner = process(launcher["pid"])
    report = {"checked_at": current.isoformat(), "next_check_at": (current + timedelta(hours=1)).isoformat(),
              "state": "healthy", "experiment": {k: v for k, v in status.items() if k != "contexts"},
              "controller_alive": bool(owner), "controller_pid": launcher["pid"], "actions": [], "alerts": [],
              "controller_restarts": previous.get("controller_restarts", 0),
              "server_restarts": previous.get("server_restarts", 0)}
    if (output / "STOP").exists() or (control / "STOP").exists():
        report["state"] = "stopped_by_request"
        return report
    if status.get("state") == "completed":
        marker = control / "evaluation_completed.json"
        if marker.exists() and read(marker)["completed"] == status["total"]:
            report["state"] = "completed"
            return report
        report["alerts"].append("Completion marker does not match the claimed coverage")
    try:
        report["server"] = endpoint(output)
    except Exception as exc:
        report["server"] = {"healthy": False, "error": str(exc)}
        report["alerts"].append(f"Serving endpoint check failed: {exc}")
    for path, expected in read(control / "input_hashes.json").items():
        if digest(path) != expected:
            report["alerts"].append(f"Pinned input changed: {path}")
            report["state"] = "needs_debugging"
            return report
    needs_restart = not owner
    if not owner:
        report["alerts"].append("Experiment controller is no longer running")
    if status.get("state") == "error":
        evidence = infrastructure_evidence(control)
        report["error_evidence"] = [e[-1600:] for e in evidence if e]
        needs_restart = any(retryable(e) for e in evidence)
        if not needs_restart:
            report["alerts"].append("Unclassified failure retained for debugging; protocol and results preserved")
    age = time.time() - (control / "status.json").stat().st_mtime
    if owner and age > 3600:
        report["alerts"].append(f"Controller heartbeat stale for {int(age)} seconds")
        needs_restart = True
    old = previous.get("experiment", {})
    if owner and old.get("phase") == status.get("phase"):
        field = "completed_steps" if status.get("phase") == "all_context_training" else "completed"
        if old.get(field) == status.get(field) and (current.timestamp() - datetime.fromisoformat(
            previous["checked_at"]).timestamp()) >= 3500:
            report["alerts"].append("No completed progress since the preceding hourly check")
            if report.get("server", {}).get("running_requests") == 0:
                needs_restart = True
    if repair and not report["server"]["healthy"] and report["server_restarts"] < 3:
        try:
            pid = restart_missing_server(output, watch)
            report["actions"].append({"restart_missing_server": pid})
            report["server_restarts"] += 1
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline and process(pid):
                time.sleep(10)
                try:
                    report["server"] = endpoint(output)
                    break
                except OSError:
                    pass
        except Exception as exc:
            report["alerts"].append(str(exc))
    if needs_restart and repair and report["server"]["healthy"] and report["controller_restarts"] < 6:
        terminated = stop_owned_controller(launcher["pid"], output, control)
        pid = start_controller(output, control)
        report["controller_restarts"] += 1
        report["actions"].append({"restart_controller": pid, "terminated_owned_processes": terminated})
    if report["actions"]:
        report["state"] = "recovered"
    elif report["alerts"]:
        report["state"] = "needs_debugging"
    return report


def save_report(watch, report):
    write(watch / "latest.json", report)
    stamp = datetime.fromisoformat(report["checked_at"]).strftime("%Y%m%dT%H%M%SZ")
    write(watch / "checks" / f"{stamp}.json", report)
    status = report["experiment"]
    line = f"[{report['checked_at']}] {report['state']} phase={status.get('phase')} " + (
        f"steps={status.get('completed_steps')}/{status.get('total_steps')} "
        if "completed_steps" in status else f"evaluations={status.get('completed')}/{status.get('total')} "
    ) + f"actions={len(report['actions'])} alerts={len(report['alerts'])} next={report['next_check_at']}"
    with (watch / "checks.log").open("a") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def main():
    def terminate(_signal, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--read-only", action="store_true")
    args = parser.parse_args()
    output, control = args.output.resolve(), args.control.resolve()
    watch = control / "hourly_watch"
    watch.mkdir(exist_ok=True)
    with FileLock(str(watch / "watch.lock"), timeout=0):
        write(watch / "process.json", {"pid": os.getpid(), "interval_seconds": 3600,
                                     "command": sys.argv, "started_at": datetime.now(UTC).isoformat()})
        while True:
            try:
                report = audit(output, control, watch, repair=not args.read_only)
                save_report(watch, report)
            except Exception as exc:
                write(watch / "watch_error.json", {"at": datetime.now(UTC).isoformat(), "error": repr(exc)})
                print(f"Audit error retained for next hourly retry: {exc!r}", flush=True)
                report = {}
            if args.once or report.get("state") in {"completed", "stopped_by_request"}:
                return
            sleeper = subprocess.Popen(["sleep", "3600"], stdin=subprocess.DEVNULL)
            write(watch / "sleep_process.json", {"pid": sleeper.pid, "seconds": 3600,
                                               "started_at": datetime.now(UTC).isoformat()})
            try:
                sleeper.wait()
            finally:
                if sleeper.poll() is None:
                    sleeper.terminate()


if __name__ == "__main__":
    main()
