# SPDX-License-Identifier: Apache-2.0
"""Start a prepared auxiliary-reward cohort after exact upstream completion proofs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import psutil

from scripts.utils import run_shared_reflection_matrix as shared


def ready(dependency, expected=None):
    plan = shared.read(dependency / "launch_manifest.json")
    if (plan.get("version") != "peace-appworld-pct-comparison-v1" or len(plan["jobs"]) != 6
            or any(job["benchmark"] != "appworld" for job in plan["jobs"])):
        raise ValueError("Dependency is not the specified six-job AppWorld PCT comparison")
    if expected and (expected["job_ids"] != [job["id"] for job in plan["jobs"]]
                     or expected["inference_profile_sha256"] != plan["inference_profile_sha256"]
                     or expected["manifest_sha256"] != plan["manifest_sha256"]):
        raise ValueError("Queued dependency identities or pinned settings changed")
    completed = sum(shared.complete(job, plan) for job in plan["jobs"])
    return completed == 6, completed


def run(output):
    plan = shared.read(output / "launch_manifest.json")
    dependency = Path(plan["waiting_for"]["output"])
    with (output / "scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        shared.write(output / "scheduler_process.json", {
            "pid": os.getpid(), "created_at": psutil.Process().create_time(), "output": str(output)})
        while True:
            finished, count = ready(dependency, plan["waiting_for"])
            shared.write(output / "schedule_status.json", {"phase": "ready" if finished else "waiting_dependency",
                         "at": shared.now(), "dependency_completed": count, "dependency_total": 6,
                         "output": str(dependency)})
            if finished:
                break
            time.sleep(15)
        old = output / "launch_process.json"
        if old.exists():
            meta = shared.read(old)
            try:
                process = psutil.Process(meta["pid"])
                if abs(process.create_time() - meta["created_at"]) < .1 and process.status() != psutil.STATUS_ZOMBIE:
                    return
            except psutil.NoSuchProcess:
                pass
        command = [str(Path(plan["snapshot"]) / ".venv/bin/python"), "-u", "-m",
                   "scripts.utils.run_peace_alignment_product", "run", "--output", str(output)]
        with (output / "controller.log").open("ab") as log:
            process = subprocess.Popen(command, cwd=plan["snapshot"], env=shared.environment(plan),
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        shared.write(output / "launch_process.json", {"pid": process.pid,
                     "created_at": psutil.Process(process.pid).create_time(), "command": command,
                     "at": shared.now(), "dependency": str(dependency), "journal_id": plan["journal_id"]})
        shared.write(output / "schedule_status.json", {"phase": "started", "at": shared.now(),
                     "dependency_completed": 6, "dependency_total": 6, "controller_pid": process.pid})
        try:
            base = "http://127.0.0.1:8878"
            with urllib.request.urlopen(base + "/api/info", timeout=5) as response:
                token = json.load(response)["csrf_token"]
            url = base + "/api/experiments/" + plan["journal_id"]
            with urllib.request.urlopen(url, timeout=5) as response:
                document = json.load(response)
            if document["status"] == "planned":
                request = urllib.request.Request(url, method="PUT", data=json.dumps({
                    "revision": document["revision"], "status": "running"}).encode(),
                    headers={"Content-Type": "application/json", "X-Journal-Token": token})
                with urllib.request.urlopen(request, timeout=5) as response:
                    result = json.load(response)
                shared.write(output / "journal_started.json", {
                    "id": plan["journal_id"], "at": shared.now(), "revision": result["revision"]})
        except (OSError, ValueError, KeyError) as exc:
            shared.write(output / "journal_start_error.json", {"at": shared.now(), "error": type(exc).__name__})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())
