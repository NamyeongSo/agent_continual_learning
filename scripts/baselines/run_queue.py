# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Freeze and schedule 18 isolated GEPA/SkillOpt runs after the existing Qwen queue."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.baselines.runtime import read, write  # noqa: E402
from scripts.utils.run_qwen_followup import acquire_gpu_lease, predecessor_finished, reuse_services  # noqa: E402


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def source_files(root):
    """Copy versioned and untracked source while honoring ignored data/installations."""
    raw = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others",
                                   "--exclude-standard", "--", "src", "scripts"])
    return [Path(os.fsdecode(path)) for path in raw.split(b"\0") if path]


def freeze(output, predecessor, *, max_running_per_method=6, join_output=None, terminalbench2_version=None):
    if terminalbench2_version not in (None, "2.0", "2.1"):
        raise ValueError(f"Unsupported Terminal-Bench release: {terminalbench2_version}")
    if (output / "launch_manifest.json").exists():
        raise ValueError("Queue already exists; use --run to continue it")
    snapshot = output / "code_snapshot"
    snapshot.mkdir(parents=True, exist_ok=False)
    for relative in source_files(ROOT):
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, snapshot / name)
    (snapshot / ".venv").symlink_to(ROOT / ".venv", target_is_directory=True)
    baseline_root = snapshot / ".exgentic/baselines"
    baseline_root.mkdir(parents=True)
    for name in ("venvs", "upstream"):
        (baseline_root / name).symlink_to(ROOT / ".exgentic/baselines" / name, target_is_directory=True)
    hashes = {str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
              for name in ("src", "scripts") for path in (snapshot / name).rglob("*") if path.is_file()}
    write(output / "code_hashes.json", hashes)
    jobs = []
    for method in ("gepa", "skillopt"):
        for benchmark in ("bfcl", "appworld", "terminalbench2"):
            for seed in (42, 43, 44):
                job_id = f"{method}_{benchmark}_{seed}"
                job_output = output / "runs" / job_id
                command = [str(snapshot / ".venv/bin/python"), str(snapshot / "scripts/baselines/run_experiment.py"),
                           "--method", method, "--benchmark", benchmark, "--seed", str(seed),
                           "--workers", "50", "--output", str(job_output)]
                if benchmark == "terminalbench2" and terminalbench2_version is not None:
                    command.extend(("--terminalbench2-version", terminalbench2_version))
                jobs.append({"id": job_id, "method": method, "benchmark": benchmark, "seed": seed,
                             "output": str(job_output), "command": command, "status": "pending",
                             **({"terminalbench2_version": terminalbench2_version}
                                if benchmark == "terminalbench2" and terminalbench2_version is not None else {})})
    if max_running_per_method not in range(1, 10):
        raise ValueError("Per-method concurrency must be between 1 and 9")
    plan = {"created_at": now(), "after_output": str(predecessor), "snapshot": str(snapshot), "jobs": jobs,
            "join_output": str(join_output) if join_output else None,
            "max_running": max_running_per_method * 2, "max_running_per_method": max_running_per_method,
            "internal_workers": 50,
            "model": "Qwen/Qwen3.6-35B-A3B", "temperature": 0.7, "total_training_limit": None,
            "termination": "single_pass_stream_exhausted"}
    if terminalbench2_version is not None:
        plan["terminalbench2_version"] = terminalbench2_version
    write(output / "launch_manifest.json", plan)
    write(output / "progress.json", {"at": now(), "jobs": jobs})
    write(output / "status.json", {"phase": "planned", "at": now(), "after_output": str(predecessor)})
    return plan


def complete(job):
    folder = Path(job["output"])
    if any(not (folder / name).exists() for name in ("summary.json", "config.json", "test_results.json")):
        return False
    config, summary = read(folder / "config.json"), read(folder / "summary.json")
    if (job.get("terminalbench2_version") is not None
            and config.get("benchmark_kwargs", {}).get("subset", "2.0") != job["terminalbench2_version"]):
        return False
    test_rows = read(folder / "test_results.json")
    online = [read(path) for path in (folder / "online").glob("*.json")]
    return (summary.get("complete") is True and summary.get("tasks") == len(config["test"])
            and len(test_rows) == len(config["test"])
            and {row["id"] for row in test_rows} == {item["task_id"] for item in config["test"]}
            and len(online) == len(config["train"])
            and {row["task_id"] for row in online} == {item["task_id"] for item in config["train"]})


def journal(output, status, message):
    """Only touch the pre-created experiment's metadata and append-only log."""
    import json
    import urllib.request

    plan = read(output / "launch_manifest.json")
    if not plan.get("journal_id"):
        return
    base = "http://127.0.0.1:8878"
    try:
        def get(path):
            return json.load(urllib.request.urlopen(base + path, timeout=10))

        token = get("/api/info")["csrf_token"]
        endpoint = "/api/experiments/" + plan["journal_id"]
        record = get(endpoint)
        for method, suffix, payload in (
            ("PUT", "", {"revision": record["revision"], "status": status}),
            ("POST", "/notes", {"text": message}),
        ):
            request = urllib.request.Request(base + endpoint + suffix, method=method,
                      data=json.dumps(payload).encode(), headers={"Content-Type": "application/json",
                                                                "X-Journal-Token": token})
            with urllib.request.urlopen(request, timeout=10) as response:
                response.read()
    except Exception as exc:
        write(output / "journal_sync_error.json", {"at": now(), "error": f"{type(exc).__name__}: {exc}"})


def run(output, poll_seconds):
    from filelock import FileLock

    plan = read(output / "launch_manifest.json")
    predecessor = Path(plan["after_output"])
    with FileLock(str(output / "controller.lock"), timeout=0):
        write(output / "controller_process.json", {"pid": os.getpid(), "started_at": now()})
        while not plan.get("join_output") and not predecessor_finished(predecessor):
            write(output / "status.json", {"phase": "waiting_predecessor", "at": now(),
                                           "after_output": str(predecessor)})
            time.sleep(poll_seconds)
        # An explicit joined run shares services already owned by the user's
        # running experiment. Its owner retains the lease and service lifecycle.
        lease = None if plan.get("join_output") else acquire_gpu_lease(output, poll_seconds)
        try:
            service_owner = Path(plan["join_output"]) if plan.get("join_output") else predecessor
            server, bridge = reuse_services(service_owner, output)
            from scripts.utils.run_qwen_isolated_when_free import preflight_model

            preflight_model(output)
            journal(output, "running", "GEPA·SkillOpt isolated 학습·최종 평가 시작. "
                    + ("사용자 요청으로 실행 중인 ACE·PREMiSE의 Qwen 서비스에 합류. " if plan.get("join_output")
                       else "선행 ACE·PREMiSE 큐 완료. ")
                    + f"방법별 동시 {plan['max_running_per_method']}개, 총 {plan['max_running']}개.")
            jobs = read(output / "progress.json")["jobs"]
            active = {}
            # Reattach controller children if the controller itself was restarted.
            for job in jobs:
                if complete(job):
                    job["status"] = "completed"
                elif job.get("pid"):
                    from scripts.utils.run_qwen_isolated_when_free import ExistingProcess

                    process = ExistingProcess(job["pid"], command=job["command"],
                                              completion=Path(job["output"]) / "summary.json")
                    if process.poll() is None:
                        active[job["id"]] = process
                    elif job["status"] == "running":
                        job["status"] = "pending"
            # Completed orphans can leave no work for the loop; persist their
            # adopted completion state before deciding whether to enter it.
            write(output / "progress.json", {"at": now(), "jobs": jobs})
            while active or any(job["status"] == "pending" for job in jobs):
                if server.poll() is not None or bridge.poll() is not None:
                    raise RuntimeError("Owned Qwen server or parser bridge stopped")
                for job in jobs:
                    process = active.get(job["id"])
                    if process is not None and process.poll() is not None:
                        if complete(job):
                            job["status"] = "completed"
                        else:
                            job["status"] = "pending" if job.get("launches", 1) < 2 else "error"
                        job["finished_at"] = now()
                        del active[job["id"]]
                counts = Counter(job["method"] for job in jobs if job["id"] in active)
                for job in jobs:
                    if (job["status"] != "pending" or len(active) >= plan["max_running"]
                            or counts[job["method"]] >= plan["max_running_per_method"]):
                        continue
                    folder = Path(job["output"])
                    folder.mkdir(parents=True, exist_ok=True)
                    from scripts.utils.run_qwen_isolated_when_free import experiment_environment

                    env = experiment_environment()
                    env.update(PYTHONUNBUFFERED="1", LITELLM_LOCAL_MODEL_COST_MAP="True")
                    env.pop("EXGENTIC_PORT_BLOCK_START", None)
                    with (folder / "worker.log").open("a", encoding="utf-8") as log:
                        process = subprocess.Popen(job["command"], cwd=plan["snapshot"], env=env,
                                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    job.update(status="running", pid=process.pid, started_at=now(),
                               launches=job.get("launches", 0) + 1)
                    active[job["id"]] = process
                    counts[job["method"]] += 1
                write(output / "progress.json", {"at": now(), "jobs": jobs})
                write(output / "status.json", {"phase": "running", "at": now(), "running": len(active),
                                               "counts": dict(Counter(j["status"] for j in jobs))})
                if active:
                    time.sleep(poll_seconds)
            errors = [job["id"] for job in jobs if job["status"] != "completed"]
            write(output / "status.json", {"phase": "blocked" if errors else "completed", "at": now(),
                                           "failed_jobs": errors})
            journal(output, "blocked" if errors else "completed",
                    f"GEPA·SkillOpt 큐 종료. 완료 {len(jobs) - len(errors)}/{len(jobs)}; 미완료 {errors}.")
        except Exception as exc:
            write(output / "status.json", {"phase": "blocked", "at": now(),
                                           "error": f"{type(exc).__name__}: {exc}"})
            journal(output, "blocked", f"GEPA·SkillOpt 스케줄러 중단: {type(exc).__name__}: {exc}")
            raise
        finally:
            if lease is not None:
                lease.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--after-output", type=Path)
    parser.add_argument("--join-output", type=Path, help="Explicitly share the owned services of an active Qwen run")
    parser.add_argument("--max-running-per-method", type=int, default=6)
    parser.add_argument("--terminalbench2-version", choices=("2.0", "2.1"), default=None,
                        help="Pin Terminal-Bench for a newly created GEPA/SkillOpt queue")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("poll-seconds must be positive")
    if args.run:
        if args.terminalbench2_version is not None:
            plan = read(args.output.resolve() / "launch_manifest.json")
            if plan.get("terminalbench2_version") != args.terminalbench2_version:
                parser.error("The frozen queue has another Terminal-Bench release; create a separate queue")
        run(args.output.resolve(), args.poll_seconds)
    else:
        if not args.after_output:
            parser.error("--after-output is required when creating a queue")
        plan = freeze(args.output.resolve(), args.after_output.resolve(),
                      max_running_per_method=args.max_running_per_method,
                      join_output=args.join_output.resolve() if args.join_output else None,
                      terminalbench2_version=args.terminalbench2_version)
        print({"jobs": len(plan["jobs"]), "snapshot": plan["snapshot"]})


if __name__ == "__main__":
    main()
