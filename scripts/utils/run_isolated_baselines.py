# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Train and test 27 isolated jobs, three seeds per method and benchmark."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json"
BENCHMARKS = ("bfcl", "appworld", "terminalbench2")
METHODS = ("ace", "premise", "ps_st_pareto")
SEEDS = (42, 43, 44)
COUNTS = {"bfcl": (50, 50), "appworld": (50, 50), "terminalbench2": (30, 29)}


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def process_identity(pid: int) -> tuple[list[str], str] | None:
    try:
        process = Path("/proc") / str(pid)
        fields = (process / "stat").read_text().rpartition(")")[2].split()
        if fields[0] in ("Z", "X"):
            return None
        argv = (process / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
        return argv, fields[19]
    except (OSError, IndexError, UnicodeDecodeError):
        return None


class ExistingWorker:
    """Track a worker that survived a previous controller process."""

    def __init__(self, pid: int, identity: tuple[list[str], str], row: dict):
        self.pid = pid
        self.identity = identity
        self.row = row

    def poll(self) -> int | None:
        if process_identity(self.pid) == self.identity:
            return None
        return 0 if job_complete(self.row) else 1

    def wait(self, timeout: int) -> int:
        deadline = time.monotonic() + timeout
        while (result := self.poll()) is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(f"isolated worker {self.pid}", timeout)
            time.sleep(0.1)
        return result


def plan_jobs(root: Path, manifest: Path) -> list[dict]:
    sys.path.insert(0, str(ROOT / "src"))
    from exgentic.utils.task_splits import select_task_split_sections, validate_task_split

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = []
    for method in METHODS:
        for benchmark in BENCHMARKS:
            train_count, test_count = COUNTS[benchmark]
            for seed in SEEDS:
                section = select_task_split_sections(payload, [benchmark], seed)[benchmark]
                split = validate_task_split(section, benchmark)
                if (len(split["train"]), len(split["test"])) != (train_count, test_count):
                    raise ValueError(f"Unexpected {benchmark}_{seed} split size")
                name = f"{method}_{benchmark}_s{seed}"
                rows.append({"name": name, "method": method, "benchmark": benchmark, "seed": seed,
                             "train_count": train_count, "test_count": test_count,
                             "train_ids_sha256": hashlib.sha256(json.dumps(split["train"]).encode()).hexdigest(),
                             "test_ids_sha256": hashlib.sha256(json.dumps(split["test"]).encode()).hexdigest(),
                             "output": str(root / benchmark / method / f"s{seed}")})
    return rows


def job_complete(row: dict) -> bool:
    output = Path(row["output"])
    benchmark = row["benchmark"]
    count = row["train_count"]
    if row["method"] == "ace":
        train = read_json(output / "train/training_complete.json")
        summary = read_json(output / "evaluation" / f"isolated_{benchmark}_s{row['seed']}" /
                            f"checkpoint_{count:03d}" / benchmark / "summary.json")
        return (train.get("completed_task_count") == count
                and len(summary.get("sessions", [])) == row["test_count"])
    state = read_json(output / "run/state.json")
    result = read_json(output / "run/test_evaluation" / f"cp{count:03d}_{benchmark}.json")
    return (state.get("cursor") == count and result.get("complete") is True
            and len(result.get("results", [])) == row["test_count"])


def base_command(row: dict, manifest: Path) -> list[str]:
    output = Path(row["output"])
    python = str(ROOT / ".venv/bin/python")
    if row["method"] == "ace":
        return [python, "-u", str(ROOT / "scripts/ace/run_experiment.py"),
                "--mode", "isolated", "--seed", str(row["seed"]), "--benchmarks", row["benchmark"],
                "--task-split-manifest", str(manifest), "--all-train-tasks", "--resume-sessions",
                "--model", "openai/Qwen/Qwen3.6-35B-A3B", "--max-tokens", "8192",
                "--disable-thinking", "--training-time", "--bfcl-max-turns", "50",
                "--appworld-max-turns", "50", "--terminalbench2-max-turns", "100",
                "--terminalbench2-actor", "terminus2", "--output-dir", str(output / "train")]
    command = [python, "-u", str(ROOT / "scripts/premise/run_experiment.py"),
               "--task-split-manifest", str(manifest), "--output-dir", str(output / "run"),
               "--benchmarks", row["benchmark"], "--mode", "isolated", "--seed", str(row["seed"]),
               "--all-train-tasks", "--past", "3", "--replay-workers", "28",
               "--skip-task-errors", "nontransient", "--unlimited-benchmark-sessions",
               "--terminalbench2-max-turns", "100", "--evaluate-test"]
    if row["method"] == "ps_st_pareto":
        command.extend(("--method", "ps_st_pareto"))
    if read_json(output / "run/state.json").get("cursor") == row["train_count"]:
        command.append("--evaluation-only")
    return command


def run_worker(row: dict, manifest: Path) -> int:
    output = Path(row["output"])
    output.mkdir(parents=True, exist_ok=True)
    if job_complete(row):
        return 0
    command = base_command(row, manifest)
    if row["method"] == "ace":
        if read_json(output / "train/training_complete.json").get("completed_task_count") != row["train_count"]:
            if subprocess.run(command, cwd=ROOT, check=False).returncode:
                return 1
        evaluate = [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "scripts/ace/evaluate_isolated.py"),
                    "--train-dir", str(output / "train"), "--output-dir", str(output / "evaluation"),
                    "--manifest", str(manifest), "--benchmark", row["benchmark"],
                    "--seed", str(row["seed"])]
        if subprocess.run(evaluate, cwd=ROOT, check=False).returncode:
            return 1
    elif subprocess.run(command, cwd=ROOT, check=False).returncode:
        return 1
    return 0 if job_complete(row) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/isolated_baselines_20260930")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--port-block-base", type=int, default=27000)
    parser.add_argument("--worker", help="Run one named training/evaluation job")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_parallel <= 3 or args.max_attempts < 1:
        parser.error("Invalid parallelism or attempt count")
    if not 1024 <= args.port_block_base <= 59535 - 1000 * args.max_parallel:
        parser.error("Port blocks exceed the allowed range")
    root = args.output_root.resolve()
    manifest = args.manifest.resolve()
    rows = plan_jobs(root, manifest)
    rows_by_name = {row["name"]: row for row in rows}
    if args.dry_run:
        print(json.dumps({"learning_jobs": len(rows), "evaluation_jobs": len(rows),
                          "method_order": METHODS, "benchmark_order": BENCHMARKS, "jobs": rows}, indent=2))
        return 0
    if args.worker:
        row = next((item for item in rows if item["name"] == args.worker), None)
        if row is None:
            parser.error(f"Unknown job: {args.worker}")
        return run_worker(row, manifest)

    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "launcher.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Isolated baseline launcher already running")
    source_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    plan = {"created_at": datetime.now(UTC).isoformat(), "source_manifest": str(manifest),
            "source_manifest_sha256": source_hash, "model": "openai/Qwen/Qwen3.6-35B-A3B",
            "method_order": METHODS, "benchmark_order": BENCHMARKS,
            "jobs": rows, "learning_jobs": 27, "evaluation_jobs": 27}
    plan_path = root / "launch_manifest.json"
    if plan_path.exists():
        old = read_json(plan_path)
        if any(old.get(key) != plan[key] for key in ("source_manifest", "source_manifest_sha256", "jobs")):
            parser.error("Existing isolated launch manifest differs")
    else:
        save_json(plan_path, plan)
    env = os.environ.copy()
    env["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + env.get("PATH", "")
    env.setdefault("OPENAI_API_BASE", "http://127.0.0.1:8006/v1")
    env.setdefault("OPENAI_API_KEY", "EMPTY")
    env.setdefault("DOCKER_HOST", "ssh://exgentic-daysi-docker:14233")
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    env["EXGENTIC_LLM_TRACE_FORMAT"] = "dedup_v1"
    progress = read_json(root / "matrix_status.json").get("jobs", {})
    for row in rows:
        progress.setdefault(row["name"], {"status": "pending", "attempts": 0})
    running: dict[str, tuple[subprocess.Popen | ExistingWorker, int]] = {}
    for row in rows:
        name = row["name"]
        record = progress[name]
        if job_complete(row):
            record["status"] = "done"
        elif record["status"] == "running" and record.get("pid"):
            pid = int(record["pid"])
            identity = process_identity(pid)
            if (identity is not None and "--worker" in identity[0] and "--output-root" in identity[0]
                    and identity[0][identity[0].index("--worker") + 1] == name
                    and identity[0][identity[0].index("--output-root") + 1] == str(root)):
                block = int(record["port_block_start"])
                slot = (block - args.port_block_base) // 1000
                if not 0 <= slot < args.max_parallel:
                    parser.error(f"Live worker {name} has an unexpected port block")
                running[name] = (ExistingWorker(pid, identity, row), slot)
                print(f"{name} adopted pid={pid} slot={slot}", flush=True)
            else:
                record["status"] = "pending" if record["attempts"] < args.max_attempts else "failed"
                record["pid"] = None
        elif record["status"] in ("done", "stopped"):
            record["status"] = "pending"
    stopped = False

    def request_stop(_signal: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def write_status() -> None:
        save_json(root / "matrix_status.json", {"updated_at": datetime.now(UTC).isoformat(),
                                                "jobs": progress, "running": list(running)})

    try:
        for method in METHODS:
            for benchmark in BENCHMARKS:
                stage = [row for row in rows if row["method"] == method and row["benchmark"] == benchmark]
                while not stopped:
                    for name, (process, _slot) in list(running.items()):
                        result = process.poll()
                        if result is None:
                            continue
                        del running[name]
                        row = rows_by_name[name]
                        if result == 0 and job_complete(row):
                            status = "done"
                        elif progress[name]["attempts"] < args.max_attempts:
                            status = "pending"
                        else:
                            status = "failed"
                        progress[name].update({"status": status, "pid": None, "returncode": result,
                                               "ended_at": datetime.now(UTC).isoformat()})
                        print(f"{name} finished rc={result} status={status}", flush=True)
                    for row in stage:
                        if job_complete(row):
                            progress[row["name"]]["status"] = "done"
                    while len(running) < args.max_parallel:
                        row = next((item for item in stage if progress[item["name"]]["status"] == "pending"), None)
                        if row is None:
                            break
                        name = row["name"]
                        slot = next(index for index in range(args.max_parallel)
                                    if index not in {value[1] for value in running.values()})
                        child_env = env | {"EXGENTIC_PORT_BLOCK_START": str(args.port_block_base + slot * 1000)}
                        log = Path(row["output"]) / "worker.log"
                        log.parent.mkdir(parents=True, exist_ok=True)
                        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                                   "--output-root", str(root), "--manifest", str(manifest), "--worker", name]
                        with log.open("a", encoding="utf-8") as stream:
                            attempt = progress[name]["attempts"] + 1
                            stream.write(f"\n[{datetime.now(UTC).isoformat()}] attempt {attempt}\n")
                            stream.flush()
                            process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdin=subprocess.DEVNULL,
                                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                        progress[name].update({"status": "running", "pid": process.pid,
                                               "attempts": progress[name]["attempts"] + 1,
                                               "started_at": datetime.now(UTC).isoformat(),
                                               "port_block_start": args.port_block_base + slot * 1000})
                        running[name] = (process, slot)
                        print(f"{name} started pid={process.pid} slot={slot}", flush=True)
                    write_status()
                    if all(progress[row["name"]]["status"] == "done" for row in stage):
                        print(f"{method} {benchmark} isolated stage complete: 3/3 evaluations", flush=True)
                        break
                    if not running and any(progress[row["name"]]["status"] == "failed" for row in stage):
                        print(f"{method} {benchmark} isolated stage failed", flush=True)
                        return 1
                    time.sleep(2)
    finally:
        if stopped:
            for process, _slot in running.values():
                os.killpg(process.pid, signal.SIGTERM)
            for process, _slot in running.values():
                process.wait(timeout=30)
            for name in running:
                progress[name]["status"] = "stopped"
            write_status()
    return 0 if not stopped else 130


if __name__ == "__main__":
    raise SystemExit(main())
