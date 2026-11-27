# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Run six BFCL/AppWorld/Terminal-Bench experiments and their 54 checkpoint test groups."""

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
DEFAULT_MANIFEST = ROOT / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json"
DEFAULT_OUTPUT = ROOT / "outputs/premise_bfcl_appworld_terminalbench2_seed42_44"
PARETO_OUTPUT = ROOT / "outputs/ps_st_pareto_bfcl_appworld_terminalbench2_seed42_44"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def experiments() -> list[dict[str, object]]:
    return [
        {"name": f"s{seed}_{mode}_t{past}", "seed": seed, "mode": mode, "past": past}
        for seed in (42, 43, 44)
        for mode in ("sequential", "interleaved")
        for past in (3,)
    ]


def build_plan(manifest: Path, rows: list[dict[str, object]], method: str = "premise") -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from exgentic.agents.premise.pipeline import manifest_order
    from exgentic.core.experiment_limits import ACTOR_OUTPUT_TOKENS, BENCHMARK_MAX_TURNS, experiment_limits
    from exgentic.utils.task_splits import select_task_split_sections, validate_task_split

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    jobs, evaluations = [], []
    benchmarks = ["bfcl", "appworld", "terminalbench2"]
    for row in rows:
        seed = int(row["seed"])
        sections = select_task_split_sections(payload, benchmarks, seed)
        splits = {slug: validate_task_split(sections[slug], slug) for slug in benchmarks}
        for slug in benchmarks:
            expected = (30, 30, 29) if slug == "terminalbench2" else (50, 50, 50)
            if tuple(len(splits[slug][part]) for part in ("train", "val", "test")) != expected:
                raise ValueError(f"Unexpected train/val/test sizes for {slug}, seed={seed}")
        order, configs = manifest_order(manifest, benchmarks, seed, None, str(row["mode"]))
        jobs.append({**row, "task_order": order, "benchmark_configs": configs})
        if not row.get("train_only", False):
            for checkpoint in (50, 100, 130):
                for slug in benchmarks:
                    evaluations.append({"experiment": row["name"], "checkpoint": checkpoint,
                                        "benchmark": slug, "seed": seed, "task_ids": splits[slug]["test"]})
    return {"method": method, "model": "openai/Qwen/Qwen3.6-35B-A3B",
            "max_tokens": ACTOR_OUTPUT_TOKENS, "terminalbench2_max_turns": BENCHMARK_MAX_TURNS["terminalbench2"],
            "experiment_limits": {slug: experiment_limits(slug) for slug in benchmarks},
            "source_manifest": str(manifest),
            "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "jobs": jobs, "evaluation_plan": evaluations,
            "training_runs": len(jobs), "evaluation_groups": len(evaluations),
            "evaluation_tasks": sum(len(item["task_ids"]) for item in evaluations)}


def checkpoint_cursor(output: Path) -> int:
    state = output / "state.json"
    if not state.is_file():
        return 0
    return int(json.loads(state.read_text(encoding="utf-8"))["cursor"])


def experiment_complete(output: Path, train_only: bool = False) -> bool:
    if checkpoint_cursor(output) < 130:
        return False
    return train_only or (output / "test_evaluation" / "complete.json").is_file()


def process_identity(pid: int) -> tuple[list[str], str] | None:
    """Identify a live Linux process without confusing a reused PID with a worker."""
    try:
        process = Path("/proc") / str(pid)
        fields = (process / "stat").read_text().rpartition(")")[2].split()
        if fields[0] in ("Z", "X"):
            return None
        argv = (process / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
        return argv, fields[19]
    except (OSError, IndexError):
        return None


class ExistingExperimentProcess:
    """Monitor a worker left running when its previous launcher was replaced."""

    def __init__(self, pid: int, identity: tuple[list[str], str], output: Path, train_only: bool):
        self.pid = pid
        self.identity = identity
        self.output = output
        self.train_only = train_only
        self.returncode: int | None = None

    def poll(self) -> int | None:
        if self.returncode is None and process_identity(self.pid) != self.identity:
            self.returncode = 0 if experiment_complete(self.output, self.train_only) else 1
        return self.returncode

    def wait(self, timeout: float) -> int:
        deadline = time.monotonic() + timeout
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(f"adopted experiment {self.pid}", timeout)
            time.sleep(0.1)
        return self.returncode


def write_status(path: Path, rows: list[dict[str, object]]) -> None:
    for row in rows:
        row["cursor"] = checkpoint_cursor(path.parent / str(row["name"]))
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"updated_at": utc_now(), "experiments": rows}, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def command(row: dict[str, object], manifest: Path, output: Path, method: str = "premise") -> list[str]:
    args = [
        sys.executable, str(ROOT / "scripts/premise/run_experiment.py"),
        "--task-split-manifest", str(manifest),
        "--output-dir", str(output),
        "--benchmarks", "bfcl,appworld,terminalbench2",
        "--mode", str(row["mode"]),
        "--seed", str(row["seed"]),
        "--all-train-tasks",
        "--past", str(row["past"]),
        "--replay-workers", "28",
        "--skip-task-errors", str(row.get("skip_task_errors", "nontransient")),
        "--unlimited-benchmark-sessions",
    ]
    if not row.get("train_only", False):
        args.append("--evaluate-test")
    if method != "premise":
        args.extend(("--method", method))
    return args


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--method", choices=("premise", "ps_st_pareto"), default="premise")
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--port-block-base", type=int, default=20000,
                        help="First 1000-port block; use distinct bases for concurrent matrices")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--skip-config", action="append", default=[],
                        help="Leave a known-failed configuration untouched while resuming the rest")
    parser.add_argument("--skip-task-errors", choices=("none", "nontransient", "all"), default="nontransient")
    parser.add_argument("--past", type=int, choices=(3, 6), default=3)
    parser.add_argument("--modes", choices=("both", "sequential", "interleaved"), default="both")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--adopt-running", action="store_true",
                        help="Keep and monitor matching live workers from the previous matrix status")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_parallel <= 6:
        parser.error("--max-parallel must be between 1 and 6")
    if not 1024 <= args.port_block_base <= 59535:
        parser.error("--port-block-base must leave room for six 1000-port blocks")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    manifest = args.manifest.resolve()
    output = (args.output_root or (PARETO_OUTPUT if args.method == "ps_st_pareto" else DEFAULT_OUTPUT)).resolve()
    if not manifest.is_file():
        parser.error(f"Manifest does not exist: {manifest}")

    rows = [row for row in experiments() if args.modes == "both" or row["mode"] == args.modes]
    for row in rows:
        row["skip_task_errors"] = args.skip_task_errors
        row["train_only"] = args.train_only
        if args.past != 3:
            row["past"] = args.past
            row["name"] = str(row["name"]).replace("_t3", f"_t{args.past}")
    plan = build_plan(manifest, rows, args.method)
    known_names = {str(row["name"]) for row in rows}
    unknown_skips = set(args.skip_config) - known_names
    if unknown_skips:
        parser.error(f"Unknown configurations to skip: {sorted(unknown_skips)}")
    if args.dry_run:
        print(
            f"Training runs={plan['training_runs']} test groups={plan['evaluation_groups']} "
            f"test tasks={plan['evaluation_tasks']}"
        )
        for index, row in enumerate(rows):
            if row["name"] in args.skip_config:
                continue
            print(f"EXGENTIC_PORT_BLOCK_START={args.port_block_base + 1000 * index} " +
                  " ".join(command(row, manifest, output / str(row["name"]), args.method)))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    lock = (output / "launcher.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error(f"A matrix launcher already holds {output / 'launcher.lock'}")

    plan_path = output / "launch_manifest.json"
    canonical = json.loads(json.dumps(plan))
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != canonical:
        parser.error("Existing launch manifest differs; choose a new output directory")
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("OPENAI_API_BASE", "http://127.0.0.1:8006/v1")
    env.setdefault("OPENAI_API_KEY", "EMPTY")
    env.setdefault("DOCKER_HOST", env.get("ACE_DOCKER_HOST", "ssh://exgentic-daysi-docker:14233"))
    stopped = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    status_path = output / "matrix_status.json"
    previous = (
        {row["name"]: row for row in json.loads(status_path.read_text())["experiments"]}
        if args.adopt_running and status_path.exists() else {}
    )
    running: dict[str, tuple[subprocess.Popen[bytes] | ExistingExperimentProcess, dict[str, object]]] = {}
    for index, row in enumerate(rows):
        # Keep each experiment's auto-selected localhost service ports in a
        # disjoint 1000-port range, below the host's ephemeral-port range.
        row["port_block_start"] = args.port_block_base + 1000 * index
        row.update({"status": "done" if experiment_complete(output / str(row["name"]), args.train_only)
                    else "skipped" if row["name"] in args.skip_config else "pending",
                    "attempts": 0, "pid": None, "returncode": None})
        old = previous.get(row["name"], {})
        pid = old.get("pid")
        identity = process_identity(int(pid)) if pid else None
        if identity is not None:
            name = str(row["name"])
            if identity[0] != command(row, manifest, output / name, args.method):
                parser.error(f"Live worker {pid} does not match the configuration for {name}")
            row.update({"status": "running", "pid": int(pid), "attempts": max(1, int(old.get("attempts", 1))),
                        "port_block_start": old.get("port_block_start", row["port_block_start"]),
                        "started_at": old.get("started_at", utc_now())})
            running[name] = (ExistingExperimentProcess(int(pid), identity, output / name, args.train_only), row)
            print(f"{utc_now()} adopted {name} pid={pid}", flush=True)

    try:
        write_status(status_path, rows)
        while not stopped:
            for name, (process, row) in list(running.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                del running[name]
                row.update({"pid": None, "returncode": returncode, "ended_at": utc_now()})
                if returncode == 0 and experiment_complete(output / name, args.train_only):
                    row["status"] = "done"
                elif int(row["attempts"]) < args.max_attempts:
                    row["status"] = "pending"
                else:
                    row["status"] = "failed"
                print(f"{utc_now()} finished {name} rc={returncode} status={row['status']}", flush=True)

            while len(running) < args.max_parallel:
                row = next((item for item in rows if item["status"] == "pending"), None)
                if row is None:
                    break
                name = str(row["name"])
                row["attempts"] = int(row["attempts"]) + 1
                child_env = env.copy()
                child_env["EXGENTIC_PORT_BLOCK_START"] = str(row["port_block_start"])
                log_path = output / "logs" / f"{name}.log"
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"\n[{utc_now()}] launcher attempt {row['attempts']}\n")
                    log.flush()
                    process = subprocess.Popen(
                        command(row, manifest, output / name, args.method), cwd=ROOT,
                        env=child_env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                row.update({"status": "running", "pid": process.pid, "started_at": utc_now()})
                running[name] = (process, row)
                print(f"{utc_now()} started {name} pid={process.pid} attempt={row['attempts']} "
                      f"port_block={row['port_block_start']}-{int(row['port_block_start']) + 999}", flush=True)

            write_status(status_path, rows)
            if not running and all(row["status"] in ("done", "failed", "skipped") for row in rows):
                break
            time.sleep(1)
    finally:
        if stopped:
            for process, row in running.values():
                os.killpg(process.pid, signal.SIGTERM)
                row["status"] = "stopped"
            for process, _row in running.values():
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            write_status(status_path, rows)

    failures = [str(row["name"]) for row in rows if row["status"] != "done"]
    print(f"{utc_now()} matrix complete; failed={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
