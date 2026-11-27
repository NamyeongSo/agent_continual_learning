# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Launch six independent ACE training runs against an existing model server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from checkpoint_recovery import session_records
from evaluate_test_checkpoints import start_controller
from run_experiment import task_order_from_train_manifest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ["bfcl", "appworld", "terminalbench2"]
SEEDS = (42, 43, 44)


def prepare_sequential_bfcl_prefix(old_base: Path, new_base: Path, jobs: list[dict], manifest_hash: str) -> dict:
    """Seed new sequential runs with their unaffected 50-task BFCL prefix."""
    old_launch = json.loads((old_base / "launch_manifest.json").read_text(encoding="utf-8"))
    if old_launch["source_manifest_sha256"] != manifest_hash:
        raise ValueError("Old and new task split manifests differ")
    old_jobs = {(item["mode"], item["seed"]): item for item in old_launch["jobs"]}
    provenance = {}
    for job in jobs:
        if job["mode"] != "sequential":
            continue
        seed = job["seed"]
        old_job = old_jobs[("sequential", seed)]
        if [tuple(item) for item in old_job["task_order"]] != [tuple(item) for item in job["task_order"]]:
            raise ValueError(f"Task order changed for sequential seed {seed}")
        if [slug for slug, _ in job["task_order"][:50]] != ["bfcl"] * 50:
            raise ValueError(f"First 50 tasks are not BFCL for sequential seed {seed}")
        source = Path(old_job["output_dir"])
        metrics = [json.loads(line) for line in (source / "online_metrics.jsonl").read_text().splitlines()]
        if len(metrics) < 50 or [
            (row["benchmark_slug"], str(row["task_id"])) for row in metrics[:50]
        ] != [tuple(item) for item in job["task_order"][:50]]:
            raise ValueError(f"BFCL metrics prefix differs for sequential seed {seed}")
        snapshot_path = old_base / "test_evaluation" / "snapshots" / f"sequential_s{seed}" / "checkpoint_050.json"
        if snapshot_path.is_file():
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            checkpoint_path = Path(snapshot["source_checkpoint"])
            checkpoint_hash = snapshot["source_checkpoint_sha256"]
        else:
            record = session_records(source).get(("bfcl", str(job["task_order"][49][1])))
            if not record or not record["checkpoint_path"]:
                raise ValueError(f"Missing last BFCL checkpoint for sequential seed {seed}")
            checkpoint_path = Path(record["checkpoint_path"])
            checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (checkpoint["store_id"] != "ace_sequential_global" or checkpoint["session_count"] != 50
                or hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint_hash):
            raise ValueError(f"Invalid BFCL checkpoint for sequential seed {seed}")
        source_session = checkpoint_path.parent.parent
        session_config = json.loads((source_session / "config.json").read_text(encoding="utf-8"))
        if (session_config["benchmark"] != "bfcl"
                or str(session_config["task_id"]) != str(job["task_order"][49][1])):
            raise ValueError(f"Last BFCL session differs for sequential seed {seed}")

        prefix = new_base / "reused_bfcl_prefix" / f"sequential_s{seed}"
        prefix.mkdir(parents=True)
        shutil.copy2(source / "experiment_config.json", prefix / "experiment_config.json")
        (prefix / "online_metrics.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metrics[:50]), encoding="utf-8"
        )
        shutil.copy2(checkpoint_path, prefix / "playbook_ace_sequential_global.json")

        # The test controller locates stage 50 from per-session checkpoint records.
        record = Path(job["output_dir"]) / "reused_bfcl_checkpoint" / "sessions" / source_session.name
        (record / "agent").mkdir(parents=True)
        for name in ("config.json", "results.json"):
            shutil.copy2(source_session / name, record / name)
        shutil.copy2(checkpoint_path, record / "agent" / "playbook_checkpoint.json")
        job["command"].extend(["--continue-from", str(prefix)])
        provenance[f"sequential_s{seed}"] = {
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": checkpoint_hash,
            "reused_train_tasks": 50,
            "continued_from": str(prefix),
        }
    return provenance


def make_jobs(args: argparse.Namespace) -> list[dict]:
    jobs = []
    for mode in ("sequential", "interleaved"):
        for seed in SEEDS:
            order = task_order_from_train_manifest(args.manifest, BENCHMARKS, seed, None, mode)
            counts = {slug: sum(name == slug for name, _ in order) for slug in BENCHMARKS}
            if counts != {"bfcl": 50, "appworld": 50, "terminalbench2": 30}:
                raise ValueError(f"Unexpected training split sizes for {mode} seed {seed}: {counts}")
            output = args.output_base / f"{mode}_s{seed}"
            command = [
                sys.executable, "-u", str(ROOT / "scripts/ace/run_experiment.py"),
                "--mode", mode, "--seed", str(seed),
                "--benchmarks", ",".join(BENCHMARKS),
                "--task-split-manifest", str(args.manifest), "--all-train-tasks",
                "--model", args.model, "--max-tokens", str(args.max_tokens),
                "--disable-thinking", "--training-time",
                "--appworld-max-turns", "50", "--bfcl-max-turns", "50",
                "--terminalbench2-max-turns", str(args.terminalbench2_max_turns),
                "--terminalbench2-actor", "terminus2", "--output-dir", str(output),
            ]
            jobs.append({
                "mode": mode, "seed": seed, "train_counts": counts,
                "task_order": order, "output_dir": str(output),
                "log": str(output / "experiment.log"),
                "port_block_start": args.port_base + 1000 * len(jobs),
                "command": command,
            })
    return jobs


def preflight(args: argparse.Namespace, env: dict[str, str], jobs: list[dict]) -> None:
    """Check shared dependencies before starting any training process."""
    import cloudpickle
    import pydantic
    import pydantic_core

    expected = [pydantic.__version__, pydantic_core.__version__, cloudpickle.__version__]
    for slug in BENCHMARKS:
        python = Path.home() / ".exgentic" / "benchmarks" / slug / "venv/bin/python"
        code = (
            "import json,pydantic,pydantic_core,cloudpickle;"
            "print(json.dumps([pydantic.__version__,pydantic_core.__version__,cloudpickle.__version__]))"
        )
        versions = json.loads(subprocess.check_output([str(python), "-c", code], text=True, timeout=20))
        if versions != expected:
            raise RuntimeError(f"{slug} RPC package versions {versions} differ from host {expected}")
    for job in jobs:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", job["port_block_start"]))
    if not env.get("DOCKER_HOST") and not Path("/var/run/docker.sock").is_socket():
        raise RuntimeError("Set DOCKER_HOST or ACE_DOCKER_HOST for Terminal-Bench")
    subprocess.run(
        ["docker", "info"], env=env, check=True, timeout=20,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    payload = {
        "model": args.model.removeprefix("openai/"),
        "messages": [{"role": "user", "content": "Call the ping tool."}],
        "tools": [{"type": "function", "function": {
            "name": "ping", "description": "Return pong",
            "parameters": {"type": "object", "properties": {}},
        }}],
        "tool_choice": "auto", "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + env["OPENAI_API_KEY"]},
    )
    for _attempt in range(3):
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        if result["choices"][0]["message"].get("tool_calls"):
            break
    else:
        raise RuntimeError("Model server did not return a tool call in three preflight checks")
    print(f"Preflight passed: RPC packages={expected}, Docker reachable, model tool calls working", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json",
    )
    parser.add_argument("--output-base", type=Path, default=ROOT / "scripts/ace/outputs" / (
        "ace_bfcl_appworld_tb2_seed42_44_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    ))
    parser.add_argument("--model", default="openai/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8006/v1"))
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--terminalbench2-max-turns", type=int, default=100)
    parser.add_argument("--port-base", type=int, default=36000)
    parser.add_argument("--reuse-sequential-bfcl-from", type=Path,
                        help="Reuse the 50-task BFCL prefix for sequential runs from a previous six-run output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.output_base = args.output_base.resolve()
    if args.reuse_sequential_bfcl_from is not None:
        args.reuse_sequential_bfcl_from = args.reuse_sequential_bfcl_from.resolve()
    if args.max_tokens < 1 or args.terminalbench2_max_turns < 1:
        parser.error("Token and turn limits must be positive")
    if not 1024 <= args.port_base <= 59535:
        parser.error("--port-base must leave room for six 1000-port blocks")
    jobs = make_jobs(args)
    for job in jobs:
        print(f"{job['mode']} seed={job['seed']}: {job['train_counts']}, ports={job['port_block_start']}", flush=True)
    if args.dry_run:
        return
    if args.output_base.exists():
        parser.error(f"Use a fresh output directory: {args.output_base}")
    env = os.environ.copy()
    env["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + env.get("PATH", "")
    env["OPENAI_API_BASE"] = args.base_url
    env.setdefault("OPENAI_API_KEY", "EMPTY")
    env.setdefault("DOCKER_HOST", env.get("ACE_DOCKER_HOST", ""))
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    env["EXGENTIC_LLM_TRACE_FORMAT"] = "dedup_v1"
    if shutil.which("docker", path=env["PATH"]) is None:
        parser.error("Docker CLI unavailable")
    preflight(args, env, jobs)
    args.output_base.mkdir(parents=True)
    reused_prefixes = (
        prepare_sequential_bfcl_prefix(
            args.reuse_sequential_bfcl_from, args.output_base, jobs,
            hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        ) if args.reuse_sequential_bfcl_from is not None else {}
    )
    manifest = {
        "created_at": datetime.now().isoformat(), "agent": "ace", "training_time": True,
        "model": args.model, "max_tokens": args.max_tokens,
        "terminalbench2_actor": "terminus2", "terminalbench2_max_turns": args.terminalbench2_max_turns,
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "reused_bfcl_prefixes": reused_prefixes,
        "jobs": jobs,
    }
    processes = []
    for job in jobs:
        output = Path(job["output_dir"])
        output.mkdir(exist_ok=True)
        run_env = env | {"EXGENTIC_PORT_BLOCK_START": str(job["port_block_start"])}
        with Path(job["log"]).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                job["command"], cwd=ROOT, env=run_env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        job["pid"] = process.pid
        processes.append(process)
        print(f"Started {job['mode']} seed={job['seed']} PID={process.pid}", flush=True)
    launch_path = args.output_base / "launch_manifest.json"
    launch_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    time.sleep(3)
    failed = [job for job, process in zip(jobs, processes, strict=True) if process.poll() is not None]
    if failed:
        failed_runs = [(job["mode"], job["seed"]) for job in failed]
        raise RuntimeError(f"Some runs exited at startup; inspect logs in {args.output_base}: {failed_runs}")
    print(f"All six processes are running. Launch manifest: {launch_path}", flush=True)
    controller_pid = start_controller(launch_path, env)
    print(f"Scheduled 54 frozen test evaluations after training; controller PID={controller_pid}", flush=True)


if __name__ == "__main__":
    main()
