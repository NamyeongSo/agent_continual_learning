# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Start BFCL checkpoint-50 test evaluation as each sequential run reaches it."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_matrix import command


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluation_command(row: dict, manifest: Path, output: Path, method: str) -> list[str]:
    args = command(row, manifest, output, method)
    args.remove("--evaluate-test")
    args.extend(("--evaluation-only", "--evaluate-test-checkpoint", "50",
                 "--evaluate-test-benchmark", "bfcl"))
    return args


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--port-block-base", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port_block_base <= 63535 or args.poll_seconds <= 0 or args.max_attempts < 1:
        parser.error("Invalid port block, poll interval, or retry limit")
    launch_path = args.launch_manifest.resolve()
    plan = json.loads(launch_path.read_text(encoding="utf-8"))
    rows = [row for row in plan["jobs"] if row["mode"] == "sequential"]
    if not rows:
        parser.error("Launch manifest has no sequential runs")
    if args.port_block_base + 1000 * len(rows) > 64535:
        parser.error("Port blocks extend past port 65535")
    root = launch_path.parent
    manifest = Path(plan["source_manifest"])
    method = str(plan["method"])
    jobs = {
        str(row["name"]): {
            "row": row, "output": root / str(row["name"]),
            "port": args.port_block_base + index * 1000,
            "attempts": 0, "status": "waiting", "pid": None,
        }
        for index, row in enumerate(rows)
    }
    if args.dry_run:
        for job in jobs.values():
            print(f"port={job['port']} " + " ".join(evaluation_command(
                job["row"], manifest, job["output"], method,
            )))
        return 0
    log_dir = root / "early_bfcl_logs"
    log_dir.mkdir(exist_ok=True)
    status_path = root / "early_bfcl_status.json"
    running: dict[str, subprocess.Popen] = {}
    env = os.environ.copy()
    while True:
        for name, job in jobs.items():
            result = job["output"] / "test_evaluation" / "cp050_bfcl.json"
            if result.is_file() and json.loads(result.read_text(encoding="utf-8")).get("complete"):
                job["status"] = "done"
            process = running.get(name)
            if process is not None and process.poll() is not None:
                del running[name]
                job["pid"] = None
                job["returncode"] = process.returncode
                if job["status"] != "done":
                    job["status"] = "waiting" if job["attempts"] < args.max_attempts else "failed"
            checkpoint = job["output"] / "checkpoints" / "checkpoint_050.json"
            if job["status"] == "waiting" and checkpoint.is_file() and name not in running:
                job["attempts"] += 1
                child_env = env | {"EXGENTIC_PORT_BLOCK_START": str(job["port"])}
                log_path = log_dir / f"{name}.log"
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"\nBFCL checkpoint-50 evaluation attempt {job['attempts']}\n")
                    log.flush()
                    process = subprocess.Popen(
                        evaluation_command(job["row"], manifest, job["output"], method),
                        cwd=Path(__file__).resolve().parents[2], env=child_env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                running[name] = process
                job["pid"] = process.pid
                job["status"] = "running"
                print(f"START {name} BFCL checkpoint=50 pid={process.pid} port={job['port']}", flush=True)
        matrix_status_path = root / "matrix_status.json"
        if matrix_status_path.is_file():
            matrix = json.loads(matrix_status_path.read_text(encoding="utf-8"))
            states = {str(row["name"]): row["status"] for row in matrix["experiments"]}
            for name, job in jobs.items():
                if job["status"] == "waiting" and states.get(name) in ("failed", "skipped"):
                    job["status"] = "failed"
        write_json(status_path, {"jobs": [{key: value for key, value in job.items() if key not in ("row", "output")}
                                          | {"name": name, "output": str(job["output"])}
                                          for name, job in jobs.items()]})
        if all(job["status"] in ("done", "failed") for job in jobs.values()):
            return 0 if all(job["status"] == "done" for job in jobs.values()) else 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
