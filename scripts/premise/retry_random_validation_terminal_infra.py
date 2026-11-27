# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Retry infrastructure-failed TerminalBench2 test tasks after the shared queue drains."""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from exgentic.agents.premise.pipeline import write_json  # noqa: E402

DEFAULT_OUTPUT = ROOT / "outputs/premise_random_val4_sequential_existing_policy_seed42_44_20261007"
QUEUE_SCRIPT = ROOT / "scripts/premise/evaluate_random_validation_test_queue.py"
INFRA_ERRORS = (
    "Failed to start tmux session",
    "Failed to download verifier directory from environment",
    "no server running on /tmp/tmux-",
    "benchmark/harbor/agent/premise_episodes.json",
)


def now() -> str:
    return datetime.now(UTC).isoformat()


def live_queue(pid: int, start_ticks: str) -> bool:
    path = Path("/proc") / str(pid)
    try:
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        command = (path / "cmdline").read_bytes()
    except (OSError, IndexError):
        return False
    return (
        fields[0] not in ("Z", "X") and fields[19] == start_ticks
        and b"evaluate_random_validation_test_queue.py" in command
    )


def errors_to_retry(output: Path) -> dict[int, list[int]]:
    rows = {}
    for seed in (42, 43, 44):
        path = output / f"s{seed}_sequential_t3/test_evaluation/cp130_terminalbench2.json"
        if not path.exists():
            rows[seed] = []
            continue
        group = json.loads(path.read_text(encoding="utf-8"))
        rows[seed] = [
            index for index, record in enumerate(group["results"])
            if record is not None and record.get("status") == "error"
            and any(marker in record.get("error", "") for marker in INFRA_ERRORS)
        ]
    return rows


def clear_infra_errors(output: Path, attempt: int) -> dict[int, list[int]]:
    rows = errors_to_retry(output)
    if not any(rows.values()):
        return rows
    archive = output / "queue_recovery" / f"attempt_{attempt}"
    archive.mkdir(parents=True, exist_ok=False)
    for seed, indices in rows.items():
        if not indices:
            continue
        source = output / f"s{seed}_sequential_t3/test_evaluation/cp130_terminalbench2.json"
        shutil.copy2(source, archive / f"s{seed}_cp130_terminalbench2_before.json")
        group = json.loads(source.read_text(encoding="utf-8"))
        for index in indices:
            group["results"][index] = None
        group["complete"] = False
        group.pop("mean_official_score", None)
        group.pop("execution_errors", None)
        write_json(source, group)
        (source.parent / "complete.json").unlink(missing_ok=True)
    (output / "queue_complete.json").unlink(missing_ok=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wait-pid", type=int, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    status_path = output / "queue_recovery_status.json"
    proc = Path("/proc") / str(args.wait_pid)
    fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    start_ticks = fields[19]
    with (output / "queue_recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(status_path, {"phase": "waiting_for_initial_queue", "updated_at": now(),
                                 "initial_queue_pid": args.wait_pid})
        while live_queue(args.wait_pid, start_ticks):
            time.sleep(15)
        for attempt, workers in enumerate((8, 4, 2), start=1):
            rows = clear_infra_errors(output, attempt)
            count = sum(map(len, rows.values()))
            if count == 0:
                if not all((output / f"s{seed}_sequential_t3/test_evaluation/complete.json").exists()
                           for seed in (42, 43, 44)):
                    write_json(status_path, {"phase": "failed", "updated_at": now(),
                                             "error": "Initial evaluation queue exited before all groups completed"})
                    return 1
                write_json(status_path, {"phase": "complete", "updated_at": now(),
                                         "remaining_infra_errors": 0})
                print("RECOVERY COMPLETE: no infrastructure errors remain", flush=True)
                return 0
            write_json(status_path, {"phase": "retrying", "updated_at": now(), "attempt": attempt,
                                     "workers": workers, "retry_task_indices": rows})
            print(f"RETRY attempt={attempt} workers={workers} tasks={count} indices={rows}", flush=True)
            with (output / f"queue_retry_attempt_{attempt}.log").open("a") as log:
                result = subprocess.run(
                    [sys.executable, "-u", str(QUEUE_SCRIPT), "--output", str(output), "--workers", str(workers)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                )
            print(f"EXIT attempt={attempt} code={result.returncode} remaining={errors_to_retry(output)}", flush=True)
            if result.returncode != 0:
                write_json(status_path, {"phase": "failed", "updated_at": now(), "attempt": attempt,
                                         "exit_code": result.returncode})
                return result.returncode
        remaining = errors_to_retry(output)
        count = sum(map(len, remaining.values()))
        write_json(status_path, {"phase": "failed" if count else "complete", "updated_at": now(),
                                 "remaining_infra_errors": count, "indices": remaining})
        return 1 if count else 0


if __name__ == "__main__":
    raise SystemExit(main())
