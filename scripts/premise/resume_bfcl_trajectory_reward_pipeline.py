# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Finish cached BFCL process-reward judgments after the shared server resumes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--not-before", required=True, help="Timezone-aware ISO timestamp")
    parser.add_argument("--concurrency", type=int, default=64)
    args = parser.parse_args()
    target = datetime.fromisoformat(args.not_before)
    if target.tzinfo is None:
        parser.error("--not-before must include a timezone")
    folder = args.folder.resolve()
    lock = (folder / "resume_pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = Path(__file__).resolve().parents[2]
    state_path = folder / "resume_status.json"
    scripts = root / "scripts/premise"
    names = [
        "run_bfcl_trajectory_reward_judging.py", "bfcl_trajectory_reward_judge.py",
        "analyze_bfcl_trajectory_rewards.py", "plot_bfcl_trajectory_rewards.py",
        "bfcl_trajectory_selection_rewards.py", "run_bfcl_trajectory_judge_reliability.py",
        "audit_bfcl_trajectory_reward_judge.py",
    ]
    hashes = {name: hashlib.sha256((scripts / name).read_bytes()).hexdigest()
              for name in names if (scripts / name).exists()}

    def write(phase: str, **extra):
        state = {"phase": phase, "updated_at": datetime.now(UTC).isoformat(), "pid": os.getpid(),
                 "not_before": target.isoformat(), "concurrency": args.concurrency,
                 "actor_rollouts": 0, "server_start_requested": False,
                 "source_sha256_at_schedule": hashes, **extra}
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)

    write("scheduled")
    while datetime.now(UTC) < target:
        write("scheduled", remaining_seconds=(target - datetime.now(UTC)).total_seconds())
        time.sleep(min(20, max(.01, (target - datetime.now(UTC)).total_seconds())))

    def execute(name, arguments):
        write(name)
        command = [sys.executable, str(scripts / name), *map(str, arguments)]
        with (folder / f"resume_{name.removesuffix('.py')}.log").open("a") as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        execute("run_bfcl_trajectory_reward_judging.py", [
            "--folder", folder, "--concurrency", args.concurrency, "--timeout", 600,
            "--retry-infrastructure", "--wait-server",
        ])
        reliability = scripts / "run_bfcl_trajectory_judge_reliability.py"
        if reliability.exists():
            execute(reliability.name, ["--folder", folder])
        audit = scripts / "audit_bfcl_trajectory_reward_judge.py"
        if audit.exists():
            execute(audit.name, ["--folder", folder])
        execute("analyze_bfcl_trajectory_rewards.py", ["--output", folder, "--skip-plots"])
        execute("plot_bfcl_trajectory_rewards.py", ["--summary", folder / "trajectory_reward_summary.json"])
        status = json.loads((folder / "judge_status.json").read_text())
        write("finished", judge_successful=status["successful"], judge_total=status["total"],
              judge_status_counts=status["status_counts"],
              source_sha256_at_finish={name: hashlib.sha256((scripts / name).read_bytes()).hexdigest()
                                       for name in names if (scripts / name).exists()})
    except Exception as error:
        write("error", error=str(error))
        raise


if __name__ == "__main__":
    main()
