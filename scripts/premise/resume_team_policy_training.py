# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Resume PREMiSE teammate-policy sequential runs and train fresh interleaved runs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exgentic.agents.premise.pipeline import write_json  # noqa: E402
from scripts.premise.restart_sequential import (  # noqa: E402
    configure_workspace_source,
    fork_interleaved_initial,
    training_command,
)

RUN_ROOT = ROOT / "outputs/ace_premise_sequential_team_policy_20261005_1943"
SOURCE_ROOT = ROOT / "outputs/premise_bfcl_appworld_terminalbench2_seed42_44_scorefix_20260928"
BASE_URL = "http://127.0.0.1:18006/v1"
SEEDS = (42, 43, 44)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_cursor(directory: Path) -> int:
    return int(json.loads((directory / "state.json").read_text())["cursor"])


def prepare(root: Path) -> tuple[Path, dict]:
    root = root.resolve()
    policy = root / "appworld_team_policy.txt"
    original = json.loads((root / "launch_manifest.json").read_text())
    if digest(policy) != original["policy_sha256"]:
        raise ValueError("Team policy changed since training started")
    pause = json.loads((root / "pause_summary.json").read_text())
    frozen = {}
    for item in pause["runs"]:
        if item["run"].startswith("premise_sequential"):
            seed = int(item["run"].rsplit("s", 1)[-1])
            checkpoint = root / "premise" / f"s{seed}_sequential_t3/checkpoints/checkpoint_050.json"
            expected = next(iter(item["checkpoints"].values()))
            if digest(checkpoint) != expected:
                raise ValueError(f"Sequential seed {seed} checkpoint 50 changed")
            frozen[str(seed)] = expected
    if set(frozen) != set(map(str, SEEDS)):
        raise ValueError("Missing sequential checkpoint 50 provenance")
    for seed in SEEDS:
        directory = root / "premise" / f"s{seed}_interleaved_t3"
        if not directory.exists():
            fork_interleaved_initial(
                SOURCE_ROOT / f"s{seed}_interleaved_t3", directory, seed, BASE_URL, "team_policy"
            )
        state = json.loads((directory / "state.json").read_text())
        if state["config"]["mode"] != "interleaved" or state["config"]["seed"] != seed:
            raise ValueError(f"Unexpected interleaved state for seed {seed}")
    identity = {
        "policy_sha256": digest(policy),
        "base_url": BASE_URL,
        "seeds": list(SEEDS),
        "modes": ["sequential", "interleaved"],
        "sequential_checkpoint_050_sha256": frozen,
        "training_targets": [100, 130],
        "cp100_appworld_evaluation_workers": 50,
        "source_interleaved": str(SOURCE_ROOT),
        "source_hashes": {
            "pipeline": digest(ROOT / "src/exgentic/agents/premise/pipeline.py"),
            "evaluator": digest(ROOT / "scripts/premise/evaluate_team_policy_cp100.py"),
        },
    }
    manifest = root / "premise_resume_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Existing PREMiSE resume plan differs")
    write_json(manifest, identity)
    return policy, identity


def controller(root: Path, workers: int) -> int:
    policy, identity = prepare(root)
    configure_workspace_source()
    environment = os.environ.copy()
    environment["OPENAI_API_BASE"] = BASE_URL
    environment.setdefault("OPENAI_API_KEY", "EMPTY")
    environment["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    environment["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + environment.get("PATH", "")
    environment.setdefault("DOCKER_HOST", "ssh://exgentic-daysi-docker:14233")
    subprocess.run(["docker", "info"], env=environment, stdout=subprocess.DEVNULL, check=True, timeout=30)
    root = root.resolve()
    status_path = root / "premise_resume_status.json"
    status_lock = threading.Lock()
    eval_lock_path = root / "premise_cp100_eval.lock"
    status = {f"s{seed}_{mode}": {"phase": "waiting", "cursor": read_cursor(root / "premise" / f"s{seed}_{mode}_t3")}
              for seed in SEEDS for mode in ("sequential", "interleaved")}

    def update(name: str, **fields):
        with status_lock:
            status[name].update(fields)
            write_json(status_path, {
                "pid": os.getpid(),
                "updated_at": datetime.now(UTC).isoformat(),
                "policy_sha256": identity["policy_sha256"],
                "jobs": status,
            })

    def command(directory: Path, target: int) -> list[str]:
        config = json.loads((directory / "state.json").read_text())["config"]
        return [*training_command(directory, config, BASE_URL, "team_policy"),
            "--appworld-policy", str(policy), "--stop-after", str(target)
        ]

    def evaluate(directory: Path, name: str):
        output = directory / "team_policy_test_cp100"
        completed = output / "complete.json"
        if completed.exists() and json.loads(completed.read_text()).get("complete"):
            update(name, evaluation="complete")
            return
        with eval_lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            log_path = directory / "team_policy_cp100.log"
            cmd = [
                str(ROOT / ".venv/bin/python"), "-u",
                str(ROOT / "scripts/premise/evaluate_team_policy_cp100.py"),
                "--run-dir", str(directory), "--policy", str(policy),
                "--workers", "50", "--base-url", BASE_URL,
            ]
            with log_path.open("a") as log:
                process = subprocess.Popen(cmd, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                update(name, phase="evaluating_cp100", eval_pid=process.pid, eval_log=str(log_path))
                code = process.wait()
            update(name, evaluation="complete" if code == 0 else "errors", eval_exit_code=code)
            if not completed.exists():
                raise RuntimeError(f"{name}: checkpoint 100 evaluation did not produce a completion record")

    def one(seed: int) -> None:
        for mode in ("sequential", "interleaved"):
            name = f"s{seed}_{mode}"
            directory = root / "premise" / f"{name}_t3"
            log_path = root / "logs" / f"premise_resume_{name}.log"
            for target in (100, 130):
                cursor = read_cursor(directory)
                if cursor < target:
                    if digest(policy) != identity["policy_sha256"]:
                        raise ValueError("Team policy changed during training")
                    with log_path.open("a") as log:
                        process = subprocess.Popen(
                            command(directory, target), cwd=ROOT, env=environment,
                            stdout=log, stderr=subprocess.STDOUT,
                        )
                        update(name, phase=f"training_to_{target}", cursor=cursor,
                               train_pid=process.pid, log=str(log_path))
                        code = process.wait()
                    cursor = read_cursor(directory)
                    update(name, cursor=cursor, train_exit_code=code)
                    if code != 0 or cursor < target:
                        raise RuntimeError(f"{name}: training stopped at {cursor}/{target}, exit={code}")
                if target == 100:
                    checkpoint = directory / "checkpoints/checkpoint_100.json"
                    if not checkpoint.is_file():
                        raise FileNotFoundError(checkpoint)
                    update(name, checkpoint_100_sha256=digest(checkpoint))
                    evaluate(directory, name)
            update(name, phase="complete", cursor=read_cursor(directory))

    update("s42_sequential")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(one, seed): seed for seed in SEEDS}
        failed = False
        for future in as_completed(pending):
            seed = pending[future]
            try:
                future.result()
            except Exception as exc:
                failed = True
                modes = [mode for mode in ("sequential", "interleaved")
                         if status[f"s{seed}_{mode}"]["phase"] != "complete"]
                name = f"s{seed}_{modes[0]}" if modes else f"s{seed}_sequential"
                update(name, phase="error", error=f"{type(exc).__name__}: {exc}")
                print(f"ERROR seed={seed}: {type(exc).__name__}: {exc}", flush=True)
    write_json(root / "premise_resume_complete.json", {
        "success": not failed, "completed_at": datetime.now(UTC).isoformat(), "jobs": status,
    })
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUN_ROOT)
    parser.add_argument("--parallel-seeds", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.parallel_seeds <= 3:
        parser.error("parallel-seeds must be 1..3")
    root = args.root.resolve()
    with (root / "premise_resume_controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return controller(root, args.parallel_seeds)


if __name__ == "__main__":
    raise SystemExit(main())
