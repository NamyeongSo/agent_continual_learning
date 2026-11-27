# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Finish frozen random-validation sequential test groups in one shared queue."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from run_experiment import resolve_docker_ssh_port  # noqa: E402

from exgentic.agents.premise.evaluation_queue import CheckpointEvaluationQueue  # noqa: E402
from exgentic.agents.premise.pipeline import PremisePipeline, write_json  # noqa: E402

DEFAULT_OUTPUT = ROOT / "outputs/premise_random_val4_sequential_existing_policy_seed42_44_20261007"
SEEDS = (42, 43, 44)
CHECKPOINTS = (50, 100, 130)
BENCHMARKS = ("bfcl", "appworld", "terminalbench2")
BASE_URL = "http://127.0.0.1:18006/v1"
EXECUTION_TAG = "random_val4_original_policy"


def now() -> str:
    return datetime.now(UTC).isoformat()


def pipeline_for(output: Path, seed: int) -> PremisePipeline:
    directory = output / f"s{seed}_sequential_t3"
    saved = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    config = saved["config"]
    if saved["cursor"] != 130 or not (directory / "training_complete.json").exists():
        raise ValueError(f"Training is incomplete for seed {seed}")
    if (config["seed"], config["mode"], config["candidate_validation"], config["validation_scope"]) != (
        seed, "sequential", "random_val4", "current",
    ):
        raise ValueError(f"Unexpected training configuration for seed {seed}")
    if tuple(config["benchmarks"]) != BENCHMARKS:
        raise ValueError(f"Unexpected benchmarks for seed {seed}")
    return PremisePipeline(
        output=directory, manifest=Path(config["manifest"]), benchmarks=config["benchmarks"],
        mode=config["mode"], seed=seed, count=config["count"], past_count=config["past_count"],
        model=config["model"], base_url=BASE_URL, replay_retries=config["replay_retries"],
        max_steps=config["max_steps"], llm_calls=32, llm_timeout=600, llm_retries=1,
        tool_calling_max_tokens=2048, execution_tag=EXECUTION_TAG,
        benchmark_limits=dict.fromkeys(config["benchmarks"]),
        candidate_validation=config["candidate_validation"], validation_scope=config["validation_scope"],
    )


def configure_runtime() -> None:
    os.environ["OPENAI_API_BASE"] = BASE_URL
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ.setdefault("DOCKER_HOST", "ssh://exgentic-daysi-docker:14233")
    os.environ["DOCKER_HOST"] = resolve_docker_ssh_port(os.environ["DOCKER_HOST"])
    os.environ["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + os.environ.get("PATH", "")
    source = str(ROOT / "src")
    os.environ["PYTHONPATH"] = source + os.pathsep + os.environ.get("PYTHONPATH", "")
    docker = shutil.which("docker")
    if docker is None:
        raise FileNotFoundError("Docker CLI is unavailable")
    subprocess.run([docker, "info"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)


def run(output: Path, workers: int) -> int:
    configure_runtime()
    pipelines = {seed: pipeline_for(output, seed) for seed in SEEDS}
    state_hashes = {
        seed: hashlib.sha256((pipeline.output / "state.json").read_bytes()).hexdigest()
        for seed, pipeline in pipelines.items()
    }
    status_path = output / "queue_status.json"
    queue = CheckpointEvaluationQueue(workers)
    keys = {}
    try:
        for seed, pipeline in pipelines.items():
            for checkpoint in CHECKPOINTS:
                keys[(seed, checkpoint)] = queue.enqueue(pipeline, checkpoint, BENCHMARKS)
        print(f"ENQUEUED pending_tasks={queue.submitted} workers={workers}", flush=True)
        while True:
            queue.poll()
            groups = {
                str(seed): {
                    f"cp{checkpoint:03d}": {
                        benchmark: (
                            "complete" if queue.complete((group_key,)) else
                            f"{sum(row is not None for row in queue.groups[group_key]['results'])}/"
                            f"{len(queue.groups[group_key]['results'])}"
                        )
                        for benchmark, group_key in zip(BENCHMARKS, keys[(seed, checkpoint)], strict=True)
                    }
                    for checkpoint in CHECKPOINTS
                }
                for seed in SEEDS
            }
            complete = all(queue.complete(group_keys) for group_keys in keys.values())
            status = {
                "phase": "complete" if complete else "running", "controller_pid": os.getpid(),
                "updated_at": now(), "queue": queue.status(), "groups": groups,
            }
            write_json(status_path, status)
            if complete:
                break
            time.sleep(2)
        for seed, pipeline in pipelines.items():
            state_path = pipeline.output / "state.json"
            if hashlib.sha256(state_path.read_bytes()).hexdigest() != state_hashes[seed]:
                raise RuntimeError(f"Training state changed during evaluation for seed {seed}")
            write_json(pipeline.output / "test_evaluation/complete.json", {
                "evaluation_groups": 9, "checkpoints": list(CHECKPOINTS), "total_tasks": 387,
            })
        write_json(output / "queue_complete.json", {"success": True, **status})
        print(f"COMPLETE finished={queue.finished} groups=27", flush=True)
        return 0
    finally:
        queue.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=50)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = args.output.resolve()
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(output, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
