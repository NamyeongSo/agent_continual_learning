# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Train three sequential PREMiSE runs using four shared validation tasks per update."""

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from exgentic.agents.premise.pipeline import manifest_order, write_json  # noqa: E402
from exgentic.utils.task_splits import select_task_split_sections, validate_task_split  # noqa: E402

MANIFEST = ROOT / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json"
DEFAULT_OUTPUT = ROOT / "outputs/premise_random_val4_sequential_existing_policy_seed42_44_20261007"
BENCHMARKS = ("bfcl", "appworld", "terminalbench2")
SEEDS = (42, 43, 44)
MODEL = "openai/Qwen/Qwen3.6-35B-A3B"
BASE_URL = "http://127.0.0.1:18006/v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat()


def plan(scope: str, parallel_seeds: int) -> dict:
    payload = json.loads(MANIFEST.read_text())
    jobs = {}
    for seed in SEEDS:
        sections = select_task_split_sections(payload, list(BENCHMARKS), seed)
        splits = {slug: validate_task_split(sections[slug], slug) for slug in BENCHMARKS}
        expected = {"bfcl": (50, 50, 50), "appworld": (50, 50, 50), "terminalbench2": (30, 30, 29)}
        if any(tuple(len(splits[slug][part]) for part in ("train", "val", "test")) != expected[slug]
               for slug in BENCHMARKS):
            raise ValueError(f"Unexpected split sizes for seed {seed}")
        order, benchmark_configs = manifest_order(MANIFEST, list(BENCHMARKS), seed, None, "sequential")
        if [slug for slug, _ in order] != ["bfcl"] * 50 + ["appworld"] * 50 + ["terminalbench2"] * 30:
            raise ValueError(f"Unexpected sequential order for seed {seed}")
        jobs[str(seed)] = {"task_count": len(order), "task_order": [list(item) for item in order],
                           "benchmark_configs": benchmark_configs,
                           "validation_task_ids": {slug: splits[slug]["val"] for slug in BENCHMARKS},
                           "test_task_ids": {slug: splits[slug]["test"] for slug in BENCHMARKS}}
    return {
        "experiment": "premise_random_validation_four_task_ablation",
        "model": MODEL, "base_url": BASE_URL, "manifest": str(MANIFEST), "manifest_sha256": digest(MANIFEST),
        "benchmarks": list(BENCHMARKS), "mode": "sequential", "seeds": list(SEEDS), "jobs": jobs,
        "appworld_policy": "built-in AppWorld policy; no override",
        "candidate_validation": "random_val4", "validation_scope": scope,
        "validation_sample_size": 4, "validation_sampling": "without replacement; shared by all candidates",
        "candidate_ranking": "mean official score across four validation tasks",
        "parallel_seeds": parallel_seeds, "replay_workers_per_seed": 28,
        "replay_retries": 2, "past_count_baseline": 3,
        "checkpoints": [50, 100, 130], "test_benchmarks": list(BENCHMARKS),
        "source_sha256": {
            name: digest(ROOT / relative)
            for name, relative in {
                "pipeline": "src/exgentic/agents/premise/pipeline.py",
                "selection": "src/exgentic/agents/premise/selection.py",
                "runner": "scripts/premise/run_experiment.py",
                "appworld_context": "src/exgentic/benchmarks/appworld/appworld_eval.py",
            }.items()
        },
    }


def command(seed: int, output: Path, scope: str) -> list[str]:
    return [
        sys.executable, "-u", str(ROOT / "scripts/premise/run_experiment.py"),
        "--task-split-manifest", str(MANIFEST), "--output-dir", str(output / f"s{seed}_sequential_t3"),
        "--benchmarks", ",".join(BENCHMARKS), "--mode", "sequential", "--seed", str(seed),
        "--all-train-tasks", "--past", "3", "--model", MODEL, "--base-url", BASE_URL,
        "--replay-workers", "28", "--replay-retries", "2", "--llm-calls", "32",
        "--llm-timeout", "600", "--llm-retries", "1", "--tool-calling-max-tokens", "2048",
        "--unlimited-benchmark-sessions", "--terminalbench2-max-steps", "100",
        "--candidate-validation", "random_val4", "--validation-scope", scope,
        "--execution-tag", "random_val4_original_policy", "--evaluate-test",
    ]


def cursor(directory: Path) -> int:
    path = directory / "state.json"
    return int(json.loads(path.read_text())["cursor"]) if path.exists() else 0


def result_complete(directory: Path) -> bool:
    if cursor(directory) != 130:
        return False
    evaluation = directory / "test_evaluation"
    if not (evaluation / "complete.json").exists():
        return False
    for checkpoint in (50, 100, 130):
        for benchmark in BENCHMARKS:
            path = evaluation / f"cp{checkpoint:03d}_{benchmark}.json"
            if not path.exists() or not json.loads(path.read_text()).get("complete"):
                return False
    return True


def controller(output: Path, identity: dict, max_attempts: int) -> int:
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "launch_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != identity:
        raise ValueError("Existing ablation launch configuration differs")
    write_json(manifest_path, identity)
    status_path = output / "status.json"
    jobs = {str(seed): {"phase": "waiting", "cursor": cursor(output / f"s{seed}_sequential_t3"),
                        "attempt": 0} for seed in SEEDS}
    env = os.environ.copy()
    env["OPENAI_API_BASE"] = BASE_URL
    env.setdefault("OPENAI_API_KEY", "EMPTY")
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    env.setdefault("DOCKER_HOST", "ssh://exgentic-daysi-docker:14233")
    env["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    def save(phase: str) -> None:
        for seed in SEEDS:
            jobs[str(seed)]["cursor"] = cursor(output / f"s{seed}_sequential_t3")
        write_json(status_path, {"phase": phase, "controller_pid": os.getpid(),
                                 "updated_at": now(), "jobs": jobs})

    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        running: dict[int, subprocess.Popen] = {}
        save("running")
        while True:
            if (output / "stop_request.json").exists():
                for process in running.values():
                    process.terminate()
                for process in running.values():
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                for row in jobs.values():
                    if row["phase"] != "complete":
                        row["phase"] = "stopped_by_user"
                save("stopped_by_user")
                return 0

            for seed, process in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                del running[seed]
                row = jobs[str(seed)]
                row["exit_code"] = code
                complete = code == 0 and result_complete(output / f"s{seed}_sequential_t3")
                row["phase"] = "complete" if complete else "waiting"
                if row["phase"] != "complete" and row["attempt"] >= max_attempts:
                    row["phase"] = "failed"
                row["finished_at"] = now()
                print(f"EXIT seed={seed} code={code} cursor={cursor(output / f's{seed}_sequential_t3')}", flush=True)

            for seed in SEEDS:
                row = jobs[str(seed)]
                if row["phase"] != "waiting" or len(running) >= identity["parallel_seeds"]:
                    continue
                directory = output / f"s{seed}_sequential_t3"
                if result_complete(directory):
                    row["phase"] = "complete"
                    continue
                row["attempt"] += 1
                log_path = output / f"s{seed}_sequential.log"
                with log_path.open("a") as log:
                    process = subprocess.Popen(command(seed, output, identity["validation_scope"]),
                                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                running[seed] = process
                row.update({"phase": "running", "pid": process.pid, "log": str(log_path), "started_at": now()})
                print(f"START seed={seed} attempt={row['attempt']} pid={process.pid}", flush=True)

            if all(row["phase"] in ("complete", "failed") for row in jobs.values()):
                success = all(row["phase"] == "complete" for row in jobs.values())
                save("complete" if success else "failed")
                write_json(output / "complete.json", {"success": success, "finished_at": now(), "jobs": jobs})
                return 0 if success else 1
            save("running")
            time.sleep(30)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-scope", choices=("current", "seen"), default="current")
    parser.add_argument("--parallel-seeds", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.parallel_seeds <= 3 or args.max_attempts < 1:
        parser.error("parallel-seeds must be 1..3 and max-attempts must be positive")
    identity = plan(args.validation_scope, args.parallel_seeds)
    if args.dry_run:
        print(json.dumps({
            "seeds": SEEDS,
            "task_counts": {seed: job["task_count"] for seed, job in identity["jobs"].items()},
            "scope": args.validation_scope, "total_test_groups": 27,
            "parallel_seeds": args.parallel_seeds,
        }))
        return 0
    return controller(args.output.resolve(), identity, args.max_attempts)


if __name__ == "__main__":
    raise SystemExit(main())
