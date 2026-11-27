# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Evaluate frozen PREMiSE sequential CP100 with an explicit policy, seeds in order."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from exgentic.agents.premise.llm import configure_llm_slots  # noqa: E402
from exgentic.agents.premise.pipeline import playbook_text  # noqa: E402
from exgentic.agents.premise.reward_data import write_json  # noqa: E402
from exgentic.core.types import RunConfig  # noqa: E402
from exgentic.interfaces.lib.api import execute  # noqa: E402
from exgentic.utils.task_splits import select_task_split_sections, validate_task_split  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_plan() -> list[dict]:
    jobs = []
    for seed in (42, 43, 44):
        root = (
            ROOT
            / "outputs"
            / (
                "premise_bfcl_appworld_terminalbench2_seed42_44_scorefix_20260928"
                if seed == 42
                else "premise_completion_contract_restart_20261003_0557"
            )
            / f"s{seed}_sequential_t3"
        )
        checkpoint = root / "checkpoints/checkpoint_100.json"
        snapshot = json.loads(checkpoint.read_text())
        previous_path = root / "test_evaluation/cp100_appworld.json"
        previous = json.loads(previous_path.read_text())
        config = snapshot["config"]
        manifest = Path(config["manifest"])
        if sha256(checkpoint) != previous["identity"]["checkpoint_sha256"]:
            raise ValueError("Original evaluation does not match the checkpoint")
        if sha256(manifest) != config["manifest_sha256"]:
            raise ValueError("Frozen task split manifest changed")
        sections = select_task_split_sections(json.loads(manifest.read_text()), config["benchmarks"], seed)
        task_ids = validate_task_split(sections["appworld"], "appworld")["test"]
        if list(task_ids) != previous["identity"]["task_ids"] or len(task_ids) != 50:
            raise ValueError("Expected exactly the same 50 AppWorld test tasks")
        jobs.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "manifest_sha256": config["manifest_sha256"],
                "model": config["model"],
                "benchmark_kwargs": config["benchmark_configs"]["appworld"],
                "max_steps": config["max_steps"]["appworld"],
                "task_ids": list(task_ids),
                "playbook": copy.deepcopy(snapshot["states"]["global"]["beam"][0]),
                "previous_evaluation": str(previous_path),
            }
        )
    return jobs


def evaluate_one(job: dict, index: int, task_id: str, policy: str, directory: Path,
                 max_tokens: int = 2048) -> dict:
    run_id = f"policy_{hashlib.sha256(str(directory).encode()).hexdigest()[:10]}_s{job['seed']}_{index:03d}"
    kwargs = copy.deepcopy(job["benchmark_kwargs"])
    kwargs["use_cache"] = False
    kwargs["env_kwargs"] = {**kwargs.get("env_kwargs", {}), "experiment_name": run_id}
    config = RunConfig(
        benchmark="appworld",
        agent="premise",
        task_ids=[task_id],
        output_dir=str(directory / "sessions"),
        run_id=run_id,
        model=job["model"],
        benchmark_kwargs=kwargs,
        agent_kwargs={
            "model": job["model"],
            "playbook": playbook_text(job["playbook"], "appworld"),
            "context_policy_override": policy,
            "runner": "direct",
            "enable_thinking": False,
            "model_settings": {"temperature": 0.0, "max_tokens": max_tokens, "timeout": 600, "num_retries": 1},
            "enable_tool_shortlisting": True,
            "max_selected_tools": 30,
        },
        max_steps=job["max_steps"],
        max_actions=max(job["max_steps"], 100),
        overwrite_sessions=True,
    )
    base = {"task_id": task_id, "run_id": run_id}
    try:
        result = execute(config=config)
        if len(result.session_results) != 1:
            raise ValueError("Expected exactly one session result")
        session = result.session_results[0]
        session_dir = directory / "sessions" / run_id / "sessions" / session.session_id
        base["session_dir"] = str(session_dir)
        if session.status.value in ("error", "cancelled"):
            raise RuntimeError(f"Session {session.status.value}: {session.details}")
        scores = json.loads((session_dir / "benchmark/scores.json").read_text())
        if not isinstance(scores.get("success"), bool):
            raise ValueError("Missing official AppWorld success flag required for TGC")
        return {
            **base,
            "status": "completed",
            "success": scores["success"],
            "tgc": int(scores["success"]),
            "partial_score": float(scores["pass_percentage"]) / 100,
            "pass_count": scores["pass_count"],
            "num_tests": scores["num_tests"],
        }
    except Exception as exc:
        return {
            **base,
            "status": "error",
            "success": False,
            "tgc": 0,
            "partial_score": 0.0,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def summarize(rows: list[dict | None]) -> dict:
    observed = [row for row in rows if row is not None]
    errors = sum(row["status"] == "error" for row in observed)
    successes = sum(row["tgc"] for row in observed)
    return {
        "processed": len(observed),
        "total": len(rows),
        "errors": errors,
        "successes": successes,
        "complete": len(observed) == len(rows),
        "tgc_with_errors_zero": successes / len(rows) if rows else None,
        "tgc": successes / len(rows) if observed and len(observed) == len(rows) and not errors else None,
        "partial_task_mean": sum(row["partial_score"] for row in observed) / len(observed) if observed else None,
    }


def run(output: Path, plan: list[dict], policy: str, workers: int = 50) -> int:
    configure_llm_slots(workers)
    policy_hash = hashlib.sha256(policy.encode()).hexdigest()
    identity = {
        "plan": plan,
        "policy_sha256": policy_hash,
        "workers": workers,
        "seed_order": [job["seed"] for job in plan],
        "temperature": 0.0,
        "max_tokens": 2048,
        "output_limit_retries": 5,
        "request_retries": 1,
        "beam_rank": 1,
        "learning": False,
    }
    manifest_path = output / "launch_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != identity:
        raise ValueError("Existing evaluation configuration differs")
    write_json(manifest_path, identity)
    (output / "policy.txt").write_text(policy)
    statuses = {str(job["seed"]): {"status": "waiting", "processed": 0, "total": len(job["task_ids"])} for job in plan}

    def save_status():
        write_json(
            output / "status.json",
            {
                "pid": os.getpid(),
                "updated_at": datetime.now(UTC).isoformat(),
                "workers": workers,
                "seed_order": identity["seed_order"],
                "seeds": statuses,
            },
        )

    save_status()
    for job in plan:
        seed = job["seed"]
        directory = output / f"s{seed}_sequential_cp100"
        directory.mkdir(parents=True, exist_ok=True)
        snapshot_copy = directory / "checkpoint_100.json"
        if snapshot_copy.exists() and sha256(snapshot_copy) != job["checkpoint_sha256"]:
            raise ValueError("Copied checkpoint changed")
        snapshot_copy.write_bytes(Path(job["checkpoint"]).read_bytes())
        previous = json.loads(Path(job["previous_evaluation"]).read_text())
        write_json(directory / "previous_evaluation.json", previous)
        previous_rows = previous["results"]
        write_json(
            directory / "previous_summary.json",
            {
                "tgc_with_errors_zero": sum(float(q["score"]) >= 1 - 1e-9 for q in previous_rows) / len(previous_rows),
                "partial_task_mean": sum(float(q["score"]) for q in previous_rows) / len(previous_rows),
                "errors": sum(q["status"] == "error" for q in previous_rows),
            },
        )
        path = directory / "evaluation.json"
        rows = json.loads(path.read_text())["results"] if path.exists() else [None] * len(job["task_ids"])
        if len(rows) != len(job["task_ids"]) or any(
            row is not None and row["task_id"] != task_id for row, task_id in zip(rows, job["task_ids"], strict=True)
        ):
            raise ValueError("Existing task results differ")

        def save(rows=rows, path=path, seed=seed, job=job):
            summary = summarize(rows)
            write_json(
                path,
                {
                    "seed": seed,
                    "checkpoint_sha256": job["checkpoint_sha256"],
                    "policy_sha256": policy_hash,
                    "results": rows,
                    **summary,
                },
            )
            statuses[str(seed)] = {"status": "complete" if summary["complete"] else "evaluating", **summary}
            save_status()

        save()
        print(f"START seed={seed} tasks={len(rows)} workers={workers}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {
                pool.submit(evaluate_one, job, i, task_id, policy, directory): i
                for i, task_id in enumerate(job["task_ids"])
                if rows[i] is None
            }
            for future in as_completed(pending):
                index = pending[future]
                rows[index] = future.result()
                save()
                print(
                    f"DONE seed={seed} index={index} status={rows[index]['status']} tgc={rows[index]['tgc']}",
                    flush=True,
                )
        print(f"FINISH seed={seed} {json.dumps(summarize(rows))}", flush=True)
    clean = all(status["errors"] == 0 for status in statuses.values())
    write_json(
        output / "complete.json", {"success": clean, "seeds": statuses, "completed_at": datetime.now(UTC).isoformat()}
    )
    return 0 if clean else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy", type=Path, default=ROOT / "scripts/premise/policies/appworld_teammate_completion.txt"
    )
    parser.add_argument("--workers", type=int, default=50)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(output, build_plan(), args.policy.read_text(), args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
