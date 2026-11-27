# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
# ruff: noqa: E501

"""Evaluate two base prompts with the built-in and teammate AppWorld policies."""

from __future__ import annotations

import argparse
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
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from exgentic.agents.premise.llm import configure_llm_slots  # noqa: E402
from exgentic.agents.premise.reward_data import write_json  # noqa: E402
from exgentic.core.types import RunConfig  # noqa: E402
from exgentic.interfaces.lib.api import execute  # noqa: E402
from exgentic.utils.task_splits import select_task_split_sections, validate_task_split  # noqa: E402

GEPA_PROMPT = (
    "You are an expert agent that completes tasks using the tools available in the environment. "
    "Read the request and tool instructions carefully. Use the appropriate tools to gather information "
    "and carry out the requested work. Inspect each result and adapt your next action accordingly. "
    "Follow the task constraints, avoid inventing tools, arguments, or results, and verify that the "
    "requested outcome is complete before responding."
)

SKILLOPT_PROMPT = """You are a careful general-purpose agent completing tasks in the environment provided.

## Understand the Task
- Read the current request and the instructions provided for the task and environment.
- Identify the requested outcome, relevant constraints, and how completion should be reported.
- Inspect relevant files, data, APIs, or tools before acting. Treat the task details and observed environment state as the source of truth.

## Use Available Tools
- Use the tools and capabilities exposed for the current task, following their documented names, schemas, and limits.
- Ground each command or action in the request and in information already observed. Do not invent paths, values, capabilities, or side effects.
- Use focused commands or calls, inspect their outputs, and use the results to guide the next step.

## Make Changes Carefully
- Make the smallest set of changes that fully completes the task.
- Preserve unrelated files, data, and state. Avoid destructive actions unless they are required by the task.
- When an action fails, inspect the error and relevant state, address the cause, then retry only when appropriate.

## Verify and Finish
- Check the resulting state with the most relevant available verification method; do not assume an action succeeded without evidence.
- Use the designated completion action after confirming the requested outcome, and provide a brief, accurate result."""

PROMPTS = {"gepa": GEPA_PROMPT, "skillopt": SKILLOPT_PROMPT}
MANIFEST = ROOT / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json"
TEAM_POLICY = ROOT / "scripts/premise/policies/appworld_teammate_completion.txt"
MODEL = "openai/Qwen/Qwen3.6-35B-A3B"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def plan_jobs(manifest: Path) -> list[dict]:
    payload = json.loads(manifest.read_text())
    jobs = []
    for seed in (42, 43):
        section = select_task_split_sections(payload, ["appworld"], seed)["appworld"]
        task_ids = validate_task_split(section, "appworld")["test"]
        if len(task_ids) != 50:
            raise ValueError(f"Expected 50 AppWorld test tasks for seed {seed}")
        for prompt_name in PROMPTS:
            for policy_name in ("existing", "teammate"):
                jobs.append({"seed": seed, "prompt": prompt_name, "policy": policy_name, "task_ids": task_ids})
    return jobs


def summarize(rows: list[dict | None]) -> dict:
    observed = [row for row in rows if row is not None]
    completed = [row for row in observed if row["status"] == "completed"]
    errors = sum(row["status"] == "error" for row in observed)
    successes = sum(row["tgc"] for row in observed)
    return {
        "processed": len(observed),
        "total": len(rows),
        "errors": errors,
        "successes": successes,
        "complete": len(observed) == len(rows),
        "tgc": successes / len(rows) if len(observed) == len(rows) and not errors else None,
        "tgc_with_errors_zero": successes / len(rows) if rows else None,
        "partial_task_mean": sum(row["partial_score"] for row in completed) / len(completed) if completed else None,
        "input_tokens": sum(row.get("input_tokens", 0) for row in observed),
        "output_tokens": sum(row.get("output_tokens", 0) for row in observed),
    }


def evaluate_one(job: dict, index: int, task_id: str, directory: Path, team_policy: str,
                 max_tokens: int = 2048) -> dict:
    name = f"s{job['seed']}_{job['prompt']}_{job['policy']}"
    run_id = f"{job.get('run_id_prefix', 'base_prompt')}_{name}_{index:03d}"
    benchmark_kwargs = {
        "subset": "test_challenge",
        "use_cache": False,
        "env_kwargs": {"experiment_name": run_id},
    }
    config = RunConfig(
        benchmark="appworld",
        agent="premise",
        task_ids=[task_id],
        output_dir=str(directory / "sessions"),
        run_id=run_id,
        model=MODEL,
        benchmark_kwargs=benchmark_kwargs,
        agent_kwargs={
            "model": MODEL,
            "playbook": "",
            "system_prompt_override": job.get("system_prompt", PROMPTS[job["prompt"]]),
            "context_policy_override": team_policy if job["policy"] == "teammate" else None,
            "runner": "direct",
            "enable_thinking": False,
            "model_settings": {"temperature": 0.0, "max_tokens": max_tokens, "timeout": 600, "num_retries": 1},
            "enable_tool_shortlisting": True,
            "max_selected_tools": 30,
        },
        max_steps=50,
        max_actions=100,
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
        tokens = {"input_tokens": 0, "output_tokens": 0}
        for report in session.cost_reports.values():
            for key in tokens:
                tokens[key] += int(report.get(key, 0) if isinstance(report, dict) else getattr(report, key, 0))
        return {
            **base,
            "status": "completed",
            "success": scores["success"],
            "tgc": int(scores["success"]),
            "partial_score": float(scores["pass_percentage"]) / 100,
            "pass_count": scores["pass_count"],
            "num_tests": scores["num_tests"],
            **tokens,
        }
    except Exception as exc:
        return {**base, "status": "error", "tgc": 0, "partial_score": 0.0,
                "error_type": type(exc).__name__, "error": str(exc)}


def run(output: Path, manifest: Path, team_policy: str, workers: int, retry_errors: bool) -> int:
    configure_llm_slots(workers)
    jobs = plan_jobs(manifest)
    identity = {
        "jobs": jobs,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256(manifest.read_bytes()),
        "prompts": PROMPTS,
        "team_policy_sha256": sha256(team_policy.encode()),
        "model": MODEL,
        "workers": workers,
        "max_steps": 50,
        "temperature": 0.0,
        "max_tokens": 2048,
        "base_url": os.environ["OPENAI_API_BASE"],
    }
    path = output / "launch_manifest.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Existing evaluation configuration differs")
    write_json(path, identity)
    (output / "gepa_prompt.txt").write_text(GEPA_PROMPT + "\n")
    (output / "skillopt_prompt.txt").write_text(SKILLOPT_PROMPT + "\n")
    (output / "teammate_policy.txt").write_text(team_policy)
    status = {}
    for job in jobs:
        name = f"s{job['seed']}_{job['prompt']}_{job['policy']}"
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory / "evaluation.json"
        rows = json.loads(result_path.read_text())["results"] if result_path.exists() else [None] * 50
        if len(rows) != 50 or any(row is not None and row["task_id"] != task_id
                                  for row, task_id in zip(rows, job["task_ids"], strict=True)):
            raise ValueError(f"Existing results do not match task split: {name}")
        if retry_errors:
            rows = [None if row is not None and row["status"] == "error" else row for row in rows]

        def save(rows=rows, result_path=result_path, name=name) -> None:
            summary = summarize(rows)
            write_json(result_path, {"job": name, "results": rows, **summary})
            status[name] = summary
            write_json(output / "status.json", {
                "pid": os.getpid(), "updated_at": datetime.now(UTC).isoformat(),
                "workers": workers, "job_order": [f"s{j['seed']}_{j['prompt']}_{j['policy']}" for j in jobs],
                "jobs": status,
            })

        save()
        print(f"START {name}: {len(rows)} tasks, workers={workers}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(evaluate_one, job, i, task_id, directory, team_policy): i
                       for i, task_id in enumerate(job["task_ids"]) if rows[i] is None}
            for future in as_completed(pending):
                index = pending[future]
                rows[index] = future.result()
                save()
                print(f"DONE {name} {index:02d}: {rows[index]['status']} tgc={rows[index]['tgc']}", flush=True)
        print(f"FINISH {name}: {json.dumps(summarize(rows))}", flush=True)
    clean = all(row["errors"] == 0 and row["complete"] for row in status.values())
    write_json(output / "complete.json", {"success": clean, "completed_at": datetime.now(UTC).isoformat(),
                                           "jobs": status})
    return 0 if clean else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--team-policy", type=Path, default=TEAM_POLICY)
    parser.add_argument("--base-url", default="http://127.0.0.1:18006/v1")
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    os.environ["OPENAI_API_BASE"] = args.base_url
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(output, args.manifest.resolve(), args.team_policy.read_text(), args.workers, args.retry_errors)


if __name__ == "__main__":
    raise SystemExit(main())
