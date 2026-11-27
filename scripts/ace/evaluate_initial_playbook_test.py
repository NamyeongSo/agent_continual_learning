# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Evaluate ACE's unchanged initial playbook on the nine frozen test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_experiment import BENCHMARK_REGISTRY  # noqa: E402

from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK, PlaybookStore  # noqa: E402
from exgentic.core.types import ModelSettings  # noqa: E402
from exgentic.interfaces.lib.api import evaluate  # noqa: E402
from exgentic.interfaces.registry import load_agent, load_benchmark  # noqa: E402
from exgentic.utils.task_splits import validate_task_split  # noqa: E402

BENCHMARKS = ("bfcl", "appworld", "terminalbench2")
SEEDS = (42, 43, 44)
EXPECTED_TEST_COUNTS = {"bfcl": 50, "appworld": 50, "terminalbench2": 29}
MAX_JOB_ATTEMPTS = 3


class IncompleteEvaluationError(RuntimeError):
    """A run finished with missing scores or execution errors that can be retried."""


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_plan(manifest: Path) -> list[dict]:
    splits = json.loads(manifest.read_text(encoding="utf-8"))
    expected_keys = {f"{slug}_{seed}" for seed in SEEDS for slug in BENCHMARKS}
    if set(splits) != expected_keys:
        raise ValueError(f"Expected exactly the nine benchmark/seed keys in {manifest}")
    jobs = []
    for seed in SEEDS:
        for slug in BENCHMARKS:
            key = f"{slug}_{seed}"
            ids = validate_task_split(splits[key], slug)["test"]
            if len(ids) != EXPECTED_TEST_COUNTS[slug]:
                raise ValueError(f"Unexpected test task count for {key}: {len(ids)}")
            jobs.append({"key": key, "benchmark": slug, "seed": seed, "task_ids": ids})
    return jobs


def run_job(job: dict, output_root: Path, config: dict) -> dict:
    key = job["key"]
    slug = job["benchmark"]
    output = output_root / key
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (summary.get("config_sha256") != config["config_sha256"]
                or summary.get("task_ids") != job["task_ids"]):
            raise ValueError(f"Existing summary has different inputs: {summary_path}")
        if summary.get("complete"):
            print(f"SKIP completed {key}: score={summary['benchmark_score']}", flush=True)
            return summary
        attempt = 1
        while (output / f"attempt_{attempt:03d}_summary.json").exists():
            attempt += 1
        shutil.copy2(summary_path, output / f"attempt_{attempt:03d}_summary.json")
        print(f"RETRY incomplete {key}: previous errors={summary['error_count']}", flush=True)

    max_steps = config["terminalbench2_max_turns"] if slug == "terminalbench2" else 50
    kwargs = dict(BENCHMARK_REGISTRY[slug]["bm_kwargs"])
    if slug == "terminalbench2":
        kwargs.update(actor="terminus2", max_interactions=max_steps)

    PlaybookStore.reset_all()
    agent = load_agent("ace")(
        model=config["model"], curator_model=config["model"],
        benchmark_id=slug, shuffle_mode="isolated", runner="direct",
        model_settings=ModelSettings(max_tokens=config["max_tokens"]),
        enable_thinking=False, training_time=False, evaluation_mode=True,
        initial_playbook=DEFAULT_PLAYBOOK, **BENCHMARK_REGISTRY[slug]["agent_kwargs"],
    )
    run_id = f"ace_initial_test_{key}"
    print(f"START {key}: {len(job['task_ids'])} test tasks, workers={config['max_workers']}", flush=True)
    results = evaluate(
        benchmark=load_benchmark(slug)(**kwargs), agent=agent, task_ids=job["task_ids"],
        max_workers=config["max_workers"], max_steps=max_steps,
        output_dir=str(output), run_id=run_id,
    )
    for store in PlaybookStore.list_stores().values():
        if store.playbook != DEFAULT_PLAYBOOK or store.session_count != 0:
            raise RuntimeError(f"Initial playbook changed while evaluating {key}")

    sessions = [{
        "task_id": str(result.task_id),
        "session_id": result.session_id,
        "score": result.score,
        "success": result.success,
        "status": result.status.value if hasattr(result.status, "value") else str(result.status),
        "steps": result.steps,
        "error": (result.details.get("session_metadata") or {}).get("error") if result.details else None,
    } for result in results.session_results]
    expected_ids = set(job["task_ids"])
    observed_ids = [item["task_id"] for item in sessions]
    error_count = sum(item["status"] == "error" for item in sessions)
    complete = (len(sessions) == len(job["task_ids"])
                and len(set(observed_ids)) == len(observed_ids)
                and set(observed_ids) == expected_ids
                and error_count == 0
                and all(item["score"] is not None for item in sessions)
                and results.benchmark_score is not None)
    summary = {
        **job, "model": config["model"], "max_workers": config["max_workers"],
        "max_tokens": config["max_tokens"], "max_steps": max_steps,
        "manifest_sha256": config["manifest_sha256"],
        "initial_playbook_sha256": config["initial_playbook_sha256"],
        "config_sha256": config["config_sha256"],
        "evaluation_mode": True, "training_time": False,
        "benchmark_score": results.benchmark_score,
        "mean_task_score": (sum(float(item["score"]) for item in sessions) / len(sessions)
                            if sessions and all(item["score"] is not None for item in sessions) else None),
        "error_count": error_count,
        "missing_task_ids": sorted(expected_ids - set(observed_ids)),
        "complete": complete,
        "sessions": sessions,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    save_json(summary_path, summary)
    print(f"DONE {key}: official={summary['benchmark_score']} complete={complete} errors={error_count}", flush=True)
    if not complete:
        raise IncompleteEvaluationError(f"Incomplete evaluation for {key}; inspect {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "scripts/utils/task_splits"
                        / "bfcl_appworld_terminalbench2_seed42_44.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="openai/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:8006/v1"))
    parser.add_argument("--max-workers", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--terminalbench2-max-turns", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_tokens < 1 or args.terminalbench2_max_turns < 1:
        parser.error("Workers, token limit, and Terminal-Bench turn limit must be positive")
    manifest = args.manifest.resolve()
    output = args.output_dir.resolve()
    jobs = build_plan(manifest)
    config = {
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "initial_playbook_sha256": hashlib.sha256(DEFAULT_PLAYBOOK.encode()).hexdigest(),
        "model": args.model, "base_url": args.base_url,
        "max_workers": args.max_workers, "max_tokens": args.max_tokens,
        "terminalbench2_max_turns": args.terminalbench2_max_turns,
        "jobs": jobs,
    }
    config["config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if args.dry_run:
        print(json.dumps({"jobs": [item["key"] for item in jobs],
                          "test_tasks": sum(len(item["task_ids"]) for item in jobs),
                          "max_workers": args.max_workers}, indent=2))
        return

    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "evaluation_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError(f"Existing output has a different evaluation configuration: {config_path}")
    else:
        save_json(config_path, config)
    os.environ["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ["OPENAI_API_BASE"] = args.base_url
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["EXGENTIC_LLM_TRACE_FORMAT"] = "dedup_v1"
    results = []
    for job in jobs:
        save_json(output / "status.json", {"phase": "running", "current_job": job["key"],
                                           "completed_jobs": [item["key"] for item in results]})
        try:
            for attempt in range(1, MAX_JOB_ATTEMPTS + 1):
                try:
                    result = run_job(job, output, config)
                except IncompleteEvaluationError:
                    if attempt == MAX_JOB_ATTEMPTS:
                        raise
                    print(f"RETRY {job['key']} attempt {attempt + 1}/{MAX_JOB_ATTEMPTS}", flush=True)
                    continue
                results.append(result)
                break
        except Exception as exc:
            save_json(output / "status.json", {"phase": "failed", "current_job": job["key"],
                                               "completed_jobs": [item["key"] for item in results],
                                               "error": str(exc)})
            raise
    save_json(output / "status.json", {"phase": "complete", "completed_jobs": [item["key"] for item in results]})
    save_json(output / "scores.json", {"manifest_sha256": config["manifest_sha256"],
                                         "initial_playbook_sha256": config["initial_playbook_sha256"],
                                         "scores": {item["key"]: item["benchmark_score"] for item in results}})


if __name__ == "__main__":
    main()
