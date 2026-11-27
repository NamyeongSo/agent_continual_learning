# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Retry failed ACE Terminal-Bench test tasks without replacing the original summaries."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_test_checkpoints import BENCHMARK_REGISTRY, build_plan, digest, save_json  # noqa: E402

from exgentic.agents.ace.playbook_store import PlaybookStore  # noqa: E402
from exgentic.core.types import ModelSettings  # noqa: E402
from exgentic.interfaces.lib.api import evaluate  # noqa: E402
from exgentic.interfaces.registry import load_agent, load_benchmark  # noqa: E402


def retry_task_ids(summary: dict) -> list[str]:
    return [entry["task_id"] for entry in summary["sessions"]
            if entry["status"] == "error" and entry.get("skip") is None]


def merge_sessions(original: dict, retried: list[dict]) -> dict:
    expected = retry_task_ids(original)
    replacements = {entry["task_id"]: entry for entry in retried}
    if len(replacements) != len(retried) or set(replacements) != set(expected):
        raise ValueError("Retried task IDs do not match the original error sessions")
    sessions = [replacements.get(entry["task_id"], entry) for entry in original["sessions"]]
    mean_score = sum(float(entry.get("score") or 0) for entry in sessions) / len(sessions)
    errors = sum(entry["status"] == "error" for entry in sessions)
    return {
        **original,
        "sessions": sessions,
        "benchmark_score": mean_score if errors == 0 else None,
        "mean_task_score": mean_score,
        "error_count": errors,
        "context_skipped_count": sum(entry.get("skip") is not None for entry in sessions),
        "retried_error_task_ids": expected,
        "original_error_count": original["error_count"],
        "original_mean_task_score": original["mean_task_score"],
        "completed_at": datetime.now(UTC).isoformat(),
    }


def retry_job(job: dict, source_root: Path, recovery_root: Path) -> dict:
    group = recovery_root / job["run"] / f"checkpoint_{job['stage']:03d}" / "terminalbench2"
    group.mkdir(parents=True, exist_ok=True)
    with FileLock(str(group / ".retry.lock")):
        return _retry_job(job, source_root, recovery_root)


def _retry_job(job: dict, source_root: Path, recovery_root: Path) -> dict:
    group = Path(job["run"]) / f"checkpoint_{job['stage']:03d}" / "terminalbench2"
    source_path = source_root / group / "summary.json"
    original = json.loads(source_path.read_text(encoding="utf-8"))
    snapshot_path = source_root / "snapshots" / job["run"] / f"checkpoint_{job['stage']:03d}.json"
    if (original["task_ids"] != job["task_ids"]
            or original["manifest_sha256"] != job["manifest_sha256"]
            or original["checkpoint_sha256"] != digest(snapshot_path)):
        raise ValueError(f"Original evaluation metadata changed: {source_path}")
    output = recovery_root / group
    summary_path = output / "summary.json"
    if summary_path.is_file():
        repaired = json.loads(summary_path.read_text(encoding="utf-8"))
        if (repaired.get("checkpoint_sha256") != original["checkpoint_sha256"]
                or repaired.get("task_ids") != original["task_ids"]
                or repaired.get("original_error_count") != original["error_count"]):
            raise ValueError(f"Recovery summary changed: {summary_path}")
        return repaired
    failed_ids = retry_task_ids(original)
    if not failed_ids:
        repaired = merge_sessions(original, [])
        save_json(summary_path, repaired)
        return repaired

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    PlaybookStore.reset_all()
    agent = load_agent("ace")(
        model=job["model"], curator_model=job["model"], shuffle_mode=job["mode"],
        benchmark_id="terminalbench2", runner="direct", model_settings=ModelSettings(max_tokens=job["max_tokens"]),
        enable_thinking=False, training_time=False, evaluation_mode=True,
        initial_playbook=snapshot["playbook"], **BENCHMARK_REGISTRY["terminalbench2"]["agent_kwargs"],
    )
    benchmark_kwargs = dict(BENCHMARK_REGISTRY["terminalbench2"]["bm_kwargs"])
    benchmark_kwargs.update(actor="terminus2", max_interactions=job["max_steps"])
    run_id = f"ace_tb2_retry_{job['run']}_c{job['stage']}"
    print(f"RETRY {job['run']} c{job['stage']} errors={len(failed_ids)}", flush=True)
    results = evaluate(
        benchmark=load_benchmark("terminalbench2")(**benchmark_kwargs), agent=agent,
        task_ids=failed_ids, max_workers=1, max_steps=job["max_steps"],
        output_dir=str(output), run_id=run_id,
    )
    for store in PlaybookStore.list_stores().values():
        if store.playbook != snapshot["playbook"] or store.session_count != 0:
            raise RuntimeError("Terminal-Bench retry modified the frozen playbook")
    retried = []
    for result in results.session_results:
        skip_path = output / run_id / "sessions" / result.session_id / "agent/task_skip.json"
        retried.append({
            "task_id": str(result.task_id), "session_id": result.session_id,
            "score": result.score, "status": str(result.status), "steps": result.steps,
            "skip": json.loads(skip_path.read_text()) if skip_path.is_file() else None,
        })
    repaired = merge_sessions(original, retried)
    repaired["source_summary"] = str(source_path)
    repaired["recovery_run_id"] = run_id
    save_json(summary_path, repaired)
    print(f"DONE {job['run']} c{job['stage']} remaining_errors={repaired['error_count']} "
          f"mean={repaired['mean_task_score']:.4f}", flush=True)
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--worker-run")
    parser.add_argument("--wait-for-summary", action="store_true")
    args = parser.parse_args()
    launch_path = args.launch_manifest.resolve()
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    source_root = launch_path.parent / "test_evaluation"
    output = (args.output_root or launch_path.parent / "test_evaluation_terminalbench2_retry").resolve()
    jobs = [job for job in build_plan(launch) if job["benchmark"] == "terminalbench2"]
    if args.worker_run:
        jobs = [job for job in jobs if job["run"] == args.worker_run]
        if len(jobs) != 3:
            parser.error("worker run must identify one of the six experiments")
    os.environ["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:8006/v1")
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["EXGENTIC_LLM_TRACE_FORMAT"] = "dedup_v1"
    output.mkdir(parents=True, exist_ok=True)
    status_dir = output / args.worker_run if args.worker_run else output
    status_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(status_dir / "controller.lock"), timeout=0):
        for index, job in enumerate(jobs):
            source = source_root / job["run"] / f"checkpoint_{job['stage']:03d}" / "terminalbench2/summary.json"
            while args.wait_for_summary and not source.exists():
                save_json(status_dir / "status.json", {"phase": "waiting_for_original",
                                                       "current_checkpoint": job["stage"],
                                                       "groups_complete": index, "groups_planned": len(jobs)})
                time.sleep(30)
            save_json(status_dir / "status.json", {
                "phase": "running", "groups_complete": index, "groups_planned": len(jobs),
                "current_run": job["run"], "current_checkpoint": job["stage"],
            })
            retry_job(job, source_root, output)
        summaries = [json.loads(path.read_text()) for path in status_dir.glob("**/terminalbench2/summary.json")]
        save_json(status_dir / "status.json", {
            "phase": "complete", "groups_complete": len(summaries), "groups_planned": len(jobs),
            "original_errors": sum(item["original_error_count"] for item in summaries),
            "remaining_errors": sum(item["error_count"] for item in summaries),
        })


if __name__ == "__main__":
    main()
