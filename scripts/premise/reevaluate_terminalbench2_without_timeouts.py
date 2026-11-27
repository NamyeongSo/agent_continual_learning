# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
# ruff: noqa: E402

"""Reevaluate deadline-affected isolated test sessions with frozen PREMiSE playbooks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from exgentic.agents.premise.pipeline import playbook_text, trajectory_from_session, write_json
from exgentic.core.types import RunConfig
from exgentic.interfaces.lib.api import execute
from exgentic.utils.context_errors import is_context_length_error


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_sources(source: Path, seeds: list[int], checkpoint: int) -> tuple[dict, list[dict]]:
    sources, jobs = {}, []
    for seed in seeds:
        root = source / f"s{seed}" / "run"
        snapshot_path = root / "checkpoints" / f"checkpoint_{checkpoint:03d}.json"
        evaluation_path = root / "test_evaluation" / f"cp{checkpoint:03d}_terminalbench2.json"
        snapshot = json.loads(snapshot_path.read_text())
        evaluation = json.loads(evaluation_path.read_text())
        state_path = root / "state.json"
        if not evaluation["complete"] or snapshot["cursor"] != checkpoint:
            raise ValueError(f"Incomplete source checkpoint/evaluation: {root}")
        if evaluation["identity"]["checkpoint_sha256"] != sha256(snapshot_path):
            raise ValueError(f"Source checkpoint hash differs: {root}")
        manifest = Path(snapshot["config"]["manifest"])
        if sha256(manifest) != evaluation["identity"]["manifest_sha256"]:
            raise ValueError(f"Source task split changed: {manifest}")
        if evaluation["identity"].get("learning") is not False:
            raise ValueError("Source evaluation must have learning disabled")
        if [r["task_id"] for r in evaluation["results"]] != evaluation["identity"]["task_ids"]:
            raise ValueError("Source evaluation task order differs")
        sources[seed] = {
            "root": str(root), "checkpoint_path": str(snapshot_path),
            "checkpoint_sha256": sha256(snapshot_path), "evaluation_sha256": sha256(evaluation_path),
            "state_sha256": sha256(state_path), "snapshot": snapshot, "evaluation": evaluation,
        }
        for index, row in enumerate(evaluation["results"]):
            paths = list(root.glob(
                f"sessions/*_test_cp{checkpoint:03d}_terminalbench2_{index:04d}_terminalbench2"
                "/sessions/*/results.json"
            ))
            if len(paths) != 1:
                raise ValueError(f"Expected one source session for seed={seed}, task={row['task_id']}")
            session = json.loads(paths[0].read_text())
            metadata = (session.get("details") or {}).get("session_metadata") or {}
            explicit_deadline = "TaskTimeLimitError" in row.get("error", "")
            agent_deadline = bool(metadata.get("agent_timed_out"))
            verifier_deadline = (
                (metadata.get("phase_timings") or {}).get("verifier", {}).get("error_type") == "TimeoutError"
            )
            if explicit_deadline or agent_deadline or verifier_deadline:
                jobs.append({
                    "id": f"s{seed}_t{index:04d}", "seed": seed, "index": index, "task_id": row["task_id"],
                    "source_session": str(paths[0].parent), "original_result": row,
                    "deadline_evidence": {"explicit_error": explicit_deadline, "agent": agent_deadline,
                                          "verifier": verifier_deadline},
                })
    # Start the explicit failures first, then interleave seeds for the remaining tasks.
    jobs.sort(key=lambda job: (not job["deadline_evidence"]["explicit_error"], job["index"], job["seed"]))
    return sources, jobs


def evaluate_job(job: dict, source: dict, output: Path, checkpoint: int) -> dict:
    started = time.monotonic()
    run_id = f"premise_no_wallclock_cp{checkpoint:03d}_{job['id']}"
    record_path = output / "records" / f"{job['id']}.json"
    write_json(record_path, {**job, "status": "running", "started_at": datetime.now(UTC).isoformat()})
    snapshot = source["snapshot"]
    config = snapshot["config"]
    step_limit = config["max_steps"]["terminalbench2"]
    benchmark_kwargs = {**config["benchmark_configs"]["terminalbench2"], "use_cache": False,
                        "actor": "terminus2", "max_interactions": step_limit, "enforce_session_timeouts": False}
    run_config = RunConfig(
        benchmark="terminalbench2", agent="premise", task_ids=[job["task_id"]],
        output_dir=str(output / "sessions"), run_id=run_id, model=config["model"],
        benchmark_kwargs=benchmark_kwargs,
        agent_kwargs={
            "model": config["model"], "runner": "direct", "enable_thinking": False,
            "playbook": playbook_text(snapshot["states"]["terminalbench2"]["beam"][0], "terminalbench2"),
            "model_settings": {"temperature": 0, "max_tokens": 8192, "timeout": 600, "num_retries": 1},
        },
        max_steps=step_limit, max_actions=max(step_limit, 100), overwrite_sessions=True,
    )
    write_json(output / "configs" / f"{job['id']}.json", run_config.model_dump(mode="json"))
    record = {**job, "run_id": run_id}
    try:
        result = execute(config=run_config)
        if len(result.session_results) != 1:
            raise RuntimeError("Expected one reevaluated session")
        session = result.session_results[0]
        metadata = (session.details or {}).get("session_metadata") or {}
        record["session_id"] = session.session_id
        record["session_dir"] = str(output / "sessions" / run_id / "sessions" / session.session_id)
        record["session_status"] = session.status.value
        record["native_score"] = session.score
        record["agent_timed_out"] = bool(metadata.get("agent_timed_out"))
        record["phase_timings"] = metadata.get("phase_timings") or {}
        if session.status.value in ("error", "cancelled"):
            raise RuntimeError(f"{session.status.value}: {metadata.get('error_type')}: {metadata.get('error')}")
        if metadata.get("enforce_session_timeouts") is not False or record["agent_timed_out"]:
            raise RuntimeError("Reevaluation did not disable the session deadline")
        trajectory = trajectory_from_session(output / "sessions", run_id, session, "terminalbench2")
        record.update(status="completed", score=trajectory["score"],
                      episodes=metadata.get("agent_metrics", {}).get("n_episodes"))
    except Exception as exc:
        record.update(status="error", score=0.0, context_error=is_context_length_error(exc),
                      error=f"{type(exc).__name__}: {exc}")
    record["elapsed_seconds"] = time.monotonic() - started
    write_json(record_path, record)
    return record


def report(output: Path, sources: dict, jobs: list[dict], records: dict, status: str) -> dict:
    groups = {}
    for seed, source in sources.items():
        original = source["evaluation"]["results"]
        merged = copy.deepcopy(original)
        selected = [j for j in jobs if j["seed"] == seed]
        for job in selected:
            record = records.get(job["id"])
            if record is not None:
                merged[job["index"]] = {k: record[k] for k in (
                    "task_id", "score", "status", "context_error", "error", "session_dir"
                ) if k in record}
        complete = all(job["id"] in records for job in selected)
        groups[str(seed)] = {
            "total_tasks": len(original), "reevaluation_tasks": len(selected),
            "finished_reevaluations": sum(job["id"] in records for job in selected), "complete": complete,
            "original_mean_official_score": sum(r["score"] for r in original) / len(original),
            "mean_official_score": sum(r["score"] for r in merged) / len(merged) if complete else None,
            "execution_errors": sum(r["status"] == "error" for r in merged) if complete else None,
            "results": merged,
        }
    summary = {
        "status": status, "pid": os.getpid(), "updated_at": datetime.now(UTC).isoformat(),
        "complete": len(records) == len(jobs), "total_reevaluations": len(jobs),
        "finished_reevaluations": len(records),
        "reevaluation_errors": sum(r["status"] == "error" for r in records.values()),
        "groups": groups,
        "protocol": "Frozen rank-1 checkpoint playbooks, no learning or summarization. Only deadline-affected test "
                    "tasks are rerun. Other results retain their original scores. This changes the official time "
                    "budget; the aggregate is a diagnostic score, not a standard Terminal-Bench score.",
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "outputs/isolated_baselines_20260930/terminalbench2/premise",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--checkpoint", type=int, default=30)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.env_file:
        os.environ.update({k: str(v) for k, v in json.loads(args.env_file.read_text()).items()})
    os.environ["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + os.environ["PATH"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources, jobs = collect_sources(args.source.resolve(), args.seeds, args.checkpoint)
    manifest = {
        "source": str(args.source.resolve()), "checkpoint": args.checkpoint, "seeds": args.seeds,
        "workers": args.workers, "enforce_session_timeouts": False, "enable_summarize": False,
        "learning": False, "script_sha256": sha256(Path(__file__)),
        "sources": {str(seed): {k: v for k, v in source.items() if k not in ("snapshot", "evaluation")}
                    for seed, source in sources.items()}, "jobs": jobs,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing reevaluation manifest differs")
    write_json(manifest_path, manifest)
    for seed, source in sources.items():
        destination = output / "frozen" / f"s{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source["checkpoint_path"], destination / "checkpoint.json")
        write_json(destination / "original_evaluation.json", source["evaluation"])
    shutil.copy2(Path(__file__), output / "source_snapshot.py")
    records = {}
    for job in jobs:
        path = output / "records" / f"{job['id']}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["status"] in ("completed", "error"):
                records[job["id"]] = record
    report(output, sources, jobs, records, "planned" if args.dry_run else "running")
    print(f"Reevaluations: {len(jobs)}; restored: {len(records)}; output: {output}", flush=True)
    if args.dry_run:
        return
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_job, job, sources[job["seed"]], output, args.checkpoint): job
                   for job in jobs if job["id"] not in records}
        for future in as_completed(futures):
            job = futures[future]
            records[job["id"]] = future.result()
            report(output, sources, jobs, records, "running")
            row = records[job["id"]]
            print(f"DONE {len(records)}/{len(jobs)} {job['id']} {job['task_id']} "
                  f"status={row['status']} score={row['score']}", flush=True)
    for seed, source in sources.items():
        if sha256(Path(source["root"]) / "state.json") != source["state_sha256"]:
            raise RuntimeError(f"Reevaluation changed training state for seed={seed}")
    summary = report(output, sources, jobs, records, "completed")
    print(json.dumps({seed: {k: v for k, v in group.items() if k != "results"}
                      for seed, group in summary["groups"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
