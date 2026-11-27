# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Evaluate frozen interleaved checkpoints across seeds in one shared benchmark queue."""

from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing
import os
import random
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.utils import evaluate_isolated_checkpoint_queue as shared  # noqa: E402
from scripts.utils.interleaved_checkpoint_recovery import collect  # noqa: E402

SPLITS = shared.SPLITS
ACE_ROOT = ROOT / "scripts/ace/outputs/ace_appworld_scorefix_resume_bfcl50_20260928"
PREMISE_ROOT = ROOT / "outputs/premise_bfcl_appworld_terminalbench2_seed42_44_scorefix_20260928"
BENCHMARKS = ("bfcl", "appworld", "terminalbench2")
SUBSETS = {"bfcl": "multi_turn_base", "appworld": "test_challenge", "terminalbench2": "2.0"}
METHODS = ("premise", "ace")
POINTS = (50, 100, 130)
SEED = 44
_OUTPUT: Path | None = None
_PLAYBOOKS: dict[str, str] = {}


def checkpoint_playbook(method: str, snapshot: dict, point: int, seed: int = SEED) -> str:
    if method == "ace":
        if snapshot["store_id"] != "ace_interleaved_global" or snapshot["session_count"] != point:
            raise ValueError("ACE snapshot is not the requested interleaved checkpoint")
        if snapshot.get("training_task_ordinal", point) != point:
            raise ValueError("ACE checkpoint ordinal differs")
        playbook = snapshot["playbook"]
    elif method == "premise":
        if (snapshot["config"]["mode"], snapshot["config"]["seed"], snapshot["cursor"]) != (
            "interleaved", seed, point,
        ):
            raise ValueError("PREMiSE snapshot is not the requested seed/mode/checkpoint")
        playbook = snapshot["states"]["global"]["beam"][0]["playbook"]
    else:
        raise ValueError(f"Unsupported method: {method}")
    if not isinstance(playbook, str) or not playbook.strip():
        raise ValueError("Checkpoint playbook is empty")
    return playbook


def prepare(
    output: Path, workers: int, base_url: str, points: tuple[int, ...] | None = None,
    seeds: tuple[int, ...] | None = None, max_tokens: int = shared.MAX_TOKENS, model_num_retries: int = 0,
) -> dict:
    from exgentic.utils.task_splits import validate_task_split

    points = POINTS if points is None else tuple(points)
    if not points or len(set(points)) != len(points) or any(point < 1 or point > 130 for point in points):
        raise ValueError("Checkpoint ordinals must be unique integers between 1 and 130")
    points = tuple(sorted(points))
    seeds = (SEED,) if seeds is None else tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed not in (42, 43, 44) for seed in seeds):
        raise ValueError("Seeds must be unique values from 42, 43, and 44")
    seeds = tuple(sorted(seeds))
    if max_tokens < 1 or model_num_retries < 0:
        raise ValueError("Output limit must be positive and request retries nonnegative")
    split_raw = SPLITS.read_bytes()
    splits = json.loads(split_raw)
    expected = {"bfcl": 50, "appworld": 50, "terminalbench2": 30}
    groups = []
    for seed in seeds:
        for benchmark in BENCHMARKS:
            section = validate_task_split(splits[f"{benchmark}_{seed}"], benchmark)
            if len(section["val"]) != expected[benchmark]:
                raise ValueError(f"Unexpected validation size for {benchmark} seed {seed}")
        recovery_output = output if seeds == (SEED,) else output / "source_recovery" / f"s{seed}"
        sources = collect(recovery_output, ACE_ROOT, PREMISE_ROOT, points, seed)
        for method in METHODS:
            for point in points:
                origin = sources[(method, point)]
                source = Path(origin["path"])
                raw = source.read_bytes()
                snapshot = json.loads(raw)
                playbook = checkpoint_playbook(method, snapshot, point, seed)
                if method == "ace" and snapshot.get("source_checkpoint_sha256"):
                    recorded_path = Path(snapshot["source_checkpoint"])
                    recorded_raw = recorded_path.read_bytes()
                    if shared.digest(recorded_raw) != snapshot["source_checkpoint_sha256"]:
                        raise ValueError("ACE export differs from its recorded training checkpoint")
                    if any(snapshot.get(key) != value for key, value in json.loads(recorded_raw).items()):
                        raise ValueError("ACE exported checkpoint state differs from its training checkpoint")
                    training_config = ACE_ROOT / f"interleaved_s{seed}/experiment_config.json"
                    config_raw = training_config.read_bytes()
                    config = json.loads(config_raw)
                    if (config["seed"], config["mode"]) != (seed, "interleaved"):
                        raise ValueError("ACE source training seed or mode differs")
                    if not recorded_path.resolve().is_relative_to(training_config.parent.resolve()):
                        raise ValueError("ACE checkpoint belongs to another training run")
                    origin = {**origin, "source_training_config": str(training_config),
                              "source_training_config_sha256": shared.digest(config_raw)}
                frozen = output / "checkpoints" / method
                if seeds != (SEED,):
                    frozen = frozen / f"s{seed}"
                frozen = frozen / f"cp{point:03d}.json"
                if frozen.exists() and frozen.read_bytes() != raw:
                    raise ValueError("Frozen checkpoint differs from its source")
                frozen.parent.mkdir(parents=True, exist_ok=True)
                frozen.write_bytes(raw)
                for benchmark in BENCHMARKS:
                    section = splits[f"{benchmark}_{seed}"]
                    groups.append({
                        "group": f"{method}/interleaved/s{seed}/cp{point:03d}/val/{benchmark}",
                        "method": method, "mode": "interleaved", "seed": seed, "checkpoint": point,
                        "benchmark": benchmark, "split": "val", "task_ids": section["val"],
                        "source_checkpoint": str(source), "checkpoint_path": str(frozen),
                        "checkpoint_origin": origin,
                        "checkpoint_sha256": shared.digest(raw), "playbook_sha256": shared.digest(playbook.encode()),
                        "model": "openai/Qwen/Qwen3.6-35B-A3B", "temperature": 0.0,
                        "subset": SUBSETS[benchmark],
                        "max_steps": 100 if benchmark == "terminalbench2" else 50,
                        "max_tokens": None if benchmark == "terminalbench2" else max_tokens,
                        "model_num_retries": model_num_retries,
                        "transport_additional_retries": 0 if benchmark == "terminalbench2" else model_num_retries,
                        "transport_retry_after": 0.5,
                        "evaluation_label": output.name,
                    })
    total = sum(len(group["task_ids"]) for group in groups)
    checkpoint_count = len(seeds) * len(METHODS) * len(points)
    if len(groups) != checkpoint_count * len(BENCHMARKS) or total != checkpoint_count * 130:
        raise ValueError("Validation queue does not cover every requested checkpoint and benchmark")
    (output / "task_split_manifest.json").write_bytes(split_raw)
    return {
        "created_at": shared.timestamp(), "mode": "interleaved", "seed": seeds[0] if len(seeds) == 1 else None,
        "seeds": list(seeds), "split": "val",
        "workers": workers, "base_url": base_url, "checkpoint_count": checkpoint_count, "total_tasks": total,
        "checkpoints": list(points),
        "validation_tasks_per_checkpoint": 130, "validation_by_benchmark": expected,
        "max_tokens": {"bfcl": max_tokens, "appworld": max_tokens, "terminalbench2": None},
        "output_length_additional_retries": {"bfcl": 2, "appworld": 2, "terminalbench2": "Harbor default"},
        "output_length_max_attempts": {"bfcl": 3, "appworld": 3, "terminalbench2": "Harbor default"},
        "transport_additional_retries": {"bfcl": model_num_retries, "appworld": model_num_retries,
                                         "terminalbench2": "Harbor default"},
        "learning": False, "premise_beam_rank": 1, "policy": "Original benchmark policies; no teammate override",
        "terminalbench2_actor": "terminus2", "terminalbench2_enforce_session_timeouts": True,
        "terminalbench2_completion_settings": "Existing Terminus-2/Harbor defaults; no evaluation output-token cap",
        "manifest_source": str(SPLITS), "manifest_sha256": shared.digest(split_raw),
        "runner_sha256": shared.digest(Path(__file__).read_bytes()),
        "worker_architecture": "One common queue; reusable spawned workers; one evaluation task per worker",
        "groups": groups,
    }


def queue_jobs(groups: list[dict], records: dict[str, list]) -> deque:
    order = [(benchmark, index) for benchmark in BENCHMARKS
             for index in range(max(len(g["task_ids"]) for g in groups if g["benchmark"] == benchmark))]
    random.Random(SEED).shuffle(order)
    jobs = deque()
    for benchmark, index in order:
        for group in groups:
            if group["benchmark"] == benchmark and records[group["group"]][index] is None:
                fields = {key: value for key, value in group.items() if key != "task_ids"}
                jobs.append({**fields, "index": index, "task_id": group["task_ids"][index]})
    return jobs


def worker_init(output: str, base_url: str) -> None:
    global _OUTPUT
    _OUTPUT = Path(output)
    shared.worker_init(output, base_url)


def task_output_limit(job: dict) -> int | None:
    if job["benchmark"] == "terminalbench2":
        return None
    limit = job.get("max_tokens", shared.MAX_TOKENS)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("BFCL/AppWorld output limit must be a positive integer")
    return limit


def playbook_identity(job: dict) -> str:
    return f"{job['method']}/s{job['seed']}/cp{job['checkpoint']:03d}/{job['checkpoint_sha256']}"


def task_config(job: dict, directory: Path, run_id: str):
    from exgentic.core.types import RunConfig

    identity = playbook_identity(job)
    if identity not in _PLAYBOOKS:
        raw = Path(job["checkpoint_path"]).read_bytes()
        if shared.digest(raw) != job["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint changed")
        playbook = checkpoint_playbook(job["method"], json.loads(raw), job["checkpoint"], job["seed"])
        if shared.digest(playbook.encode()) != job["playbook_sha256"]:
            raise ValueError("Frozen playbook changed")
        _PLAYBOOKS[identity] = playbook
    playbook = _PLAYBOOKS[identity]
    kwargs = {
        "model": job["model"], "runner": "direct", "enable_thinking": False,
        "enable_tool_shortlisting": job["benchmark"] == "appworld", "max_selected_tools": 30,
        "model_settings": {
            "temperature": job["temperature"], "max_tokens": task_output_limit(job), "timeout": 600,
            "num_retries": job.get("model_num_retries", 0), "retry_after": job.get("transport_retry_after", 0.5),
            "retry_strategy": "exponential_backoff_retry",
        },
    }
    if job["benchmark"] == "terminalbench2":
        kwargs["model_settings"].pop("max_tokens")
    if job["method"] == "ace":
        kwargs.update(initial_playbook=playbook, shuffle_mode="interleaved", benchmark_id=job["benchmark"],
                      evaluation_mode=True, training_time=False, curator_model=job["model"])
    else:
        kwargs["playbook"] = playbook
    benchmark_kwargs = {"subset": job["subset"], "use_cache": False}
    if job["benchmark"] == "appworld":
        benchmark_kwargs["env_kwargs"] = {"experiment_name": run_id}
    elif job["benchmark"] == "terminalbench2":
        benchmark_kwargs.update(actor="terminus2", max_interactions=100, enforce_session_timeouts=True)
    return RunConfig(
        benchmark=job["benchmark"], agent=job["method"], model=job["model"],
        task_ids=[job["task_id"]], output_dir=str(directory), run_id=run_id,
        agent_kwargs=kwargs, benchmark_kwargs=benchmark_kwargs, max_workers=1,
        max_steps=job["max_steps"], max_actions=100,
    )


def evaluate_task(job: dict) -> dict:
    from exgentic.agents.ace.playbook_store import PlaybookStore
    from exgentic.interfaces.lib.api import execute

    key, index = job["group"], job["index"]
    directory = _OUTPUT / "sessions" / key
    run_id = f"{job['split']}{job['seed']}_{job['method']}_cp{job['checkpoint']:03d}_{job['benchmark']}_{index:03d}"
    if job.get("evaluation_label"):
        run_id += f"_{job['evaluation_label']}"
    audit_path = _OUTPUT / "request_audits" / key / f"{index:03d}.json"
    audit = {"task_id": job["task_id"], "group": key, "worker_pid": os.getpid(),
             "audit_path": str(audit_path), "model_calls": 0, "output_length_errors": 0,
             "output_length_retries": 0, "length_exhausted": False, "max_tokens": task_output_limit(job),
             "transport_additional_retries": job.get("transport_additional_retries", 0),
             "transport_retry_after": job.get("transport_retry_after", 0.5), "transport_retries": 0}
    shared._AUDIT = audit
    shared.write_json(audit_path, audit)
    shared.write_json(_OUTPUT / "active_workers" / f"{os.getpid()}.json", {
        "phase": "running", "started_at": shared.timestamp(), **job, "run_id": run_id,
    })
    row = {**job, "worker_pid": os.getpid(), "started_at": shared.timestamp(), "run_id": run_id}
    try:
        PlaybookStore.reset_all()
        result = execute(config=task_config(job, directory, run_id))
        if len(result.session_results) != 1:
            raise ValueError("Expected exactly one session result")
        session = result.session_results[0]
        session_dir = directory / run_id / "sessions" / session.session_id
        row.update(session_id=session.session_id, session_dir=str(session_dir), session_status=session.status.value)
        if audit["length_exhausted"]:
            raise shared.OutputLengthRetriesExhaustedError("Output length failed after two additional retries")
        if session.status.value in ("error", "cancelled") or session.score is None:
            raise RuntimeError(str(session.details))
        if job["method"] == "ace":
            playbook = _PLAYBOOKS[playbook_identity(job)]
            if any(store.playbook != playbook or store.session_count != 0
                   for store in PlaybookStore.list_stores().values()):
                raise RuntimeError("Frozen ACE playbook changed during evaluation")
        if job["benchmark"] == "appworld":
            scores = shared.read_json(session_dir / "benchmark/scores.json")
            if not isinstance(scores.get("success"), bool):
                raise ValueError("Missing official AppWorld success flag")
            hard = float(scores["success"])
        else:
            hard = float(session.score)
        row.update(status="completed", score=hard, partial_score=float(session.score), steps=session.steps)
    except Exception as exc:
        row.update(status="error", score=0.0, partial_score=0.0, error_type=type(exc).__name__,
                   error=str(exc), output_length_failure=audit["length_exhausted"])
    terminal = job["benchmark"] == "terminalbench2"
    row.update(completed_at=shared.timestamp(), model_calls=None if terminal else audit["model_calls"],
               output_length_retries=None if terminal else audit["output_length_retries"],
               transport_retries=None if terminal else audit["transport_retries"])
    shared.write_json(audit_path, audit)
    shared.write_json(_OUTPUT / "results" / key / f"{index:03d}.json", row)
    shared.write_json(_OUTPUT / "active_workers" / f"{os.getpid()}.json", {
        "phase": "idle", "completed_at": shared.timestamp(), "last_group": key,
    })
    return row


def save_scores(output: Path, groups: list[dict], records: dict[str, list]) -> None:
    scores = []
    for method in METHODS:
        identities = sorted({(g["seed"], g["checkpoint"]) for g in groups if g["method"] == method})
        for seed, point in identities:
            selected = [g for g in groups if (g["method"], g["seed"], g["checkpoint"]) == (method, seed, point)]
            rows = [r for g in selected for r in records[g["group"]] if r is not None]
            total = sum(len(g["task_ids"]) for g in selected)
            complete = len(rows) == total
            scores.append({
                "method": method, "seed": seed, "mode": "interleaved", "checkpoint": point,
                "split": selected[0]["split"],
                "processed": len(rows), "total": total, "complete": complete,
                "execution_errors": sum(r["status"] == "error" for r in rows),
                "hard": sum(r["score"] for r in rows) / total if complete else None,
                "soft": sum(r["partial_score"] for r in rows) / total if complete else None,
                "by_benchmark": {g["benchmark"]: shared.summarize(g, records[g["group"]]) for g in selected},
                "std": None, "std_reason": "This checkpoint result represents one seed",
            })
    shared.write_json(output / "scores.json", {"updated_at": shared.timestamp(), "scores": scores})


def available_workers(launch: dict) -> int:
    dependency = launch.get("capacity_dependency")
    if dependency is None:
        return launch["workers"]
    output = Path(dependency["output"])
    if shared.digest((output / "launch_manifest.json").read_bytes()) != dependency["launch_sha256"]:
        raise ValueError("Shared-capacity evaluation inputs changed")
    status = shared.read_json(output / "status.json")
    if status["total_tasks"] != dependency["total_tasks"] or status["workers"] != dependency["workers"]:
        raise ValueError("Shared-capacity evaluation size or concurrency changed")
    if status["phase"] == "complete":
        if status["processed_tasks"] != dependency["total_tasks"] or status["active_tasks"] != 0:
            raise ValueError("Shared-capacity completion status is inconsistent")
        reserved = 0
    elif status["phase"] == "running" and status["queued_tasks"] == 0:
        reserved = status["active_tasks"]
        if reserved + status["processed_tasks"] != dependency["total_tasks"]:
            raise ValueError("Draining evaluation task counts are inconsistent")
    else:
        reserved = dependency["workers"]
    return max(0, min(launch["workers"], dependency["global_workers"] - reserved))


def run(output: Path, launch: dict) -> int:
    groups = launch["groups"]
    for group in groups:
        if task_output_limit(group) != launch["max_tokens"][group["benchmark"]]:
            raise ValueError("Task output limit differs from the launch manifest")
        if group["benchmark"] != "terminalbench2" and group.get("transport_additional_retries", 0) != (
            launch["transport_additional_retries"][group["benchmark"]]
        ):
            raise ValueError("Task request retry budget differs from the launch manifest")
    by_key = {g["group"]: g for g in groups}
    records = {g["group"]: [None] * len(g["task_ids"]) for g in groups}
    for group in groups:
        for index, task_id in enumerate(group["task_ids"]):
            path = output / "results" / group["group"] / f"{index:03d}.json"
            if path.exists():
                row = shared.read_json(path)
                if any(row.get(key) != group[key] for key in ("method", "seed", "mode", "checkpoint", "benchmark")):
                    raise ValueError("Existing task seed or checkpoint identity differs from frozen evaluation")
                if (row["task_id"], row["checkpoint_sha256"], row["split"], row["status"]) not in [
                    (task_id, group["checkpoint_sha256"], group["split"], "completed"),
                    (task_id, group["checkpoint_sha256"], group["split"], "error"),
                ]:
                    raise ValueError("Existing task result differs from frozen evaluation")
                if task_output_limit(row) != task_output_limit(group):
                    raise ValueError("Existing task output limit differs from frozen evaluation")
                if any(row.get(key, default) != group.get(key, default) for key, default in (
                    ("temperature", 0.0), ("model_num_retries", 0), ("transport_additional_retries", 0),
                )):
                    raise ValueError("Existing task sampling or retry settings differ from frozen evaluation")
                records[group["group"]][index] = row
    jobs = queue_jobs(groups, records)
    pending = {}

    def status(phase="running"):
        rows = [r for values in records.values() for r in values if r is not None]
        shared.write_json(output / "status.json", {
            "controller_pid": os.getpid(), "phase": phase, "workers": launch["workers"],
            "active_tasks": len(pending), "queued_tasks": len(jobs), "processed_tasks": len(rows),
            "total_tasks": launch["total_tasks"], "checkpoint_count": launch["checkpoint_count"],
            "stage": launch["split"],
            "available_workers": available_workers(launch),
            "execution_errors": sum(r["status"] == "error" for r in rows), "updated_at": shared.timestamp(),
        })

    status("starting")
    save_scores(output, groups, records)
    print(f"START common queue: tasks={len(jobs)} checkpoints={launch['checkpoint_count']} "
          f"split={launch['split']} concurrency={launch['workers']}", flush=True)
    with ProcessPoolExecutor(max_workers=launch["workers"], mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_init, initargs=(str(output), launch["base_url"])) as pool:
        while jobs or pending:
            limit = available_workers(launch)
            while jobs and len(pending) < limit:
                job = jobs.popleft()
                pending[pool.submit(evaluate_task, job)] = job
            status()
            if not pending:
                time.sleep(5)
                continue
            done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                row = future.result()
                records[job["group"]][job["index"]] = row
                shared.write_json(output / "summaries" / job["group"] / "summary.json",
                                  shared.summarize(by_key[job["group"]], records[job["group"]]))
                print(f"DONE {job['group']} task={job['index']:02d} status={row['status']}", flush=True)
            if done:
                save_scores(output, groups, records)
    save_scores(output, groups, records)
    status("complete")
    marker = "validation_complete.json" if launch["split"] == "val" else "test_complete.json"
    shared.write_json(output / marker, {
        "completed_at": shared.timestamp(), "all_tasks_terminal": True, "processed_tasks": launch["total_tasks"],
        "checkpoint_count": launch["checkpoint_count"],
    })
    print(f"FINISH {launch['split']} queue: all {launch['total_tasks']} tasks terminal", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=150)
    parser.add_argument("--base-url", default="http://127.0.0.1:18006/v1")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--checkpoints", type=int, nargs="+")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--num-retries", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / "launch_manifest.json"
        if path.exists():
            launch = shared.read_json(path)
            if (launch["workers"], launch["base_url"]) != (args.workers, args.base_url):
                raise ValueError("Resume settings differ")
            if args.seeds is not None and sorted(args.seeds) != launch.get("seeds", [launch["seed"]]):
                raise ValueError("Resume seeds differ")
            if args.checkpoints is not None and sorted(args.checkpoints) != launch["checkpoints"]:
                raise ValueError("Resume checkpoints differ")
            if args.max_tokens is not None and args.max_tokens != launch["max_tokens"]["bfcl"]:
                raise ValueError("Resume output limit differs")
            if args.num_retries is not None and args.num_retries != launch["transport_additional_retries"]["bfcl"]:
                raise ValueError("Resume request retry budget differs")
        else:
            launch = prepare(
                output, args.workers, args.base_url, points=args.checkpoints, seeds=args.seeds,
                max_tokens=shared.MAX_TOKENS if args.max_tokens is None else args.max_tokens,
                model_num_retries=0 if args.num_retries is None else args.num_retries,
            )
            shared.write_json(path, launch)
            jobs = queue_jobs(launch["groups"], {g["group"]: [None] * len(g["task_ids"]) for g in launch["groups"]})
            (output / "queue.jsonl").write_text("".join(json.dumps(job) + "\n" for job in jobs))
        if args.dry_run:
            print(json.dumps({k: v for k, v in launch.items() if k != "groups"}, indent=2))
            return 0
        return run(output, launch)


if __name__ == "__main__":
    raise SystemExit(main())
