# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Reevaluate interleaved GEPA/SkillOpt validation winners on AppWorld across seeds."""

from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.utils import evaluate_team_prompt_queue as team  # noqa: E402
from scripts.utils.evaluate_interleaved_validation_queue import available_workers  # noqa: E402

shared = team.shared
HISTORY_RUNS = {
    "gepa": "07_gepa_interleaved_seed44_legacy",
    "skillopt": "16_skillopt_interleaved_seed44_legacy",
}
HISTORY_RUNS_BY_SEED = {
    42: {"gepa": "02_gepa_interleaved_seed42_legacy", "skillopt": "10_skillopt_interleaved_seed42_legacy"},
    43: {"gepa": "05_gepa_interleaved_seed43_legacy", "skillopt": "13_skillopt_interleaved_seed43_legacy"},
    44: HISTORY_RUNS,
}


def _prepare_seed(
    source: Path, splits_path: Path, output: Path, workers: int, base_url: str,
    seed: int, max_tokens: int, model_num_retries: int,
) -> dict:
    from exgentic.utils.task_splits import validate_task_split

    split_raw = splits_path.read_bytes()
    section = validate_task_split(json.loads(split_raw)[f"appworld_{seed}"], "appworld")
    tasks = section["val"]
    if len(tasks) != 50:
        raise ValueError(f"Expected the full 50-task seed-{seed} AppWorld validation set")
    groups = []
    for method, run in HISTORY_RUNS_BY_SEED[seed].items():
        history = source / "training_history_20261009/runs" / run / "shared/raw"
        if method == "gepa":
            result = shared.read_json(history / "gepa_result.json")
            selected = result["best_idx"]
            scores = result["val_aggregate_scores"]
            if scores[selected] != max(scores):
                raise ValueError("GEPA export does not select the highest validation score")
            config = shared.read_json(history / "experiment_config.json")
            if (config.get("seed", seed), config.get("mode", "interleaved")) != (seed, "interleaved"):
                raise ValueError("GEPA training seed or mode differs")
            recorded_tasks = [r["task_id"] for r in config["validation_task_ids"] if r["benchmark"] == "appworld"]
            if set(recorded_tasks) != set(tasks):
                raise ValueError("GEPA validation task set differs from the frozen split")
            original = source / f"gepa_interleaved_seed{seed}/cp130_shared/candidate.json"
            if shared.read_json(original)["best_candidate"] != result["candidates"][selected]:
                raise ValueError("GEPA checkpoint differs from the highest-scoring candidate")
            selection = {"candidate_index": selected, "metric": "soft", "score": scores[selected],
                         "n": len(config["validation_task_ids"])}
        else:
            summary = shared.read_json(history / "summary.json")
            selected = summary["best_step"]
            step = history / "steps" / f"step_{selected:04d}"
            original = source / f"skillopt_interleaved_seed{seed}/cp130_shared_best/skill.md"
            if original.read_bytes() != (history / "best_skill.md").read_bytes():
                raise ValueError("SkillOpt export differs from its selected best skill")
            if original.read_bytes() != (step / "candidate_skill.md").read_bytes():
                raise ValueError("SkillOpt best skill differs from its selected step")
            results = shared.read_json(step / "selection_eval/results.json")
            score = sum(r["hard"] for r in results) / len(results)
            recorded_tasks = [r["task_id"] for r in results if r["benchmark"] == "appworld"]
            if len(recorded_tasks) != 20 or not set(recorded_tasks).issubset(tasks):
                raise ValueError("SkillOpt selection set is not the expected validation subset")
            evaluations = [history / "selection_eval_baseline/results.json",
                           *sorted(history.glob("steps/step_*/selection_eval/results.json"))]
            peak = max(sum(r["hard"] for r in rows) / len(rows)
                       for p in evaluations if p.exists() for rows in [shared.read_json(p)])
            if score != peak or score != summary["best_selection_hard"]:
                raise ValueError("SkillOpt checkpoint does not have the highest recorded validation hard score")
            selection = {"step": selected, "metric": "hard", "score": score, "n": len(results)}
        raw = original.read_bytes()
        provenance = shared.read_json(original.with_name("provenance.json"))
        if (provenance["method"], provenance["mode"], provenance["seed"], provenance["checkpoint"]) != (
            method, "interleaved", seed, 130,
        ) or shared.digest(raw) != provenance["sha256"]:
            raise ValueError("Selected checkpoint provenance differs")
        prompt = team.exported_prompt(method, raw)
        frozen_directory = output / "checkpoints" / method
        if seed != 44:
            frozen_directory = frozen_directory / f"s{seed}"
        frozen = frozen_directory / original.name
        if frozen.exists() and frozen.read_bytes() != raw:
            raise ValueError("Frozen checkpoint differs from the selected optimizer winner")
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(raw)
        shared.write_json(frozen.with_name("provenance.json"), provenance)
        groups.append({
            "group": f"appworld/{method}/interleaved/s{seed}/cp130/val", "benchmark": "appworld",
            "method": method, "mode": "interleaved", "seed": seed, "checkpoint": 130, "split": "val",
            "task_ids": tasks, "checkpoint_path": str(frozen), "checkpoint_sha256": shared.digest(raw),
            "prompt_sha256": shared.digest(prompt.encode()), "source": str(original),
            "provenance": provenance, "selection": selection, "history_source": str(history),
            "model": "openai/Qwen/Qwen3.6-35B-A3B", "temperature": 0.0, "subset": "test_challenge",
            "max_tokens": max_tokens, "model_num_retries": model_num_retries,
            "transport_additional_retries": model_num_retries, "transport_retry_after": 0.5,
            "evaluation_label": output.name,
        })
    (output / "task_split_manifest.json").write_bytes(split_raw)
    return {
        "created_at": shared.timestamp(), "source": str(source), "manifest_source": str(splits_path),
        "manifest_sha256": shared.digest(split_raw), "workers": workers, "base_url": base_url,
        "checkpoint_count": 2, "total_tasks": 100, "validation_tasks_per_checkpoint": 50,
        "seed": seed, "seeds": [seed], "mode": "interleaved", "split": "val", "benchmark": "appworld",
        "max_tokens": max_tokens, "model_num_retries": model_num_retries,
        "output_length_additional_retries": shared.LENGTH_RETRIES,
        "output_length_max_attempts": 3, "transport_additional_retries": model_num_retries, "learning": False,
        "policy": "Original AppWorld benchmark policy; no teammate override", "max_steps": 50, "max_actions": 100,
        "prompt_application": "Exact exported full system prompt through frozen PREMiSE tool-calling runtime",
        "enable_tool_shortlisting": True, "max_selected_tools": 30, "temperature": 0.0,
        "runner_sha256": shared.digest(Path(__file__).read_bytes()),
        "worker_runner_sha256": shared.digest(Path(team.__file__).read_bytes()),
        "shared_runner_sha256": shared.digest(Path(shared.__file__).read_bytes()), "groups": groups,
    }


def prepare(
    source: Path, splits_path: Path, output: Path, workers: int, base_url: str,
    seeds: tuple[int, ...] = (44,), max_tokens: int = shared.MAX_TOKENS, model_num_retries: int = 0,
    capacity_after: Path | None = None,
) -> dict:
    seeds = tuple(sorted(seeds))
    if not seeds or len(set(seeds)) != len(seeds) or any(seed not in HISTORY_RUNS_BY_SEED for seed in seeds):
        raise ValueError("Seeds must be unique values from 42, 43, and 44")
    if max_tokens < 1 or model_num_retries < 0 or workers < 1:
        raise ValueError("Output limit and workers must be positive; retries must be nonnegative")
    output.mkdir(parents=True, exist_ok=True)
    launches = [_prepare_seed(source, splits_path, output, workers, base_url, seed, max_tokens, model_num_retries)
                for seed in seeds]
    launch = launches[0]
    groups = [group for item in launches for group in item["groups"]]
    launch.update(groups=groups, seed=seeds[0] if len(seeds) == 1 else None, seeds=list(seeds),
                  checkpoint_count=len(groups), total_tasks=sum(len(group["task_ids"]) for group in groups))
    if capacity_after is not None:
        dependency = shared.read_json(capacity_after / "launch_manifest.json")
        if dependency["base_url"] != base_url:
            raise ValueError("Shared-capacity evaluations must use the same model endpoint")
        launch["capacity_dependency"] = {
            "output": str(capacity_after),
            "launch_sha256": shared.digest((capacity_after / "launch_manifest.json").read_bytes()),
            "total_tasks": dependency["total_tasks"], "workers": dependency["workers"], "global_workers": workers,
        }
        available_workers(launch)
    return launch


def save_scores(output: Path, groups: list[dict], records: dict[str, list]) -> None:
    scores = []
    for group in groups:
        summary = shared.summarize(group, records[group["group"]])
        shared.write_json(output / "summaries" / group["group"] / "summary.json", summary)
        scores.append({
            "method": group["method"], "seed": group["seed"], "mode": group["mode"],
            "checkpoint": group["checkpoint"], "benchmark": group["benchmark"], "split": group["split"],
            "selection": group["selection"],
            "processed": summary["processed"], "total": summary["total"], "complete": summary["complete"],
            "execution_errors": summary["execution_errors"],
            "hard": summary["score_with_errors_zero"] if summary["complete"] else None,
            "soft": summary["partial_score_with_errors_zero"] if summary["complete"] else None,
            "error_scoring": "zero", "std": None, "std_reason": "This checkpoint result represents one seed",
        })
    shared.write_json(output / "scores.json", {"updated_at": shared.timestamp(), "scores": scores})


def run(output: Path, launch: dict) -> int:
    groups = launch["groups"]
    benchmark = launch.get("benchmark", "appworld")
    partition = launch.get("split", "val")
    for group in groups:
        if group.get("max_tokens", shared.MAX_TOKENS) != launch["max_tokens"]:
            raise ValueError("Task output limit differs from the launch manifest")
        if group.get("transport_additional_retries", 0) != launch["transport_additional_retries"]:
            raise ValueError("Task request retry budget differs from the launch manifest")
    records = {g["group"]: [None] * len(g["task_ids"]) for g in groups}
    for group in groups:
        for index, task in enumerate(group["task_ids"]):
            path = output / "results" / group["group"] / f"{index:03d}.json"
            if path.exists():
                row = shared.read_json(path)
                fields = ("method", "seed", "mode", "checkpoint", "benchmark", "checkpoint_sha256",
                          "prompt_sha256", "split", "subset", "model", "temperature")
                if row["task_id"] != task or any(row[k] != group[k] for k in fields):
                    raise ValueError("Stored result differs from the frozen validation inputs")
                if any(row.get(key, default) != group.get(key, default) for key, default in (
                    ("max_tokens", shared.MAX_TOKENS), ("model_num_retries", 0), ("transport_additional_retries", 0),
                )):
                    raise ValueError("Stored result sampling or retry settings differ")
                if row["status"] not in ("completed", "error"):
                    raise ValueError("Stored result has no terminal status")
                records[group["group"]][index] = row
    queue = shared.queue_jobs(groups, records)
    (output / "queue.jsonl").write_text("".join(json.dumps(job) + "\n" for job in queue))
    save_scores(output, groups, records)
    pending = {}
    print(f"START {benchmark} {partition}: checkpoints={launch['checkpoint_count']} tasks={len(queue)} "
          f"concurrency={launch['workers']}", flush=True)
    with ProcessPoolExecutor(max_workers=launch["workers"], mp_context=multiprocessing.get_context("spawn"),
                             initializer=team.worker_init, initargs=(str(output), launch["base_url"])) as pool:
        while queue or pending:
            limit = available_workers(launch)
            while queue and len(pending) < limit:
                job = queue.popleft()
                pending[pool.submit(team.evaluate_task, job)] = job
            observed = [r for rows in records.values() for r in rows if r is not None]
            shared.write_json(output / "status.json", {
                "controller_pid": os.getpid(), "phase": "waiting_capacity" if not pending and not limit else "running",
                "stage": f"{benchmark}_{'validation' if partition == 'val' else partition}", "available_workers": limit,
                "workers": launch["workers"], "active_tasks": len(pending), "queued_tasks": len(queue),
                "processed_tasks": len(observed), "total_tasks": launch["total_tasks"],
                "checkpoint_count": launch["checkpoint_count"], "evaluation_group_count": len(groups),
                "execution_errors": sum(r["status"] == "error" for r in observed), "updated_at": shared.timestamp(),
            })
            if not pending:
                time.sleep(5)
                continue
            done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                row = future.result()
                records[job["group"]][job["index"]] = row
                print(f"DONE {job['method']} s{job['seed']} task={job['index']:02d} status={row['status']}", flush=True)
            if done:
                save_scores(output, groups, records)
    save_scores(output, groups, records)
    errors = sum(r["status"] == "error" for rows in records.values() for r in rows)
    completion = {"controller_pid": os.getpid(), "phase": "complete", "all_tasks_terminal": True,
                  "processed_tasks": launch["total_tasks"], "total_tasks": launch["total_tasks"],
                  "checkpoint_count": launch["checkpoint_count"], "evaluation_group_count": len(groups),
                  "execution_errors": errors, "workers": launch["workers"],
                  "active_tasks": 0, "queued_tasks": 0, "updated_at": shared.timestamp()}
    marker = "validation_complete.json" if partition == "val" else "evaluation_complete.json"
    shared.write_json(output / marker, completion)
    shared.write_json(output / "status.json", completion)
    print(f"FINISH {benchmark} {partition}: {launch['total_tasks']}/{launch['total_tasks']} terminal; "
          f"errors={errors}", flush=True)
    return 0 if not errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "team_prompt")
    parser.add_argument("--splits", type=Path, default=shared.SPLITS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=100)
    parser.add_argument("--base-url", default="http://127.0.0.1:18006/v1")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--num-retries", type=int)
    parser.add_argument("--capacity-after", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "automatic_resume_disabled.json").exists() and not args.dry_run:
            print("Validation evaluation remains paused; no workers started.", flush=True)
            return 0
        manifest = output / "launch_manifest.json"
        if manifest.exists():
            launch = shared.read_json(manifest)
            if (launch["source"], launch["manifest_source"], launch["workers"], launch["base_url"]) != (
                str(args.source.resolve()), str(args.splits.resolve()), args.workers, args.base_url,
            ):
                raise ValueError("Resume settings differ from the frozen validation run")
            if args.seeds is not None and sorted(args.seeds) != launch.get("seeds", [launch["seed"]]):
                raise ValueError("Resume seeds differ")
            if args.max_tokens is not None and args.max_tokens != launch["max_tokens"]:
                raise ValueError("Resume output limit differs")
            if args.num_retries is not None and args.num_retries != launch["transport_additional_retries"]:
                raise ValueError("Resume request retry budget differs")
            if args.capacity_after is not None and str(args.capacity_after.resolve()) != (
                launch.get("capacity_dependency", {}).get("output")
            ):
                raise ValueError("Resume capacity dependency differs")
        else:
            launch = prepare(
                args.source.resolve(), args.splits.resolve(), output, args.workers, args.base_url,
                seeds=(44,) if args.seeds is None else tuple(args.seeds),
                max_tokens=shared.MAX_TOKENS if args.max_tokens is None else args.max_tokens,
                model_num_retries=0 if args.num_retries is None else args.num_retries,
                capacity_after=None if args.capacity_after is None else args.capacity_after.resolve(),
            )
            shared.write_json(manifest, launch)
        if args.dry_run:
            print(json.dumps({k: v for k, v in launch.items() if k != "groups"}, indent=2))
            return 0
        return run(output, launch)


if __name__ == "__main__":
    raise SystemExit(main())
