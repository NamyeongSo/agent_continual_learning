# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
# ruff: noqa: E402

"""Train sequential/interleaved reward judges with one frozen, task-disjoint validation set."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from exgentic.agents.premise.reward_data import BENCHMARKS, prepare_dataset, write_json
from exgentic.agents.premise.reward_pipeline import RewardLLM, RewardPipeline


class RewardValidationQueue:
    def __init__(self, workers: int = 50):
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="reward-val")
        self.lock = threading.Lock()
        self.groups, self.futures = {}, {}

    @staticmethod
    def execute(pipeline: RewardPipeline, record: dict, rules: list[dict]) -> dict:
        try:
            result = pipeline.scored_judge(record["id"], rules)
            return {"pair_id": record["id"], "benchmark": record["benchmark"], "status": "completed", **result}
        except Exception as exc:
            return {
                "pair_id": record["id"],
                "benchmark": record["benchmark"],
                "status": "error",
                "reward": 0,
                "error_type": type(exc).__name__,
            }

    def enqueue(self, pipeline: RewardPipeline, checkpoint: Path) -> None:
        checkpoint_bytes = checkpoint.read_bytes()
        snapshot = json.loads(checkpoint_bytes)
        cursor = snapshot["cursor"]
        key = (pipeline.mode, cursor)
        records = [r for r in pipeline.manifest["records"] if r["split"] == "val"]
        identity = {
            "checkpoint": cursor,
            "mode": pipeline.mode,
            "seed": pipeline.seed,
            "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "dataset_sha256": pipeline.config["manifest_sha256"],
            "beam_rank": 1,
            "learning": False,
            "pair_ids": [r["id"] for r in records],
            "benchmarks": [r["benchmark"] for r in records],
        }
        path = pipeline.output / "validation" / f"cp{cursor:05d}.json"
        with self.lock:
            if key in self.groups:
                return
            results = [None] * len(records)
            if path.exists():
                previous = json.loads(path.read_text())
                if previous["identity"] != identity:
                    raise ValueError("Existing validation differs from frozen checkpoint")
                results = previous["results"]
                if len(results) != len(records) or any(
                    row is not None and row["pair_id"] != record["id"]
                    for row, record in zip(results, records, strict=True)
                ):
                    raise ValueError("Existing validation pair list differs")
            group = {"identity": identity, "results": results, "path": path}
            self.groups[key] = group
            rules = copy.deepcopy(snapshot["beam"][0]["rules"])
            self.save(group)
            for index, record in enumerate(records):
                if results[index] is None:
                    future = self.pool.submit(self.execute, pipeline, record, rules)
                    self.futures[future] = (key, index)

    @staticmethod
    def save(group: dict) -> None:
        records = group["results"]
        by_benchmark = {}
        for benchmark in BENCHMARKS:
            positions = [i for i, slug in enumerate(group["identity"]["benchmarks"]) if slug == benchmark]
            rows = [records[i] for i in positions if records[i] is not None]
            valid = [row for row in rows if row["status"] == "completed"]
            errors = len(rows) - len(valid)
            complete = len(rows) == len(positions)
            correct = sum(row["reward"] for row in valid)
            by_benchmark[benchmark] = {
                "total": len(positions),
                "processed": len(rows),
                "errors": errors,
                "correct": correct,
                "complete": complete,
                "accuracy": correct / len(positions) if positions and complete and not errors else None,
                "valid_accuracy": correct / len(valid) if valid else None,
            }
        write_json(
            group["path"],
            {
                "identity": group["identity"],
                "results": records,
                "complete": all(row is not None for row in records),
                "by_benchmark": by_benchmark,
            },
        )

    def poll(self) -> None:
        with self.lock:
            for future, (key, index) in list(self.futures.items()):
                if future.done():
                    self.groups[key]["results"][index] = future.result()
                    del self.futures[future]
                    self.save(self.groups[key])

    def status(self) -> dict:
        with self.lock:
            return {
                "running": sum(f.running() for f in self.futures),
                "queued": sum(not f.running() for f in self.futures),
                "groups": {
                    f"{mode}:cp{cp}": {
                        "processed": sum(row is not None for row in group["results"]),
                        "total": len(group["results"]),
                        "errors": sum(row is not None and row["status"] == "error" for row in group["results"]),
                    }
                    for (mode, cp), group in self.groups.items()
                },
            }


def run(
    output: Path,
    dataset: Path,
    base_url: str,
    model: str,
    seed: int,
    workers: int,
    timeout: float,
    checkpoints: list[int] | None = None,
    *,
    revise_prompts: bool = False,
    parent_count: int = 3,
    children_per_parent: int = 3,
    past_count: int = 10,
    shared_slots: threading.BoundedSemaphore | None = None,
) -> int:
    slots = shared_slots if shared_slots is not None else threading.BoundedSemaphore(workers)
    validation = RewardValidationQueue(workers)
    pipelines = {}
    for mode in ("sequential", "interleaved"):
        directory = output / f"s{seed}_{mode}"
        llm = RewardLLM(base_url, model, directory / "llm_calls", slots, timeout=timeout)
        pipelines[mode] = RewardPipeline(
            dataset,
            directory,
            mode,
            seed,
            llm,
            past_count=past_count,
            revise_prompts=revise_prompts,
            parent_count=parent_count,
            children_per_parent=children_per_parent,
        )
    manifest = pipelines["sequential"].manifest
    counts = manifest["counts"]
    natural_points = [
        counts["bfcl"]["train_pairs"],
        counts["bfcl"]["train_pairs"] + counts["appworld"]["train_pairs"],
        sum(counts[slug]["train_pairs"] for slug in BENCHMARKS),
    ]
    points = sorted(set(checkpoints or natural_points))
    if any(point < 1 or point > len(pipelines["sequential"].order) for point in points):
        raise ValueError("Checkpoint is outside the training order")
    points = sorted({0, *points, len(pipelines["sequential"].order)})
    from exgentic.agents.premise import reward_prompts

    write_json(
        output / "launch_manifest.json",
        {
            "method": "premise_reward_bandit",
            "model": model,
            "seed": seed,
            "dataset": str(dataset.resolve()),
            "source_seed": manifest["source_seed"],
            "checkpoint_steps": points,
            "validation_workers": workers,
            "max_concurrent_llm_calls": workers,
            "parent_count": parent_count,
            "children_per_parent": children_per_parent,
            "past_count": past_count,
            "mutation_operations": ["ADD", "UPDATE"],
            "max_rules": None,
            "max_rule_chars": None,
            "reward": "Official environment preference agreement (1 correct, 0 incorrect)",
            "alignment_scoring": False,
            "counts": counts,
            "prompts": {
                "judge": reward_prompts.JUDGE_SYSTEM,
                "reflector": reward_prompts.REFLECTOR_SYSTEM,
                "mutator": reward_prompts.MUTATOR_SYSTEM,
            },
            "validation_policy": "Frozen beam rank 1; no feedback or validation examples enter training",
        },
    )
    for pipeline in pipelines.values():
        for point in points:
            path = pipeline.output / "checkpoints" / f"checkpoint_{point:05d}.json"
            if path.exists():
                validation.enqueue(pipeline, path)

    def train(pipeline: RewardPipeline) -> None:
        while pipeline.cursor < len(pipeline.order):
            pipeline.step()
            if pipeline.cursor in points:
                validation.enqueue(pipeline, pipeline.checkpoint())
        write_json(
            pipeline.output / "training_complete.json",
            {"cursor": pipeline.cursor, "completed_at": datetime.now(UTC).isoformat()},
        )

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="reward-train") as trainers:
        futures = {mode: trainers.submit(train, pipeline) for mode, pipeline in pipelines.items()}
        while True:
            validation.poll()
            statuses = {}
            for mode, pipeline in pipelines.items():
                future = futures[mode]
                statuses[mode] = {
                    "cursor": pipeline.cursor,
                    "total": len(pipeline.order),
                    "rules": len(pipeline.beam[0]["rules"]),
                    "status": "failed"
                    if future.done() and future.exception()
                    else "trained"
                    if future.done()
                    else "training",
                }
                if future.done() and future.exception():
                    statuses[mode]["error_type"] = type(future.exception()).__name__
            progress = validation.status()
            write_json(
                output / "status.json",
                {
                    "updated_at": datetime.now(UTC).isoformat(),
                    "pid": os.getpid(),
                    "training": statuses,
                    "validation": progress,
                },
            )
            if all(future.done() for future in futures.values()) and not progress["running"] and not progress["queued"]:
                break
            time.sleep(2)
    validation.pool.shutdown(wait=True)
    errors = sum(group["errors"] for group in validation.status()["groups"].values())
    success = all(not future.exception() for future in futures.values()) and not errors
    write_json(
        output / "complete.json",
        {"success": success, "validation_errors": errors, "completed_at": datetime.now(UTC).isoformat()},
    )
    return 0 if success else 1


def run_comparison(
    output: Path,
    dataset: Path,
    base_url: str,
    model: str,
    seed: int,
    workers: int,
    timeout: float,
    past_count: int = 10,
) -> int:
    """Run both requested beams/orders with one global request semaphore and frozen dataset."""
    slots = threading.BoundedSemaphore(workers)
    variants = [(3, 3), (2, 2)]
    write_json(
        output / "comparison_manifest.json",
        {
            "dataset": str(dataset.resolve()),
            "seed": seed,
            "past_count": past_count,
            "max_concurrent_llm_calls": workers,
            "variants": [
                {"name": f"p{parents}_c{children}", "parents": parents, "children_per_parent": children}
                for parents, children in variants
            ],
            "orders": ["sequential", "interleaved"],
            "comparison": "Only parent_count and children_per_parent differ",
        },
    )
    with ThreadPoolExecutor(max_workers=len(variants), thread_name_prefix="reward-variant") as executor:
        futures = {
            f"p{parents}_c{children}": executor.submit(
                run,
                output / f"p{parents}_c{children}",
                dataset,
                base_url,
                model,
                seed,
                workers,
                timeout,
                parent_count=parents,
                children_per_parent=children,
                past_count=past_count,
                shared_slots=slots,
            )
            for parents, children in variants
        }
        while True:
            statuses = {}
            for name, future in futures.items():
                path = output / name / "status.json"
                state = json.loads(path.read_text()) if path.exists() else {}
                statuses[name] = {
                    "training": state.get("training", {}),
                    "validation": state.get("validation", {}),
                    "status": "failed"
                    if future.done() and future.exception()
                    else "complete"
                    if future.done()
                    else "running",
                }
                if future.done() and future.exception():
                    statuses[name]["error_type"] = type(future.exception()).__name__
                    statuses[name]["error_message"] = str(future.exception())
            write_json(
                output / "status.json",
                {"updated_at": datetime.now(UTC).isoformat(), "pid": os.getpid(), "variants": statuses},
            )
            if all(future.done() for future in futures.values()):
                break
            time.sleep(2)
    success = all(not future.exception() and future.result() == 0 for future in futures.values())
    write_json(output / "complete.json", {"success": success, "completed_at": datetime.now(UTC).isoformat()})
    return 0 if success else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-pairs",
        type=Path,
        default=ROOT / "outputs/premise_reward_pair_count_20261004/non_tie_pair_metadata.jsonl",
    )
    parser.add_argument("--source-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--parents", type=int, default=3)
    parser.add_argument("--children-per-parent", type=int, default=3)
    parser.add_argument("--past-count", type=int, default=10)
    parser.add_argument(
        "--compare-beams", action="store_true", help="Run 3x3 and 2x2, sharing the LLM concurrency limit"
    )
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--revise-prompts", action="store_true", help="Record changed prompts and retain learned state")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.parents < 1 or args.children_per_parent < 1 or args.past_count < 0:
        parser.error("parents and children-per-parent must be positive; past-count must be nonnegative")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("This reward experiment already has a live controller")
        dataset = output / "dataset"
        if not (dataset / "manifest.json").exists():
            tokenizer = args.tokenizer_json
            if tokenizer is None:
                cached = sorted(
                    (Path.home() / ".cache/huggingface/hub" / "models--Qwen--Qwen3.6-35B-A3B/snapshots").glob(
                        "*/tokenizer.json"
                    )
                )
                if len(cached) != 1:
                    parser.error("Specify the acting model's tokenizer with --tokenizer-json")
                tokenizer = cached[0]
            manifest = prepare_dataset(
                args.source_pairs,
                dataset,
                source_seed=args.source_seed,
                split_seed=args.seed,
                val_fraction=args.val_fraction,
                tokenizer_path=tokenizer,
            )
        else:
            if not args.resume:
                parser.error("Dataset already exists; use --resume to preserve and continue it")
            manifest = json.loads((dataset / "manifest.json").read_text())
            if (
                manifest["source_seed"] != args.source_seed
                or manifest["split_seed"] != args.seed
                or manifest["val_fraction"] != args.val_fraction
            ):
                parser.error("Existing dataset split configuration differs")
        print(json.dumps({"output": str(output), "counts": manifest["counts"]}), flush=True)
        if args.prepare_only:
            return 0
        if args.compare_beams:
            return run_comparison(
                output, dataset, args.base_url, args.model, args.seed, args.workers, args.timeout, args.past_count
            )
        return run(
            output,
            dataset,
            args.base_url,
            args.model,
            args.seed,
            args.workers,
            args.timeout,
            revise_prompts=args.revise_prompts,
            parent_count=args.parents,
            children_per_parent=args.children_per_parent,
            past_count=args.past_count,
        )


if __name__ == "__main__":
    raise SystemExit(main())
