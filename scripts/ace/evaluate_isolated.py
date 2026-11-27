# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Evaluate one finished ACE isolated playbook on its own frozen test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from exgentic.utils.task_splits import select_task_split_sections, validate_task_split  # noqa: E402
from scripts.ace.evaluate_test_checkpoints import digest, run_job, save_json  # noqa: E402
from scripts.ace.run_experiment import task_order_from_train_manifest  # noqa: E402


def prepare_job(train_dir: Path, output: Path, manifest: Path, benchmark: str, seed: int) -> dict:
    config = json.loads((train_dir / "experiment_config.json").read_text(encoding="utf-8"))
    expected_order = task_order_from_train_manifest(manifest, [benchmark], seed, None, "isolated")
    if (config.get("mode") != "isolated" or config.get("seed") != seed
            or config.get("benchmarks") != [benchmark]
            or [tuple(item) for item in config.get("task_order", [])] != expected_order
            or config.get("task_split_manifest_sha256") != digest(manifest)
            or config.get("training_time") is not True):
        raise ValueError("ACE isolated training configuration differs from frozen test plan")
    complete = json.loads((train_dir / "training_complete.json").read_text(encoding="utf-8"))
    if complete.get("completed_task_count") != len(expected_order):
        raise ValueError("ACE isolated training is incomplete")
    source = train_dir / f"playbook_ace_isolated_{benchmark}.json"
    checkpoint = json.loads(source.read_text(encoding="utf-8"))
    if checkpoint.get("store_id") != f"ace_isolated_{benchmark}":
        raise ValueError("Wrong isolated playbook store")
    split = json.loads(manifest.read_text(encoding="utf-8"))
    section = select_task_split_sections(split, [benchmark], seed)[benchmark]
    ids = validate_task_split(section, benchmark)["test"]
    expected = 29 if benchmark == "terminalbench2" else 50
    if len(ids) != expected:
        raise ValueError(f"Unexpected {benchmark} test split size")
    stage = len(expected_order)
    name = f"isolated_{benchmark}_s{seed}"
    snapshot = output / "snapshots" / name / f"checkpoint_{stage:03d}.json"
    payload = {**checkpoint, "training_task_ordinal": stage, "source_checkpoint": str(source.resolve()),
               "source_checkpoint_sha256": digest(source)}
    if snapshot.is_file() and json.loads(snapshot.read_text(encoding="utf-8")) != payload:
        raise ValueError("Frozen isolated snapshot changed")
    if not snapshot.is_file():
        save_json(snapshot, payload)
    return {
        "run": name, "mode": "isolated", "seed": seed, "stage": stage, "benchmark": benchmark,
        "train_dir": str(train_dir.resolve()), "task_ids": ids, "model": config["model"],
        "max_tokens": config["max_tokens"],
        "max_steps": config["benchmark_max_turns"].get(benchmark) or (100 if benchmark == "terminalbench2" else 50),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "actor_context": config.get("actor_context", "legacy"),
        "actor_context_metadata": config.get("actor_context_metadata"),
        "benchmark_kwargs": config.get("benchmark_configs", {}).get(benchmark),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("bfcl", "appworld", "terminalbench2"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 12:
        parser.error("--max-workers must be between 1 and 12")
    os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:8006/v1")
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    job = prepare_job(args.train_dir, args.output_dir, args.manifest, args.benchmark, args.seed)
    summary = run_job(job, args.output_dir, args.max_workers)
    print(json.dumps({"benchmark": args.benchmark, "seed": args.seed, "test_tasks": len(job["task_ids"]),
                      "score": summary["benchmark_score"], "execution_errors": summary["error_count"]}), flush=True)


if __name__ == "__main__":
    main()
