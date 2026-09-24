#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Create a reusable train/validation task-ID manifest without running tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from task_ordering import select_train_val_task_splits


BENCHMARK_CONFIGS = {
    "appworld": {"bm_kwargs": {"subset": "test_challenge"}},
    "bfcl": {"bm_kwargs": {"subset": "multi_turn_base"}},
    "swebench": {"bm_kwargs": {"subset": "princeton-nlp/SWE-bench_Verified"}},
}


def build_manifest(*, seed: int, train_count: int, val_count: int) -> dict:
    splits = select_train_val_task_splits(
        BENCHMARK_CONFIGS,
        seed=seed,
        train_count=train_count,
        val_count=val_count,
    )
    return {
        "schema_version": 1,
        "policy": "sort task IDs; shuffle with benchmark-specific seed; take train then val",
        "seed": seed,
        "train_count_per_benchmark": train_count,
        "val_count_per_benchmark": val_count,
        "benchmarks": {
            slug: {
                "benchmark_kwargs": BENCHMARK_CONFIGS[slug]["bm_kwargs"],
                "train": splits[slug]["train"],
                "val": splits[slug]["val"],
            }
            for slug in sorted(splits)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=50)
    parser.add_argument("--val-count", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(
        seed=args.seed,
        train_count=args.train_count,
        val_count=args.val_count,
    )
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != serialized:
            parser.error(f"refusing to overwrite a different split manifest: {args.output}")
        print(f"Unchanged: {args.output}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    for slug, split in manifest["benchmarks"].items():
        print(f"{slug}: train={len(split['train'])} val={len(split['val'])}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
