# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

#!/usr/bin/env python3

"""Create a frozen train/validation/test task-ID manifest without running tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from task_ordering import select_train_val_task_splits

BENCHMARK_CONFIGS = {
    "appworld": {"bm_kwargs": {"subset": "test_challenge", "interaction_mode": "code_generation"}},
    "bfcl": {"bm_kwargs": {"subset": "multi_turn_base"}},
    "swebench": {"bm_kwargs": {"subset": "princeton-nlp/SWE-bench_Verified"}},
    "terminalbench2": {"bm_kwargs": {"subset": "2.0"}},
}


def build_manifest(
    *,
    seed: int,
    train_count: int | None = None,
    val_count: int | None = None,
    test_count: int | None = None,
    benchmarks: tuple[str, ...] = ("bfcl", "appworld", "swebench", "terminalbench2"),
) -> dict:
    selected = {slug: BENCHMARK_CONFIGS[slug] for slug in benchmarks}
    counts = {}
    for slug in benchmarks:
        defaults = {"train": 30, "val": 30, "test": 29} if slug == "terminalbench2" else {
            "train": 50, "val": 50, "test": 50,
        }
        counts[slug] = {
            name: defaults[name] if override is None else override
            for name, override in (("train", train_count), ("val", val_count), ("test", test_count))
        }
    splits = select_train_val_task_splits(
        selected,
        seed=seed,
        per_benchmark_counts=counts,
    )
    return {
        "schema_version": 2,
        "policy": "sort task IDs; shuffle with benchmark-specific seed; take train then val then test",
        "seed": seed,
        "benchmarks": {
            slug: {
                "benchmark_kwargs": selected[slug]["bm_kwargs"],
                "split_counts": counts[slug],
                "train": splits[slug]["train"],
                "val": splits[slug]["val"],
                "test": splits[slug]["test"],
            }
            for slug in sorted(splits)
        },
    }


def build_seeded_task_splits(
    *, seeds: tuple[int, ...], benchmarks: tuple[str, ...],
    train_count: int | None = None, val_count: int | None = None, test_count: int | None = None,
) -> dict:
    """Export one independent split per benchmark and seed, keyed by slug_seed."""
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must contain distinct integers")
    result = {}
    for seed in seeds:
        manifest = build_manifest(
            seed=seed, benchmarks=benchmarks, train_count=train_count,
            val_count=val_count, test_count=test_count,
        )
        for slug in benchmarks:
            section = manifest["benchmarks"][slug]
            result[f"{slug}_{seed}"] = {name: section[name] for name in ("train", "val", "test")}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    seed_args = parser.add_mutually_exclusive_group()
    seed_args.add_argument("--seed", type=int, default=42)
    seed_args.add_argument("--seeds", type=int, nargs="+", help="Export flat benchmark_seed keys for these split seeds")
    parser.add_argument(
        "--train-count", type=int, default=None, help="Override train size for every selected benchmark"
    )
    parser.add_argument("--val-count", type=int, default=None, help="Override val size for every selected benchmark")
    parser.add_argument("--test-count", type=int, default=None, help="Override test size for every selected benchmark")
    parser.add_argument(
        "--benchmarks", default=None,
        help="Comma-separated slugs; defaults are 50/50/50, or 30/30/29 for Terminal-Bench 2.0",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    default_benchmarks = "bfcl,appworld,terminalbench2" if args.seeds else "bfcl,appworld,swebench,terminalbench2"
    benchmarks = tuple(slug.strip() for slug in (args.benchmarks or default_benchmarks).split(",") if slug.strip())
    if not benchmarks or len(set(benchmarks)) != len(benchmarks) or set(benchmarks) - set(BENCHMARK_CONFIGS):
        parser.error(f"Invalid --benchmarks; available: {', '.join(sorted(BENCHMARK_CONFIGS))}")

    count_args = {"train_count": args.train_count, "val_count": args.val_count, "test_count": args.test_count}
    if args.seeds:
        if len(args.seeds) != len(set(args.seeds)):
            parser.error("--seeds must contain distinct integers")
        manifest = build_seeded_task_splits(seeds=tuple(args.seeds), benchmarks=benchmarks, **count_args)
        sections = manifest
    else:
        manifest = build_manifest(seed=args.seed, benchmarks=benchmarks, **count_args)
        sections = manifest["benchmarks"]
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != serialized:
            parser.error(f"refusing to overwrite a different split manifest: {args.output}")
        print(f"Unchanged: {args.output}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    for slug, split in sections.items():
        print(f"{slug}: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
