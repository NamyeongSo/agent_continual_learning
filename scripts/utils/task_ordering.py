# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""
Seed-controlled task selection and ordering for evaluation experiments.

Guarantees:
  - Same seed → same task set for every benchmark, regardless of mode.
  - Within-benchmark task order is identical across isolated / sequential / interleaved.
  - Interleaved only interleaves *between* benchmarks; within-benchmark order is preserved.
"""

from __future__ import annotations

import hashlib
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

# Allow importing exgentic from the repo source tree
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from exgentic.interfaces.registry import load_benchmark


SEQUENTIAL_BENCHMARK_ORDER = ("bfcl", "appworld", "swebench", "terminalbench2")


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def get_unified_task_order(
    benchmark_configs: dict[str, dict[str, Any]],
    num_tasks_per_benchmark: int,
    seed: int,
    mode: str,
) -> list[tuple[str, str]]:
    """Return a deterministic, mode-aware task ordering.

    Parameters
    ----------
    benchmark_configs : dict
        ``{slug: {"bm_kwargs": {...}, "agent_kwargs": {...}}}``
    num_tasks_per_benchmark : int
        How many tasks to select from each benchmark (e.g. 50).
    seed : int
        Ordering seed.  Controls within-benchmark task order and
        interleaved interleaving.  Task *selection* is always fixed at
        seed=42 so all experiments use the same task set.
    mode : str
        ``"isolated"`` | ``"sequential"`` | ``"interleaved"``.

    Returns
    -------
    list of (benchmark_slug, task_id)
        Ordered task sequence. For sequential mode the three experiment
        benchmarks run BFCL -> AppWorld -> SWE-bench (other benchmarks follow
        alphabetically). Isolated mode groups alphabetically; interleaved
        mode preserves within-benchmark order.
    """
    # Always use seed=42 for task SELECTION (which tasks to pick),
    # use the provided seed only for ORDERING (task sequence).
    _SELECTION_SEED = 42
    per_bm_tasks = _select_tasks(benchmark_configs, num_tasks_per_benchmark, _SELECTION_SEED)

    # Re-shuffle within-benchmark order using the provided seed
    if seed != _SELECTION_SEED:
        for slug in per_bm_tasks:
            order_seed = _derive_seed(seed, slug)
            rng = random.Random(order_seed)
            rng.shuffle(per_bm_tasks[slug])

    if mode in ("isolated", "sequential"):
        result: list[tuple[str, str]] = []
        if mode == "sequential":
            slugs = [slug for slug in SEQUENTIAL_BENCHMARK_ORDER if slug in per_bm_tasks]
            slugs.extend(sorted(set(per_bm_tasks) - set(slugs)))
        else:
            slugs = sorted(per_bm_tasks)
        for slug in slugs:
            for tid in per_bm_tasks[slug]:
                result.append((slug, tid))
        return result

    if mode == "interleaved":
        return _interleave_preserving_order(per_bm_tasks, seed)

    raise ValueError(f"Unknown mode: {mode!r}")


def select_tasks_only(
    benchmark_configs: dict[str, dict[str, Any]],
    num_tasks_per_benchmark: int,
    seed: int,
) -> dict[str, list[str]]:
    """Return the selected tasks per benchmark (no ordering applied)."""
    return _select_tasks(benchmark_configs, num_tasks_per_benchmark, seed)


def split_task_ids(
    task_ids: list[str],
    *,
    benchmark_slug: str,
    seed: int,
    train_count: int = 50,
    val_count: int = 50,
) -> dict[str, list[str]]:
    """Deterministically split one benchmark's tasks into disjoint train/val sets.

    Task IDs are sorted before shuffling so the split is independent of the
    evaluator's enumeration order. The same per-benchmark seed derivation is
    used by the existing experiment selector, but this is a separate policy:
    existing experiments are not silently switched to a new task set.
    """
    if train_count < 1 or val_count < 1:
        raise ValueError("train_count and val_count must both be positive")
    ids = [str(task_id) for task_id in task_ids]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{benchmark_slug}: duplicate task IDs in evaluator listing")
    required = train_count + val_count
    if len(ids) < required:
        raise ValueError(
            f"{benchmark_slug}: need {required} tasks for train/val, found {len(ids)}"
        )

    shuffled = sorted(ids)
    random.Random(_derive_seed(seed, benchmark_slug)).shuffle(shuffled)
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count:required],
    }


def select_train_val_task_splits(
    benchmark_configs: dict[str, dict[str, Any]],
    *,
    seed: int,
    train_count: int = 50,
    val_count: int = 50,
) -> dict[str, dict[str, list[str]]]:
    """Discover task IDs for each benchmark and apply ``split_task_ids``."""
    splits: dict[str, dict[str, list[str]]] = {}
    for slug in sorted(benchmark_configs):
        bm_kwargs = benchmark_configs[slug].get("bm_kwargs", {})
        benchmark = load_benchmark(slug)(**bm_kwargs)
        try:
            evaluator = benchmark.get_evaluator()
            try:
                task_ids = evaluator.list_tasks()
            finally:
                evaluator.close()
        finally:
            benchmark.close()
        splits[slug] = split_task_ids(
            task_ids,
            benchmark_slug=slug,
            seed=seed,
            train_count=train_count,
            val_count=val_count,
        )
    return splits


# ──────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────

def _select_tasks(
    benchmark_configs: dict[str, dict[str, Any]],
    num_tasks: int,
    seed: int,
) -> dict[str, list[str]]:
    """Select *num_tasks* tasks per benchmark using per-benchmark derived seeds."""
    per_bm: dict[str, list[str]] = {}
    for slug in sorted(benchmark_configs):
        bm_kwargs = benchmark_configs[slug].get("bm_kwargs", {})
        bm = load_benchmark(slug)(**bm_kwargs)
        evaluator = bm.get_evaluator()
        try:
            all_ids = [str(t) for t in evaluator.list_tasks()]
        finally:
            try:
                evaluator.close()
            except Exception:
                pass
            bm.close()

        bm_seed = _derive_seed(seed, slug)
        rng = random.Random(bm_seed)
        rng.shuffle(all_ids)
        per_bm[slug] = all_ids[:num_tasks]
    return per_bm


def _interleave_preserving_order(
    per_bm_tasks: dict[str, list[str]],
    seed: int,
) -> list[tuple[str, str]]:
    """Interleave tasks across benchmarks, preserving within-benchmark order.

    At each step, randomly pick a non-empty benchmark queue and pop
    its next task.  This ensures the relative order within each
    benchmark is the same as in isolated/sequential mode.
    """
    queues = {slug: deque(tasks) for slug, tasks in per_bm_tasks.items()}
    rng = random.Random(seed)
    result: list[tuple[str, str]] = []
    while any(queues.values()):
        available = sorted(s for s, q in queues.items() if q)
        slug = rng.choice(available)
        result.append((slug, queues[slug].popleft()))
    return result


def _derive_seed(master_seed: int, slug: str) -> int:
    """Derive a deterministic per-benchmark seed from master seed + slug."""
    h = hashlib.md5(f"{master_seed}_{slug}".encode()).hexdigest()
    return int(h, 16) % (2**31)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def group_by_benchmark(
    task_order: list[tuple[str, str]],
) -> list[tuple[str, list[str]]]:
    """Group a task order list into (slug, [task_ids]) preserving order."""
    groups: list[tuple[str, list[str]]] = []
    current_slug: str | None = None
    current_ids: list[str] = []
    for slug, tid in task_order:
        if slug != current_slug:
            if current_slug is not None:
                groups.append((current_slug, current_ids))
            current_slug = slug
            current_ids = [tid]
        else:
            current_ids.append(tid)
    if current_slug is not None:
        groups.append((current_slug, current_ids))
    return groups


# ──────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick sanity check without requiring benchmark data
    print("=== task_ordering.py self-test ===\n")

    # Simulate with fake data
    fake_per_bm = {
        "bfcl": ["b1", "b2", "b3", "b4", "b5"],
        "tau2": ["t1", "t2", "t3", "t4", "t5"],
        "browsecompplus": ["c1", "c2", "c3", "c4", "c5"],
    }

    # Test interleave preserving order
    interleaved = _interleave_preserving_order(fake_per_bm, seed=42)
    print("Interleaved interleave (seed=42):")
    for slug, tid in interleaved:
        print(f"  {slug}: {tid}")

    # Verify within-benchmark order is preserved
    for slug in fake_per_bm:
        original = fake_per_bm[slug]
        fused = [tid for s, tid in interleaved if s == slug]
        assert fused == original, f"{slug}: order changed! {original} → {fused}"
        print(f"  ✓ {slug} order preserved: {fused}")

    # Verify different seeds produce different interleaving
    interleaved2 = _interleave_preserving_order(fake_per_bm, seed=123)
    order1 = [(s, t) for s, t in interleaved]
    order2 = [(s, t) for s, t in interleaved2]
    print(f"\n  seed=42 vs seed=123 differ: {order1 != order2}")

    print("\nAll checks passed.")
