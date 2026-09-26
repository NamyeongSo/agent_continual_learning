# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "utils"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "ace"))

import task_ordering  # noqa: E402
from task_ordering import split_task_ids  # noqa: E402
from run_experiment import task_order_from_train_manifest  # noqa: E402


def test_split_is_deterministic_disjoint_and_order_independent():
    ids = [f"task-{i:03d}" for i in range(150)]
    first = split_task_ids(ids, benchmark_slug="bfcl", seed=42)
    assert first == split_task_ids(list(reversed(ids)), benchmark_slug="bfcl", seed=42)
    assert first == split_task_ids(ids, benchmark_slug="bfcl", seed=42)
    assert len(first["train"]) == len(first["val"]) == 50
    assert not set(first["train"]) & set(first["val"])
    assert first != split_task_ids(ids, benchmark_slug="bfcl", seed=43)


def test_split_uses_independent_benchmark_seed():
    ids = [f"task-{i:03d}" for i in range(150)]
    bfcl = split_task_ids(ids, benchmark_slug="bfcl", seed=42)
    appworld = split_task_ids(ids, benchmark_slug="appworld", seed=42)
    assert bfcl != appworld


@pytest.mark.parametrize("ids", [
    [f"task-{i}" for i in range(99)],
    [f"task-{i}" for i in range(99)] + ["task-0"],
])
def test_split_rejects_insufficient_or_duplicate_ids(ids):
    with pytest.raises(ValueError):
        split_task_ids(ids, benchmark_slug="bfcl", seed=42)


def test_split_rejects_nonpositive_counts():
    with pytest.raises(ValueError, match="positive"):
        split_task_ids(["a", "b"], benchmark_slug="bfcl", seed=42, train_count=0, val_count=1)


def test_sequential_benchmark_order(monkeypatch):
    monkeypatch.setattr(
        task_ordering,
        "_select_tasks",
        lambda configs, count, seed: {
            "swebench": ["s1"], "appworld": ["a1"], "bfcl": ["b1"]
        },
    )
    configs = {slug: {} for slug in ("appworld", "bfcl", "swebench")}
    assert task_ordering.get_unified_task_order(configs, 1, 42, "sequential") == [
        ("bfcl", "b1"), ("appworld", "a1"), ("swebench", "s1")
    ]


def test_manifest_training_order_keeps_ids_and_excludes_validation(tmp_path):
    manifest = {
        "train_count_per_benchmark": 2,
        "benchmarks": {
            slug: {"train": [f"{slug}-a", f"{slug}-b"], "val": [f"{slug}-val"]}
            for slug in ("bfcl", "appworld", "swebench")
        },
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    benchmarks = ["swebench", "appworld", "bfcl"]
    orders = [task_order_from_train_manifest(path, benchmarks, seed, 2) for seed in (42, 43, 44)]
    assert all(len(order) == 6 for order in orders)
    assert all([slug for slug, _ in order[::2]] == ["bfcl", "appworld", "swebench"] for order in orders)
    assert all({item for item in order} == {item for item in orders[0]} for order in orders)
    assert all(not task_id.endswith("-val") for order in orders for _, task_id in order)

    for seed, sequential in zip((42, 43, 44), orders):
        interleaved = task_order_from_train_manifest(path, benchmarks, seed, 2, "interleaved")
        assert len(interleaved) == 6
        assert set(interleaved) == set(sequential)
        for slug in ("bfcl", "appworld", "swebench"):
            assert [task for bm, task in interleaved if bm == slug] == [
                task for bm, task in sequential if bm == slug
            ]


def test_manifest_allows_five_task_subset_without_using_validation(tmp_path):
    manifest = {
        "train_count_per_benchmark": 8,
        "benchmarks": {
            "terminalbench2": {
                "train": [f"train-{i}" for i in range(8)],
                "val": ["val-0", "val-1"],
            },
        },
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    order = task_order_from_train_manifest(path, ["terminalbench2"], seed=42, count=5)
    assert len(order) == 5
    assert len(set(order)) == 5
    assert {task for _, task in order} <= set(manifest["benchmarks"]["terminalbench2"]["train"])


def test_terminalbench2_pilot_tasks_come_from_frozen_train_split():
    root = Path(__file__).resolve().parents[2] / "scripts" / "utils" / "task_splits"
    full = json.loads((root / "seed42_train50_val39_terminalbench2.json").read_text(encoding="utf-8"))
    pilot = json.loads((root / "seed42_terminalbench2_pilot5.json").read_text(encoding="utf-8"))
    full_split = full["benchmarks"]["terminalbench2"]
    pilot_split = pilot["benchmarks"]["terminalbench2"]
    assert len(pilot_split["train"]) == 5
    assert len(set(pilot_split["train"])) == 5
    assert set(pilot_split["train"]) <= set(full_split["train"])
    assert pilot_split["val"] == full_split["val"]
