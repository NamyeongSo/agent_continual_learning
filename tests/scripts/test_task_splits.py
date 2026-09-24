from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "utils"))

import task_ordering  # noqa: E402
from task_ordering import split_task_ids  # noqa: E402


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
