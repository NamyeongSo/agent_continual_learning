# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "utils"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "ace"))

import task_ordering  # noqa: E402
from task_ordering import split_task_ids  # noqa: E402
from run_experiment import task_order_from_train_manifest  # noqa: E402
from exgentic.agents.premise.pipeline import manifest_order  # noqa: E402
from exgentic.utils.task_splits import validate_task_split  # noqa: E402


def test_split_is_deterministic_disjoint_and_order_independent():
    ids = [f"task-{i:03d}" for i in range(150)]
    first = split_task_ids(ids, benchmark_slug="bfcl", seed=42)
    assert first == split_task_ids(list(reversed(ids)), benchmark_slug="bfcl", seed=42)
    assert first == split_task_ids(ids, benchmark_slug="bfcl", seed=42)
    assert all(len(first[name]) == 50 for name in ("train", "val", "test"))
    assert len(set(first["train"] + first["val"] + first["test"])) == 150
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


def test_terminalbench2_split_covers_all_89_tasks_without_overlap():
    ids = [f"task-{i:03d}" for i in range(89)]
    split = split_task_ids(
        ids, benchmark_slug="terminalbench2", seed=42, train_count=30, val_count=30, test_count=29,
    )
    assert [len(split[name]) for name in ("train", "val", "test")] == [30, 30, 29]
    assert set(split["train"] + split["val"] + split["test"]) == set(ids)
    assert split == split_task_ids(
        list(reversed(ids)), benchmark_slug="terminalbench2", seed=42,
        train_count=30, val_count=30, test_count=29,
    )
    with pytest.raises(ValueError, match="need 89 tasks"):
        split_task_ids(
            ids[:-1], benchmark_slug="terminalbench2", seed=42,
            train_count=30, val_count=30, test_count=29,
        )


def test_nine_benchmark_seed_keys_have_frozen_disjoint_partitions():
    root = Path(__file__).resolve().parents[2] / "scripts" / "utils" / "task_splits"
    seeded = json.loads((root / "bfcl_appworld_terminalbench2_seed42_44.json").read_text())
    canonical = json.loads((root / "seed42_train_val_test.json").read_text())["benchmarks"]
    slugs = ("bfcl", "appworld", "terminalbench2")
    assert set(seeded) == {f"{slug}_{seed}" for slug in slugs for seed in (42, 43, 44)}
    for slug in slugs:
        expected = [30, 30, 29] if slug == "terminalbench2" else [50, 50, 50]
        for seed in (42, 43, 44):
            split = seeded[f"{slug}_{seed}"]
            assert set(split) == {"train", "val", "test"}
            validate_task_split(split, slug)
            assert [len(split[name]) for name in ("train", "val", "test")] == expected
        assert seeded[f"{slug}_42"] == {
            name: canonical[slug][name] for name in ("train", "val", "test")
        }
        train_sets = {frozenset(seeded[f"{slug}_{seed}"]["train"]) for seed in (42, 43, 44)}
        assert len(train_sets) == 3
    tb_pools = [
        set(sum(seeded[f"terminalbench2_{seed}"].values(), [])) for seed in (42, 43, 44)
    ]
    assert len(tb_pools[0]) == 89
    assert tb_pools[0] == tb_pools[1] == tb_pools[2]


def test_seed42_runtime_preserves_legacy_large_benchmark_train_and_val():
    root = Path(__file__).resolve().parents[2] / "scripts" / "utils" / "task_splits"
    old = json.loads((root / "seed42_train50_val50.json").read_text())["benchmarks"]
    new = json.loads((root / "seed42_train_val_test.json").read_text())["benchmarks"]
    for slug in ("bfcl", "appworld", "swebench"):
        assert new[slug]["train"] == old[slug]["train"]
        assert new[slug]["val"] == old[slug]["val"]
        validate_task_split(new[slug], slug)
    pilot = json.loads((root / "seed42_terminalbench2_train30_pilot5.json").read_text())["benchmarks"]
    assert len(pilot["terminalbench2"]["train"]) == 5
    assert set(pilot["terminalbench2"]["train"]) <= set(new["terminalbench2"]["train"])
    for name in ("val", "test"):
        assert pilot["terminalbench2"][name] == new["terminalbench2"][name]


@pytest.mark.parametrize("reader", [task_order_from_train_manifest, manifest_order])
@pytest.mark.parametrize("mode", ["sequential", "interleaved"])
def test_manifest_readers_use_different_train_sizes_without_held_out_tasks(tmp_path, reader, mode):
    sections = {
        slug: {
            "train": [f"{slug}-{i}" for i in range(size)],
            "val": [f"{slug}-val"], "test": [f"{slug}-test"],
            "split_counts": {"train": size, "val": 1, "test": 1},
        }
        for slug, size in (("bfcl", 50), ("terminalbench2", 30))
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"benchmarks": sections}))
    result = reader(path, list(sections), 43, None, mode)
    order = result[0] if reader is manifest_order else result
    assert len(order) == 80
    assert set(order) == {(slug, task) for slug, section in sections.items() for task in section["train"]}


@pytest.mark.parametrize("reader", [task_order_from_train_manifest, manifest_order])
@pytest.mark.parametrize("first,second", [("train", "val"), ("train", "test"), ("val", "test")])
def test_manifest_readers_reject_overlap_in_any_partition(tmp_path, reader, first, second):
    section = {name: [name] for name in ("train", "val", "test")}
    section[second] = section[first]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"benchmarks": {"bfcl": section}}))
    with pytest.raises(ValueError, match="overlapping"):
        reader(path, ["bfcl"], 42, 1, "sequential")


def test_split_validator_rejects_duplicate_test_ids_and_wrong_counts():
    with pytest.raises(ValueError, match="duplicate test"):
        validate_task_split({"train": ["train"], "val": ["val"], "test": ["test", "test"]}, "bfcl")
    with pytest.raises(ValueError, match="split_counts"):
        validate_task_split({"train": ["train"], "split_counts": {"train": 50}}, "bfcl")


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_ace_reads_matching_seed_keys_and_preserves_order_between_modes(seed):
    path = Path(__file__).resolve().parents[2] / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json"
    splits = json.loads(path.read_text())
    slugs = ["bfcl", "appworld", "terminalbench2"]
    sequential = task_order_from_train_manifest(path, slugs, seed, None, "sequential")
    interleaved = task_order_from_train_manifest(path, slugs, seed, None, "interleaved")
    assert len(sequential) == len(interleaved) == 130
    assert [slug for slug, _ in sequential] == ["bfcl"] * 50 + ["appworld"] * 50 + ["terminalbench2"] * 30
    assert set(interleaved) == set(sequential)
    for slug in slugs:
        ordered = [task for name, task in sequential if name == slug]
        assert set(ordered) == set(splits[f"{slug}_{seed}"]["train"])
        assert [task for name, task in interleaved if name == slug] == ordered


def test_ace_rejects_missing_seed_in_flat_split_file():
    path = Path(__file__).resolve().parents[2] / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json"
    with pytest.raises(ValueError, match="Missing task split key: bfcl_45"):
        task_order_from_train_manifest(path, ["bfcl"], 45, None)
