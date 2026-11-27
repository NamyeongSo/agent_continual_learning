# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Validate frozen task partitions before using them for learning or evaluation."""

from __future__ import annotations

from typing import Any


def select_task_split_sections(
    manifest: dict[str, Any], benchmark_slugs: list[str], seed: int
) -> dict[str, dict[str, Any]]:
    """Select a runtime manifest or the matching benchmark_seed entries."""
    if "benchmarks" in manifest:
        return {slug: manifest["benchmarks"][slug] for slug in benchmark_slugs}
    sections = {}
    for slug in benchmark_slugs:
        key = f"{slug}_{seed}"
        if key not in manifest:
            raise ValueError(f"Missing task split key: {key}")
        sections[slug] = manifest[key]
    return sections


def validate_task_split(section: dict[str, Any], benchmark_slug: str) -> dict[str, list[str]]:
    """Accept legacy train/val manifests and check all available partitions."""
    partitions = {}
    seen: set[str] = set()
    counts = section.get("split_counts", {})
    for name in ("train", "val", "test"):
        raw = section.get(name, [])
        if not isinstance(raw, list):
            raise ValueError(f"{benchmark_slug}: {name} must be a task-ID list")
        ids = [str(item) for item in raw]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{benchmark_slug}: duplicate {name} task IDs")
        if seen & set(ids):
            raise ValueError(f"{benchmark_slug}: overlapping train/val/test task IDs")
        if name in counts and len(ids) != counts[name]:
            raise ValueError(f"{benchmark_slug}: {name} length differs from split_counts")
        partitions[name] = ids
        seen.update(ids)
    if not partitions["train"]:
        raise ValueError(f"{benchmark_slug}: train must contain at least one task")
    return partitions
