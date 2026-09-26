# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Deterministic benchmark-balanced replay sampling and Pareto beam ranking."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any


def sample_past(memory: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    """Balance across benchmarks, then sample uniformly without replacement."""
    if count < 0:
        raise ValueError("count must be nonnegative")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in memory:
        groups[str(item["benchmark"])].append(item)
    target = min(count, len(memory))
    quotas = {slug: 0 for slug in groups}
    # Draw whole rounds while possible. Random ordering resolves remainders and
    # depleted groups, including the t=3, two-benchmark 1/2 case.
    while sum(quotas.values()) < target:
        eligible = [slug for slug, items in groups.items() if quotas[slug] < len(items)]
        if not eligible:
            break
        rng.shuffle(eligible)
        for slug in eligible:
            if sum(quotas.values()) >= target:
                break
            quotas[slug] += 1
    selected: list[dict[str, Any]] = []
    for slug in sorted(groups):
        selected.extend(rng.sample(groups[slug], quotas[slug]))
    rng.shuffle(selected)
    return selected


def pareto_fronts(candidates: list[dict[str, Any]], values: dict[str, tuple[float, ...]]) -> list[list[dict[str, Any]]]:
    remaining = list(candidates)
    fronts: list[list[dict[str, Any]]] = []
    while remaining:
        front = []
        for candidate in remaining:
            own = values[candidate["id"]]
            if not any(
                other is not candidate
                and all(b + 1e-9 >= a for a, b in zip(own, values[other["id"]]))
                and any(b > a + 1e-9 for a, b in zip(own, values[other["id"]]))
                for other in remaining
            ):
                front.append(candidate)
        if not front:
            raise RuntimeError("Pareto sorting produced an empty front")
        fronts.append(front)
        ids = {id(item) for item in front}
        remaining = [item for item in remaining if id(item) not in ids]
    return fronts


def select_beam(candidates: list[dict[str, Any]], beam_size: int = 2) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exact alignment-then-PCT Pareto layering used by premise."""
    valid = [item for item in candidates if item.get("valid", False)]
    if len(valid) < beam_size:
        raise RuntimeError(f"Need {beam_size} valid candidates, found {len(valid)}")
    use_past = any(item.get("past_alignment") is not None for item in valid)

    def vector(item: dict[str, Any], kind: str) -> tuple[float, ...]:
        current = float(item[f"current_{kind}"])
        if not use_past:
            return (current,)
        past = item.get(f"past_{kind}")
        return (current, float(past) if past is not None else -math.inf)

    alignment = {item["id"]: vector(item, "alignment") for item in valid}
    pct = {item["id"]: vector(item, "pct") for item in valid}
    alignment_layers = pareto_fronts(valid, alignment)
    ordered = []
    performance_layers = []
    for rank, layer in enumerate(alignment_layers, 1):
        sublayers = pareto_fronts(layer, pct)
        performance_layers.append({"alignment_rank": rank, "fronts": [[c["id"] for c in front] for front in sublayers]})
        for front in sublayers:
            front.sort(key=lambda c: (
                -sum(pct[c["id"]]), -min(pct[c["id"]]),
                -sum(alignment[c["id"]]), -min(alignment[c["id"]]),
                int(c["id"].rsplit("_", 1)[-1]), c["id"],
            ))
            ordered.extend(front)
    return ordered[:beam_size], {
        "objectives": {c["id"]: {"alignment": alignment[c["id"]], "performance": pct[c["id"]]} for c in valid},
        "alignment_fronts": [[c["id"] for c in layer] for layer in alignment_layers],
        "performance_fronts": performance_layers,
        "ordered": [c["id"] for c in ordered],
        "selected": [c["id"] for c in ordered[:beam_size]],
    }
