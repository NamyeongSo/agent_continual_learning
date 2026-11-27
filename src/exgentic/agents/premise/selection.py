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


def select_official_score_beam(candidates: list[dict[str, Any]], beam_size: int = 2):
    """Remove alignment while retaining PREMiSE's official-score tie breakers."""
    return _select_score_beam(candidates, beam_size, "official_score", "official_current_past_without_alignment")


def select_policy_shift_beam(candidates: list[dict[str, Any]], beam_size: int = 2):
    """Retain the current/past Pareto fronts and tie breakers for inference-only scores."""
    return _select_score_beam(candidates, beam_size, "evaluation_score", "span_policy_shift_current_past")


def _select_score_beam(candidates, beam_size, score_key, criterion):
    valid = [item for item in candidates if item.get("valid", False)]
    if len(valid) < beam_size:
        raise RuntimeError(f"Need {beam_size} valid candidates, found {len(valid)}")
    use_past = any(item.get(f"past_{score_key}") is not None for item in valid)
    scores = {item["id"]: ((float(item[f"current_{score_key}"]),
               float(item[f"past_{score_key}"]) if item.get(f"past_{score_key}") is not None else -math.inf)
              if use_past else (float(item[f"current_{score_key}"]),)) for item in valid}
    fronts = pareto_fronts(valid, scores)
    ordered = []
    for front in fronts:
        ordered.extend(sorted(front, key=lambda candidate: (
            -sum(scores[candidate["id"]]), -min(scores[candidate["id"]]),
            int(candidate["id"].rsplit("_", 1)[-1]), candidate["id"],
        )))
    return ordered[:beam_size], {
        "criterion": criterion, "alignment_scoring": False,
        "objectives": {item["id"]: {score_key: scores[item["id"]]} for item in valid},
        "performance_fronts": [[item["id"] for item in front] for front in fronts],
        "ordered": [item["id"] for item in ordered],
        "selected": [item["id"] for item in ordered[:beam_size]],
    }


def select_beam(
    candidates: list[dict[str, Any]], beam_size: int = 2, *,
    alignment_scoring: bool = True, score_key: str = "official_score",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exact alignment-then-official-score Pareto layering used by premise."""
    if not alignment_scoring:
        if score_key != "official_score":
            return _select_score_beam(candidates, beam_size, score_key, "candidate_current_past_without_alignment")
        return select_official_score_beam(candidates, beam_size)
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
    official_scores = {item["id"]: vector(item, score_key) for item in valid}
    alignment_layers = pareto_fronts(valid, alignment)
    ordered = []
    performance_layers = []
    for rank, layer in enumerate(alignment_layers, 1):
        sublayers = pareto_fronts(layer, official_scores)
        performance_layers.append({"alignment_rank": rank, "fronts": [[c["id"] for c in front] for front in sublayers]})
        for front in sublayers:
            front.sort(key=lambda c: (
                -sum(official_scores[c["id"]]), -min(official_scores[c["id"]]),
                -sum(alignment[c["id"]]), -min(alignment[c["id"]]),
                int(c["id"].rsplit("_", 1)[-1]), c["id"],
            ))
            ordered.extend(front)
    return ordered[:beam_size], {
        "benchmark_score_key": score_key,
        "objectives": {
            c["id"]: {"alignment": alignment[c["id"]], score_key: official_scores[c["id"]]}
            for c in valid
        },
        "alignment_fronts": [[c["id"] for c in layer] for layer in alignment_layers],
        "performance_fronts": performance_layers,
        "ordered": [c["id"] for c in ordered],
        "selected": [c["id"] for c in ordered[:beam_size]],
    }


def select_alignment_product(candidates: list[dict[str, Any]], *, score_key: str = "official_score"):
    """Choose one child by mean alignment times mean chosen benchmark reward.

    Current and each sampled past replay have equal weight. The initial step
    uses only the current replay. A stable child ID breaks exact product ties.
    """
    valid = [item for item in candidates if item.get("valid", False)]
    if not valid:
        raise RuntimeError("Need at least one valid child for alignment_product")
    objectives = {}
    for child in valid:
        results = [child["current_result"], *child.get("past_results", [])]
        alignment = [float(result["alignment"]["score"]) for result in results]
        official = [float(result[score_key]) for result in results]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in alignment + official):
            raise ValueError("Product selection requires finite alignment and official scores in [0, 1]")
        a, b = sum(alignment) / len(results), sum(official) / len(results)
        objectives[child["id"]] = {"alignment": a, "benchmark_score": b, "product": a * b,
                                    "evaluation_count": len(results), "score_key": score_key,
                                    "benchmark_score_metrics": sorted({
                                        result.get("candidate_score_metric", "official_score")
                                        if score_key == "candidate_score" else "official_score"
                                        for result in results})}
        child["selection_objective"] = objectives[child["id"]]
    ordered = sorted(valid, key=lambda child: (-objectives[child["id"]]["product"], child["id"]))
    return ordered[:1], {
        "criterion": "alignment_product", "parent_in_selection_pool": False,
        "benchmark_score_key": score_key,
        "aggregation": "mean_over_current_and_each_past_replay_then_product",
        "tie_break": "child_id_ascending", "objectives": objectives,
        "ordered": [item["id"] for item in ordered], "selected": [ordered[0]["id"]],
    }


def select_ps_st_pareto_beam(
    candidates: list[dict[str, Any]], beam_size: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select parents from official current/past score Pareto fronts only."""
    valid = [item for item in candidates if item.get("valid", False)]
    if len(valid) < beam_size:
        raise RuntimeError(f"Need {beam_size} valid candidates, found {len(valid)}")
    use_past = any(item.get("past_official_score") is not None for item in valid)
    scores = {
        item["id"]: (
            (float(item["current_official_score"]), float(item["past_official_score"]))
            if use_past else (float(item["current_official_score"]),)
        )
        for item in valid
    }
    fronts = pareto_fronts(valid, scores)
    ordered = [
        item
        for front in fronts
        for item in sorted(front, key=lambda candidate: (
            -sum(scores[candidate["id"]]) / len(scores[candidate["id"]]), candidate["id"],
        ))
    ]
    return ordered[:beam_size], {
        "objectives": {item["id"]: {
            "current": scores[item["id"]][0],
            "past": scores[item["id"]][1] if use_past else None,
        } for item in valid},
        "performance_fronts": [[item["id"] for item in front] for front in fronts],
        "ordered": [item["id"] for item in ordered],
        "selected": [item["id"] for item in ordered[:beam_size]],
    }


def select_validation_beam(
    candidates: list[dict[str, Any]], beam_size: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rank candidates by mean official score on their shared held-out tasks."""
    valid = [item for item in candidates if item.get("valid", False)]
    if len(valid) < beam_size:
        raise RuntimeError(f"Need {beam_size} valid candidates, found {len(valid)}")
    ordered = sorted(valid, key=lambda item: (-float(item["validation_mean_official_score"]), item["id"]))
    return ordered[:beam_size], {
        "criterion": "mean_official_score_on_four_shared_validation_tasks",
        "scores": {item["id"]: item["validation_mean_official_score"] for item in valid},
        "ordered": [item["id"] for item in ordered],
        "selected": [item["id"] for item in ordered[:beam_size]],
    }
