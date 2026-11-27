# SPDX-License-Identifier: Apache-2.0
"""Controlled two-survivor alternatives to PEACE's historical IFS-first rule."""

from __future__ import annotations

import math
import random
from typing import Any

from .selection import pareto_fronts, select_beam

ABLATION_SELECTION_RULES = ("pfs_first", "joint_pareto", "random")


def select_ablation_beam(
    candidates: list[dict[str, Any]], rule: str, *, score_key: str = "official_score",
    seed: int = 42, iteration: int = 0, beam_size: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    valid = [item for item in candidates if item.get("valid", False)]
    if len(valid) < beam_size:
        raise RuntimeError(f"Need {beam_size} valid candidates, found {len(valid)}")
    if rule == "random":
        # Selection draws never advance the shared proposal/history-sampling RNG.
        rng = random.Random(f"peace-random-selection-v1:{seed}:{iteration}")
        selected = rng.sample(valid, beam_size)
        return selected, {
            "criterion": rule, "selected": [item["id"] for item in selected],
            "eligible": [item["id"] for item in valid], "without_replacement": True,
            "selection_seed": seed, "iteration": iteration, "candidate_scoring": False,
        }
    if rule not in ABLATION_SELECTION_RULES:
        raise ValueError(f"Unknown ablation selection rule: {rule}")
    use_past = any(item.get("past_alignment") is not None for item in valid)
    objectives = {}
    for item in valid:
        axes = {}
        for name, field in [("ifs", "alignment"), ("pfs", score_key)]:
            values = [float(item[f"current_{field}"])]
            if use_past:
                values.append(float(item[f"past_{field}"]))
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                raise ValueError("Scored selection requires finite measured values in [0, 1]")
            axes[name] = tuple(values)
        objectives[item["id"]] = axes
    if rule == "pfs_first":
        swapped = []
        for item in valid:
            clone = dict(item)
            for scope in ["current", "past"]:
                clone[f"{scope}_alignment"] = item.get(f"{scope}_{score_key}")
                clone[f"{scope}_{score_key}"] = item.get(f"{scope}_alignment")
            swapped.append(clone)
        _, evidence = select_beam(swapped, beam_size, score_key=score_key)
        ids = evidence["selected"]
        by_id = {item["id"]: item for item in valid}
        return [by_id[key] for key in ids], {
            "criterion": rule, "objectives": objectives, "selected": ids,
            "ordered": evidence["ordered"], "performance_fronts": evidence["alignment_fronts"],
            "alignment_within_performance_fronts": evidence["performance_fronts"],
            "tie_break_order": ["ifs_sum", "ifs_min", "pfs_sum", "pfs_min", "generation_order"],
        }
    vectors = {key: value["ifs"] + value["pfs"] for key, value in objectives.items()}
    fronts = pareto_fronts(valid, vectors)
    ordered = []
    for front in fronts:
        ordered.extend(sorted(front, key=lambda item: (
            -sum(objectives[item["id"]]["pfs"]), -min(objectives[item["id"]]["pfs"]),
            -sum(objectives[item["id"]]["ifs"]), -min(objectives[item["id"]]["ifs"]),
            int(item["id"].rsplit("_", 1)[-1]), item["id"],
        )))
    return ordered[:beam_size], {
        "criterion": rule, "objectives": objectives, "selected": [item["id"] for item in ordered[:beam_size]],
        "ordered": [item["id"] for item in ordered],
        "joint_fronts": [[item["id"] for item in front] for front in fronts],
        "joint_axes": ["current_ifs", "past_ifs", "current_pfs", "past_pfs"] if use_past else [
            "current_ifs", "current_pfs"],
        "tie_break_order": ["pfs_sum", "pfs_min", "ifs_sum", "ifs_min", "generation_order"],
    }
