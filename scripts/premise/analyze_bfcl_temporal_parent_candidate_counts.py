# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Exact candidate-subset expectations for LLM-free temporal parent preservation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.analyze_bfcl_temporal_parent_preservation import (  # noqa: E402
    CASES,
    MODELS,
    WEIGHTS,
    allocation,
    selected_indices,
    selection_scores,
    task_weights,
    temporal_prefixes,
)
from scripts.premise.analyze_bfcl_temporal_selection import write_csv  # noqa: E402
from scripts.premise.bfcl_trajectory_selection_rewards import ordering_ranks  # noqa: E402


def _choose(n: int, k: int) -> int:
    return math.comb(n, k) if 0 <= k <= n else 0


def selection_lookup() -> np.ndarray:
    """P(candidate selected | lower-ranked count, tie count, subset size).

    All size-lambda subsets of five proposals are equally probable. Uniform tie
    breaking occurs only within the sampled subset's maximal rank. This computes
    the exact subset expectation, without sampling subsets or scoring them again.
    """
    lookup = np.full((5, 5, 5), np.nan)
    for lower in range(5):
        for tied in range(1, 6 - lower):
            for count in range(1, 6):
                numerator = sum(_choose(tied - 1, others) * _choose(lower, count - 1 - others) / (others + 1)
                                for others in range(tied))
                lookup[lower, tied - 1, count - 1] = numerator / _choose(5, count)
    return lookup


def candidate_count_probabilities(ranks: np.ndarray, lookup: np.ndarray | None = None) -> np.ndarray:
    """Map average ranks (draws, 5) to exact probabilities (draws, 5, lambda=1..5)."""
    ranks = np.asarray(ranks, dtype=float)
    if ranks.ndim != 2 or ranks.shape[1] != 5 or not np.isfinite(ranks).all():
        raise ValueError("Expected finite average ranks for exactly five candidates")
    tied = (ranks[:, :, None] == ranks[:, None, :]).sum(axis=-1)
    raw_lower = ranks - (tied + 1) / 2
    lower = np.rint(raw_lower).astype(int)
    if (not np.allclose(raw_lower, lower, rtol=0, atol=1e-12)
            or np.any(lower < 0) or np.any(lower + tied > 5)):
        raise ValueError("Ranks must be valid ascending average ranks 1..5")
    # Check consistency between the rank-derived lower count and actual ordering.
    actual_lower = (ranks[:, :, None] > ranks[:, None, :]).sum(axis=-1)
    if not np.array_equal(lower, actual_lower):
        raise ValueError("Inconsistent average rank partition")
    values = selection_lookup() if lookup is None else lookup
    probabilities = values[lower, tied - 1]
    if not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("Candidate probabilities must sum to one for each subset size")
    return probabilities


def calculate_model(model: dict, keys: list[str], trials: int, seed: int) -> tuple[list[dict], list[dict]]:
    rows, audit = [], []
    lookup = selection_lookup()
    points = [point for point in model["points"] if point["key"] in keys]
    for number, point in enumerate(points, 1):
        if number == 1 or number % 10 == 0:
            print(f"{model['label']} candidate counts: {number}/{len(points)} ({point['key']})", flush=True)
        success, turns = np.asarray(point["success"], float), np.asarray(point["turns"], float)
        if success.shape != (6, 50) or turns.shape != success.shape:
            raise ValueError("Expected parent plus five candidates on all fifty chronological training tasks")
        prefixes = temporal_prefixes(point, trials, seed)
        memo = {}
        for n in range(1, 11):
            for case in CASES:
                parts = allocation(case, n)
                if parts is None:
                    continue
                indices = selected_indices(prefixes, parts)
                query_digest = hashlib.sha256(indices.astype(np.uint8).tobytes()).hexdigest()
                for weight in WEIGHTS:
                    weights = task_weights(parts, weight)
                    memo_key = (query_digest, tuple(np.round(weights, 15)))
                    if memo_key not in memo:
                        sampled_success = success[:, indices].transpose(1, 0, 2)
                        sampled_turns = turns[:, indices].transpose(1, 0, 2)
                        scores = selection_scores(sampled_success, sampled_turns, weights)["parent_preservation"]
                        probabilities = candidate_count_probabilities(ordering_ranks(scores), lookup)
                        memo[memo_key] = probabilities.mean(axis=0)
                    means = memo[memo_key]
                    # Target scores become available only after all subset choices are computed.
                    val = np.asarray(point["targets"]["full_validation"], float)
                    remaining = np.asarray(point["targets"]["validation_remaining40"], float)
                    lifts = means.T @ val - val.mean()
                    remaining_lifts = means.T @ remaining - remaining.mean()
                    if not np.allclose(means[:, 0], .2, rtol=0, atol=1e-12):
                        raise ValueError("One uniformly chosen proposal must equal the proposal mean")
                    lifts[0], remaining_lifts[0] = 0, 0
                    for count in range(1, 6):
                        rows.append({"model": model["label"], "key": point["key"], "seed": point["seed"],
                                     "checkpoint": point["checkpoint"], "n": n, "actual_n": sum(parts),
                                     "case": case, "weight": weight, "candidate_count": count,
                                     "method": "parent_preservation", "trials": trials,
                                     "choice_probabilities": means[:, count - 1].tolist(),
                                     "validation_lift": float(lifts[count - 1]),
                                     "validation_remaining40_lift": float(remaining_lifts[count - 1])})
                    audit.append({"model": model["label"], "key": point["key"], "case": case,
                                  "weight": weight, "n": n, "actual_n": sum(parts),
                                  "queries_sha256": query_digest, "counts_current_past_future": list(parts)})
    return rows, audit


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["case"], row["weight"], row["n"], row["candidate_count"])].append(row)
    output = []
    for (model, case, weight, n, count), points in grouped.items():
        row = {"model": model, "case": case, "weight": weight, "n": n,
               "candidate_count": count, "actual_n": sum(allocation(case, n)),
               "method": "parent_preservation", "scope": "all", "checkpoints": len(points)}
        for metric in ("validation_lift", "validation_remaining40_lift"):
            values = np.asarray([point[metric] for point in points])
            std = float(values.std(ddof=1)) if len(values) > 1 else None
            row[metric] = {"mean": float(values.mean()), "std": std,
                           "checkpoint_std": std,
                           "checkpoint_sem": std / math.sqrt(len(values)) if std is not None else None,
                           "uncertainty_kind": "std across checkpoint expected lifts, not Monte Carlo draw std"}
        output.append(row)
    return output


def compare_main(summary: dict, main_path: Path, *, tolerance: float = 1e-12) -> dict:
    if not main_path.exists():
        return {"status": "main_not_yet_available"}
    main = json.loads(main_path.read_text())
    if (main["trials"], main["sampling_seed"], set(main["fixed_checkpoint_keys"])) != (
            summary["trials"], summary["sampling_seed"], set(summary["fixed_checkpoint_keys"])):
        raise ValueError("Main comparison must use exactly the same cohort, draws, and seed")
    indexed = {(row["model"], row["case"], row["weight"], row["n"]): row
               for row in main["aggregate"] if row["method"] == "parent_preservation" and row["scope"] == "all"}
    differences = []
    for row in summary["aggregate"]:
        if row["candidate_count"] != 5:
            continue
        other = indexed[row["model"], row["case"], row["weight"], row["n"]]
        for metric in ("validation_lift", "validation_remaining40_lift"):
            difference = abs(row[metric]["mean"] - other[metric]["mean"])
            differences.append(difference)
            if difference > tolerance:
                raise ValueError(f"All-five result does not match main curve: {difference}")
    return {"status": "matched", "metrics_compared": len(differences), "max_abs_difference": max(differences),
            "tolerance": tolerance, "path": str(main_path.resolve()),
            "sha256": hashlib.sha256(main_path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Existing temporal parent preservation directory")
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    source = args.output / "full_train_metrics.json"
    raw = source.read_bytes()
    cache = json.loads(raw)
    models = [model for model in cache["models"] if model["label"] in MODELS]
    if {model["label"] for model in models} != set(MODELS):
        raise ValueError("Need Qwen and Gemma archived caches")
    keys = [point["key"] for point in models[0]["points"]
            if point["checkpoint"] >= 10 and 49 - point["checkpoint"] >= 10]
    if len(keys) != 36:
        raise ValueError("Candidate surfaces must share the fixed thirty-six-checkpoint cohort")
    rows, audit = [], []
    for model in models:
        points, queries = calculate_model(model, keys, args.trials, args.seed)
        rows.extend(points)
        audit.extend(queries)
    summary = {"created_at": datetime.now(UTC).isoformat(), "models": list(MODELS),
               "cases": list(CASES), "weights": list(WEIGHTS), "method": "parent_preservation",
               "n_values": list(range(1, 11)), "candidate_counts": list(range(1, 6)),
               "trials": args.trials, "sampling_seed": args.seed, "fixed_checkpoint_keys": keys,
               "checkpoints_per_model": len(keys), "new_model_calls": 0, "llm_judgments_used": False,
               "source": str(source.resolve()), "source_sha256": hashlib.sha256(raw).hexdigest(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "selection": "Same temporal queries and lexicographic parent preservation as main analysis.",
               "candidate_subsets": "Exact average over all uniformly chosen lambda-subsets of the five archived "
                                    "proposals, including uniform maximal-score tie breaking within each subset.",
               "probability_formula": "P_i(lambda)=sum_j C(k-1,j) C(l,lambda-1-j)/(j+1)/C(5,lambda), "
                                      "where l is the lower-ranked count and k is the including-self tie count.",
               "lift": "Selected validation score minus the mean validation score of all five proposals. "
                       "lambda=1 is zero exactly. No parent validation score is used.",
               "uncertainty": "Only expected choice probabilities are kept per checkpoint. Reported std is "
                              "across checkpoint expected lifts, not task-sampling Monte Carlo standard deviation.",
               "current_only": "The same single current task is repeated across nominal n=1..10.",
               "future_condition": "Future-containing task sampling is retrospective and unavailable online.",
               "aggregate": aggregate_rows(rows), "checkpoint_rows": rows, "query_audit": audit}
    summary["all_five_main_curve_check"] = compare_main(summary, args.output / "summary.json")
    target = args.output / "candidate_count_surface_summary.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    flattened = [{**{key: value for key, value in row.items() if not isinstance(value, list)},
                  **{f"candidate_{index + 1}_probability": probability
                     for index, probability in enumerate(row["choice_probabilities"])}}
                 for row in rows]
    write_csv(args.output / "candidate_count_checkpoint_metrics.csv", flattened)
    print(f"Saved {len(rows)} checkpoint rows and {len(summary['aggregate'])} aggregate rows", flush=True)


if __name__ == "__main__":
    main()
