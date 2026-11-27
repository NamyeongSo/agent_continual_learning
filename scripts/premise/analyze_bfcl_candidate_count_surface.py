# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Replay candidate counts 1..5 using archived n-task trajectory reward scores.

Each candidate count averages over every equally likely subset of the original
five candidates and breaks maximizing-score ties uniformly within that subset.
No reward is recomputed and no inference, environment execution, or new task
sampling occurs. Full-set targets are used only after selection probabilities
have been determined from archived reward scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.analyze_bfcl_temporal_selection import write_csv  # noqa: E402
from scripts.premise.analyze_bfcl_trajectory_rewards import MODELS, _describe, _flatten  # noqa: E402
from scripts.premise.bfcl_trajectory_selection_rewards import ordering_ranks  # noqa: E402

COHORTS = ("primary", "objective", "process_ablation")
METRICS = ("validation_lift", "validation_score", "train_proposal_lift", "train_parent_gain",
           "selection_total_variation_vs_uniform")


def subset_choice_probabilities(scores: np.ndarray, candidate_count: int) -> np.ndarray:
    """Return marginal probabilities over the five original candidate identities.

    Scalar scores have shape (5,), lexicographic scores (5, coordinates). The
    returned probability averages C(5, candidate_count) subsets exactly; this
    function deliberately accepts no validation or training target arguments.
    """
    if (not isinstance(candidate_count, int | np.integer) or isinstance(candidate_count, bool)
            or not 1 <= candidate_count <= 5):
        raise ValueError("candidate_count must be an integer from 1 to 5")
    scores = np.asarray(scores, float)
    if scores.ndim not in (1, 2) or scores.shape[0] != 5:
        raise ValueError("scores must contain five scalar or lexicographic candidate scores")
    ranks = ordering_ranks(scores[None])[0]
    probabilities = np.zeros(5, float)
    for subset in combinations(range(5), candidate_count):
        indices = np.asarray(subset)
        winners = indices[ranks[indices] == ranks[indices].max()]
        probabilities[winners] += 1 / (comb(5, candidate_count) * len(winners))
    if not np.isclose(probabilities.sum(), 1, atol=1e-12):
        raise AssertionError("Exact-subset selection probabilities must sum to one")
    return probabilities


def _target_vector(row: dict, name: str) -> np.ndarray:
    values = np.asarray(row[name], float)
    if values.shape != (5,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"{name} must contain five finite performance fractions")
    return values


def replay_rows(source_rows: list[dict]) -> list[dict]:
    """Expand archived checkpoint rows without allowing targets to select candidates."""
    result, seen, identity_targets = [], set(), {}
    for source in source_rows:
        model, cohort, n = (source[key] for key in ("model", "cohort", "n"))
        if model not in MODELS or cohort not in COHORTS or not isinstance(n, int) or not 1 <= n <= 10:
            raise ValueError("Unsupported model, cohort, or selection sample size")
        identity = (model, source["key"], cohort, source["recipe"], n)
        if identity in seen:
            raise ValueError(f"Duplicate archived checkpoint reward row: {identity}")
        seen.add(identity)
        score = np.asarray(source["candidate_scores"], float)
        ranks = ordering_ranks(score[None])[0]
        full_validation = _target_vector(source, "full_validation_scores")
        full_train = _target_vector(source, "full_train_scores")
        target_identity = (model, source["key"])
        stored = identity_targets.setdefault(target_identity, (full_validation, full_train))
        if not np.array_equal(stored[0], full_validation) or not np.array_equal(stored[1], full_train):
            raise ValueError("Full-set targets must agree across n, cohorts, and reward rules")
        parent_train = float(full_train.mean() - source["proposal_mean_train_gain"])
        if not np.isfinite(parent_train) or not -1e-12 <= parent_train <= 1 + 1e-12:
            raise ValueError("Archived parent training performance is invalid")
        for count in range(1, 6):
            probability = subset_choice_probabilities(score, count)
            validation_score = float(probability @ full_validation)
            validation_lift = float(validation_score - full_validation.mean())
            train_lift = float(probability @ full_train - full_train.mean())
            # Remove only arithmetic noise, never a measured small improvement.
            validation_lift = 0. if abs(validation_lift) < 1e-14 else validation_lift
            train_lift = 0. if abs(train_lift) < 1e-14 else train_lift
            if count == 5:
                if (not np.allclose(probability, source["choice_probabilities"], atol=1e-12)
                        or not np.isclose(validation_lift, source["validation_lift"], atol=1e-12)):
                    raise ValueError("Five-candidate replay differs from the archived full-pool selection")
            result.append({
                **{key: source[key] for key in ("model", "key", "seed", "checkpoint", "n", "cohort", "recipe")},
                "candidate_count": count, "score_type": source["score_type"],
                "validation_lift": validation_lift, "validation_score": validation_score,
                "train_proposal_lift": train_lift,
                "train_parent_gain": float(probability @ full_train - parent_train),
                "proposal_mean_validation_score": float(full_validation.mean()),
                "proposal_mean_train_gain": source["proposal_mean_train_gain"],
                "selection_total_variation_vs_uniform": float(np.abs(probability - .2).sum() / 2),
                "choice_probabilities": probability.tolist(), "candidate_scores": score.tolist(),
                "candidate_ranks": ranks.tolist(), "full_validation_scores": full_validation.tolist(),
                "full_train_scores": full_train.tolist(), "query_ids": source.get("query_ids", []),
                "subsets_enumerated": comb(5, count),
            })
    return result


def summarize(source_rows: list[dict], recipe_catalog: dict | None = None) -> dict:
    """Aggregate equal checkpoint weights while preserving each original cohort."""
    rows = replay_rows(source_rows)
    recipes = recipe_catalog or {
        cohort: sorted({row["recipe"] for row in source_rows if row["cohort"] == cohort}) for cohort in COHORTS
    }
    indexed, cohorts = defaultdict(list), defaultdict(dict)
    for row in rows:
        indexed[row["model"], row["cohort"], row["recipe"], row["n"], row["candidate_count"]].append(row)
        key = (row["model"], row["cohort"], row["n"], row["candidate_count"])
        cohorts[key].setdefault(row["recipe"], set()).add(row["key"])
    for sets in cohorts.values():
        if any(points != next(iter(sets.values())) for points in sets.values()):
            raise ValueError("All reward recipes require matched checkpoint cohorts at each n")
    aggregate, seed_rows = [], []
    for model in MODELS:
        for cohort, names in recipes.items():
            for recipe in names:
                for n in range(1, 11):
                    for count in range(1, 6):
                        selected = indexed[model, cohort, recipe, n, count]
                        for seed in ("all", 42, 43, 44):
                            subset = selected if seed == "all" else [row for row in selected if row["seed"] == seed]
                            if seed != "all" and not subset:
                                continue
                            record = {"model": model, "cohort": cohort, "recipe": recipe, "n": n,
                                      "candidate_count": count, "checkpoints": len(subset),
                                      **{name: _describe([row[name] for row in subset]) for name in METRICS}}
                            if seed == "all":
                                aggregate.append(record)
                            else:
                                seed_rows.append({**record, "seed": seed})
    coverage = [{"model": model, "cohort": cohort, "n": n,
                 "checkpoints": len(next(iter(sets.values()))) if sets else 0}
                for (model, cohort, n, count), sets in sorted(cohorts.items()) if count == 5]
    return {"checkpoint_scores": rows, "aggregate": aggregate, "seed_scores": seed_rows,
            "coverage": coverage, "recipes_by_cohort": recipes,
            "source_checkpoint_reward_rows": len(source_rows)}


def fixed_cohort_rows(source_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep checkpoints observed at all ten n values, separately by model/cohort."""
    indexed = defaultdict(set)
    for row in source_rows:
        indexed[row["model"], row["cohort"], row["n"]].add(row["key"])
    fixed, description = {}, []
    for model in MODELS:
        for cohort in COHORTS:
            sets = [indexed[model, cohort, n] for n in range(1, 11)]
            keep = set.intersection(*sets)
            fixed[model, cohort] = keep
            description.append({"model": model, "cohort": cohort, "checkpoints": len(keep),
                                "checkpoint_keys": sorted(keep)})
    return ([row for row in source_rows if row["key"] in fixed[row["model"], row["cohort"]]], description)


def write_report(summary: dict, output: Path) -> None:
    lines = ["# Candidate-count surfaces from archived trajectory reward scores", "",
             "The x/y coordinates vary selection tasks n=1..10 and candidate count lambda=1..5. "
             "Task IDs and all reward scores are reused from the original uniform training-pool experiment. "
             "This is not the requested current-plus-past experiment, which remains deferred until inference "
             "is available. No new task samples, rollouts, or LLM judgments were produced here.", "",
             "For each checkpoint and lambda, every C(5,lambda) subset of the original five candidates has "
             "equal probability. Selection maximizes the archived scalar or lexicographic reward within "
             "that subset, with uniform ties. Marginal candidate probabilities are averaged exactly; full "
             "validation scores are consulted only afterward. Lambda=1 gives uniform proposal selection "
             "and zero validation lift; lambda=5 exactly reproduces the archived result.", "",
             "Validation lift is selected full-validation performance minus the mean of the original five "
             "proposals, in fractions (multiply by 100 for pp). The expected subset proposal mean equals "
             "this five-proposal mean. No parent validation baseline exists. SD measures heterogeneity "
             "between checkpoints, not uncertainty from subset Monte Carlo: all subsets were enumerated.", "",
             "Primary semantic, objective-only, and process-ablation cohorts remain separate. Their "
             "checkpoint coverage can differ, and semantic coverage changes across n because judging is "
             "incomplete. Within each cohort and n, reward rules and candidate counts use matched "
             "checkpoint identities. A companion fixed-cohort summary restricts every n to checkpoint keys "
             "present at all ten n values, separately by model/cohort. Use that companion to compare the "
             "n axis without changing checkpoint composition. Missing judgments have not been assigned zero. "
             "The surfaces describe "
             "subsets of an existing five-candidate pool; they are not independent new proposal runs at "
             "each lambda and cannot establish the bivariate-normal proposition or causal sample complexity.", "",
             "## n=10, lambda=5 validation lift (pp)", "",
             "|Model|Cohort|Reward|CPs|Lift|", "|---|---|---|---:|---:|"]
    for row in summary["aggregate"]:
        if row["n"] == 10 and row["candidate_count"] == 5:
            lift = row["validation_lift"]["mean"]
            value = "pending" if lift is None else f"{100 * lift:+.3f}"
            lines.append(f"|{row['model']}|{row['cohort']}|{row['recipe']}|{row['checkpoints']}|{value}|")
    (output / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="Original trajectory_reward_summary.json")
    parser.add_argument("--process-summary", type=Path, help="Optional original process_ablation_summary.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sources, source_rows = [], []
    for path in (args.summary, args.process_summary):
        if path is None:
            continue
        data = path.read_bytes()
        decoded = json.loads(data)
        source_rows.extend(decoded["checkpoint_scores"])
        sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
                        "created_at": decoded.get("created_at"),
                        "checkpoint_reward_rows": len(decoded["checkpoint_scores"])})
    summary = summarize(source_rows)
    metadata = {"created_at": datetime.now(UTC).isoformat(), "sources": sources,
                    "models": list(MODELS), "excluded": ["OSS"], "new_actor_rollouts": 0,
                    "new_model_calls": 0, "candidate_counts": [1, 2, 3, 4, 5],
                    "task_sample_sizes": list(range(1, 11)),
                    "sampling": "Original uniform ten-task frozen prefixes; not current-plus-past",
                    "subset_policy": "All C(5,lambda) subsets equally likely; uniform maximizing-score ties",
                    "aggregation": "Equal checkpoint weights, original n-specific complete-case cohorts",
                    "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    summary.update({**metadata, "cohort_scope": "available_per_n", "fixed_cohort": False})
    output = args.output or args.summary.parent / "candidate_count_surfaces_20261022"
    output.mkdir(parents=True, exist_ok=True)
    (output / "candidate_count_surface_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name in ("aggregate", "checkpoint_scores", "seed_scores", "coverage"):
        if summary[name]:
            write_csv(output / f"{name}.csv", _flatten(summary[name]))
    write_report(summary, output)
    fixed_rows, fixed_description = fixed_cohort_rows(source_rows)
    fixed_summary = summarize(fixed_rows, summary["recipes_by_cohort"])
    fixed_summary.update({**metadata, "cohort_scope": "fixed_across_n", "fixed_cohort": True,
                          "fixed_cohort_checkpoints": fixed_description,
                          "aggregation": "Equal checkpoint weights; same checkpoint keys for n=1..10 "
                          "and lambda=1..5 within each model/cohort"})
    (output / "candidate_count_surface_fixed_cohort_summary.json").write_text(
        json.dumps(fixed_summary, indent=2, allow_nan=False) + "\n",
    )
    for name in ("aggregate", "checkpoint_scores", "seed_scores", "coverage"):
        if fixed_summary[name]:
            write_csv(output / f"fixed_cohort_{name}.csv", _flatten(fixed_summary[name]))
    print(json.dumps({"output": str(output.resolve()), "checkpoint_rows": len(summary["checkpoint_scores"]),
                      "aggregate_rows": len(summary["aggregate"]), "coverage": summary["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
