# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Replay archived success-mean choices on past+current and future training tasks.

No inference is performed. The seven samplers, both weightings, checkpoint cohort,
and exact Monte Carlo query prefixes come from the temporal-preservation figure.
Parent-relative gains and proposal-relative selection lifts are distinct metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.analyze_bfcl_alignment_significance import (  # noqa: E402
    holm_adjust,
    paired_block_contributions,
    rng_for,
    sign_flip_test,
)
from scripts.premise.analyze_bfcl_temporal_parent_preservation import (  # noqa: E402
    CASES,
    MODELS,
    WEIGHTS,
    allocation,
    selected_indices,
    task_weights,
)
from scripts.premise.analyze_bfcl_temporal_selection import correlations, describe, write_csv  # noqa: E402
from scripts.premise.bfcl_trajectory_selection_rewards import choice_probabilities  # noqa: E402

TARGETS = ("past_current", "future")
METRICS = (
    "selected_gain", "proposal_gain", "selection_lift", "selected_gain_unqueried",
    "proposal_gain_unqueried", "selection_lift_unqueried", "rho",
)


def target_task_indices(task_count: int, current: int) -> dict[str, np.ndarray]:
    """Include the proposal task in past+current, and exclude it from future."""
    if not isinstance(current, int) or not 0 <= current < task_count - 1:
        raise ValueError("Both past+current and future target sets must be nonempty")
    return {"past_current": np.arange(current + 1), "future": np.arange(current + 1, task_count)}


def success_selection_scores(sampled_success: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Only n queried candidate outcomes are visible to this selector."""
    sampled_success = np.asarray(sampled_success, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (sampled_success.ndim != 3 or sampled_success.shape[1] != 5
            or not 1 <= sampled_success.shape[2] <= 10
            or weights.shape != (sampled_success.shape[2],)):
        raise ValueError("Expected queried successes (draws, 5, n=1..10) and n weights")
    if not np.isin(sampled_success, (0, 1)).all():
        raise ValueError("Success observations must be binary")
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Query weights must be finite, nonnegative, and sum to one")
    return sampled_success @ weights


def target_metrics(
    success: np.ndarray, queries: np.ndarray, probabilities: np.ndarray, target: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate choices already made; also remove queried target tasks per draw.

    The proposal baseline on the unqueried set uses exactly the same per-draw
    remainder as the selected candidate, so gain = proposal gain + selection lift.
    """
    success = np.asarray(success, dtype=float)
    queries = np.asarray(queries)
    probabilities = np.asarray(probabilities, dtype=float)
    target = np.asarray(target)
    if success.ndim != 2 or success.shape[0] != 6 or not np.isin(success, (0, 1)).all():
        raise ValueError("Expected binary parent plus five-candidate task outcomes")
    if (queries.ndim != 2 or not np.issubdtype(queries.dtype, np.integer)
            or (queries < 0).any() or (queries >= success.shape[1]).any()
            or not np.all(np.diff(np.sort(queries, axis=1), axis=1) > 0)):
        raise ValueError("Each draw requires distinct, valid integer query indices")
    if (target.ndim != 1 or not len(target) or not np.issubdtype(target.dtype, np.integer)
            or len(np.unique(target)) != len(target) or (target < 0).any()
            or (target >= success.shape[1]).any()):
        raise ValueError("Expected a nonempty set of distinct valid target task indices")
    if (probabilities.shape != (len(queries), 5) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("Expected a probability distribution over five candidates per draw")
    delta = success[1:] - success[:1]
    target_sum = delta[:, target].sum(axis=1)
    gains = target_sum / len(target)
    overlap = np.isin(queries, target)
    remaining = len(target) - overlap.sum(axis=1)
    if (remaining <= 0).any():
        raise ValueError("Selection-unqueried targets must remain nonempty")
    queried_delta = delta[:, queries].transpose(1, 0, 2)
    unqueried = (target_sum - (queried_delta * overlap[:, None, :]).sum(axis=-1)) / remaining[:, None]
    selected = probabilities @ gains
    proposal = np.full(len(queries), gains.mean())
    selected_unqueried = (probabilities * unqueried).sum(axis=1)
    proposal_unqueried = unqueried.mean(axis=1)
    return {
        "selected_gain": selected, "proposal_gain": proposal, "selection_lift": selected - proposal,
        "selected_gain_unqueried": selected_unqueried,
        "proposal_gain_unqueried": proposal_unqueried,
        "selection_lift_unqueried": selected_unqueried - proposal_unqueried,
    }


def _mean(vector: np.ndarray) -> float | None:
    finite = vector[np.isfinite(vector)]
    return float(finite.mean()) if finite.size else None


def calculate_model(model: dict, old: dict, prefixes: dict) -> tuple[list, list, list, dict]:
    """Replay every archived selector, failing if any candidate choice changes."""
    keys = old["fixed_checkpoint_keys"]
    points = [point for point in model["points"] if point["key"] in keys]
    if len(points) != len(keys) or {p["key"] for p in points} != set(keys):
        raise ValueError("The exact original checkpoint cohort is required")
    archived = {(row["key"], row["case"], row["weight"], row["n"]): row
                for row in old["checkpoint_rows"]
                if row["model"] == model["label"] and row["method"] == "success_mean"}
    sums, counts, rows, reference = {}, {}, [], []
    max_choice_difference, choices_compared = 0., 0
    for number, point in enumerate(points, 1):
        print(f"{model['label']}: checkpoint {number}/{len(points)} ({point['key']})", flush=True)
        success = np.asarray(point["success"], dtype=float)
        targets = target_task_indices(success.shape[1], point["current_index"])
        query_prefix = {period: prefixes[f"{point['key']}__{period}"]
                        for period in ("current", "past", "future")}
        if any(len(p) != old["trials"] for p in query_prefix.values()):
            raise ValueError("Frozen prefix draw count differs from the archived figure")
        # The validation selector sees the same archived, fixed ten tasks as before.
        val_scores = np.asarray(point["targets"]["val10"], dtype=float)
        val_probability = choice_probabilities(val_scores[None])[0]
        for target, indices in targets.items():
            delta = (success[1:, indices] - success[:1, indices]).mean(axis=1)
            selected_gain, proposal_gain = float(val_probability @ delta), float(delta.mean())
            reference.append({
                "model": model["label"], "key": point["key"], "seed": point["seed"],
                "checkpoint": point["checkpoint"], "target": target, "n": 10,
                "selected_gain": selected_gain, "proposal_gain": proposal_gain,
                "selection_lift": selected_gain - proposal_gain,
                "selected_gain_unqueried": selected_gain, "proposal_gain_unqueried": proposal_gain,
                "selection_lift_unqueried": selected_gain - proposal_gain,
                "rho": _mean(correlations(val_scores[None], delta)),
            })
        for n in range(1, 11):
            for case in CASES:
                parts = allocation(case, n)
                if parts is None:
                    continue
                queries = selected_indices(query_prefix, parts)
                sampled = success[1:, queries].transpose(1, 0, 2)
                for weight in WEIGHTS:
                    scores = success_selection_scores(sampled, task_weights(parts, weight))
                    probability = choice_probabilities(scores)
                    mean_probability = probability.mean(axis=0)
                    previous = archived[point["key"], case, weight, n]
                    difference = float(np.abs(mean_probability - previous["choice_probabilities"]).max())
                    if difference > 1e-12:
                        raise ValueError(f"Archived candidate selection changed by {difference}")
                    max_choice_difference = max(max_choice_difference, difference)
                    choices_compared += 1
                    for target, indices in targets.items():
                        metrics = target_metrics(success, queries, probability, indices)
                        delta = (success[1:, indices] - success[:1, indices]).mean(axis=1)
                        metrics["rho"] = correlations(scores, delta)
                        row = {
                            "model": model["label"], "key": point["key"], "seed": point["seed"],
                            "checkpoint": point["checkpoint"], "n": n, "actual_n": sum(parts),
                            "case": case, "weight": weight, "target": target, "target_size": len(indices),
                            "queried_target_tasks": sum(parts[:2]) if target == "past_current" else parts[2],
                            "choice_probabilities": mean_probability.tolist(),
                            **{name: _mean(values) for name, values in metrics.items()},
                            "rho_defined_fraction": float(np.isfinite(metrics["rho"]).mean()),
                        }
                        rows.append(row)
                        key = (case, weight, n, target)
                        for name, values in metrics.items():
                            metric_key = (*key, name)
                            finite = np.isfinite(values)
                            if metric_key not in sums:
                                sums[metric_key] = np.zeros(old["trials"])
                                counts[metric_key] = np.zeros(old["trials"], dtype=np.int16)
                            sums[metric_key] += np.where(finite, values, 0)
                            counts[metric_key] += finite
    aggregate = aggregate_rows(rows)
    for row in aggregate:
        if row["scope"] != "all":
            continue
        key = (row["case"], row["weight"], row["n"], row["target"])
        for name in METRICS:
            sums_for_metric, denominator = sums[*key, name], counts[*key, name]
            means = np.full(old["trials"], np.nan)
            np.divide(sums_for_metric, denominator, out=means, where=denominator > 0)
            cp_mean = row[name]["mean"]
            # rho weights defined checkpoints within each draw, as the original protocol.
            row[name].update(describe(means), mean_of_checkpoint_means=cp_mean,
                             minimum_defined_checkpoints=int(denominator.min()))
    audit = {"model": model["label"], "selections_compared": choices_compared,
             "max_absolute_choice_probability_difference": max_choice_difference,
             "archived_choices_matched": True}
    return rows, aggregate, reference, audit


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        for scope in ("all", row["seed"]):
            grouped[(row["model"], row.get("case"), row.get("weight"),
                     row["n"], row["target"], scope)].append(row)
    output = []
    for (model, case, weight, n, target, scope), group in grouped.items():
        row = {"model": model, "case": case, "weight": weight, "n": n, "target": target,
               "scope": scope, "checkpoints": len(group)}
        for metric in METRICS:
            vector = np.asarray([r[metric] if r[metric] is not None else np.nan for r in group])
            row[metric] = {"mean": _mean(vector), "std": None,
                           "checkpoint_std": float(np.nanstd(vector, ddof=1))
                           if np.isfinite(vector).sum() > 1 else None,
                           "defined_checkpoints": int(np.isfinite(vector).sum())}
        output.append(row)
    return output


def statistical_tests(rows: list[dict], seed: int) -> list[dict]:
    """Exploratory n=10 current+past/task-mean tests; preserve seed boundaries."""
    output = []
    metrics = ("selected_gain", "selection_lift", "selected_gain_unqueried", "selection_lift_unqueried")
    for model in MODELS:
        for block in (2, 3, 4, 5, "whole_seed"):
            family = []
            for target in TARGETS:
                points = sorted([r for r in rows if r["model"] == model and r["case"] == "current_past"
                                 and r["weight"] == "task_mean" and r["n"] == 10 and r["target"] == target],
                                key=lambda r: (r["seed"], r["checkpoint"]))
                # Current+past never queries future; full and unqueried future
                # are the same hypothesis and must not be counted twice.
                tested_metrics = metrics if target == "past_current" else metrics[:2]
                for metric in tested_metrics:
                    differences = np.asarray([p[metric] for p in points])
                    if block == "whole_seed":
                        contributions = np.asarray([sum(v for p, v in zip(points, differences, strict=True)
                                                        if p["seed"] == s) / len(points) for s in (42, 43, 44)])
                    else:
                        contributions, _ = paired_block_contributions(points, points, differences, block)
                    result = sign_flip_test(contributions, 100000,
                                            rng_for(seed, f"split-gain:{model}:{target}:{metric}:{block}"))
                    effect = result.pop("mean_rho_difference")
                    family.append({"model": model, "target": target, "case": "current_past",
                                   "weight": "task_mean", "n": 10, "metric": metric, "effect": effect,
                                   "block_size": block, "checkpoints": len(points), **result})
            for row, adjusted in zip(family, holm_adjust([r["p_two_sided"] for r in family]), strict=True):
                row.update(p_holm=adjusted, family_size=len(family))
            output.extend(family)
    return output


def verify_recombination(old: dict, rows: list[dict]) -> dict:
    """The size-weighted target gains must recover every archived full-train gain."""
    indexed = {tuple(r[k] for k in ("model", "key", "case", "weight", "n", "target")): r for r in rows}
    differences = []
    for row in old["checkpoint_rows"]:
        if row["method"] != "success_mean" or row["model"] not in MODELS:
            continue
        key = tuple(row[k] for k in ("model", "key", "case", "weight", "n"))
        first, second = indexed[*key, "past_current"], indexed[*key, "future"]
        count = first["target_size"] + second["target_size"]
        reconstructed = sum(r["target_size"] * r["selected_gain"] for r in (first, second)) / count
        differences.append(abs(reconstructed - row["train_parent_gain"]))
    maximum = max(differences)
    if maximum > 1e-12:
        raise ValueError("Split target gains do not recover the archived full-training gain")
    return {"selections_compared": len(differences), "max_absolute_difference": maximum, "matched": True}


def _flatten(rows: list[dict]) -> list[dict]:
    return [{**{k: v for k, v in row.items() if not isinstance(v, (list, dict))},
             **{f"{k}_{field}": value for k, item in row.items() if isinstance(item, dict)
                for field, value in item.items()}} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="temporal_parent_preservation directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_paths = [args.source / name for name in ("summary.json", "full_train_metrics.json", "query_prefixes.npz")]
    old, cache = [json.loads(path.read_text()) for path in source_paths[:2]]
    if hashlib.sha256(source_paths[1].read_bytes()).hexdigest() != old["source_sha256"]:
        raise ValueError("Full-task outcomes differ from those used in the archived figure")
    if old["new_model_calls"] != 0 or old["llm_judgments_used"] is not False:
        raise ValueError("Expected the archived LLM-free experiment")
    models = [m for m in cache["models"] if m["label"] in MODELS]
    if {m["label"] for m in models} != set(MODELS):
        raise ValueError("Both Qwen and Gemma are required")
    rows, aggregate, reference, audits = [], [], [], []
    with np.load(source_paths[2], allow_pickle=False) as prefixes:
        for model in models:
            cp, agg, ref, audit = calculate_model(model, old, prefixes)
            rows.extend(cp)
            aggregate.extend(agg)
            reference.extend(ref)
            audits.append(audit)
    summary = {
        "created_at": datetime.now(UTC).isoformat(), "models": list(MODELS), "excluded_models": ["OSS"],
        "cases": list(CASES), "weights": list(WEIGHTS), "targets": list(TARGETS), "method": "success_mean",
        "trials": old["trials"], "sampling_seed": old["sampling_seed"],
        "fixed_checkpoint_keys": old["fixed_checkpoint_keys"],
        "checkpoints_per_model": len(old["fixed_checkpoint_keys"]),
        "source_files": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in source_paths],
        "new_model_calls": 0, "new_environment_calls": 0, "llm_judgments_used": False,
        "target_definition": {"past_current": "Chronological train_order[:current_index+1]",
                              "future": "Chronological train_order[current_index+1:]"},
        "gain_definition": "Mean candidate success minus mean parent success on the SAME target tasks.",
        "lift_definition": "Selected gain minus mean gain of all five proposals on the SAME target tasks.",
        "unqueried_definition": "Remove selected evaluation tasks from each target per draw before computing "
                                "parent-relative gains and proposal baselines. "
                                "Current task still influenced proposals.",
        "selection": "Exactly the original success_mean selector; uniform tie breaking; only queried n candidate "
                     "outcomes are used. No parent outcomes or unqueried scores enter selection.",
        "allocation": old["allocation"], "weighting": old["weighting"], "sampling": old["sampling"],
        "future_condition": old["future_condition"],
        "cohort": old["cohort"],
        "uncertainty": "All-scope std bands describe task-sampling variation of checkpoint-averaged metrics, "
                       "with ties integrated out; not seed confidence intervals. Seed rows use checkpoint means.",
        "rho": "Within-checkpoint Pearson correlation across five candidates; constants undefined, never zero-imputed. "
               "Per-draw means average defined checkpoints; coverage is saved separately.",
        "validation_reference": "Fixed archived val10, displayed only at n=10. Its selected candidate is evaluated "
                                "on the two training targets. Validation tasks and these targets are disjoint.",
        "statistics": "Exploratory current+past/task_mean n10: within-seed chronological block sign flips "
                      "(2/3/4/5 checkpoints), plus exact whole-seed flips. Two-sided, Holm over six distinct "
                      "gain/lift metrics per model/block size (future unqueried duplicates omitted). "
                      "Blocks need symmetry and independence; "
                      "only three independent training seeds, minimum whole-seed two-sided p=0.25.",
        "historical_limits": {"max_actor_steps": cache["historical_actor_max_steps"],
                              "original_infrastructure_errors_keep_score_zero": True,
                              "new_actor_trajectories": 0},
        "aggregate": aggregate, "checkpoint_rows": rows,
        "validation_reference_aggregate": aggregate_rows(reference),
        "validation_reference_checkpoint_rows": reference,
        "selection_replay_audit": audits,
        "full_training_recombination_audit": verify_recombination(old, rows),
        "tests_n10": statistical_tests(rows, old["sampling_seed"]),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_csv(args.output / "checkpoint_metrics.csv", _flatten(rows))
    write_csv(args.output / "aggregate_metrics.csv", _flatten(aggregate))
    write_csv(args.output / "validation_reference_metrics.csv", _flatten(summary["validation_reference_aggregate"]))
    write_csv(args.output / "tests_n10.csv", summary["tests_n10"])
    n10 = [r for r in aggregate if r["scope"] == "all" and r["n"] == 10]
    write_csv(args.output / "n10_metrics.csv", _flatten(n10))
    print(f"Saved {len(rows)} checkpoint metrics, {len(aggregate)} aggregates to {args.output}", flush=True)


if __name__ == "__main__":
    main()
