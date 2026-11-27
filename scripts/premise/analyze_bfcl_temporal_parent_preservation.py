# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Offline temporal sampling ablation for LLM-free parent-preserving selection.

The selection rule reads only the queried candidate/parent successes and successful
trajectory steps. Full-set scores are read after candidate choice. Future tasks
are a retrospective oracle condition, not observations available during training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.analyze_bfcl_alignment_significance import (  # noqa: E402
    holm_adjust,
    paired_block_contributions,
    rng_for,
    sign_flip_test,
)
from scripts.premise.analyze_bfcl_temporal_selection import correlations, describe, write_csv  # noqa: E402
from scripts.premise.bfcl_trajectory_selection_rewards import choice_probabilities, ordering_ranks  # noqa: E402

CASES = ("current", "past", "future", "current_future", "current_past", "past_future", "current_past_future")
WEIGHTS = ("task_mean", "period_mean")
METHODS = ("success_mean", "parent_preservation")
MODELS = ("Qwen", "Gemma")
TARGETS = ("full_train", "full_validation", "validation_remaining40")
METRICS = ("validation_lift", "validation_remaining40_lift", "train_parent_gain", "selected_regressions",
           "selected_repairs", "selected_relative_efficiency", "selection_tv_vs_success")


def allocation(case: str, n: int) -> tuple[int, int, int] | None:
    """Allocate (current, past, future); an odd remainder belongs to the past."""
    if case not in CASES or not isinstance(n, int) or not 1 <= n <= 10:
        raise ValueError("Expected a known case and integer n=1..10")
    if case == "current":
        return 1, 0, 0
    if case == "past":
        return 0, n, 0
    if case == "future":
        return 0, 0, n
    if case == "current_past" and n >= 2:
        return 1, n - 1, 0
    if case == "current_future" and n >= 2:
        return 1, 0, n - 1
    if case == "past_future" and n >= 2:
        return 0, (n + 1) // 2, n // 2
    if case == "current_past_future" and n >= 3:
        return 1, n // 2, (n - 1) // 2
    return None


def task_weights(counts: tuple[int, int, int], mode: str) -> np.ndarray:
    if mode not in WEIGHTS or any(count < 0 for count in counts) or sum(counts) < 1:
        raise ValueError("Expected nonempty period counts and a known weighting mode")
    if mode == "task_mean":
        return np.full(sum(counts), 1 / sum(counts))
    periods = sum(count > 0 for count in counts)
    return np.concatenate([np.full(count, 1 / (periods * count)) for count in counts if count])


def temporal_prefixes(point: dict, trials: int, seed: int, maximum_n: int = 10) -> dict[str, np.ndarray]:
    """Model-independent, nested uniform permutations; all cases share each pool."""
    if trials < 1 or maximum_n < 1:
        raise ValueError("Positive trials and maximum prefix size are required")
    t = point["checkpoint"]
    if point.get("current_index", t) != t or len(point["train_order"]) != 50:
        raise ValueError("Expected chronological fifty-task training order")
    pools = {"past": np.arange(t), "future": np.arange(t + 1, 50)}
    output = {"current": np.full((trials, 1), t, dtype=np.int16)}
    for period, pool in pools.items():
        generator = rng_for(seed, f"temporal-parent:{point['key']}:{period}")
        order = np.argsort(generator.random((trials, len(pool))), axis=1)
        output[period] = pool[order[:, :maximum_n]].astype(np.int16)
    return output


def selected_indices(prefixes: dict, counts: tuple[int, int, int]) -> np.ndarray:
    arrays = []
    for period, count in zip(("current", "past", "future"), counts, strict=True):
        if count:
            if prefixes[period].shape[1] < count:
                raise ValueError(f"Not enough {period} tasks for this condition")
            arrays.append(prefixes[period][:, :count])
    return np.concatenate(arrays, axis=1)


def selection_scores(success: np.ndarray, turns: np.ndarray, weights: np.ndarray) -> dict:
    """Read sampled data only: arrays (trials, parent+five candidates, actual_n).

    The parent-preserving tuple maximizes (-regression, repair, efficiency).
    Both event rates and joint-success relative step reduction use the supplied
    task weights. A missing step on a failed trajectory contributes nothing;
    a missing/nonpositive successful step is an invalid observation.
    """
    success = np.asarray(success, dtype=float)
    turns = np.asarray(turns, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (success.ndim != 3 or success.shape[1] != 6 or success.shape[2] < 1
            or turns.shape != success.shape or weights.shape != (success.shape[2],)):
        raise ValueError("Expected success/turns (trials, 6, n) and weights (n,)")
    if not np.isin(success, (0, 1)).all():
        raise ValueError("Success must be binary")
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Weights must be finite, nonnegative and sum to one")
    successful = success == 1
    if np.any(successful & (~np.isfinite(turns) | (turns <= 0) | (turns != np.floor(turns)))):
        raise ValueError("Successful trajectories need positive integer step counts")
    parent, candidate = success[:, :1], success[:, 1:]
    joint = (parent == 1) & (candidate == 1)
    # Invalid failed counts are not imputed as observed zero-step trajectories.
    pt = np.where(parent == 1, turns[:, :1], 1)
    ct = np.where(candidate == 1, turns[:, 1:], 1)
    efficiency = np.where(joint, np.clip((pt - ct) / np.maximum(1, pt), -1, 1), 0)
    regressions = ((parent == 1) & (candidate == 0)) @ weights
    repairs = ((parent == 0) & (candidate == 1)) @ weights
    return {
        "success_mean": candidate @ weights,
        "parent_preservation": np.stack((-regressions, repairs, efficiency @ weights), axis=-1),
        "regressions": regressions,
        "repairs": repairs,
        "efficiency": efficiency @ weights,
    }


def _summary(values: np.ndarray, cp_values: list[float] | None = None) -> dict:
    result = describe(values)
    if cp_values is not None:
        finite = np.asarray([value for value in cp_values if value is not None], dtype=float)
        finite = finite[np.isfinite(finite)]
        result["checkpoint_std"] = float(finite.std(ddof=1)) if finite.size > 1 else None
        result["checkpoint_sem"] = float(finite.std(ddof=1) / math.sqrt(finite.size)) if finite.size > 1 else None
    return result


def calculate(model: dict, keys: list[str], trials: int, seed: int) -> tuple[list, list, list, dict]:
    points = [point for point in model["points"] if point["key"] in keys]
    sums, rho_sums, rho_counts, counts = {}, {}, {}, defaultdict(int)
    cp_rows, audits, queries = [], [], {}
    for number, point in enumerate(points, 1):
        if number == 1 or number % 10 == 0:
            print(f"{model['label']}: checkpoint {number}/{len(points)} ({point['key']})", flush=True)
        success = np.asarray(point["success"], float)
        turns = np.asarray(point["turns"], float)
        if success.shape != (6, 50) or turns.shape != (6, 50):
            raise ValueError("Cache needs all six contexts on fifty tasks")
        prefixes = temporal_prefixes(point, trials, seed)
        for period, values in prefixes.items():
            queries[f"{point['key']}__{period}"] = values.astype(np.uint8)
        scopes = ("all", point["seed"])
        for n in range(1, 11):
            for case in CASES:
                parts = allocation(case, n)
                if parts is None:
                    continue
                indices = selected_indices(prefixes, parts)
                candidate = success[:, indices].transpose(1, 0, 2)
                sampled_turns = turns[:, indices].transpose(1, 0, 2)
                query_digest = hashlib.sha256(indices.astype(np.uint8).tobytes()).hexdigest()
                if not np.all(np.diff(np.sort(indices, axis=1), axis=1) > 0):
                    raise ValueError("A sampled task appears more than once")
                for weight in WEIGHTS:
                    values = selection_scores(candidate, sampled_turns, task_weights(parts, weight))
                    ranks = {method: ordering_ranks(values[method]) for method in METHODS}
                    maxima = {method: ranks[method] == ranks[method].max(axis=-1, keepdims=True)
                              for method in METHODS}
                    probabilities = {method: maxima[method] / maxima[method].sum(axis=-1, keepdims=True)
                                     for method in METHODS}
                    rhos = {}
                    for target in TARGETS:
                        truth = rankdata(point["targets"][target])
                        rhos[target] = {method: correlations(ranks[method], truth) for method in METHODS}
                    # Evaluation targets are read after both selectors have finished.
                    val = np.asarray(point["targets"]["full_validation"], float)
                    remaining = np.asarray(point["targets"]["validation_remaining40"], float)
                    train = np.asarray(point["targets"]["full_train"], float)
                    parent = float(point["parent_scores"]["full_train"])
                    metrics = {}
                    for method in METHODS:
                        p = probabilities[method]
                        metrics[method] = {
                            "validation_lift": p @ val - val.mean(),
                            "validation_remaining40_lift": p @ remaining - remaining.mean(),
                            "train_parent_gain": p @ train - parent,
                            "selected_regressions": (p * values["regressions"]).sum(-1),
                            "selected_repairs": (p * values["repairs"]).sum(-1),
                            "selected_relative_efficiency": (p * values["efficiency"]).sum(-1),
                            "selection_tv_vs_success": .5 * np.abs(p - probabilities["success_mean"]).sum(-1),
                        }
                        row = {"model": model["label"], "key": point["key"], "seed": point["seed"],
                               "checkpoint": point["checkpoint"], "n": n, "actual_n": sum(parts),
                               "case": case, "weight": weight, "method": method,
                               "choice_probabilities": p.mean(axis=0).tolist(),
                               **{metric: float(vector.mean()) for metric, vector in metrics[method].items()}}
                        for target in TARGETS:
                            valid = np.isfinite(rhos[target][method])
                            row[f"spearman_{target}"] = describe(rhos[target][method])["mean"]
                            row[f"spearman_{target}_defined_fraction"] = float(valid.mean())
                        cp_rows.append(row)
                    for scope in scopes:
                        prefix = (scope, case, weight, n)
                        counts[prefix] += 1
                        for method in (*METHODS, "parent_minus_success"):
                            for metric in METRICS:
                                vector = (metrics[method][metric] if method in METHODS else
                                          metrics["parent_preservation"][metric] - metrics["success_mean"][metric])
                                k = (*prefix, method, metric)
                                if k not in sums:
                                    sums[k] = np.zeros(trials)
                                sums[k] += vector
                        for target in TARGETS:
                            common = np.logical_and.reduce([np.isfinite(rhos[target][m]) for m in METHODS])
                            k = (*prefix, target)
                            if k not in rho_counts:
                                rho_counts[k] = np.zeros(trials, dtype=np.int16)
                            rho_counts[k] += common
                            for method in METHODS:
                                k = (*prefix, method, target)
                                if k not in rho_sums:
                                    rho_sums[k] = np.zeros(trials)
                                rho_sums[k] += np.where(common, rhos[target][method], 0)
                    audits.append({"model": model["label"], "key": point["key"], "n": n,
                                   "actual_n": sum(parts), "case": case, "weight": weight,
                                   "counts_current_past_future": list(parts), "queries_sha256": query_digest,
                                   "candidate_outcomes_read": 5 * sum(parts), "parent_outcomes_read": sum(parts),
                                   "trials": trials})
    aggregate, contrasts = [], []
    for (scope, case, weight, n), count in counts.items():
        selected_rows = [r for r in cp_rows if r["case"] == case and r["weight"] == weight and r["n"] == n
                         and (scope == "all" or r["seed"] == scope)]
        for method in (*METHODS, "parent_minus_success"):
            prefix = (scope, case, weight, n)
            row = {"model": model["label"], "scope": scope, "case": case, "weight": weight, "n": n,
                   "actual_n": sum(allocation(case, n)), "method": method, "checkpoints": count}
            for metric in METRICS:
                if method in METHODS:
                    cp_values = [r[metric] for r in selected_rows if r["method"] == method]
                else:
                    by_cp = {(r["key"], r["method"]): r[metric] for r in selected_rows}
                    cp_values = [by_cp[key, "parent_preservation"] - by_cp[key, "success_mean"]
                                 for key in {r["key"] for r in selected_rows}]
                row[metric] = _summary(sums[*prefix, method, metric] / count, cp_values)
            if method in METHODS:
                for target in TARGETS:
                    denominator = rho_counts[*prefix, target]
                    rho = np.full(trials, np.nan)
                    np.divide(rho_sums[*prefix, method, target], denominator, out=rho, where=denominator > 0)
                    row[f"spearman_{target}"] = _summary(rho)
                    row[f"spearman_{target}_common_checkpoints"] = _summary(denominator)
                aggregate.append(row)
            else:
                contrasts.append(row)
    return aggregate, contrasts, cp_rows, {"audit": audits, "queries": queries}


def statistical_tests(rows: list[dict], seed: int) -> list[dict]:
    """n=10 paired block sign flips; seven temporal contrasts per model/weight."""
    output = []
    for model in MODELS:
        for weight in WEIGHTS:
            for block_size in (2, 3, 4, 5, "whole_seed"):
                family = []
                for case in CASES:
                    selected = [r for r in rows if r["model"] == model and r["weight"] == weight
                                and r["case"] == case and r["n"] == 10]
                    indexed = {(r["key"], r["method"]): r for r in selected}
                    baseline = sorted([r for r in selected if r["method"] == "success_mean"],
                                      key=lambda r: (r["seed"], r["checkpoint"], r["key"]))
                    differences = np.asarray([
                        indexed[r["key"], "parent_preservation"]["validation_lift"] - r["validation_lift"]
                        for r in baseline])
                    if block_size == "whole_seed":
                        contributions = np.asarray([differences[[i for i, r in enumerate(baseline)
                                                               if r["seed"] == s]].sum() / len(baseline)
                                                    for s in (42, 43, 44)])
                    else:
                        contributions, _ = paired_block_contributions(baseline, baseline, differences, block_size)
                    result = sign_flip_test(contributions, 100000,
                                            rng_for(seed, f"parent-temporal:{model}:{weight}:{case}:{block_size}"))
                    effect = result.pop("mean_rho_difference")
                    family.append({"model": model, "weight": weight, "case": case, "n": 10,
                                   "block_size": block_size, "difference": effect, "checkpoints": len(baseline),
                                   "reference": "success_mean", "method": "parent_preservation", **result})
                for row, adjusted in zip(family, holm_adjust([r["p_two_sided"] for r in family]), strict=True):
                    row.update(p_holm=adjusted, family_size=len(family))
                output.extend(family)
    return output


def validation_reference(models: list[dict], keys: list[str], previous: dict | None = None) -> list[dict]:
    output = []
    for model in models:
        points = [p for p in model["points"] if p["key"] in keys]
        for target in ("full_validation", "validation_remaining40"):
            if target == "full_validation" and previous is not None:
                continue
            lifts = []
            for point in points:
                p = choice_probabilities(np.asarray(point["targets"]["val10"])[None])[0]
                truth = np.asarray(point["targets"][target])
                lifts.append(float(p @ truth - truth.mean()))
            output.append({"model": model["label"], "n": 10, "target": target,
                           "selector": "archived_val10", "selected_lift": float(np.mean(lifts)),
                           "checkpoints": len(points), "checkpoint_lifts": lifts,
                           "overlap": target == "full_validation"})
    if previous is not None:
        if set(previous["fixed_checkpoint_keys"]) != set(keys):
            raise ValueError("Archived validation curve uses a different checkpoint cohort")
        for model in previous["models"]:
            if model["label"] not in MODELS:
                continue
            for row in model["validation_reference"]:
                if row["checkpoints"] != len(keys):
                    raise ValueError("Archived validation reference checkpoint count differs")
                output.append({"model": model["label"], "n": row["n"], "target": "full_validation",
                               "selector": "random_validation_n", "selected_lift": row["selection_lift"]["mean"],
                               "sampling_distribution": row["selection_lift"], "checkpoints": len(keys),
                               "overlap": True, "validation_pool_size": row["validation_pool_size"]})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trials", default=10000, type=int)
    parser.add_argument("--seed", default=20261022, type=int)
    args = parser.parse_args()
    source = args.output / "full_train_metrics.json"
    raw = source.read_bytes()
    cache = json.loads(raw)
    models = [model for model in cache["models"] if model["label"] in MODELS]
    if {m["label"] for m in models} != set(MODELS):
        raise ValueError("Need Qwen and Gemma caches")
    keys = [p["key"] for p in models[0]["points"] if p["checkpoint"] >= 10 and 49 - p["checkpoint"] >= 10]
    if len(keys) != 36:
        raise ValueError(f"Expected the prior fixed 36-checkpoint cohort, found {len(keys)}")
    for model in models:
        indexed = {p["key"]: p for p in model["points"]}
        for first in models[0]["points"]:
            other = indexed[first["key"]]
            if first["train_order"] != other["train_order"] or first["checkpoint"] != other["checkpoint"]:
                raise ValueError("Models must share the same checkpoint task chronology")
    aggregate, contrasts, rows, audit, queries = [], [], [], [], {}
    for model in models:
        a, c, r, extra = calculate(model, keys, args.trials, args.seed)
        aggregate.extend(a)
        contrasts.extend(c)
        rows.extend(r)
        audit.extend(extra["audit"])
        for key, value in extra["queries"].items():
            if key in queries and not np.array_equal(queries[key], value):
                raise ValueError("Models did not share sampled tasks")
            queries[key] = value
    np.savez_compressed(args.output / "query_prefixes.npz", **queries)
    print("Running paired n=10 block/seed tests", flush=True)
    tests = statistical_tests(rows, args.seed)
    reference_path = args.output.parents[2] / "temporal_selection_summary.json"
    # output -> trajectory_reward_design -> hypotheses_training_only -> temporal_selection
    if not reference_path.exists():
        reference_path = args.output.parents[3] / "temporal_selection_summary.json"
    reference_blob = reference_path.read_bytes()
    summary = {
        "models": list(MODELS), "excluded_models": ["OSS"], "cases": list(CASES), "weights": list(WEIGHTS),
        "methods": list(METHODS), "trials": args.trials, "sampling_seed": args.seed,
        "fixed_checkpoint_keys": keys, "checkpoints_per_model": len(keys), "n_values": list(range(1, 11)),
        "source": str(source.resolve()), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "new_model_calls": 0, "llm_judgments_used": False,
        "rule": "Lexicographically maximize (-weighted parent-success/candidate-failure rate, "
                "weighted parent-failure/candidate-success rate, weighted joint-success relative step reduction). "
                "Relative reduction=clip((parent_steps-candidate_steps)/max(1,parent_steps),-1,1).",
        "baseline": "Maximize weighted candidate success on exactly the same sampled tasks and weights.",
        "sampling": "Uniform without replacement within entire chronological past/future pools; shared nested "
                    "permutations across n, cases, methods, weights and models. Current is exactly the proposal task.",
        "allocation": "Current-only actual_n=1 for every displayed n; two-period mixtures start n=2; "
                      "three-period mixture starts n=3. Past/future split balanced, past gets an odd extra task.",
        "weighting": {"task_mean": "Every sampled task has weight 1/actual_n.",
                      "period_mean": "Each present period has equal mass; tasks within it have equal weight."},
        "cohort": "Same fixed 36 checkpoints, each with at least 10 past and 10 future tasks, for every result.",
        "lift": "Selected full-validation candidate score minus mean score of all five proposed candidates. "
                "No parent validation score is available; this is not parent-relative validation gain.",
        "uncertainty": "std/q025/q975 describe task-sampling variability of the checkpoint-aggregated curve "
                       "over Monte Carlo draws; not a confidence interval over seeds/checkpoints. "
                       "checkpoint_std/sem additionally describe checkpoint expected values.",
        "correlation": "Spearman of the scalar baseline or lexicographic ordering with target scores; "
                       "averages use jointly defined checkpoint/draws across both methods within each condition. "
                       "No arbitrary scalar encoding/Pearson correlation is applied to lexicographic tuples.",
        "infrastructure": "Archived infrastructure failures retain original score 0; missing failed steps are "
                          "never treated as zero-step successes. Steps are used only on joint-success tasks.",
        "statistics": "Exploratory n=10 parent-preservation minus matched-success comparisons. "
                      "Sign-flip chronological checkpoint blocks 2/3/4/5 within seed, plus exact whole-seed flips. "
                      "Holm correction over seven cases separately by model/weight/block-size; three seeds only. "
                      "Reused validation outcomes do not provide fresh confirmatory evidence.",
        "future_condition": "Future-containing samples are retrospective oracle ablations, unavailable online.",
        "aggregate": aggregate, "paired_parent_minus_success": contrasts, "checkpoint_rows": rows,
        "validation_reference": validation_reference(models, keys, json.loads(reference_blob)),
        "validation_reference_source": str(reference_path.resolve()),
        "validation_reference_sha256": hashlib.sha256(reference_blob).hexdigest(),
        "paired_block_tests_n10": tests,
        "query_audit": audit,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    flattened = [{**{k: v for k, v in row.items() if not isinstance(v, list)},
                  **{f"candidate_{i + 1}_probability": p for i, p in enumerate(row["choice_probabilities"])}}
                 for row in rows]
    write_csv(args.output / "checkpoint_metrics.csv", flattened)
    write_csv(args.output / "paired_block_tests_n10.csv", tests)
    print(f"Saved {len(rows)} checkpoint rows and {len(aggregate)} aggregate rows", flush=True)


if __name__ == "__main__":
    main()
