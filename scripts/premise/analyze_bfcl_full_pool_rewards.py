# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Compare budgeted sample selection and semantic rewards over all fifty BFCL train tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.analyze_bfcl_alignment_significance import rng_for  # noqa: E402
from scripts.premise.analyze_bfcl_temporal_selection import correlations, describe, write_csv  # noqa: E402
from scripts.premise.analyze_bfcl_training_selection_hypotheses import (  # noqa: E402
    public_metadata,
    selection_probabilities,
)
from scripts.premise.bfcl_full_pool_rewards import REWARD_KEYS, reward_metadata, reward_scores  # noqa: E402
from scripts.premise.bfcl_full_pool_selection import STRATEGIES, prepare_features, sample_prefixes  # noqa: E402

MODELS = ("Qwen", "Gemma")
TARGETS = ("full_train", "full_validation")
METRICS = ("validation_lift", "train_parent_gain", "unselected_train_parent_gain", "selected_repairs",
           "selected_regressions", "selection_tv_vs_mean", "mean_top_probability", "safe_selected_probability")
N_VALUES = tuple(range(1, 11))


def population_weights(similarity: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Predict the population mean using only public affinities and the queried labels."""
    kernel = similarity[:, indices].transpose(1, 0, 2)
    return softmax(kernel / 0.25, axis=-1).mean(axis=1)


def calculate(model: dict, parents: dict, metadata: dict, trials: int, seed: int,
              feature_cache: dict, sampler_cache: dict) -> tuple:
    if model["label"] not in MODELS or trials < 1:
        raise ValueError("Only Qwen/Gemma and positive trial counts are supported")
    scopes = ("all", 42, 43, 44)
    sums = {(scope, sampler, reward, n, metric): np.zeros(trials) for scope in scopes for sampler in STRATEGIES
            for reward in REWARD_KEYS for n in N_VALUES for metric in METRICS}
    rho_sums = {(scope, sampler, reward, n, target, kind): np.zeros(trials) for scope in scopes
                for sampler in STRATEGIES for reward in REWARD_KEYS for n in N_VALUES for target in TARGETS
                for kind in ("rho", "spearman")}
    coverage = {(scope, sampler, n, target): np.zeros(trials) for scope in scopes for sampler in STRATEGIES
                for n in N_VALUES for target in TARGETS}
    counts = dict.fromkeys(scopes, 0)
    rows_n10, rows_all, audit_rows, saved_queries = [], [], [], {}
    for number, point in enumerate(model["points"], 1):
        if number == 1 or number % 10 == 0:
            print(f"{model['label']} CP {number}/{len(model['points'])}: {point['key']}", flush=True)
        order = tuple(point["train_order"])
        if order not in feature_cache:
            feature_cache[order] = prepare_features(list(order), metadata, random_state=seed)
        similarity, clusters = feature_cache[order]
        proportions = np.bincount(clusters, minlength=5) / len(clusters)
        if len(similarity) != 50 or np.any(proportions == 0):
            raise ValueError("This experiment requires fifty tasks and five nonempty public-metadata clusters")
        matrix = np.asarray(point["train_rewards"], float)
        parent_all = np.asarray(parents[model["label"], point["key"]]["parent_rewards"], float)
        scopes_here = ("all", point["seed"])
        for scope in scopes_here:
            counts[scope] += 1
        for sampler in STRATEGIES:
            cache_key = (point["key"], sampler)
            audit = {}
            if sampler != "adaptive_disagreement" and cache_key in sampler_cache:
                indices_full = sampler_cache[cache_key]
            else:
                reward_source = matrix if sampler == "adaptive_disagreement" else None
                indices_full = sample_prefixes(
                    sampler, similarity, clusters, reward_source, trials, 10,
                    rng_for(seed, f"full-pool:{point['key']}:{sampler}"), audit=audit)
                if sampler != "adaptive_disagreement":
                    sampler_cache[cache_key] = indices_full
            saved_queries[f"{model['label']}__{point['key']}__{sampler}"] = indices_full.astype(np.uint8)
            digest = hashlib.sha256(indices_full.astype(np.uint8).tobytes()).hexdigest()
            audit_rows.append({
                "model": model["label"], "key": point["key"], "seed": point["seed"], "sampler": sampler,
                "trials": trials, "max_n": 10, "unique_tasks_per_prefix": True, "queries_sha256": digest,
                "sampler_outcomes_read_before_nth_choice": "n-1" if sampler == "adaptive_disagreement" else "0",
                "total_observed_candidate_values_for_n_prefix": "5*n",
                "parent_values_for_parent_aware_reward": "n, on same queried tasks",
            })
            for n in N_VALUES:
                indices = indices_full[:, :n]
                candidate = matrix[:, indices].transpose(1, 0, 2)
                parent = parent_all[indices]
                observed_similarity = similarity[indices[:, :, None], indices[:, None, :]]
                selected_clusters = clusters[indices]
                cluster_coverage = float((selected_clusters[:, :, None] == np.arange(5)).any(1).sum(1).mean())
                parent_strata_fraction = float(((parent.min(-1) == 0) & (parent.max(-1) == 1)).mean())
                weights = population_weights(similarity, indices)
                scores = reward_scores(candidate, parent, selected_clusters, observed_similarity, weights, proportions)
                probability = {key: selection_probabilities(value) for key, value in scores.items()}
                if n == 1 and any(np.max(np.abs(p - probability["mean"])) > 1e-12 for p in probability.values()):
                    raise ValueError("A reward changed the n=1 choices from the identical binary observation")
                # Full-set outcomes are read as targets only after the n-observation selector has finished.
                val = np.asarray(point["scores"]["full_validation"])
                train = np.asarray(point["scores"]["full_train"])
                delta = candidate - parent[:, None, :]
                repaired, broken = (delta > 0).sum(-1), (delta < 0).sum(-1)
                safe = broken == 0
                held_delta = (50 * (train - point["parent_scores"]["full_train"])[None, :] - delta.sum(-1)) / (50 - n)
                point_rho = {}
                for key, values in scores.items():
                    ranks = rankdata(np.round(values, 12), method="average", axis=-1)
                    for target in TARGETS:
                        truth = np.asarray(point["scores"][target])
                        point_rho[key, target, "rho"] = correlations(values, truth)
                        point_rho[key, target, "spearman"] = correlations(ranks, rankdata(truth))
                for key, p in probability.items():
                    metrics = {
                        "validation_lift": p @ val - val.mean(),
                        "train_parent_gain": p @ train - point["parent_scores"]["full_train"],
                        "unselected_train_parent_gain": (p * held_delta).sum(-1),
                        "selected_repairs": (p * repaired).sum(-1),
                        "selected_regressions": (p * broken).sum(-1),
                        "selection_tv_vs_mean": 0.5 * np.abs(p - probability["mean"]).sum(-1),
                        "mean_top_probability": (p * (probability["mean"] > 0)).sum(-1),
                        "safe_selected_probability": (p * safe).sum(-1),
                    }
                    row = {"model": model["label"], "key": point["key"], "seed": point["seed"],
                           "checkpoint": point["checkpoint"], "sampler": sampler, "reward": key, "n": n,
                           **{metric: float(value.mean()) for metric, value in metrics.items()},
                           "cluster_coverage": cluster_coverage,
                           "parent_strata_covered_fraction": parent_strata_fraction}
                    for target in TARGETS:
                        for kind in ("rho", "spearman"):
                            row[f"{kind}_{target}"] = describe(point_rho[key, target, kind])["mean"]
                    rows_all.append(row)
                    if n == 10:
                        rows_n10.append({**row, **{f"candidate_{i + 1}_probability": float(p[:, i].mean())
                                                  for i in range(5)}})
                    for scope in scopes_here:
                        for metric, values in metrics.items():
                            sums[scope, sampler, key, n, metric] += values
                for target in TARGETS:
                    common = np.logical_and.reduce([np.isfinite(point_rho[key, target, "rho"]) for key in REWARD_KEYS])
                    for scope in scopes_here:
                        coverage[scope, sampler, n, target] += common
                        for key in REWARD_KEYS:
                            for kind in ("rho", "spearman"):
                                rho_sums[scope, sampler, key, n, target, kind] += np.where(
                                    common, point_rho[key, target, kind], 0)
    aggregate = []
    for scope, count in counts.items():
        if not count:
            continue
        for sampler in STRATEGIES:
            for key in REWARD_KEYS:
                for n in N_VALUES:
                    row = {"model": model["label"], "seed": scope, "sampler": sampler, "reward": key,
                           "n": n, "checkpoints": count,
                           **{metric: describe(sums[scope, sampler, key, n, metric] / count) for metric in METRICS}}
                    for target in TARGETS:
                        defined = coverage[scope, sampler, n, target]
                        row[f"rho_{target}_common_checkpoints"] = describe(defined)
                        for kind in ("rho", "spearman"):
                            values = np.full(trials, np.nan)
                            np.divide(rho_sums[scope, sampler, key, n, target, kind], defined,
                                      out=values, where=defined > 0)
                            row[f"{kind}_{target}"] = describe(values)
                    aggregate.append(row)
    return aggregate, rows_n10, rows_all, audit_rows, saved_queries


def flatten(rows: list[dict]) -> list[dict]:
    return [{out_key: number for key, value in row.items() for out_key, number in
             ([(f"{key}_{stat}", number) for stat, number in value.items()] if isinstance(value, dict)
              else [(key, value)])} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    plan_path = args.output / "analysis_plan.json"
    plan = json.loads(plan_path.read_text())
    source_path, parent_path = Path(plan["source"]), Path(plan["parents"])
    source_blob, parent_blob = source_path.read_bytes(), parent_path.read_bytes()
    if hashlib.sha256(source_blob).hexdigest() != plan["source_sha256"]:
        raise ValueError("Candidate outcomes differ from the frozen plan")
    if hashlib.sha256(parent_blob).hexdigest() != plan["parents_sha256"]:
        raise ValueError("Parent outcomes differ from the frozen plan")
    raw = json.loads(source_blob)
    models = [m for m in raw["models"] if m["label"] in MODELS]
    parents = {(m["label"], p["key"]): p for m in json.loads(parent_blob)["models"] for p in m["points"]}
    metadata_path = ROOT / ("src/exgentic/benchmarks/bfcl/installation/gorilla/berkeley-function-call-leaderboard/"
                            "bfcl_eval/data/BFCL_v4_multi_turn_base.json")
    wanted = {task for model in models for point in model["points"] for task in point["train_order"]}
    metadata = public_metadata(metadata_path, wanted)
    aggregate, n10, all_rows, audits, queries = [], [], [], [], {}
    features, samples = {}, {}
    for model in models:
        values, rows, full_rows, model_audits, model_queries = calculate(
            model, parents, metadata, plan["trials"], plan["seed"], features, samples)
        aggregate.extend(values)
        n10.extend(rows)
        all_rows.extend(full_rows)
        audits.extend(model_audits)
        queries.update(model_queries)
    np.savez_compressed(args.output / "query_prefixes.npz", **queries)
    from scripts.premise.bfcl_full_pool_statistics import leave_one_seed_out, statistical_tests

    print("Computing n=10 paired-block tests over 48 methods", flush=True)
    statistics = statistical_tests(n10, plan["seed"])
    summary = {
        "models": list(MODELS), "excluded_models": ["OSS"], "n_values": list(N_VALUES),
        "samplers": list(STRATEGIES), "rewards": reward_metadata(), "seed": plan["seed"], "trials": plan["trials"],
        "new_model_calls": 0, "source": str(source_path), "source_sha256": plan["source_sha256"],
        "parent_source": str(parent_path), "parent_sha256": plan["parents_sha256"],
        "analysis_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "metadata_source": str(metadata_path),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "aggregate": aggregate, "checkpoint_scores_n10": n10, "paired_block_tests": statistics,
        "leave_one_seed_out": leave_one_seed_out(n10), "query_audit": audits,
        "coverage": "Pearson/Spearman averages use CPs jointly defined across12 rewards within each sampler/n. "
        "Different samplers can have different common coverage; comparison across samplers is descriptive. "
        "Validation lift always uses all50 CPs. No undefined correlations are replaced by zero.",
        "budget": "Each method sees only the first n distinct queried training-task outcomes per candidate. "
        "All50 public descriptions/tools may guide selection. Parent-aware rewards see parent on the same n. "
        "Adaptive next-task selection reads at most n-1 previous outcomes. Stored10-prefixes replay smaller n "
        "without future-observation influence; stopped-run equivalence tested.",
        "uncertainty": "SD is task-sampling variability over1000 draws, not independent-seed confidence. "
        "All targets and validation are reused retrospective data; LOSO is a diagnostic, not fresh validation.",
        "implementation_hashes": {name: hashlib.sha256((ROOT / "scripts" / "premise" / name).read_bytes()).hexdigest()
                                  for name in ("bfcl_full_pool_rewards.py", "bfcl_full_pool_selection.py",
                                               "bfcl_full_pool_statistics.py", Path(__file__).name)},
    }
    (args.output / "full_pool_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_csv(args.output / "method_scores.csv", flatten(aggregate))
    write_csv(args.output / "checkpoint_scores_n10.csv", n10)
    write_csv(args.output / "checkpoint_scores_all_n.csv", all_rows)
    write_csv(args.output / "query_audit.csv", audits)
    write_csv(args.output / "paired_block_tests_n10.csv", statistics)
    write_csv(args.output / "leave_one_seed_out_n10.csv", summary["leave_one_seed_out"])
    write_csv(args.output / "reward_formulas.csv", reward_metadata())
    if not args.skip_plots:
        from scripts.premise.plot_bfcl_full_pool_rewards import plot

        plot(summary, args.output / "figures")
    for model in MODELS:
        rows = [r for r in aggregate if r["model"] == model and r["seed"] == "all" and r["n"] == 10]
        best = sorted(rows, key=lambda r: r["validation_lift"]["mean"], reverse=True)[:5]
        print(model, [(r["sampler"], r["reward"], round(100*r["validation_lift"]["mean"], 5)) for r in best],
              flush=True)


if __name__ == "__main__":
    main()
