# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Matched uniform-training reference for the offline temporal preservation study.

Only archived successes/steps on the queried n tasks enter selection. The old
semantic 5-checkpoint, LLM-assisted surface is not a matched sampling baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.analyze_bfcl_alignment_significance import rng_for  # noqa: E402
from scripts.premise.analyze_bfcl_temporal_parent_preservation import METHODS, MODELS, selection_scores  # noqa: E402
from scripts.premise.analyze_bfcl_temporal_selection import describe, write_csv  # noqa: E402
from scripts.premise.bfcl_trajectory_selection_rewards import choice_probabilities  # noqa: E402

METRICS = ("validation_lift", "validation_remaining40_lift", "train_parent_gain")


def uniform_prefixes(key: str, trials: int, seed: int) -> np.ndarray:
    """Ten nested uniform-without-replacement queries from all fifty tasks."""
    if trials < 1:
        raise ValueError("Positive Monte Carlo trial count required")
    generator = rng_for(seed, f"uniform-parent-reference:{key}")
    return np.argsort(generator.random((trials, 50)), axis=1)[:, :10].astype(np.uint8)


def calculate(models: list[dict], keys: list[str], trials: int, seed: int) -> tuple:
    prefixes = {key: uniform_prefixes(key, trials, seed) for key in keys}
    aggregate, cp_rows = [], []
    # n=10 is calculated first to expose the matched primary comparison promptly.
    for n in (10, *range(1, 10)):
        for model in models:
            points = [point for point in model["points"] if point["key"] in keys]
            sums = {(scope, method, metric): np.zeros(trials)
                    for scope in ("all", 42, 43, 44) for method in METHODS for metric in METRICS}
            counts = dict.fromkeys(("all", 42, 43, 44), 0)
            here = []
            for point in points:
                indices = prefixes[point["key"]][:, :n]
                success = np.asarray(point["success"], float)[:, indices].transpose(1, 0, 2)
                turns = np.asarray(point["turns"], float)[:, indices].transpose(1, 0, 2)
                scores = selection_scores(success, turns, np.full(n, 1 / n))
                probabilities = {method: choice_probabilities(scores[method]) for method in METHODS}
                # Target scores do not enter either sampled-data selector above.
                validation = np.asarray(point["targets"]["full_validation"])
                remaining = np.asarray(point["targets"]["validation_remaining40"])
                train = np.asarray(point["targets"]["full_train"])
                parent_train = point["parent_scores"]["full_train"]
                for scope in ("all", point["seed"]):
                    counts[scope] += 1
                for method, probability in probabilities.items():
                    metrics = {
                        "validation_lift": probability @ validation - validation.mean(),
                        "validation_remaining40_lift": probability @ remaining - remaining.mean(),
                        "train_parent_gain": probability @ train - parent_train,
                    }
                    row = {"model": model["label"], "key": point["key"], "seed": point["seed"],
                           "checkpoint": point["checkpoint"], "n": n, "actual_n": n, "weight": "uniform",
                           "method": method, "sampler": "uniform_training50", "trials": trials,
                           "choice_probabilities": probability.mean(axis=0).tolist(),
                           **{metric: float(vector.mean()) for metric, vector in metrics.items()}}
                    here.append(row)
                    for scope in ("all", point["seed"]):
                        for metric, vector in metrics.items():
                            sums[scope, method, metric] += vector
            cp_rows.extend(here)
            for scope, count in counts.items():
                for method in METHODS:
                    row = {"model": model["label"], "scope": scope, "n": n, "actual_n": n,
                           "weight": "uniform", "sampler": "uniform_training50", "method": method,
                           "checkpoints": count}
                    for metric in METRICS:
                        row[metric] = describe(sums[scope, method, metric] / count)
                        values = [r[metric] for r in here if r["method"] == method
                                  and (scope == "all" or r["seed"] == scope)]
                        row[metric]["checkpoint_std"] = float(np.std(values, ddof=1)) if count > 1 else None
                        row[metric]["checkpoint_sem"] = (float(np.std(values, ddof=1) / np.sqrt(count))
                                                         if count > 1 else None)
                    aggregate.append(row)
                    if scope == "all":
                        lift = 100 * row["validation_lift"]["mean"]
                        print(f"{model['label']} n={n} {method}: full-validation lift {lift:+.6f}pp", flush=True)
    return aggregate, cp_rows, prefixes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    source_path, temporal_path = args.output / "full_train_metrics.json", args.output / "summary.json"
    source_blob, temporal_blob = source_path.read_bytes(), temporal_path.read_bytes()
    cache, temporal = json.loads(source_blob), json.loads(temporal_blob)
    models = [model for model in cache["models"] if model["label"] in MODELS]
    if {model["label"] for model in models} != set(MODELS):
        raise ValueError("Both Qwen and Gemma are required")
    keys = temporal["fixed_checkpoint_keys"]
    if len(keys) != 36:
        raise ValueError("Matched temporal reference requires the fixed 36 checkpoints")
    if temporal["source_sha256"] != hashlib.sha256(source_blob).hexdigest():
        raise ValueError("Uniform and temporal analyses must use the exact same raw success/step cache")
    first = {p["key"]: p for p in models[0]["points"]}
    for model in models:
        indexed = {p["key"]: p for p in model["points"]}
        for key in keys:
            if indexed[key]["train_order"] != first[key]["train_order"]:
                raise ValueError("Models must use the same fifty-task chronology")
    aggregate, rows, prefixes = calculate(models, keys, args.trials, args.seed)
    query_path = args.output / "uniform_training_query_prefixes.npz"
    np.savez_compressed(query_path, **prefixes)
    output = {
        "models": list(MODELS), "excluded_models": ["OSS"], "fixed_checkpoint_keys": keys,
        "checkpoints_per_model": len(keys), "n_values": list(range(1, 11)), "trials": args.trials,
        "sampling_seed": args.seed, "weight": "uniform", "sampler": "uniform_training50",
        "sampling": "Uniform without replacement from all fifty training tasks, with no temporal restriction. "
                    "Nested prefixes across n and identical draws for both models and both selection methods.",
        "weighting": "One pooled uniform average, 1/n per task. This reference is identical in temporal "
                     "task_mean and period_mean panels; it does not reweight temporal periods.",
        "rule": temporal["rule"], "baseline": temporal["baseline"],
        "source": str(source_path.resolve()), "source_sha256": hashlib.sha256(source_blob).hexdigest(),
        "temporal_source": str(temporal_path.resolve()),
        "temporal_source_sha256": hashlib.sha256(temporal_blob).hexdigest(),
        "analysis_source": str(Path(__file__).resolve()),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "query_prefixes": str(query_path.resolve()),
        "query_prefixes_sha256": hashlib.sha256(query_path.read_bytes()).hexdigest(),
        "new_model_calls": 0, "llm_judgments_used": False,
        "lift": temporal["lift"], "uncertainty": temporal["uncertainty"],
        "scope": "Matched descriptive sampling reference only; no new significance tests. "
                 "Full training pool includes future tasks as an offline oracle condition.",
        "old_surface_comparison": "The old semantic_parent_safe_surface_validation.pdf used a different "
                                  "five-checkpoint cohort, a single random ten-task draw and a four-coordinate "
                                  "LLM-assisted rule. Its values cannot isolate sampling effects against "
                                  "the current 36-checkpoint LLM-free three-coordinate rule.",
        "aggregate": aggregate, "checkpoint_rows": rows,
        "query_audit": [{"key": key, "trials": args.trials, "pool_size": 50, "maximum_n": 10,
                         "queries_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                         "same_models_methods_n_prefix": True} for key, value in prefixes.items()],
    }
    path = args.output / "uniform_training_reference.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    flattened = [{**{k: v for k, v in row.items() if not isinstance(v, list)},
                  **{f"candidate_{i + 1}_probability": p for i, p in enumerate(row["choice_probabilities"])}}
                 for row in rows]
    write_csv(args.output / "uniform_training_checkpoint_metrics.csv", flattened)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
