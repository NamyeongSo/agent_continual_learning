# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Permute checkpoint order within seeds to test Qwen same-history policy-shift trends."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.analyze_bfcl_alignment_significance import holm_adjust, rng_for  # noqa: E402
from scripts.utils.plot_qwen_policy_shift import POINTS, SEEDS, aggregate  # noqa: E402

METRICS = ("strict_accuracy", "strict_f1")
STATISTICS = ("pearson_r", "slope_per_task")


def shift_graphs(rows: list[dict], checkpoints: tuple[int, ...], seeds: tuple[int, ...]) -> np.ndarray:
    """Each checkpoint is a vertex; all its pairwise comparisons move together."""
    expected = {(seed, left, right) for seed in seeds
                for index, left in enumerate(checkpoints) for right in checkpoints[index + 1:]}
    keys = [(int(row["seed"]), int(row["left"]), int(row["right"])) for row in rows]
    if set(keys) != expected or len(keys) != len(set(keys)):
        raise ValueError("Expected complete, unique within-seed checkpoint pairs")
    graphs = np.zeros((len(seeds), len(checkpoints), len(checkpoints), len(METRICS)))
    for row in rows:
        seed = seeds.index(int(row["seed"]))
        left, right = checkpoints.index(int(row["left"])), checkpoints.index(int(row["right"]))
        if int(row["distance"]) != checkpoints[right] - checkpoints[left]:
            raise ValueError("Incorrect checkpoint distance")
        values = np.asarray([float(row[metric]) for metric in METRICS])
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("Invalid action agreement score")
        graphs[seed, left, right] = graphs[seed, right, left] = 1 - values
    return graphs


def distance_means(graphs: np.ndarray) -> np.ndarray:
    count = graphs.shape[-2]
    means = []
    for distance in range(1, count):
        left = np.arange(count - distance)
        means.append(graphs[..., left, left + distance, :].mean(axis=-2))
    return np.stack(means, axis=-2)


def trend_statistics(curves: np.ndarray, distances: np.ndarray) -> np.ndarray:
    x = np.asarray(distances, dtype=float) - np.mean(distances)
    centered = curves - curves.mean(axis=-2, keepdims=True)
    numerator = np.einsum("...dm,d->...m", centered, x)
    denominator = np.sqrt((x ** 2).sum() * (centered ** 2).sum(axis=-2))
    rho = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=rho, where=denominator > 0)
    # Zero encodes a genuinely flat trend for randomization, not an imputed candidate correlation.
    slope = numerator / (x ** 2).sum()
    return np.stack((rho, slope), axis=-1)


def order_draws(rng: np.random.Generator, trials: int, seeds: int, checkpoints: int) -> np.ndarray:
    # CP0 is a distinct initial context and stays at position zero in every permutation.
    draws = np.empty((trials, seeds, checkpoints), dtype=int)
    draws[..., 0] = 0
    draws[..., 1:] = np.argsort(rng.random((trials, seeds, checkpoints - 1)), axis=-1) + 1
    return draws


def analyze_graphs(graphs: np.ndarray, distances: np.ndarray, trials: int, rng: np.random.Generator) -> list[dict]:
    if trials < 1 or graphs.shape[0] != len(SEEDS):
        raise ValueError("Expected positive trials and three seed graphs")
    curves = distance_means(graphs)
    observed = trend_statistics(np.concatenate((curves, curves.mean(axis=0, keepdims=True))), distances)
    greater = np.zeros_like(observed, dtype=np.int64)
    less = np.zeros_like(observed, dtype=np.int64)
    seed_indices = np.arange(graphs.shape[0])[None, :, None]
    left, right = np.triu_indices(graphs.shape[1], k=1)
    weights = np.zeros((len(left), len(distances)))
    for index in range(len(distances)):
        paired = right - left == index + 1
        weights[paired, index] = 1 / paired.sum()
    for start in range(0, trials, 1000):
        draws = order_draws(rng, min(1000, trials - start), graphs.shape[0], graphs.shape[1])
        # The same node permutation moves both ACC and F1 matrices, preserving their dependence.
        permuted = graphs[seed_indices, draws[:, :, left], draws[:, :, right]]
        sampled_curves = np.einsum("bspe,pd->bsde", permuted, weights)
        sampled_curves = np.concatenate((sampled_curves, sampled_curves.mean(axis=1, keepdims=True)), axis=1)
        sampled = trend_statistics(sampled_curves, distances)
        greater += (sampled >= observed - 1e-12).sum(axis=0)
        less += (sampled <= observed + 1e-12).sum(axis=0)
    rows = []
    for index, seed in enumerate((*SEEDS, "all")):
        for metric_index, metric in enumerate(METRICS):
            for statistic_index, statistic in enumerate(STATISTICS):
                p_greater = (greater[index, metric_index, statistic_index] + 1) / (trials + 1)
                p_less = (less[index, metric_index, statistic_index] + 1) / (trials + 1)
                value = float(observed[index, metric_index, statistic_index])
                row = {"seed": str(seed), "metric": metric, "statistic": statistic, "value": value,
                       "permutations": trials, "p_greater": float(p_greater), "p_less": float(p_less),
                       "p_two_sided": float(min(1, 2 * min(p_greater, p_less)))}
                if statistic == "slope_per_task":
                    row["slope_pp_per_ten_tasks"] = 1000 * value
                rows.append(row)
    return rows


def run(sources: list[tuple[str, Path]], trials: int, seed: int) -> dict:
    rows, provenance = [], []
    for benchmark, folder in sources:
        figure_data, summary = aggregate(folder, benchmark)
        source = folder / "pairwise_metrics.csv"
        with source.open(newline="") as stream:
            graphs = shift_graphs(list(csv.DictReader(stream)), POINTS, SEEDS)
        distances = np.asarray(POINTS[1:], dtype=float)
        curves = distance_means(graphs).mean(axis=0)
        for index, metric in enumerate(METRICS):
            expected = [row["mean_shift"] for row in figure_data if row["metric"] == metric]
            if not np.allclose(curves[:, index], expected, atol=1e-12, rtol=0):
                raise ValueError("Policy-shift curve differs from the plotted figure")
        for row in analyze_graphs(graphs, distances, trials, rng_for(seed, f"policy-shift:{benchmark}")):
            row["benchmark"] = benchmark
            rows.append(row)
        provenance.append({"benchmark": benchmark, "analysis": str(folder.resolve()),
                           "coverage": summary["coverage"], "source_hashes": {
                               name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                               for name in ("summary.json", "pairwise_metrics.csv")}})
    for scope in (*map(str, SEEDS), "all"):
        family = [row for row in rows if row["seed"] == scope]
        for row, adjusted in zip(family, holm_adjust([row["p_two_sided"] for row in family]), strict=True):
            row["holm_family_size"] = len(family)
            row["p_holm_all_policy_shift_tests"] = adjusted
    seed_family = [row for row in rows if row["seed"] != "all"]
    adjusted = holm_adjust([row["p_two_sided"] for row in seed_family])
    for row, value in zip(seed_family, adjusted, strict=True):
        row["p_holm_all_seed_policy_shift_tests"] = value
    return {"model": "Qwen", "rng_seed": seed, "checkpoint_steps": POINTS, "seeds": SEEDS,
            "permutations": trials,
            "method": "Within each seed, permute the ten learned-checkpoint labels jointly for the entire "
            "pairwise ACC/F1 matrices. Keep CP0 fixed. Recompute ten distance-bin means and their Pearson "
            "correlation and linear slope. All 55 checkpoint pairs move consistently with their vertices.",
            "null_assumption": "Conditional on the observed comparison matrices and the initial checkpoint, "
            "the temporal labels of learned checkpoints are jointly exchangeable within each seed. This tests "
            "association with learned-checkpoint order; it is not a causal randomized training experiment.",
            "p_definition": "Two-sided p is twice the smaller Monte Carlo tail, capped at one, with "
            "(extreme+1)/(B+1) in each tail. The CP0-fixed null need not be centered at zero.",
            "multiple_testing": "Holm correction across 8 tests at each scope: 2 benchmarks x 2 action metrics "
            "x 2 trend statistics. Also report a conservative family of all 24 seed-specific tests.",
            "scope": "Conditional on observed contexts and common saved histories; no new environment runs "
            "or model calls. Preserves shared-checkpoint dependence, but assumes checkpoint-order "
            "exchangeability under the null and is not inference over new training seeds.",
            "sources": provenance, "tests": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfcl-analysis", type=Path, default=ROOT / "outputs"
                        / "bfcl_same_history_policy_shift_cp5_steps20_queue150_20261011/qwen/analysis")
    parser.add_argument("--appworld-analysis", type=Path, default=ROOT / "outputs"
                        / "appworld_ace_qwen_policy_shift_cp5_restored_history_queue150_20261013/analysis")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    result = run([("BFCL", args.bfcl_analysis), ("AppWorld", args.appworld_analysis)], args.trials, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["tests"]:
        if row["seed"] == "all":
            print(f"{row['benchmark']:<8} {row['metric']:<15} {row['statistic']:<15} "
                  f"value={row['value']:.8f} p={row['p_two_sided']:.6g} "
                  f"Holm(8)={row['p_holm_all_policy_shift_tests']:.6g}")


if __name__ == "__main__":
    main()
