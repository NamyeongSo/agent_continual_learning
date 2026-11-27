# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Conditional, within-seed permutation tests for the saved BFCL n=10 alignment results."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

TARGETS = ("full_train", "full_validation")
SELECTORS = ("train10", "val10")


def unit_centered(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(centered, axis=-1, keepdims=True)
    if (norm == 0).any():
        raise ValueError("Correlation is undefined for constant candidate scores")
    return centered / norm


def paired_points(points: list[dict], target: str) -> list[dict]:
    """Keep the same jointly defined checkpoint cohort as the n=10 figure."""
    keys = [point["key"] for point in points]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate checkpoint keys")
    paired = []
    for point in sorted(points, key=lambda point: (point["seed"], point["checkpoint"])):
        arrays = [np.asarray(point["scores"][key], dtype=float) for key in (*SELECTORS, target)]
        if any(values.shape != (5,) or not np.isfinite(values).all() for values in arrays):
            raise ValueError(f"Expected five finite candidate scores: {point['key']}")
        if all(np.ptp(values) > 0 for values in arrays):
            paired.append(point)
    return paired


def correlation_permutations(points: list[dict], target: str) -> np.ndarray:
    """Permute one full-set target vector jointly against both evaluator vectors."""
    permutations = np.asarray(list(itertools.permutations(range(5))))
    scores = unit_centered(np.asarray([[point["scores"][key] for key in SELECTORS] for point in points]))
    targets = unit_centered(np.asarray([point["scores"][target] for point in points]))
    return np.einsum("psi,pki->pks", scores, targets[:, permutations])


def rng_for(seed: int, key: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "big"))


def association_tests(points: list[dict], target: str, trials: int, seed: int) -> list[dict]:
    table = correlation_permutations(points, target)
    seeds = sorted({point["seed"] for point in points})
    scopes = [(str(value), [i for i, point in enumerate(points) if point["seed"] == value]) for value in seeds]
    scopes.append(("all", list(range(len(points)))))
    observed = {label: table[indices, 0].mean(axis=0) for label, indices in scopes}
    counts = {label: np.zeros(2, dtype=np.int64) for label, _ in scopes}
    rng = rng_for(seed, target)
    columns = np.arange(len(points))[None, :]
    for start in range(0, trials, 10000):
        draws = rng.integers(120, size=(min(10000, trials - start), len(points)))
        sampled = table[columns, draws]
        for label, indices in scopes:
            statistic = sampled[:, indices].mean(axis=1)
            counts[label] += (np.abs(statistic) >= np.abs(observed[label]) - 1e-12).sum(axis=0)
    rows = []
    for label, indices in scopes:
        for index, selector in enumerate(SELECTORS):
            rows.append({"target": target, "selector": selector, "seed": label,
                         "checkpoints": len(indices), "mean_rho": float(observed[label][index]),
                         "extreme_permutations": int(counts[label][index]), "permutations": trials,
                         "p_two_sided": float((counts[label][index] + 1) / (trials + 1))})
    return rows


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    previous = 0.0
    for index, position in enumerate(order):
        previous = max(previous, min(1.0, values[position] * (len(values) - index)))
        adjusted[position] = previous
    return adjusted.tolist()


def paired_block_contributions(
    all_points: list[dict], valid_points: list[dict], differences: np.ndarray, block_size: int,
) -> tuple[np.ndarray, list[dict]]:
    """Define chronological blocks before excluding undefined checkpoints."""
    if block_size < 1:
        raise ValueError("Block size must be positive")
    values = {point["key"]: float(value) for point, value in zip(valid_points, differences, strict=True)}
    blocks = []
    for seed in sorted({point["seed"] for point in all_points}):
        ordered = sorted(
            (point for point in all_points if point["seed"] == seed), key=lambda point: point["checkpoint"],
        )
        for offset in range(0, len(ordered), block_size):
            block = ordered[offset:offset + block_size]
            valid = [point for point in block if point["key"] in values]
            if valid:
                blocks.append({"seed": seed, "all_checkpoint_keys": [point["key"] for point in block],
                               "valid_checkpoint_keys": [point["key"] for point in valid],
                               "contribution": sum(values[point["key"]] for point in valid) / len(valid_points)})
    return np.asarray([block["contribution"] for block in blocks]), blocks


def sign_flip_test(contributions: np.ndarray, trials: int, rng: np.random.Generator) -> dict:
    contributions = np.asarray(contributions, dtype=float)
    if contributions.ndim != 1 or not contributions.size or not np.isfinite(contributions).all():
        raise ValueError("Expected nonempty finite block contributions")
    observed = float(contributions.sum())
    exact = contributions.size <= 20
    count = 2 ** contributions.size if exact else trials
    extreme = 0
    bit_indices = np.arange(contributions.size, dtype=np.uint64)
    for start in range(0, count, 10000):
        size = min(10000, count - start)
        if exact:
            numbers = np.arange(start, start + size, dtype=np.uint64)[:, None]
            signs = 2 * ((numbers >> bit_indices) & 1).astype(float) - 1
        else:
            signs = 2 * rng.integers(2, size=(size, contributions.size)) - 1
        statistic = np.einsum("ij,j->i", signs, contributions)
        extreme += int((np.abs(statistic) >= abs(observed) - 1e-12).sum())
    return {"mean_rho_difference": observed, "blocks": int(contributions.size), "exact": exact,
            "permutations": count, "extreme_permutations": extreme,
            "p_two_sided": extreme / count if exact else (extreme + 1) / (count + 1)}


def analyze(model: dict, trials: int, seed: int, block_sizes: list[int]) -> dict:
    if trials < 1:
        raise ValueError("Permutation trials must be positive")
    points = model["checkpoints"]
    associations, comparisons = [], []
    for target in TARGETS:
        valid = paired_points(points, target)
        if not valid:
            raise ValueError(f"No jointly valid checkpoints: {target}")
        table = correlation_permutations(valid, target)
        aggregate = model["sizes"][-1]["targets"][target]
        if model["sizes"][-1]["n"] != 10 or len(valid) != aggregate["paired_checkpoint_count_mean"]:
            raise ValueError(f"n=10 cohort differs from the figure: {target}")
        for index, selector in enumerate(SELECTORS):
            if not np.isclose(table[:, 0, index].mean(), aggregate[selector]["mean_pearson"], atol=1e-12, rtol=0):
                raise ValueError(f"n=10 correlation differs from the figure: {target}/{selector}")
        associations.extend(association_tests(valid, target, trials, seed))
        differences = table[:, 0, 0] - table[:, 0, 1]
        aligned = "train10"
        if target == "full_validation":
            differences = -differences
            aligned = "val10"
        for scope in [*sorted({point["seed"] for point in points}), "all"]:
            selected = [i for i, point in enumerate(valid) if scope == "all" or point["seed"] == scope]
            scope_points = [point for point in points if scope == "all" or point["seed"] == scope]
            for block_size in block_sizes:
                contributions, blocks = paired_block_contributions(
                    scope_points, [valid[i] for i in selected], differences[selected], block_size,
                )
                result = sign_flip_test(contributions, trials, rng_for(seed, f"{target}:{scope}:{block_size}"))
                comparisons.append({"target": target, "seed": str(scope), "aligned_selector": aligned,
                                    "checkpoints": len(selected), "block_size": block_size,
                                    **result, "block_details": blocks})
    for scope in sorted({row["seed"] for row in associations}):
        family = [row for row in associations if row["seed"] == scope]
        for row, adjusted in zip(family, holm_adjust([row["p_two_sided"] for row in family]), strict=True):
            row["p_holm_four_associations"] = adjusted
    seed_rows = [row for row in associations if row["seed"] != "all"]
    for row, adjusted in zip(seed_rows, holm_adjust([row["p_two_sided"] for row in seed_rows]), strict=True):
        row["p_holm_all_seed_associations"] = adjusted
    for scope in sorted({row["seed"] for row in comparisons}):
        for block_size in block_sizes:
            family = [row for row in comparisons if row["seed"] == scope and row["block_size"] == block_size]
            for row, adjusted in zip(family, holm_adjust([row["p_two_sided"] for row in family]), strict=True):
                row["p_holm_two_target_comparisons"] = adjusted
    return {"model": model["label"], "n": 10, "checkpoints": len(points), "rng_seed": seed,
            "association_permutations": trials,
            "association_method": "Within each checkpoint, permute the five full-set candidate scores jointly "
            "against both evaluator score vectors. Average checkpoint rho separately inside each seed, and "
            "also compute the checkpoint-weighted aggregate without moving data across seeds or checkpoints.",
            "association_null": "Conditional on the fixed score vectors, candidate-to-target pairings are "
            "exchangeable under independent within-checkpoint permutations. This is a no-association null, "
            "stronger than a weak null of zero Pearson correlation alone.",
            "comparison_method": "Compare aligned minus cross-distribution rho at the same checkpoint. "
            "Swap both evaluator labels jointly for all checkpoints in each chronological block within a seed. "
            "Blocks are formed from all observed checkpoints, before excluding undefined correlations.",
            "comparison_null": "Block difference vectors are jointly sign-symmetric and exchangeable under "
            "independent block sign flips. Arbitrary within-block dependence is retained; dependence between "
            "different blocks is not guaranteed to be retained. Block-size sensitivity is not a model-free "
            "solution to serial dependence.",
            "p_definition": "Two-sided absolute-statistic tail with tolerance 1e-12; exact enumeration when "
            "at most 20 paired sign blocks; Monte Carlo uses (extreme+1)/(permutations+1).",
            "scope": "Conditional evidence for the observed candidate sets and task pools, not inference "
            "over new training seeds. Full targets include their respective ten-task evaluator pools; "
            "positive same-distribution correlations include an overlap component. A nonsignificant "
            "cross-distribution test does not establish equivalence to zero.",
            "associations": associations, "paired_rho_comparisons": comparisons}


def gain_measurements(model: dict) -> list[dict]:
    """Parent-relative gains and proposal-relative lifts use every checkpoint, including score ties."""
    points = model["checkpoints"]
    measurements = []
    for target in TARGETS:
        values = np.asarray([point["scores"][target] for point in points], dtype=float)
        if values.shape != (len(points), 5) or not np.isfinite(values).all():
            raise ValueError(f"Expected five finite full-set candidate scores: {target}")
        proposal = values.mean(axis=1)
        parents = [point["parent_scores"][target] for point in points]
        have_parents = all(value is not None for value in parents)
        if have_parents:
            parent = np.asarray(parents, dtype=float)
            measurements.append({"target": target, "metric": "proposal_mean_gain_vs_parent", "selector": None,
                                 "values": proposal - parent})
        selected = {}
        for selector in SELECTORS:
            scores = np.asarray([point["scores"][selector] for point in points], dtype=float)
            if scores.shape != values.shape or not np.isfinite(scores).all():
                raise ValueError(f"Expected five finite evaluator candidate scores: {selector}")
            tied = scores == scores.max(axis=1, keepdims=True)
            selected[selector] = (tied * values).sum(axis=1) / tied.sum(axis=1)
            if have_parents:
                measurements.append({"target": target, "metric": "selected_gain_vs_parent", "selector": selector,
                                     "values": selected[selector] - parent})
            measurements.append({"target": target, "metric": "selection_lift_vs_proposal_mean", "selector": selector,
                                 "values": selected[selector] - proposal})
            reference = model["sizes"][-1]["targets"][target][selector]
            if not np.isclose(selected[selector].mean(), reference["mean_selected_performance"], atol=1e-12, rtol=0):
                raise ValueError(f"n=10 selected performance differs from the figure: {target}/{selector}")
        aligned, cross = ("train10", "val10") if target == "full_train" else ("val10", "train10")
        measurements.append({"target": target, "metric": "aligned_minus_cross_selected_performance",
                             "selector": aligned, "values": selected[aligned] - selected[cross]})
    return measurements


def analyze_gains(model: dict, trials: int, seed: int, block_sizes: list[int]) -> dict:
    if trials < 1:
        raise ValueError("Permutation trials must be positive")
    points = model["checkpoints"]
    if model["sizes"][-1]["n"] != 10:
        raise ValueError("Expected n=10 as the final sample size")
    measurements = gain_measurements(model)
    rows = []
    for scope in [*sorted({point["seed"] for point in points}), "all"]:
        indices = [i for i, point in enumerate(points) if scope == "all" or point["seed"] == scope]
        subset = [points[i] for i in indices]
        for measurement in measurements:
            values = measurement["values"][indices]
            for block_size in block_sizes:
                contributions, blocks = paired_block_contributions(subset, subset, values, block_size)
                key = f"gain:{measurement['target']}:{measurement['metric']}:{measurement['selector']}"
                test = sign_flip_test(contributions, trials, rng_for(seed, f"{key}:{scope}:{block_size}"))
                effect = test.pop("mean_rho_difference")
                rows.append({"target": measurement["target"], "metric": measurement["metric"],
                             "selector": measurement["selector"], "seed": str(scope), "block_size": block_size,
                             "checkpoints": len(indices), "mean_effect": effect, "mean_effect_pp": 100 * effect,
                             **test, "block_details": blocks})
    for scope in sorted({row["seed"] for row in rows}):
        for block_size in block_sizes:
            family = [row for row in rows if row["seed"] == scope and row["block_size"] == block_size]
            adjusted = holm_adjust([row["p_two_sided"] for row in family])
            for row, value in zip(family, adjusted, strict=True):
                row["holm_family_size"] = len(family)
                row["p_holm_all_gain_tests"] = value
    return {"model": model["label"], "n": 10, "checkpoints": len(points), "rng_seed": seed,
            "maximum_monte_carlo_trials": trials,
            "method": "Within-seed chronological block sign-flip tests on checkpoint-level differences. "
            "Parent-relative gain, proposal-relative selection lift, and paired aligned-minus-cross selection "
            "are distinct contrasts. Constant evaluator scores retain uniform tie-breaking and all checkpoints.",
            "null_assumption": "Under the relevant zero-effect null, checkpoint difference vectors in "
            "different blocks are exchangeable under independent sign flips. Within-block dependence is retained. "
            "This requires block sign symmetry and does not allow arbitrary dependence between different blocks.",
            "scope": "Observed checkpoints/candidates/task pools; not a test over new training seeds. "
            "Validation parent performance was not measured, so validation parent-relative gain cannot be tested. "
            "Non-rejection for proposal mean gain does not establish equivalence to zero.",
            "multiple_testing": "Holm correction across all available gain contrasts at each scope/block size. "
            "Block sizes are sensitivity analyses, not independent replications.",
            "checkpoint_effects": [{"target": row["target"], "metric": row["metric"], "selector": row["selector"],
                                    "values": row["values"].tolist()} for row in measurements],
            "tests": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model", default="Qwen")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[1, 2, 3, 4, 5, 1000])
    parser.add_argument("--gain-only", action="store_true")
    args = parser.parse_args()
    data = args.summary.read_bytes()
    summary = json.loads(data)
    model = next(model for model in summary["models"] if model["label"] == args.model)
    result = (analyze_gains if args.gain_only else analyze)(model, args.trials, args.seed, args.block_sizes)
    result["source"] = str(args.summary.resolve())
    result["source_sha256"] = hashlib.sha256(data).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.gain_only:
        for row in result["tests"]:
            if row["seed"] == "all" and row["block_size"] in (3, 1000):
                print(f"{row['target']:<15} {row['metric']:<44} {row['selector']!s:>7} "
                      f"block={row['block_size']:4d} effect={row['mean_effect_pp']:+.6f}pp "
                      f"p={row['p_two_sided']:.6g} Holm={row['p_holm_all_gain_tests']:.6g}")
        return
    for row in result["associations"]:
        print(f"{row['seed']:>3} {row['selector']:>7} -> {row['target']:<15} "
              f"rho={row['mean_rho']:+.6f} p={row['p_two_sided']:.6g} "
              f"Holm(4)={row['p_holm_four_associations']:.6g}")
    for row in result["paired_rho_comparisons"]:
        if row["seed"] == "all":
            print(f"all {row['target']:<15} block={row['block_size']:4d} "
                  f"difference={row['mean_rho_difference']:.6f} p={row['p_two_sided']:.6g} "
                  f"Holm(2)={row['p_holm_two_target_comparisons']:.6g}")


if __name__ == "__main__":
    main()
