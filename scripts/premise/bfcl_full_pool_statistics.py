# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Paired exploratory inference for the fixed 48-method, n=10 full-train-pool study.

Checkpoint task-draw averages are the observations. Draws are never counted as
independent repetitions. Leave-one-seed-out selection reuses an inspected archive
and is a retrospective stability diagnostic, rather than a fresh validation test.
"""

from __future__ import annotations

import numpy as np

from scripts.premise.analyze_bfcl_alignment_significance import (
    holm_adjust,
    paired_block_contributions,
    rng_for,
    sign_flip_test,
)
from scripts.premise.bfcl_full_pool_selection import STRATEGIES

MODELS = ("Qwen", "Gemma")
SEEDS = (42, 43, 44)
REFERENCE = "uniform:mean"


def _validated_models(rows: list[dict]) -> list[tuple]:
    """Require one complete fixed method family on exactly paired checkpoints."""
    if any(row["model"] not in MODELS for row in rows):
        raise ValueError("Only Qwen/Gemma are in scope")
    output = []
    for model in MODELS:
        subset = [row for row in rows if row["model"] == model]
        if not subset:
            continue
        samplers = list(dict.fromkeys(row["sampler"] for row in subset))
        rewards = list(dict.fromkeys(row["reward"] for row in subset))
        if set(samplers) != set(STRATEGIES) or len(rewards) != 12:
            raise ValueError("Expected all four samplers and twelve prespecified rewards")
        methods = [(sampler, reward) for sampler in samplers for reward in rewards]
        by_method = {f"{sampler}:{reward}": {} for sampler, reward in methods}
        for row in subset:
            if row["seed"] not in SEEDS or ("n" in row and row["n"] != 10):
                raise ValueError("Only n=10 checkpoint rows from seeds 42/43/44 are accepted")
            if not np.isfinite(row["validation_lift"]):
                raise ValueError("Checkpoint validation lift must be finite")
            selected = by_method[f"{row['sampler']}:{row['reward']}"]
            if row["key"] in selected:
                raise ValueError("Duplicate checkpoint/method row")
            selected[row["key"]] = row
        if REFERENCE not in by_method:
            raise ValueError("The uniform:mean reference is missing")
        baseline = by_method[REFERENCE]
        if {row["seed"] for row in baseline.values()} != set(SEEDS):
            raise ValueError("Every model must contain checkpoints from all three seeds")
        for indexed in by_method.values():
            if set(indexed) != set(baseline):
                raise ValueError("All 48 methods must use the same checkpoint keys")
            if any((row["seed"], row["checkpoint"]) != (baseline[key]["seed"], baseline[key]["checkpoint"])
                   for key, row in indexed.items()):
                raise ValueError("Checkpoint seed/time identity differs between methods")
        output.append((model, methods, by_method))
    return output


def statistical_tests(rows: list[dict], seed: int) -> list[dict]:
    """Compare 47 alternatives with uniform mean; apply Holm separately per model/block size."""
    output = []
    for model, methods, by_method in _validated_models(rows):
        baseline = sorted(by_method[REFERENCE].values(), key=lambda row: (row["seed"], row["checkpoint"], row["key"]))
        for block_size in (2, 3, 4, 5):
            family = []
            for sampler, reward in methods:
                method = f"{sampler}:{reward}"
                if method == REFERENCE:
                    continue
                differences = np.asarray([
                    by_method[method][row["key"]]["validation_lift"] - row["validation_lift"] for row in baseline
                ])
                contributions, _ = paired_block_contributions(baseline, baseline, differences, block_size)
                test = sign_flip_test(contributions, 100000, rng_for(seed, f"fullpool:{model}:{method}:{block_size}"))
                effect = test.pop("mean_rho_difference")
                family.append({"model": model, "method": method, "sampler": sampler, "reward": reward,
                               "reference": REFERENCE, "n": 10, "block_size": block_size,
                               "difference": effect, "checkpoints": len(baseline), **test})
            adjusted = holm_adjust([row["p_two_sided"] for row in family])
            for row, p_value in zip(family, adjusted, strict=True):
                row["p_holm"] = p_value
                row["family_size"] = len(family)
            output.extend(family)
    return output


def leave_one_seed_out(rows: list[dict]) -> list[dict]:
    """Choose by two development-seed means; evaluate the third seed only after choosing."""
    output = []
    for model, methods, by_method in _validated_models(rows):
        choices = [f"{sampler}:{reward}" for sampler, reward in methods]
        for held in SEEDS:
            development = {
                method: float(np.mean([
                    np.mean([row["validation_lift"] for row in by_method[method].values() if row["seed"] == other])
                    for other in SEEDS if other != held
                ])) for method in choices
            }
            chosen = max(choices, key=lambda method: development[method])
            sampler, reward = next((sampler, reward) for sampler, reward in methods if f"{sampler}:{reward}" == chosen)
            held_rows = [row for row in by_method[chosen].values() if row["seed"] == held]
            reference = [row for row in by_method[REFERENCE].values() if row["seed"] == held]
            held_lift = float(np.mean([row["validation_lift"] for row in held_rows]))
            reference_lift = float(np.mean([row["validation_lift"] for row in reference]))
            output.append({"model": model, "held_out_seed": held, "method": chosen,
                           "sampler": sampler, "reward": reward, "development_lift": development[chosen],
                           "held_out_lift": held_lift, "reference_lift": reference_lift,
                           "difference": held_lift - reference_lift, "checkpoints": len(held_rows),
                           "retrospective": True})
    return output
