# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Post hoc process diagnostics; keep the six frozen semantic comparisons intact."""

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

from scripts.premise.analyze_bfcl_temporal_selection import write_csv  # noqa: E402
from scripts.premise.analyze_bfcl_trajectory_rewards import (  # noqa: E402
    MODELS,
    REFERENCE,
    _checkpoint_rows,
    _describe,
    _flatten,
    _observed_arrays,
    analyze,
    process_prevalence,
    statistical_tests,
)
from scripts.premise.bfcl_trajectory_selection_rewards import objective_reward_scores  # noqa: E402

RECIPES = (REFERENCE, "all_trajectory_loop_penalty", "paired_success_efficiency")
COHORT = "process_ablation"


def process_ablation_scores(
    success: np.ndarray, turns: np.ndarray, avoidable: np.ndarray,
    parent_success: np.ndarray, parent_turns: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reward only the supplied n tasks; no target/full-set arguments are accepted.

    These two rules were added after inspecting process-feature prevalence, not
    selected by validation lift. Neither changes one-additional-success priority.
    """
    count = objective_reward_scores(success, turns, avoidable)[REFERENCE]
    success, turns, avoidable = (np.asarray(value, float) for value in (success, turns, avoidable))
    parent_shape = (success.shape[0], success.shape[2])
    parent_success, parent_turns = np.asarray(parent_success, float), np.asarray(parent_turns, float)
    if (parent_success.shape != parent_shape or parent_turns.shape != parent_shape
            or not np.isin(parent_success, (0, 1)).all() or not np.isfinite(parent_turns).all()
            or not ((parent_turns >= 0) & (parent_turns == np.floor(parent_turns))).all()):
        raise ValueError("Parent must have the same n tasks, binary success and finite integer turns")
    joint_success = success * parent_success[:, None, :]
    relative_cost = np.clip(
        (parent_turns[:, None, :] - turns) / np.maximum(1, parent_turns[:, None, :]), -1, 1,
    )
    return {
        REFERENCE: count,
        "all_trajectory_loop_penalty": count - .25 * (np.minimum(avoidable, 20) / 20).mean(axis=-1),
        "paired_success_efficiency": count + .25 * (joint_success * relative_cost).mean(axis=-1),
    }


def replay(groups: list[dict], targets: dict, seed: int = 20261021) -> dict:
    """Use only the existing matched trace-complete cohort for both diagnostics."""
    observed = analyze(groups, targets, {})
    coverage = observed["checkpoint_coverage"]
    ready = {(row["model"], row["key"], row["n"]) for row in coverage if row["objective_complete"]}
    indexed = defaultdict(list)
    for group in groups:
        indexed[group["model"], group["key"]].append(group)
    rows = []
    for identity, point_groups in indexed.items():
        point_groups.sort(key=lambda group: group["query_position"])
        for n in range(1, 11):
            if (*identity, n) not in ready:
                continue
            prefix = point_groups[:n]
            success, turns, avoidable, _ = _observed_arrays(prefix)
            scores = process_ablation_scores(success[None, 1:], turns[None, 1:], avoidable[None, 1:],
                                              success[None, 0], turns[None, 0])
            rows.extend(_checkpoint_rows(prefix, targets[identity], {}, COHORT, score_overrides=scores))
    numeric = sorted({key for row in rows for key, value in row.items()
                      if key not in {"seed", "checkpoint", "n"} and isinstance(value, int | float)})
    aggregate, seed_rows = [], []
    for model in MODELS:
        for n in range(1, 11):
            for recipe in RECIPES:
                selected = [row for row in rows if (row["model"], row["n"], row["recipe"]) == (model, n, recipe)]
                for stream in ("all", 42, 43, 44):
                    subset = selected if stream == "all" else [row for row in selected if row["seed"] == stream]
                    if not subset:
                        continue
                    row = {"model": model, "n": n, "recipe": recipe, "cohort": COHORT,
                           "checkpoints": len(subset),
                           **{metric: _describe([item.get(metric) for item in subset]) for metric in numeric}}
                    if stream == "all":
                        aggregate.append(row)
                    else:
                        seed_rows.append({**row, "seed": stream})
    return {
        "created_at": datetime.now(UTC).isoformat(), "status": "post_hoc_process_diagnostics",
        "analysis_phase_model_calls": 0, "new_actor_rollouts": 0,
        "recipes": list(RECIPES), "checkpoint_scores": rows,
        "aggregate": aggregate, "seed_scores": seed_rows, "checkpoint_coverage": coverage,
        "coverage": observed["coverage"], "process_prevalence": process_prevalence(groups),
        "paired_block_tests": statistical_tests(rows, seed, coverage, recipe_families={COHORT: RECIPES}),
        "sampling": "Same frozen uniform ten-task ordering per checkpoint; n uses only its first n tasks.",
        "motivation": "Success-gated avoidable-repeat penalties are almost constant because repeated identical "
                      "tool errors mostly occur on failed runs. Compare penalizing those loops on all runs; "
                      "separately control task difficulty through parent-relative cost on joint successes.",
        "inference": "Exploratory Holm2 comparisons per model and seed-local block setting, plus separate whole-seed "
                     "exact sign flips. These are additional diagnostics, outside the frozen six-recipe family.",
    }


def write_report(summary: dict, output: Path) -> None:
    lines = ["# Additional process reward diagnostics", "",
             "These two exploratory rules were added after observing that almost all avoidable identical-error "
             "loops occurred in failed trajectories. They are separate from the original six reward hypotheses "
             "and do not replace their results. No new actor rollout or LLM call is needed for this replay.", "",
             "K is sampled success count, u is avoidable identical-error repetition, and t is environment steps. "
             "Every rule sees only n sampled training tasks (n=1..10) from the same frozen uniform ten-task "
             "ordering. Parent-relative cost uses the parent on exactly those n tasks. Full validation is read "
             "after selection to measure lift over the mean of the five proposals.", "",
             "* All-run loops: `K - 0.25/n * sum(min(u,20)/20)`, including failed executions.",
             "* Paired cost: `K + 0.25/n * sum(joint_success * clip((t_parent-t)/max(1,t_parent),-1,1))`.", "",
             "One additional sampled success dominates either bounded secondary term. Paired cost only credits "
             "executions where both candidate and parent succeed; short failures receive no efficiency credit. "
             "The loop rule compares process behavior even among failures and does not claim that avoiding a "
             "loop proves task progress.", "",
             "## n=10 results", "",
             "|Model|Rule|CPs|Validation lift (pp)|Delta vs success (pp)|Changed selections|",
             "|---|---|---:|---:|---:|---:|"]
    for model in MODELS:
        subset = [row for row in summary["aggregate"] if row["model"] == model and row["n"] == 10]
        baseline = next(row for row in subset if row["recipe"] == REFERENCE)
        for row in subset:
            lift = row["validation_lift"]["mean"]
            changed = round(row["checkpoints"] * (1 - row["choice_distribution_agreement"]["mean"]))
            lines.append(f"|{model}|{row['recipe']}|{row['checkpoints']}|{100 * lift:+.4f}|"
                         f"{100 * (lift - baseline['validation_lift']['mean']):+.4f}|{changed}|")
    min_adjusted = min(row["p_holm"] for row in summary["paired_block_tests"])
    lines.extend(["", f"The minimum n=10 Holm2-adjusted p-value is {min_adjusted:.4f} across chronological "
                  "seed-local block sizes 2, 3, 4 and 5, and the separate whole-seed sign flips. "
                  "These comparisons are exploratory and do not provide independent confirmation. "
                  "With three independent "
                  "training seeds the exact whole-seed test has only eight patterns (minimum two-sided "
                  "p=0.25). Checkpoint blocks make an additional exchangeability assumption.", "",
                  "Selection changes compare the entire tie-breaking distribution against uniform selection "
                  "among maximal success counts; gains use its exact expectation. Reported execution costs "
                  "are diagnostics on the selected training samples, not independent held-out efficiency "
                  "estimates. Correlations exclude undefined constant candidate vectors and retain their "
                  "defined-checkpoint counts in JSON/CSV. n=1..9 results are descriptive.", ""])
    (output / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    folder = args.folder.resolve()
    output = folder / "process_ablations"
    output.mkdir(exist_ok=True)
    cache_bytes = (folder / "group_features.jsonl").read_bytes()
    groups = [json.loads(line) for line in cache_bytes.decode().splitlines() if line.strip()]
    source = Path(json.loads((folder / "task_groups.json").read_text())["source"])
    target_bytes = source.read_bytes()
    raw = json.loads(target_bytes)
    targets = {(model["label"], point["key"]): point for model in raw["models"] if model["label"] in MODELS
               for point in model["points"]}
    summary = replay(groups, targets, args.seed)
    summary.update({"cache_sha256": hashlib.sha256(cache_bytes).hexdigest(),
                    "source_sha256": hashlib.sha256(target_bytes).hexdigest(),
                    "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (output / "process_ablation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name in ("checkpoint_scores", "aggregate", "seed_scores", "paired_block_tests"):
        write_csv(output / f"{name}.csv", _flatten(summary[name]))
    write_report(summary, output)
    print(json.dumps({"output": str(output), "n10": [
        {"model": row["model"], "recipe": row["recipe"], "checkpoints": row["checkpoints"],
         "validation_lift_pp": 100 * row["validation_lift"]["mean"],
         "choice_agreement": row["choice_distribution_agreement"]["mean"]}
        for row in summary["aggregate"] if row["n"] == 10
    ]}, indent=2))


if __name__ == "__main__":
    main()
