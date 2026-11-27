# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Analyze frozen BFCL trajectory rewards; read cached traces/judgments without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
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
from scripts.premise.analyze_bfcl_temporal_selection import write_csv  # noqa: E402
from scripts.premise.bfcl_trajectory_reward_judge import RISK_METRICS  # noqa: E402
from scripts.premise.bfcl_trajectory_selection_rewards import (  # noqa: E402
    LEXICOGRAPHIC_KEYS,
    REWARD_KEYS,
    choice_probabilities,
    objective_reward_scores,
    ordering_ranks,
    reward_scores,
)
from scripts.premise.cache_bfcl_trajectory_reward_features import CONTEXTS  # noqa: E402

MODELS = ("Qwen", "Gemma")
REFERENCE = "success_count"
OBJECTIVE = (REFERENCE, "success_efficiency", "successful_recovery")
PROCESS_FEATURES = {
    "environment_steps": "selected_environment_steps", "tool_invocations": "selected_tool_invocations",
    "tool_error_observations": "selected_tool_error_observations", "invalid_actions": "selected_invalid_actions",
    "avoidable_error_repeat_calls": "selected_avoidable_error_repeats",
    "visible_turn_advances": "selected_visible_turn_advances",
}


def _describe(values: list) -> dict:
    numbers = np.asarray([value for value in values if value is not None and np.isfinite(value)], float)
    if not len(numbers):
        return {"mean": None, "std": None, "min": None, "max": None, "defined_checkpoints": 0}
    return {"mean": float(numbers.mean()), "std": float(numbers.std()), "min": float(numbers.min()),
            "max": float(numbers.max()), "defined_checkpoints": len(numbers)}


def _correlation(first, second) -> float | None:
    first, second = np.asarray(first, float), np.asarray(second, float)
    if first.std() < 1e-12 or second.std() < 1e-12:
        return None
    return float(np.clip(np.corrcoef(first, second)[0, 1], -1, 1))


def _retryable_infrastructure(record: dict) -> bool:
    """Match the driver's three actual completion attempts, excluding failed tokenization."""
    if record.get("ok") or record.get("status") != "judge_transport_error":
        return False
    sent = 0
    for attempt in record.get("attempts", []):
        result = attempt["result"]
        started = result.get("chat_request_started")
        if started is None:
            started = (result.get("prompt_tokens_counted", 24001) <= 24000
                       and result.get("status") != "input_budget_exceeded")
        sent += bool(started)
    return sent < 3


def _validated_judgments(group: dict, record: dict | None) -> dict | None:
    if not record or not record.get("ok"):
        return None
    parsed = record.get("result", record.get("judge_result", record))
    if not parsed.get("ok") or set(parsed.get("judgments", {})) != set(CONTEXTS):
        return None
    judgments = parsed["judgments"]
    turns = len(group["public_task"]["user_turns"])
    for context in CONTEXTS:
        metrics = judgments[context].get("metrics", {})
        fulfillment = metrics.get("fulfillment", {})
        lower = fulfillment.get("lower_bound")
        if (not isinstance(lower, int | float) or not np.isfinite(lower) or not 0 <= lower <= 1
                or fulfillment.get("shared_turn_count") != turns
                or not isinstance(fulfillment.get("unknown_turns"), int)
                or not 0 <= fulfillment["unknown_turns"] <= turns
                or "material_violation" not in metrics
                or metrics["material_violation"].get("value") not in (0, 1, None)):
            return None
    return judgments


def _observed_arrays(groups: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[dict]]]:
    by_context = [[next(trace for trace in group["traces"] if trace["context"] == context)
                   for group in groups] for context in CONTEXTS]
    success = np.asarray([[trace["official_success"] for trace in traces] for traces in by_context], float)
    turns = np.asarray([[trace["features"]["environment_steps"] for trace in traces] for traces in by_context], float)
    avoidable = np.asarray([[trace["features"]["avoidable_error_repeat_calls"] for trace in traces]
                            for traces in by_context], float)
    return success, turns, avoidable, by_context


def _semantic_arrays(groups: list[dict], judgments: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    progress, material, progress_unknown, material_unknown = [], [], [], []
    for context in CONTEXTS:
        metrics = [judgments[group["id"]][context]["metrics"] for group in groups]
        progress.append([row["fulfillment"]["lower_bound"] for row in metrics])
        material.append([1 if row["material_violation"]["value"] == 1 else 0 for row in metrics])
        progress_unknown.append([row["fulfillment"]["unknown_turns"] / row["fulfillment"]["shared_turn_count"]
                                 for row in metrics])
        material_unknown.append([row["material_violation"]["value"] is None for row in metrics])
    return tuple(np.asarray(array, float) for array in (progress, material, progress_unknown, material_unknown))


def _checkpoint_rows(
    groups: list[dict], target: dict, judgments: dict, cohort: str, *, score_overrides: dict | None = None,
) -> list[dict]:
    success, turns, avoidable, traces = _observed_arrays(groups)
    semantic = cohort == "primary"
    if semantic:
        progress, material, q_unknown, h_unknown = _semantic_arrays(groups, judgments)
        scores = reward_scores(success[None, 1:], turns[None, 1:], progress[None, 1:], material[None, 1:],
                               avoidable[None, 1:], success[None, 0], turns[None, 0],
                               progress[None, 0], material[None, 0])
    else:
        scores = objective_reward_scores(success[None, 1:], turns[None, 1:], avoidable[None, 1:])
    if score_overrides is not None:
        if semantic or REFERENCE not in score_overrides:
            raise ValueError("Only nonsemantic process ablations may override scores; baseline is mandatory")
        scores = score_overrides
    baseline_probability = choice_probabilities(scores[REFERENCE])[0]
    success_counts = success[1:].sum(axis=-1)
    tied_pairs = int(sum(success_counts[first] == success_counts[second]
                         for first in range(5) for second in range(first + 1, 5)))
    baseline_choice_count = int(np.count_nonzero(baseline_probability))
    first, rows = groups[0], []
    full_train, full_validation = (np.asarray(target["scores"][name], float)
                                   for name in ("full_train", "full_validation"))
    for recipe, score in scores.items():
        probability, ranks = choice_probabilities(score)[0], ordering_ranks(score)[0]
        selected_success = float(probability @ success[1:].mean(axis=-1))
        success_mass = float((probability[:, None] * success[1:]).sum())
        row = {"model": first["model"], "key": first["key"], "seed": first["seed"],
               "checkpoint": first["checkpoint"], "n": len(groups), "recipe": recipe, "cohort": cohort,
               "score_type": "lexicographic" if recipe in LEXICOGRAPHIC_KEYS else "scalar",
               "validation_lift": float(probability @ full_validation - full_validation.mean()),
               "validation_score": float(probability @ full_validation),
               "train_parent_gain": float(probability @ full_train - target["parent_scores"]["full_train"]),
               "train_proposal_lift": float(probability @ full_train - full_train.mean()),
               "proposal_mean_train_gain": float(full_train.mean() - target["parent_scores"]["full_train"]),
               "proposal_mean_validation_score": float(full_validation.mean()),
               "sample_success_rate": selected_success, "parent_sample_success_rate": float(success[0].mean()),
               "selected_sample_regressions": float(probability @ ((success[0] == 1) & (success[1:] == 0)).sum(-1)),
               "selected_sample_repairs": float(probability @ ((success[0] == 0) & (success[1:] == 1)).sum(-1)),
               "choice_overlap": float(np.minimum(probability, baseline_probability).sum()),
               "baseline_tied_candidate_pairs": tied_pairs,
               "baseline_best_candidate_tie_count": baseline_choice_count,
               "baseline_has_best_tie": int(baseline_choice_count > 1),
               "selected_candidate_tie_count": int(np.count_nonzero(probability)),
               "choice_distribution_agreement": float(np.allclose(probability, baseline_probability, atol=1e-12)),
               "selection_total_variation_vs_baseline": float(np.abs(probability - baseline_probability).sum() / 2),
               "selected_successful_environment_steps": float((probability[:, None] * success[1:] * turns[1:]).sum()
                                                              / success_mass) if success_mass else None,
               "candidate_scores": score[0].tolist(), "candidate_ranks": ranks.tolist(),
               "choice_probabilities": probability.tolist(), "full_train_scores": full_train.tolist(),
               "full_validation_scores": full_validation.tolist(), "query_ids": [group["id"] for group in groups]}
        for name, destination in PROCESS_FEATURES.items():
            values = np.asarray([[trace["features"][name] for trace in context] for context in traces[1:]], float)
            row[destination] = float(probability @ values.mean(axis=-1))
        for name, values in (("full_train", full_train), ("full_validation", full_validation)):
            target_ranks = ordering_ranks(values[None])[0]
            row[f"rho_{name}"] = None if recipe in LEXICOGRAPHIC_KEYS else _correlation(score[0], values)
            row[f"spearman_{name}"] = _correlation(ranks, target_ranks)
        if semantic:
            row.update({"selected_progress_lower_bound": float(probability @ progress[1:].mean(axis=-1)),
                        "selected_confirmed_material_rate": float(probability @ material[1:].mean(axis=-1)),
                        "selected_progress_unknown_turn_fraction": float(probability @ q_unknown[1:].mean(axis=-1)),
                        "selected_material_unknown_fraction": float(probability @ h_unknown[1:].mean(axis=-1))})
            for name in RISK_METRICS:
                numer, denom = 0.0, 0.0
                for candidate, context in enumerate(CONTEXTS[1:]):
                    for group in groups:
                        value = judgments[group["id"]][context]["metrics"].get(name, {}).get("value")
                        if value is not None:
                            numer += probability[candidate] * value
                            denom += probability[candidate]
                row[f"selected_{name}_known_mean"] = numer / denom if denom else None
                row[f"selected_{name}_known_fraction"] = denom / len(groups)
        rows.append(row)
    return rows


def analyze(groups: list[dict], targets: dict, judge_results: dict) -> dict:
    """Join frozen task groups to evaluation-only targets and form n-specific complete-case cohorts."""
    grouped, valid_judgments = defaultdict(list), {}
    for group in groups:
        if group["model"] not in MODELS:
            raise ValueError("Only Qwen/Gemma are in scope")
        identity = (group["model"], group["key"])
        if identity not in targets:
            raise ValueError("Frozen query checkpoint has no full-set evaluation target")
        target = targets[identity]
        index = group["train_order_index"]
        if target["train_order"][index] != group["task_id"]:
            raise ValueError("Frozen task identity differs from the target training order")
        if {trace["context"] for trace in group["traces"]} != set(CONTEXTS) or len(group["traces"]) != 6:
            raise ValueError("Every task group must contain exactly the parent and five candidates")
        for number, context in enumerate(CONTEXTS[1:]):
            trace = next(row for row in group["traces"] if row["context"] == context)
            if trace["official_success"] != target["train_rewards"][number][index]:
                raise ValueError("Sampled training reward differs from its cached evaluation target identity")
        grouped[identity].append(group)
        judgment = _validated_judgments(group, judge_results.get(group["id"]))
        if judgment is not None:
            valid_judgments[group["id"]] = judgment
    rows, coverage = [], []
    for identity, point_groups in grouped.items():
        point_groups.sort(key=lambda group: group["query_position"])
        if [group["query_position"] for group in point_groups] != list(range(10)):
            raise ValueError("Each checkpoint needs one frozen query ordering of exactly ten tasks")
        if len({group["task_id"] for group in point_groups}) != 10:
            raise ValueError("Frozen query prefixes must sample distinct training tasks")
        for n in range(1, 11):
            prefix = point_groups[:n]
            objective_ready = all(trace["available"] and trace.get("features") is not None
                                  for group in prefix for trace in group["traces"])
            semantic_ready = objective_ready and all(group["id"] in valid_judgments for group in prefix)
            coverage.append({"model": identity[0], "key": identity[1], "seed": prefix[0]["seed"],
                             "checkpoint": prefix[0]["checkpoint"], "n": n,
                             "objective_complete": objective_ready, "primary_complete": semantic_ready,
                             "missing_trace_groups": sum(not all(trace["available"] for trace in group["traces"])
                                                         for group in prefix),
                             "missing_or_invalid_judgment_groups": sum(group["id"] not in valid_judgments
                                                                       for group in prefix)})
            if objective_ready:
                rows.extend(_checkpoint_rows(prefix, targets[identity], valid_judgments, "objective"))
            if semantic_ready:
                rows.extend(_checkpoint_rows(prefix, targets[identity], valid_judgments, "primary"))
    aggregate, seed_scores = [], []
    excluded = {"seed", "checkpoint", "n"}
    numeric = sorted({key for row in rows for key, value in row.items()
                      if key not in excluded and isinstance(value, int | float)})
    for model in MODELS:
        for n in range(1, 11):
            for cohort, recipes in (("primary", REWARD_KEYS), ("objective", OBJECTIVE)):
                for recipe in recipes:
                    chosen = [row for row in rows if (row["model"], row["n"], row["cohort"], row["recipe"])
                              == (model, n, cohort, recipe)]
                    if not chosen:
                        continue
                    for seed in ("all", 42, 43, 44):
                        subset = chosen if seed == "all" else [row for row in chosen if row["seed"] == seed]
                        if not subset:
                            continue
                        record = {"model": model, "n": n, "cohort": cohort, "recipe": recipe,
                                  "checkpoints": len(subset),
                                  **{metric: _describe([row.get(metric) for row in subset]) for metric in numeric}}
                        if seed == "all":
                            aggregate.append(record)
                        else:
                            seed_scores.append({**record, "seed": seed})
    coverage_aggregate = []
    for model in MODELS:
        for n in range(1, 11):
            chosen = [row for row in coverage if row["model"] == model and row["n"] == n]
            if chosen:
                coverage_aggregate.append({"model": model, "n": n, "expected_checkpoints": len(chosen),
                                           "objective_checkpoints": sum(row["objective_complete"] for row in chosen),
                                           "primary_checkpoints": sum(row["primary_complete"] for row in chosen)})
    return {"checkpoint_scores": rows, "aggregate": aggregate, "seed_scores": seed_scores,
            "checkpoint_coverage": coverage, "coverage": coverage_aggregate,
            "judged_groups_valid": len(valid_judgments), "query_groups": len(groups)}


def process_prevalence(groups: list[dict]) -> list[dict]:
    """Describe observed candidate features without using them to alter a reward."""
    result = []
    for model in MODELS:
        traces = [trace for group in groups if group["model"] == model for trace in group["traces"]
                  if trace["context"] != "parent" and trace["available"] and trace.get("features") is not None]
        for success in (0, 1):
            subset = [trace for trace in traces if trace["official_success"] == success]
            if not subset:
                continue
            for feature in (*PROCESS_FEATURES, "immediate_exact_repeat_calls"):
                values = [trace["features"][feature] for trace in subset]
                result.append({"model": model, "official_success": success, "feature": feature,
                               "traces": len(subset), "sum": int(sum(values)),
                               "positive_traces": sum(value > 0 for value in values),
                               "mean": float(np.mean(values))})
    return result


def statistical_tests(
    rows: list[dict], seed: int, coverage: list[dict], *, include_primary: bool = False,
    recipe_families: dict[str, tuple[str, ...]] | None = None,
) -> list[dict]:
    """Test selection lift with seed-local chronological blocks defined before exclusions.

    Trace-only tests form a family of two contrasts; the primary semantic family
    has six. Never mix the two different matched checkpoint populations in a test.
    """
    output = []
    families = recipe_families or {"objective": OBJECTIVE, "primary": REWARD_KEYS}
    for model in MODELS:
        all_points = [row for row in coverage if row["model"] == model and row["n"] == 10]
        for cohort, recipes in families.items():
            if cohort == "primary" and not include_primary:
                continue
            selected = [row for row in rows if row["model"] == model and row["n"] == 10
                        and row["cohort"] == cohort]
            baseline = sorted([row for row in selected if row["recipe"] == REFERENCE],
                              key=lambda row: (row["seed"], row["checkpoint"]))
            if not baseline:
                continue
            for block in (2, 3, 4, 5, "whole_seed"):
                family = []
                for recipe in recipes[1:]:
                    compared = {row["key"]: row for row in selected if row["recipe"] == recipe}
                    if set(compared) != {row["key"] for row in baseline}:
                        raise ValueError("Reward tests require exactly matched checkpoint cohorts")
                    differences = np.asarray([compared[row["key"]]["validation_lift"] - row["validation_lift"]
                                              for row in baseline])
                    if block == "whole_seed":
                        blocks = [{"seed": stream,
                                   "all_checkpoint_keys": [row["key"] for row in all_points if row["seed"] == stream],
                                   "valid_checkpoint_keys": [row["key"] for row in baseline if row["seed"] == stream],
                                   "contribution": float(sum(value for row, value in zip(
                                       baseline, differences, strict=True,
                                   ) if row["seed"] == stream) / len(baseline))}
                                  for stream in sorted({row["seed"] for row in baseline})]
                        contributions = np.asarray([row["contribution"] for row in blocks])
                    else:
                        contributions, blocks = paired_block_contributions(all_points, baseline, differences, block)
                    test = sign_flip_test(contributions, 100000,
                                          rng_for(seed, f"trajectory:{model}:{cohort}:{recipe}:{block}"))
                    difference = test.pop("mean_rho_difference")
                    family.append({"model": model, "cohort": cohort, "n": 10, "recipe": recipe,
                                   "reference": REFERENCE, "block_size": block, "difference": difference,
                                   "checkpoints": len(baseline), "seeds": sorted({row["seed"] for row in baseline}),
                                   "test_unit": "seed" if block == "whole_seed" else "seed_local_checkpoint_block",
                                   "chronological_blocks": blocks, **test})
                for row, p in zip(family, holm_adjust([row["p_two_sided"] for row in family]), strict=True):
                    row.update({"p_holm": p, "family_size": len(family)})
                output.extend(family)
    return output


def _flatten(rows: list[dict]) -> list[dict]:
    flattened = [{name: number for key, value in row.items() for name, number in
                  ([(f"{key}_{stat}", number) for stat, number in value.items()] if isinstance(value, dict)
                   else [(key, json.dumps(value) if isinstance(value, list) else value)])} for row in rows]
    # Objective and semantic rows intentionally have different observable fields.
    # A missing semantic field stays an empty CSV cell, never a fabricated zero.
    columns = list(dict.fromkeys(key for row in flattened for key in row))
    return [{key: row.get(key) for key in columns} for row in flattened]


def write_report(summary: dict, output: Path) -> None:
    lines = ["# Observed-trajectory reward design", "",
             "This is a retrospective replay on Qwen and Gemma, using one fixed uniform ten-task ordering per "
             "checkpoint from the full fifty-task training pool. n uses exactly the first n task executions, "
             "shared by the parent and five candidates. Full validation is evaluated only after selection.", "",
             "The primary cohort at each n requires all six traces and valid shared judgments on every queried "
             "task. All seven recipes, including success count, use that same cohort. Objective-only results "
             "are separate and can have larger coverage. Missing judgments are never substituted with zero.", "",
             "Verified progress uses the judge's lower bound on a shared public user-request inventory. Unknown "
             "attainment contributes no verified progress while retaining its unknown flag. An unknown material "
             "violation is unconfirmed (h=0 for the confirmed-violation indicator), and its unknown fraction is "
             "reported separately; that is not evidence of safety.", "",
             "Validation lift is selected full-validation score minus the mean of five proposals, not "
             "parent-relative validation improvement. Train parent gain uses the archived full-train parent "
             "score. All reward/gain numbers in JSON/CSV are fractions, so multiply by 100 for pp.", "",
             "Scalar recipes report within-checkpoint Pearson and Spearman; lexicographic recipes report "
             "Spearman only. Undefined correlations are excluded with their defined checkpoint count. Changes "
             "in Pearson under different nonlinear scales do not alone establish better selection. SD is "
             "between-checkpoint heterogeneity, not a confidence interval or independent task-draw variation.", "",
             "## Coverage", "", "|Model|n|Expected CPs|Objective CPs|Primary CPs|", "|---|---:|---:|---:|---:|"]
    lines.extend(f"|{row['model']}|{row['n']}|{row['expected_checkpoints']}|{row['objective_checkpoints']}|"
                 f"{row['primary_checkpoints']}|" for row in summary["coverage"])
    lines.extend(["", "## Reward definitions", "",
                  "All rules maximize the score. K is sampled-task success count, q is verified progress, "
                  "h indicates a confirmed material violation, t is environment steps, and u is immediate "
                  "avoidable repetition of an identical erroneous tool call. The historical cap T is 20. "
                  "The bounded quarter-point bonus/penalty cannot outweigh an additional sampled success "
                  "in the completion-primary scalar recipes.", "",
                  "|Recipe|Selection rule|", "|---|---|",
                  "|Success|Maximize K; uniform random tie breaking.|",
                  "|Efficient completion|K + 0.25 * mean(1 - min(t,20)/20) over successful executions.|",
                  "|Grounded progress|K + 0.25 * mean((1-success)q) over all sampled tasks.|",
                  "|Certified completion|Count success(1-h), then add bounded progress/efficiency credit "
                  "only where h=0. This deliberately changes the completion objective.|",
                  "|Parent preservation|Lexicographically minimize sampled regressions, maximize repairs, "
                  "then parent-relative progress on joint failures and efficiency on joint successes.|",
                  "|Recovery|K - 0.25 * mean(success * min(u,20)/20). Initial tool failures and changed "
                  "arguments are not penalized as avoidable repeats.|",
                  "|Quality gates|Lexicographically maximize (K, -sum h, rounded ten-bin mean q, "
                  "-successful repeats, -successful steps).|", "",
                  "The LLM receives six anonymized executions of the same task, in randomized order, "
                  "with one shared public request inventory. It receives no official success labels, "
                  "model identities, playbook text, private initial state or reference answer. This shared "
                  "group rubric replaces isolated per-trajectory judging in the initial review."])
    for cohort, title in (("objective", "Trace-only"), ("primary", "Semantic complete-case")):
        lines.extend(["", f"## {title} n=10 validation lift (pp)", "",
                      "|Model|Recipe|Checkpoints|Lift|", "|---|---|---:|---:|"])
        lines.extend(f"|{row['model']}|{row['recipe']}|{row['checkpoints']}|"
                     f"{100 * row['validation_lift']['mean']:+.3f}|" for row in summary["aggregate"]
                     if row["n"] == 10 and row["cohort"] == cohort)
    lines.extend(["", "## Inference", "", f"Status: {summary['inference_status']}. "
                  "The primary n=10 test compares six recipes against matched success count using paired "
                  "chronological checkpoint blocks inside each seed (sizes 2, 3, 4, 5). Holm6 applies separately "
                  "per model/block setting. The trace-only family compares its two recipes against success "
                  "count with Holm2. Chronological blocks are formed on all scheduled checkpoints within "
                  "each seed before excluding missing traces or judgments. Block sign exchangeability is "
                  "an assumption; three seeds remain "
                  "only three independent training streams. All previously inspected validation and these "
                  "follow-up mechanisms are exploratory.", "",
                  "A separate exact sign flip uses one aggregate contribution per seed, preserving equal "
                  "checkpoint weighting. With three independent seeds it has only eight sign patterns and "
                  "a minimum two-sided p-value of 0.25; the checkpoint-block results cannot replace that "
                  "limited independent replication.", "",
                  "Process efficiency is conditional on observed success in the success-efficiency recipe; "
                  "short failed trajectories do not earn its efficiency bonus. Tool errors and repetition "
                  "remain observable proxies, not automatic material constraint violations. Source hashes, "
                  "candidate score vectors, tie probabilities, seed rows, coverage and unknown rates are "
                  "saved alongside this report.", ""])
    if summary.get("judge_retryable_infrastructure_groups"):
        lines.extend(["## Pending semantic measurements", "",
                      f"{summary['judged_groups_valid']}/{summary['query_groups']} task groups have valid judgments; "
                      f"{summary['judge_retryable_infrastructure_groups']} groups encountered transport failures. "
                      "Their terminal error records are not completed semantic measurements. Semantic curves "
                      "are descriptive on the available prefix cohort, which can favor early queue checkpoints. "
                      "No primary semantic significance tests are reported until judging resumes and finishes "
                      "unless partial inference is explicitly requested.", ""])
    (output / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--allow-partial-inference", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    output = args.output
    plan_path, cache_path = output / "analysis_plan.json", output / "group_features.jsonl"
    if not plan_path.exists():
        raise ValueError("Freeze the trajectory reward plan before analyzing outcomes")
    manifest = json.loads((output / "task_groups.json").read_text())
    targets_path = args.targets or Path(manifest["source"])
    target_bytes, cache_bytes = targets_path.read_bytes(), cache_path.read_bytes()
    cache_digest = hashlib.sha256(cache_bytes).hexdigest()
    groups = [json.loads(line) for line in cache_bytes.decode().splitlines() if line.strip()]
    raw = json.loads(target_bytes)
    targets = {(model["label"], point["key"]): point for model in raw["models"] if model["label"] in MODELS
               for point in model["points"]}
    judges, terminal = {}, 0
    for group in groups:
        path = output / "judge_results" / f"{group['id']}.json"
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        if row.get("id") != group["id"] or row.get("source_cache_sha256") != cache_digest:
            raise ValueError("Cached judge belongs to a different group/trajectory cache")
        judges[group["id"]] = row
        terminal += 1
    summary = analyze(groups, targets, judges)
    judge_status_path = output / "judge_status.json"
    judge_status = json.loads(judge_status_path.read_text()) if judge_status_path.exists() else {}
    statuses = Counter(row.get("status", "unknown") for row in judges.values())
    infrastructure_failures = sum(_retryable_infrastructure(row) for row in judges.values())
    # A terminal file can merely record a refused connection. It does not mean
    # the scheduled semantic measurements have been completed.
    complete = terminal == len(groups) and infrastructure_failures == 0
    semantic_inference = complete or args.allow_partial_inference
    summary.update({"created_at": datetime.now(UTC).isoformat(), "models": list(MODELS), "excluded": ["OSS"],
                    "analysis_phase_model_calls": 0, "new_actor_rollouts": 0,
                    "judge_status_path": str(judge_status_path.resolve()),
                    "recorded_judge_usage": judge_status.get("usage", {}),
                    "judge_call_note": "Analysis is read-only. Cached LLM judging used additional measured "
                    "model calls; zero analysis-phase calls does not mean a zero-inference experiment.",
                    "source": str(targets_path.resolve()),
                    "source_sha256": hashlib.sha256(target_bytes).hexdigest(),
                    "cache_sha256": cache_digest,
                    "analysis_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                    "reward_source_sha256": hashlib.sha256(
                        (ROOT / "scripts/premise/bfcl_trajectory_selection_rewards.py").read_bytes()).hexdigest(),
                    "judge_results_present": terminal, "judge_queue_complete": complete,
                    "judge_terminal_records_complete": terminal == len(groups),
                    "judge_status_counts": dict(statuses),
                    "judge_transport_error_groups": statuses.get("judge_transport_error", 0),
                    "judge_retryable_infrastructure_groups": infrastructure_failures,
                    "semantic_measurement_complete": summary["judged_groups_valid"] == len(groups),
                    "analysis_state": "waiting_server" if infrastructure_failures else (
                        "completed" if complete else "judging_in_progress"),
                    "inference_status": "complete-case exploratory inference" if semantic_inference
                    else "pending semantic measurements; primary results descriptive only",
                    "objective_inference_status": "complete-case exploratory Holm2 comparisons",
                    "process_prevalence": process_prevalence(groups),
                    "reference": REFERENCE,
                    "recipes": list(REWARD_KEYS), "lexicographic_recipes": sorted(LEXICOGRAPHIC_KEYS),
                    "aggregation": "Equal checkpoint weighting; SD between checkpoints, not a confidence interval. "
                    "Primary seven-recipe complete-case cohort matched separately at each n; objective-only cohort "
                    "reported separately. Unknown q uses verified lower bound; unknown h is unconfirmed, not safe."})
    summary["paired_block_tests"] = statistical_tests(
        summary["checkpoint_scores"], args.seed, summary["checkpoint_coverage"],
        include_primary=semantic_inference,
    )
    (output / "trajectory_reward_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name, rows in (("checkpoint_scores", summary["checkpoint_scores"]), ("aggregate", summary["aggregate"]),
                       ("seed_scores", summary["seed_scores"]), ("coverage", summary["coverage"]),
                       ("paired_block_tests", summary["paired_block_tests"])):
        if rows:
            write_csv(output / f"{name}.csv", _flatten(rows))
    write_report(summary, output)
    if not args.skip_plots:
        from scripts.premise.plot_bfcl_trajectory_rewards import plot

        plot(summary, output / "figures")
    print(json.dumps({"query_groups": len(groups), "judge_results_present": terminal,
                      "judge_groups_valid": summary["judged_groups_valid"], "coverage_n10": [
                          row for row in summary["coverage"] if row["n"] == 10]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
