# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Replay meaning-driven BFCL selection rewards on Qwen/Gemma; no new inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.analyze_bfcl_alignment_significance import (  # noqa: E402
    holm_adjust,
    paired_block_contributions,
    rng_for,
    sign_flip_test,
)
from scripts.premise.analyze_bfcl_similarity_extremes import mixed_indices, ranked_draws  # noqa: E402
from scripts.premise.analyze_bfcl_temporal_selection import correlations, describe, write_csv  # noqa: E402
from scripts.premise.analyze_bfcl_training_selection_hypotheses import (  # noqa: E402
    public_metadata,
    selection_probabilities,
    similarities,
)

MODELS = ("Qwen", "Gemma")
TARGETS = ("full_train", "full_validation")
REFERENCE = "period_mean"
CONTROLS = ("period_mean", "task_mean", "balanced_mean", "paired_linear", "log_mean")
NOVEL = ("repair_aversion", "preserve_barrier", "geometric_balance", "soft_worst", "conflict_aversion",
         "paired_risk", "repair_retention_harmonic", "transfer_breadth")
REWARDS = (*CONTROLS, *NOVEL)
TEST_KEYS = ("task_mean", "balanced_mean", *NOVEL)
METRICS = ("validation_lift", "validation_score", "train_parent_gain", "train_proposal_lift",
           "selected_repaired_parent_failures", "selected_broken_parent_successes", "selected_past_regressions",
           "selected_near_reward", "selected_far_reward", "selected_current_reward", "selected_near_far_gap",
           "safe_selected_probability", "selection_total_variation_vs_period", "unselected_train_parent_gain",
           "unselected_train_repaired_fraction", "unselected_train_broken_fraction")


def reward_metadata() -> list[dict]:
    definitions = [
        ("period_mean", "Current / replay mean", "Current priority and average past competence", "B", False),
        ("task_mean", "Task mean", "Every task has equal value", "A", False),
        ("balanced_mean", "Group mean", "Each of current, similar, dissimilar groups has equal value",
         "(C+N+F)/3", False),
        ("paired_linear", "Linear parent advantage", "Control: shared parent subtraction is constant",
         "mean(d)", True),
        ("log_mean", "Log mean", "Control: monotone transformation leaves selection unchanged", "log(1+A)", False),
        ("repair_aversion", "Repair / regression", "A lost parent success costs twice a rescued failure",
         "sum(w*(max(d,0)-2*max(-d,0)))", True),
        ("preserve_barrier", "Preservation barrier", "Prefer candidates breaking no past parent success",
         "B-2*I(any past d<0); all-unsafe falls back to B", True),
        ("geometric_balance", "Geometric balance", "Joint competence values improving weak groups",
         "((C+.1)*(N+.1)*(F+.1))**(1/3)-.1", False),
        ("soft_worst", "Soft worst group", "Smooth worst-group preference",
         "-.25*log(mean(exp(-[C,N,F]/.25)))", False),
        ("conflict_aversion", "Specialization penalty", "Penalize similar/dissimilar competence gap",
         "(C+N+F)/3-.5*abs(N-F)", False),
        ("paired_risk", "Paired risk penalty", "Discount unstable parent-relative improvements",
         "mean(d)-std(d,ddof=1)/sqrt(10)", True),
        ("repair_retention_harmonic", "Repair / retention", "Require repair of failures and retention of successes",
         "2*u*v/(u+v), u=(rescues+.5)/(parent_failures+1), v=(retained+.5)/(parent_successes+1)", True),
        ("transfer_breadth", "Transfer breadth", "Bonus for positive improvement in each past group",
         "mean(d)+.1*I(mean(d_near)>0)+.1*I(mean(d_far)>0)", True),
    ]
    return [{"reward": key, "label": label, "meaning": meaning, "formula": formula, "requires_parent": parent,
             "role": "novel" if key in NOVEL else "equivalent_control" if key in CONTROLS[3:] else "baseline"}
            for key, label, meaning, formula, parent in definitions]


def reward_scores(candidate: np.ndarray, parent: np.ndarray) -> tuple[dict, dict]:
    """Read only selected binary tasks: draw,candidate,task and draw,task arrays."""
    candidate, parent = np.asarray(candidate, float), np.asarray(parent, float)
    if candidate.ndim != 3 or candidate.shape[-1] != 10 or parent.shape != (candidate.shape[0], 10):
        raise ValueError("Expected draw-by-candidate-by-ten rewards and matching draw-by-ten parent tasks")
    if not np.isin(candidate, [0, 1]).all() or not np.isin(parent, [0, 1]).all():
        raise ValueError("Rescue/retention designs require finite binary task rewards")
    current, near, far = candidate[..., 0], candidate[..., 1:6].mean(-1), candidate[..., 6:].mean(-1)
    average = candidate.mean(-1)
    period = 0.5 * current + 0.5 * (5 * near + 4 * far) / 9
    group = np.stack([current, near, far], axis=-1)
    delta = candidate - parent[:, None, :]
    repair, regression = (delta > 0).sum(-1), (delta < 0).sum(-1)
    past_regression = (delta[..., 1:] < 0).sum(-1)
    safe = past_regression == 0
    weights = np.array([0.5, *([1 / 18] * 9)])
    failed, succeeded = (parent == 0).sum(-1)[:, None], (parent == 1).sum(-1)[:, None]
    rescue_utility = (repair + 0.5) / (failed + 1)
    retained = (candidate * parent[:, None, :]).sum(-1)
    retention_utility = (retained + 0.5) / (succeeded + 1)
    scores = {
        "period_mean": period, "task_mean": average, "balanced_mean": group.mean(-1),
        "paired_linear": delta.mean(-1), "log_mean": np.log1p(average),
        "repair_aversion": ((np.maximum(delta, 0) - 2 * np.maximum(-delta, 0)) * weights).sum(-1),
        "preserve_barrier": period - 2 * ~safe,
        "geometric_balance": np.prod(group + 0.1, axis=-1) ** (1 / 3) - 0.1,
        "soft_worst": -0.25 * (logsumexp(-group / 0.25, axis=-1) - np.log(3)),
        "conflict_aversion": group.mean(-1) - 0.5 * np.abs(near - far),
        "paired_risk": delta.mean(-1) - delta.std(axis=-1, ddof=1) / np.sqrt(10),
        "repair_retention_harmonic": 2 * rescue_utility * retention_utility / (rescue_utility + retention_utility),
        "transfer_breadth": delta.mean(-1) + 0.1 * (delta[..., 1:6].mean(-1) > 1e-12)
        + 0.1 * (delta[..., 6:].mean(-1) > 1e-12),
    }
    diagnostics = {"selected_repaired_parent_failures": repair, "selected_broken_parent_successes": regression,
                   "selected_past_regressions": past_regression, "selected_near_reward": near,
                   "selected_far_reward": far, "selected_current_reward": current,
                   "selected_near_far_gap": np.abs(near - far), "safe_selected_probability": safe.astype(float)}
    return scores, diagnostics


def load_parents(raw: dict, output: Path) -> dict:
    models = []
    for model in raw["models"]:
        if model["label"] not in MODELS:
            continue
        points = []
        for point in model["points"]:
            folder = Path(point["source"])
            context = folder / "frozen" / point["key"] / "parent.json"
            context_hash = hashlib.sha256(context.read_bytes()).hexdigest()
            lookup, digest = {}, hashlib.sha256()
            paths = sorted((folder / "results" / point["key"] / "parent" / "train").glob("*.json"))
            if len(paths) != 50:
                raise ValueError(f"Expected fifty parent evaluations: {point['key']}")
            for path in paths:
                blob = path.read_bytes()
                row = json.loads(blob)
                if (row["checkpoint_key"], row["context"], row["partition"], row["context_sha256"]) != (
                        point["key"], "parent", "train", context_hash):
                    raise ValueError(f"Parent context/result identity mismatch: {path}")
                if row["task_id"] in lookup or row["status"] not in ("completed", "error"):
                    raise ValueError(f"Duplicate parent task or incomplete result: {path}")
                if row["score"] not in (0, 1) or (row["status"] == "error" and row["score"] != 0):
                    raise ValueError(f"Parent reward is not terminal binary: {path}")
                lookup[row["task_id"]] = row
                digest.update(hashlib.sha256(blob).hexdigest().encode())
            if set(lookup) != set(point["train_order"]):
                raise ValueError(f"Parent task set differs: {point['key']}")
            rewards = [lookup[task]["score"] for task in point["train_order"]]
            if not np.isclose(np.mean(rewards), point["parent_scores"]["full_train"], atol=1e-12, rtol=0):
                raise ValueError(f"Parent full-train mean differs: {point['key']}")
            points.append({"key": point["key"], "train_order": point["train_order"], "parent_rewards": rewards,
                           "parent_error_flags": [lookup[t]["status"] == "error" for t in point["train_order"]],
                           "parent_context_sha256": context_hash,
                           "result_digest_sorted_index_order": digest.hexdigest()})
        models.append({"label": model["label"], "points": points})
    cache = {"new_model_calls": 0, "models": models, "cost": "Ten extra parent-task scores per CP shared by five "
             "candidates, all already evaluated. Parent validation unavailable; no abstention option."}
    output.write_text(json.dumps(cache, indent=2, allow_nan=False) + "\n")
    return {(model["label"], point["key"]): point for model in models for point in model["points"]}


def task_draws(point: dict, metadata: dict, trials: int, seed: int) -> tuple[np.ndarray, str]:
    t = point["checkpoint"]
    sim = similarities(point["train_order"][:t + 1], metadata)
    ranking = ranked_draws(sim[-1, :-1], trials, rng_for(seed, f"near-far-ties:{point['key']}"))
    indices = np.concatenate((np.full((trials, 1), t), mixed_indices(ranking, 5)), axis=-1)
    return indices, hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def calculate(model: dict, parents: dict, metadata: dict, matched: set[str], trials: int, seed: int) -> tuple:
    if model["label"] not in MODELS or trials < 1:
        raise ValueError("Only Qwen/Gemma with positive draw counts are in scope")
    scopes = [(cohort, group) for cohort in ("cp36", "cp50") for group in ("all", 42, 43, 44)]
    counts = dict.fromkeys(scopes, 0)
    sums = {(cohort, group, reward, metric): np.zeros(trials) for cohort, group in scopes
            for reward in REWARDS for metric in METRICS}
    rho_sums = {(cohort, group, reward, target, kind): np.zeros(trials) for cohort, group in scopes
                for reward in REWARDS for target in TARGETS for kind in ("rho", "spearman")}
    coverage = {(cohort, group, target): np.zeros(trials) for cohort, group in scopes for target in TARGETS}
    rows, samples = [], []
    for point in model["points"]:
        indices, digest = task_draws(point, metadata, trials, seed)
        candidate = np.asarray(point["train_rewards"])[:, indices].transpose(1, 0, 2)
        parent_all = np.asarray(parents[model["label"], point["key"]]["parent_rewards"])
        parent = parent_all[indices]
        scores, diagnostic = reward_scores(candidate, parent)
        probabilities = {key: selection_probabilities(value) for key, value in scores.items()}
        equivalent_errors = {key: float(np.max(np.abs(probabilities[key] - probabilities["task_mean"])))
                             for key in ("paired_linear", "log_mean")}
        if max(equivalent_errors.values()) > 1e-12:
            raise ValueError("Negative-control scores changed the candidate tie distribution")
        full_delta = np.asarray(point["train_rewards"]) - parent_all[None, :]
        held_repair = ((full_delta > 0).sum(-1)[None, :] - diagnostic["selected_repaired_parent_failures"]) / 40
        held_break = ((full_delta < 0).sum(-1)[None, :] - diagnostic["selected_broken_parent_successes"]) / 40
        cohorts = ("cp50", "cp36") if point["key"] in matched else ("cp50",)
        scopes_here = [(cohort, group) for cohort in cohorts for group in ("all", point["seed"])]
        for scope in scopes_here:
            counts[scope] += 1
        val, train = (np.asarray(point["scores"][name]) for name in ("full_validation", "full_train"))
        point_rhos = {}
        for key, value in scores.items():
            ranks = rankdata(np.round(value, 12), axis=-1, method="average")
            for target in TARGETS:
                target_values = np.asarray(point["scores"][target])
                point_rhos[key, target, "rho"] = correlations(value, target_values)
                point_rhos[key, target, "spearman"] = correlations(ranks, rankdata(target_values))
        samples.append({"model": model["label"], "key": point["key"], "task_indices_sha256": digest,
                        "safe_feasible_fraction": float((diagnostic["safe_selected_probability"] > 0).any(1).mean()),
                        "paired_linear_max_choice_error": equivalent_errors["paired_linear"],
                        "log_mean_max_choice_error": equivalent_errors["log_mean"],
                        "parent_sample_successes_mean": float(parent.sum(-1).mean())})
        for key, probability in probabilities.items():
            val_score, train_score = probability @ val, probability @ train
            metrics = {"validation_lift": val_score - val.mean(), "validation_score": val_score,
                       "train_parent_gain": train_score - point["parent_scores"]["full_train"],
                       "train_proposal_lift": train_score - train.mean(),
                       "selection_total_variation_vs_period":
                       0.5 * np.abs(probability - probabilities[REFERENCE]).sum(-1),
                       "unselected_train_parent_gain": (probability * (held_repair - held_break)).sum(-1),
                       "unselected_train_repaired_fraction": (probability * held_repair).sum(-1),
                       "unselected_train_broken_fraction": (probability * held_break).sum(-1),
                       **{name: (probability * value).sum(-1) for name, value in diagnostic.items()}}
            row = {"model": model["label"], "key": point["key"], "seed": point["seed"],
                   "checkpoint": point["checkpoint"], "matched36": point["key"] in matched, "reward": key,
                   **{name: float(value.mean()) for name, value in metrics.items()},
                   "safe_feasible_fraction": samples[-1]["safe_feasible_fraction"],
                   **{f"candidate_{i + 1}_probability": float(probability[:, i].mean()) for i in range(5)}}
            for target in TARGETS:
                for kind in ("rho", "spearman"):
                    values = point_rhos[key, target, kind]
                    row[f"{kind}_{target}"] = describe(values)["mean"]
                    row[f"{kind}_{target}_defined_fraction"] = float(np.isfinite(values).mean())
            rows.append(row)
            for cohort, group in scopes_here:
                for metric, value in metrics.items():
                    sums[cohort, group, key, metric] += value
        for target in TARGETS:
            common = np.logical_and.reduce([np.isfinite(point_rhos[key, target, "rho"]) for key in REWARDS])
            for cohort, group in scopes_here:
                coverage[cohort, group, target] += common
                for key in REWARDS:
                    for kind in ("rho", "spearman"):
                        rho_sums[cohort, group, key, target, kind] += np.where(common, point_rhos[key, target, kind], 0)
    aggregate = []
    for (cohort, group), count in counts.items():
        if not count:
            continue
        for key in REWARDS:
            row = {"model": model["label"], "cohort": cohort, "seed": group, "reward": key, "checkpoints": count,
                   **{metric: describe(sums[cohort, group, key, metric] / count) for metric in METRICS}}
            for target in TARGETS:
                defined = coverage[cohort, group, target]
                row[f"rho_{target}_common_checkpoints"] = describe(defined)
                for kind in ("rho", "spearman"):
                    values = np.full(trials, np.nan)
                    np.divide(rho_sums[cohort, group, key, target, kind], defined, out=values, where=defined > 0)
                    row[f"{kind}_{target}"] = describe(values)
            aggregate.append(row)
    return aggregate, rows, samples


def statistical_tests(rows: list[dict], seed: int) -> list[dict]:
    output = []
    for model in MODELS:
        for cohort in ("cp36", "cp50"):
            subset = [r for r in rows if r["model"] == model and (cohort == "cp50" or r["matched36"])]
            baseline = sorted([r for r in subset if r["reward"] == REFERENCE],
                              key=lambda r: (r["seed"], r["checkpoint"]))
            if not baseline:
                continue
            for block in (2, 3, 4, 5):
                family = []
                for key in TEST_KEYS:
                    compared = {r["key"]: r for r in subset if r["reward"] == key}
                    differences = np.array([compared[r["key"]]["validation_lift"] - r["validation_lift"]
                                            for r in baseline])
                    contributions, _ = paired_block_contributions(baseline, baseline, differences, block)
                    test = sign_flip_test(contributions, 100000,
                                          rng_for(seed, f"semantic:{model}:{cohort}:{key}:{block}"))
                    effect = test.pop("mean_rho_difference")
                    family.append({"model": model, "cohort": cohort, "reward": key, "reference": REFERENCE,
                                   "block_size": block, "difference": effect, **test})
                for row, p in zip(family, holm_adjust([r["p_two_sided"] for r in family]), strict=True):
                    row["p_holm"] = p
                    row["family_size"] = len(family)
                output.extend(family)
    return output


def leave_one_seed_out(rows: list[dict]) -> list[dict]:
    choices, output = (REFERENCE, *TEST_KEYS), []
    for model in MODELS:
        for cohort in ("cp36", "cp50"):
            subset = [r for r in rows if r["model"] == model and (cohort == "cp50" or r["matched36"])]
            for held in (42, 43, 44):
                means = {key: np.mean([np.mean([r["validation_lift"] for r in subset
                                               if r["reward"] == key and r["seed"] == other])
                                      for other in (42, 43, 44) if other != held]) for key in choices}
                selected = max(choices, key=lambda key: means[key])
                chosen = [r for r in subset if r["seed"] == held and r["reward"] == selected]
                baseline = [r for r in subset if r["seed"] == held and r["reward"] == REFERENCE]
                output.append({"model": model, "cohort": cohort, "held_out_seed": held, "reward": selected,
                               "development_lift": float(means[selected]),
                               "held_out_lift": float(np.mean([r["validation_lift"] for r in chosen])),
                               "reference_lift": float(np.mean([r["validation_lift"] for r in baseline])),
                               "checkpoints": len(chosen)})
    return output


def flatten(rows: list[dict]) -> list[dict]:
    return [{out_key: number for key, value in row.items() for out_key, number in
             ([(f"{key}_{stat}", number) for stat, number in value.items()] if isinstance(value, dict)
              else [(key, value)])} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    plan_path = args.output / "analysis_plan.json"
    if not plan_path.exists():
        raise ValueError("Freeze the analysis plan before computing results")
    source_bytes = args.source.read_bytes()
    raw = json.loads(source_bytes)
    raw["models"] = [model for model in raw["models"] if model["label"] in MODELS]
    if hashlib.sha256(source_bytes).hexdigest() != json.loads(plan_path.read_text())["source_sha256"]:
        raise ValueError("Source differs from the frozen analysis plan")
    matched_path = args.source.parent / "temporal_selection_summary.json"
    matched = set(json.loads(matched_path.read_text())["fixed_checkpoint_keys"])
    metadata_path = ROOT / ("src/exgentic/benchmarks/bfcl/installation/gorilla/berkeley-function-call-leaderboard/"
                            "bfcl_eval/data/BFCL_v4_multi_turn_base.json")
    wanted = {task for model in raw["models"] for point in model["points"] for task in point["train_order"]}
    metadata = public_metadata(metadata_path, wanted)
    parent_path = args.output / "parent_training_rewards.json"
    parents = load_parents(raw, parent_path)
    aggregate, checkpoints, samples = [], [], []
    for model in raw["models"]:
        print(f"Scoring {model['label']}: eight semantic rewards, five controls", flush=True)
        values, rows, draws = calculate(model, parents, metadata, matched, args.trials, args.seed)
        aggregate.extend(values)
        checkpoints.extend(rows)
        samples.extend(draws)
    summary = {"source": str(args.source.resolve()), "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
               "analysis_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
               "parent_source": str(parent_path.resolve()),
               "parent_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
               "metadata_source": str(metadata_path),
               "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
               "models": list(MODELS), "excluded_models": ["OSS"], "seed": args.seed, "trials": args.trials,
               "new_model_calls": 0, "rewards": reward_metadata(), "aggregate": aggregate,
               "checkpoint_scores": checkpoints, "sample_draws": samples,
               "paired_block_tests": statistical_tests(checkpoints, args.seed),
               "leave_one_seed_out": leave_one_seed_out(checkpoints),
               "limitations": "Retrospective reused checkpoints and validation. Taskdraw SD is not independent seed "
               "confidence. CP blocks require independent sign exchangeability between blocks. Paired risk is "
               "not a formal confidence bound. Pairing means same task, not common random numbers. Preservation "
               "protects sampled tasks only. Parent val unknown; always select an existing candidate. Nonlinear "
               "Pearson rho depends on reward scale; rank correlations and exact choices are also reported."}
    (args.output / "semantic_reward_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_csv(args.output / "reward_scores.csv", flatten(aggregate))
    write_csv(args.output / "checkpoint_scores.csv", checkpoints)
    write_csv(args.output / "paired_block_tests.csv", summary["paired_block_tests"])
    write_csv(args.output / "leave_one_seed_out.csv", summary["leave_one_seed_out"])
    if not args.skip_plots:
        from scripts.premise.plot_bfcl_semantic_selection_rewards import plot

        plot(summary, args.output / "figures")
    for model in MODELS:
        rows = [r for r in aggregate if r["model"] == model and r["cohort"] == "cp50" and r["seed"] == "all"]
        print(model, [(r["reward"], round(100 * r["validation_lift"]["mean"], 5)) for r in rows], flush=True)


if __name__ == "__main__":
    main()
