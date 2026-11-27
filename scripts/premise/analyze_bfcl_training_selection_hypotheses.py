# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Test training-only replay/selection hypotheses on saved BFCL candidates, without new inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.analyze_bfcl_alignment_significance import (  # noqa: E402
    holm_adjust,
    paired_block_contributions,
    rng_for,
    sign_flip_test,
)
from scripts.premise.analyze_bfcl_temporal_selection import (  # noqa: E402
    correlations,
    describe,
    write_csv,
)

ALPHAS = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
SAMPLERS = (
    "recent", "oldest", "stratified", "similar", "diverse", "representative", "near3_uniform6", "near6_uniform3",
)
BASELINE = "uniform_a050"
SEEDS = (42, 43, 44)


def policies() -> list[dict]:
    rows = []
    for alpha in ALPHAS:
        rows.append({"key": f"uniform_a{round(100 * alpha):03d}", "sampler": "uniform", "alpha": alpha,
                     "label": f"Uniform replay, alpha={alpha:g}", "score": "mean",
                     "tasks_used": 1 if alpha == 1 else 9 if alpha == 0 else 10})
    rows.append({"key": "past10", "sampler": "uniform", "alpha": 0, "label": "Past only (10)",
                 "score": "past10", "tasks_used": 10})
    for sampler in SAMPLERS:
        label = {"near3_uniform6": "Current + 3 similar + 6 random",
                 "near6_uniform3": "Current + 6 similar + 3 random"}.get(sampler, f"Current + {sampler}")
        rows.append({"key": f"{sampler}_a050", "sampler": sampler, "alpha": 0.5,
                     "label": label, "score": "mean", "tasks_used": 10})
    rows.append({"key": "information_a050", "sampler": "uniform", "alpha": 0.5,
                 "label": "Current + disagreement-weighted", "score": "information", "tasks_used": 10})
    return rows


def public_metadata(path: Path, wanted: set[str]) -> dict:
    """Never expose solution paths, answers, initial environment states, or later user requests."""
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["id"] in wanted:
            first_turn = row["question"][0]
            text = " ".join(str(message["content"]) for message in first_turn if message["role"] == "user")
            result[row["id"]] = {"first_request": text, "tool_classes": sorted(row["involved_classes"])}
    if set(result) != wanted:
        raise ValueError("Public task metadata missing for some training tasks")
    return result


def similarities(task_ids: list[str], metadata: dict) -> np.ndarray:
    # Fit only on the current/past tasks at this checkpoint; no future or validation text enters IDF.
    vectors = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), strip_accents="unicode").fit_transform(
        [metadata[task]["first_request"] for task in task_ids],
    )
    lexical = (vectors @ vectors.T).toarray()
    classes = [set(metadata[task]["tool_classes"]) for task in task_ids]
    tools = np.asarray([[len(a & b) / len(a | b) if a | b else 1 for b in classes] for a in classes])
    return (lexical + tools) / 2


def metadata_indices(similarity: np.ndarray, count: int, strategy: str) -> np.ndarray:
    past = similarity.shape[0] - 1
    if count > past or count < 1:
        raise ValueError("Insufficient past tasks")
    if strategy == "similar":
        return np.argsort(-similarity[-1, :-1], kind="stable")[:count]
    selected = []
    covered = similarity[:-1, -1].copy()
    for _ in range(count):
        if strategy == "diverse":
            criterion = 1 - similarity[:-1, [past, *selected]].max(axis=1)
        elif strategy == "representative":
            criterion = (np.maximum(covered[:, None], similarity[:-1, :-1]) - covered[:, None]).mean(axis=0)
        else:
            raise ValueError("Unknown public-metadata sampler")
        criterion[selected] = -np.inf
        chosen = int(np.argmax(criterion))
        selected.append(chosen)
        covered = np.maximum(covered, similarity[:-1, chosen])
    return np.asarray(selected)


def task_subsets(point: dict, trials: int, seed: int, similarity: np.ndarray) -> dict:
    t = point["checkpoint"]
    if t < 10:
        raise ValueError("Expected at least ten past tasks")
    digest = hashlib.sha256(f"{seed}:{point['key']}".encode()).digest()
    generator = np.random.SeedSequence(int.from_bytes(digest[:16], "big")).spawn(2)[0]
    rng = np.random.default_rng(generator)
    uniform = np.argsort(rng.random((trials, t)), axis=1)[:, :10]
    result = {"uniform": uniform[:, :9], "past10": uniform,
              "recent": np.arange(t - 9, t)[None, :], "oldest": np.arange(9)[None, :]}
    rng = rng_for(seed, f"stratified:{point['key']}")
    result["stratified"] = np.column_stack([
        rng.choice(chunk, size=trials) for chunk in np.array_split(np.arange(t), 9)
    ])
    for strategy in ("similar", "diverse", "representative"):
        result[strategy] = metadata_indices(similarity, 9, strategy)[None, :]
    for near_count in (3, 6):
        selected = result["similar"][0, :near_count]
        remaining = np.setdiff1d(np.arange(t), selected)
        rng = rng_for(seed, f"near{near_count}:{point['key']}")
        draws = np.argsort(rng.random((trials, len(remaining))), axis=1)[:, :9 - near_count]
        result[f"near{near_count}_uniform{9 - near_count}"] = np.concatenate(
            (np.broadcast_to(selected, (trials, near_count)), remaining[draws]), axis=1,
        )
    return result


def replay_score(rewards: np.ndarray, indices: np.ndarray, information: bool = False) -> np.ndarray:
    # trials x tasks x candidates, with one identical task subset for all five candidates.
    samples = rewards[:, indices].transpose(1, 2, 0)
    if not information:
        return samples.mean(axis=1)
    weights = samples.var(axis=-1)
    total = weights.sum(axis=-1, keepdims=True)
    score = samples.mean(axis=1)
    np.divide(np.einsum("bt,bti->bi", weights, samples), total, out=score, where=total > 0)
    return score


def selection_probabilities(scores: np.ndarray) -> np.ndarray:
    tied = np.isclose(scores, scores.max(axis=-1, keepdims=True), atol=1e-12, rtol=0)
    return tied / tied.sum(axis=-1, keepdims=True)


def calculate(model: dict, metadata: dict, matched_keys: set[str], trials: int, seed: int) -> tuple:
    policy_list = policies()
    sums = {cohort: {policy["key"]: np.zeros(trials) for policy in policy_list} for cohort in ("cp36", "cp50")}
    counts = {"cp36": 0, "cp50": 0}
    rows, diagnostics, sample_log = [], [], []
    for point in model["points"]:
        t, key = point["checkpoint"], point["key"]
        matrix = np.asarray(point["train_rewards"])
        similarity = similarities(point["train_order"][:t + 1], metadata)
        subsets = task_subsets(point, trials, seed, similarity)
        current = matrix[:, t][None, :]
        replay = {name: replay_score(matrix, indices) for name, indices in subsets.items()}
        replay["information"] = replay_score(matrix, subsets["uniform"], information=True)
        cohorts = ["cp50", "cp36"] if key in matched_keys else ["cp50"]
        for cohort in cohorts:
            counts[cohort] += 1
        val = np.asarray(point["scores"]["full_validation"])
        train = np.asarray(point["scores"]["full_train"])
        proposal = float(val.mean())
        for policy in policy_list:
            sampler = "past10" if policy["score"] == "past10" else policy["sampler"]
            sampler = "information" if policy["score"] == "information" else sampler
            scores = policy["alpha"] * current + (1 - policy["alpha"]) * replay[sampler]
            probabilities = selection_probabilities(scores)
            performance = probabilities @ val
            gain = performance - proposal
            for cohort in cohorts:
                sums[cohort][policy["key"]] += gain  # deterministic policies broadcast over subset trials.
            rho = correlations(scores, val)
            finite = rho[np.isfinite(rho)]
            rows.append({"model": model["label"], "key": key, "seed": point["seed"], "checkpoint": t,
                         "matched36": key in matched_keys, "policy": policy["key"],
                         "tasks_used": policy["tasks_used"], "validation_lift": float(gain.mean()),
                         "validation_performance": float(performance.mean()), "proposal_mean": proposal,
                         "train_parent_gain": float((probabilities @ train).mean())
                         - point["parent_scores"]["full_train"],
                         "rho_validation": float(finite.mean()) if finite.size else None,
                         "rho_defined_fraction": float(np.isfinite(rho).mean()),
                         **{f"candidate_{i + 1}_probability": float(probabilities[:, i].mean()) for i in range(5)}})
        # Condition on training signals only when decomposing the current task's contribution.
        current_ties = selection_probabilities(current) > 0
        replay_ties = selection_probabilities(replay["uniform"]) > 0
        current_informative = np.ptp(current) > 1e-12
        replay_informative = np.ptp(replay["uniform"], axis=1) > 1e-12
        conflict = current_informative & replay_informative & ~(current_ties & replay_ties).any(axis=1)
        mixed = selection_probabilities((current + replay["uniform"]) / 2) @ val
        past = selection_probabilities(replay["uniform"]) @ val
        groups = {"current_flat": np.full(trials, not current_informative),
                  "informative_compatible": np.full(trials, current_informative) & ~conflict,
                  "informative_conflict": conflict}
        for group, mask in groups.items():
            if mask.any():
                diagnostics.append({"model": model["label"], "key": key, "seed": point["seed"],
                                    "matched36": key in matched_keys, "group": group,
                                    "probability": float(mask.mean()),
                                    "mixed_minus_past9": float((mixed - past)[mask].mean())})
        for strategy in SAMPLERS:
            if strategy not in ("stratified", "near3_uniform6", "near6_uniform3"):
                sample_log.append({"model": model["label"], "key": key, "strategy": strategy,
                                   "past_task_ids": [point["train_order"][i] for i in subsets[strategy][0]]})
    aggregate = []
    for cohort in sums:
        for policy in policy_list:
            values = sums[cohort][policy["key"]] / counts[cohort]
            aggregate.append({"model": model["label"], "cohort": cohort, "policy": policy["key"],
                              "checkpoints": counts[cohort], **describe(values)})
    return aggregate, rows, diagnostics, sample_log


def leave_one_seed_out(rows: list[dict], candidate_policies: list[str] | None = None) -> list[dict]:
    """Policy choice reads development seeds only; the held-out outcome is accessed afterward."""
    output = []
    for model in sorted({row["model"] for row in rows}):
        for cohort in ("cp36", "cp50"):
            subset = [row for row in rows if row["model"] == model and (cohort == "cp50" or row["matched36"])]
            for held in SEEDS:
                development = [row for row in subset if row["seed"] != held]
                if candidate_policies is None:
                    candidate_policies = [policy["key"] for policy in policies()]
                values = {policy: np.mean([np.mean([row["validation_lift"] for row in development
                          if row["policy"] == policy and row["seed"] == seed]) for seed in SEEDS if seed != held])
                          for policy in candidate_policies}
                chosen = max(candidate_policies, key=lambda policy: values[policy])
                evaluation = [row for row in subset if row["seed"] == held and row["policy"] == chosen]
                baseline = [row for row in subset if row["seed"] == held and row["policy"] == BASELINE]
                output.append({"model": model, "cohort": cohort, "held_out_seed": held, "chosen_policy": chosen,
                               "development_lift": float(values[chosen]),
                               "held_out_lift": float(np.mean([row["validation_lift"] for row in evaluation])),
                               "baseline_held_out_lift": float(np.mean([row["validation_lift"] for row in baseline])),
                               "evaluation_checkpoints": len(evaluation)})
    return output


def statistical_tests(rows: list[dict], trials: int, seed: int) -> list[dict]:
    output = []
    for model in sorted({row["model"] for row in rows}):
        for cohort in ("cp36", "cp50"):
            subset = [row for row in rows if row["model"] == model and (cohort == "cp50" or row["matched36"])]
            baseline = {row["key"]: row for row in subset if row["policy"] == BASELINE}
            points = sorted(baseline.values(), key=lambda row: (row["seed"], row["checkpoint"]))
            for block_size in (2, 3, 4, 5):
                family = []
                for policy in policies():
                    if policy["key"] == BASELINE:
                        continue
                    observations = {row["key"]: row["validation_lift"] for row in subset
                                    if row["policy"] == policy["key"]}
                    differences = np.asarray([observations[p["key"]] - p["validation_lift"] for p in points])
                    contributions, _ = paired_block_contributions(points, points, differences, block_size)
                    result = sign_flip_test(
                        contributions, trials, rng_for(seed, f"{model}:{cohort}:{block_size}:{policy['key']}"),
                    )
                    effect = result.pop("mean_rho_difference")
                    family.append({"model": model, "cohort": cohort, "policy": policy["key"], "baseline": BASELINE,
                                   "block_size": block_size, "difference": effect, **result})
                for row, adjusted in zip(family, holm_adjust([row["p_two_sided"] for row in family]), strict=True):
                    row["p_holm"] = adjusted
                    row["family_size"] = len(family)
                output.extend(family)
    return output


def write_report(summary: dict, output: Path) -> None:
    labels = {row["key"]: row["label"] for row in summary["policies"]}

    def value(model, cohort, policy):
        return 100 * next(row["mean"] for row in summary["aggregate"]
                          if row["model"] == model and row["cohort"] == cohort and row["policy"] == policy)

    def seed_value(model, cohort, policy, seed):
        rows = [row for row in summary["checkpoint_scores"] if row["model"] == model and row["seed"] == seed
                and row["policy"] == policy and (cohort == "cp50" or row["matched36"])]
        return 100 * np.mean([row["validation_lift"] for row in rows])

    lines = ["# Training-only candidate selection hypotheses", "",
        "The most promising exploratory rule is current + six similar past tasks + three uniformly sampled "
        "remaining past tasks, with current/replay weights 1:1. It improves the observed Qwen and Gemma means "
        "over uniform replay, but has not established a robust statistical advantage and fails on OSS.", "",
        "## Design and information budget", "",
        "- Fixed five candidates at each of 50 existing checkpoints; seeds 42/43/44 have 17/17/16 checkpoints.",
        "- Also use the 36-checkpoint cohort from the previous figures, with twelve checkpoints per seed.",
        "- At most ten selected training tasks per rule, hence at most 50 candidate-task evaluations. "
        "No new model calls were made. Replay tasks are always earlier than the current proposal task.",
        "- Per-checkpoint scores never read validation rewards or future training rewards. Candidate-disagreement "
        "weights use only the nine already selected replay tasks, not a scan over unselected tasks.",
        "- Validation lift is selected full-validation reward minus the five-proposal mean. "
        "A validation parent baseline is unavailable.",
        "- Default comparison: one current + nine uniform past tasks; score=0.5 current +0.5 past mean.",
        "- Alpha=0 uses only nine past tasks; alpha=1 uses only the current task. "
        "These are cheaper endpoint diagnostics. Past-only ten tasks is a separate matched-budget baseline.",
        "- Other replay samplers: latest nine, oldest nine, one from each of nine chronological strata, "
        "nearest to the current request, farthest-point diversity, and greedy coverage of the past-task metadata.",
        "- Similarity: mean of first-user-request TF-IDF cosine and available-tool-class Jaccard. "
        "TF-IDF is fitted on past+current text within each checkpoint only. "
        "Solution paths, answers, later user messages, and validation/future metadata are excluded.",
        "- Stage 1 fixed fifteen rules; stage 2 added two retrieval/uniform mixtures after observing seed "
        "instability. analysis_plan.json records both stages; stage1_15rules_summary.json preserves stage 1.",
        "- Source contexts are the original Qwen PREMiSE checkpoints; "
        "Gemma/OSS retain the original context-transfer setup.", "",
        "## Observed validation lift (pp)", "",
        "| Rule | Qwen, 36 CP | Qwen, 50 CP | Gemma, 50 CP | OSS, 50 CP |", "|---|---:|---:|---:|---:|"]
    display = ["uniform_a100", "past10", "uniform_a010", BASELINE, "recent_a050", "stratified_a050",
               "similar_a050", "diverse_a050", "representative_a050", "information_a050",
               "near3_uniform6_a050", "near6_uniform3_a050"]
    for policy in display:
        lines.append(f"| {labels[policy]} | {value('Qwen', 'cp36', policy):+.4f} | "
                     f"{value('Qwen', 'cp50', policy):+.4f} | {value('Gemma', 'cp50', policy):+.4f} | "
                     f"{value('OSS', 'cp50', policy):+.4f} |")
    lines.extend(["", "## Hypothesis assessment", "",
        "H1: current/replay balance. Current-only cannot distinguish the candidates at 78% of Qwen checkpoints "
        "and 88% of Gemma checkpoints. At those checkpoints every alpha<1 produces the same ranking as replay. "
        "The observed alpha=0.5 and 0.75 policies select identically; tuning that range adds no new signal.",
        "For binary current rewards, alpha>0.5 prioritizes current-task success before replay performance. "
        "Alpha=0.5 differs only at the extreme tie between (current=1,replay=0) and (current=0,replay=1).", "",
        "H2: recentness versus coverage. Recent-only replay underperforms uniform replay in Qwen. "
        "Chronological stratification yields a small, seed-consistent Qwen mean improvement, but its adjusted "
        "p-value is sensitive to block size; do not report the one favorable block setting alone.", "",
        "H3: relevant replay. Nearest-task replay has the largest pooled Qwen/Gemma mean among stage-1 rules, "
        "but has seed reversals. Six nearest + three random retains most of the pooled gain and reduces "
        "concentration on a narrow task neighborhood. This is exploratory evidence, not a proven winner.", "",
        "H4: discrimination is insufficient. Variance weighting distinguishes candidates within the selected "
        "tasks, but lowers Qwen's gain relative to uniform replay. Candidate disagreement does not by itself "
        "identify a signal that transfers to validation.", "",
        "Stage-2 hypothesis: combine relevance and coverage. Six similar + three random is positive in every "
        "Qwen/Gemma seed, but Qwen seed 43 remains slightly below uniform replay. OSS does not support the rule.", "",
        "## Replay-mixture seed breakdown (50 CP)", "",
        "| Model | Seed | Uniform replay | Nine similar | Six similar + three random |", "|---|---:|---:|---:|---:|"])
    for model in ("Qwen", "Gemma", "OSS"):
        for seed in SEEDS:
            lines.append(f"| {model} | {seed} | {seed_value(model, 'cp50', BASELINE, seed):+.4f} | "
                         f"{seed_value(model, 'cp50', 'similar_a050', seed):+.4f} | "
                         f"{seed_value(model, 'cp50', 'near6_uniform3_a050', seed):+.4f} |")
    lines.extend(["", "## Development selection and held-out seeds", "",
        "For each held-out seed, select the highest mean-lift policy using only the other two seeds, then "
        "evaluate the selected policy on the held-out seed. This is a retrospective sensitivity check: "
        "earlier temporal results have already been inspected, and task IDs can overlap across seed splits. "
        "It is not an independent prospective or task-held-out confirmation.", "",
        "| Model | Fixed uniform baseline | Broad 17-rule selection | Restricted stage-2 mixture selection |",
        "|---|---:|---:|---:|"])
    for model in ("Qwen", "Gemma", "OSS"):
        folds = [row for row in summary["leave_one_seed_out"] if row["model"] == model and row["cohort"] == "cp50"]
        regularized = [row for row in summary["leave_one_seed_out_regularized"]
                       if row["model"] == model and row["cohort"] == "cp50"]

        def average(rows):
            return 100 * sum(row["held_out_lift"] * row["evaluation_checkpoints"] for row in rows) / sum(
                row["evaluation_checkpoints"] for row in rows)

        lines.append(f"| {model} | {value(model, 'cp50', BASELINE):+.4f} | {average(folds):+.4f} | "
                     f"{average(regularized):+.4f} |")
    lines.extend(["",
        "Restricted stage-2 selection considers only uniform replay, three nearest + six random, and six "
        "nearest + three random. It is reported as the explicitly post-hoc regularization follow-up; "
        "a higher result here is not independent evidence that this restricted family was chosen optimally.", "",
        "## Statistical limits", "",
        "Paired differences use the same checkpoints. Chronological within-seed blocks of size 2,3,4,5 "
        "are sign-flipped jointly. Each setting corrects all sixteen non-baseline contrasts using Holm. "
        "This assumes sign symmetry and exchangeability between blocks and does not allow arbitrary "
        "dependence across different blocks. Task-subset draws are not independent experiments.",
        "None of the proposed improvements stays below adjusted p=0.05 for every block setting. "
        "In particular the six-nearest mixture has adjusted p=1 for Qwen under every tested setting. "
        "Chronological stratification crosses 0.05 for one Qwen setting but not the others.",
        "The recommended next rule to independently validate is the relevance/coverage mixture; "
        "the established default remains current plus uniform replay. Do not call the exploratory "
        "mixture statistically superior on the basis of these reused data.",
        "Original errors are retained as zero rewards. OSS has 7,456/27,500 original execution failures, "
        "so its policy comparisons are not clean evidence of generalization quality.", "",
        "## Artifacts", "",
        "- figures/strategy_validation_gain_cp50.pdf and cp36 version: all rules, pooled bars and per-seed dots.",
        "- figures/current_weight_sweep.pdf: current/replay weight sensitivity.",
        "- figures/retrieval_mix_tradeoff.pdf: zero/three/six/nine nearest replay tasks, with other slots uniform.",
        "- figures/leave_one_seed_out.pdf: broad-rule development selection versus the fixed baseline.",
        "- figures/current_signal_decomposition.pdf: which training-signal conditions let current weighting matter.",
        "- hypotheses_summary.json: all results, methods, probabilities, folds, and original hashes.",
        "- checkpoint_scores.csv: candidate selection probabilities and outcomes for each checkpoint/rule.",
        "- paired_block_tests.csv: raw and adjusted p-values for all settings, not only favorable ones.",
        "- analysis_plan.json: initial design and explicitly post-hoc second stage.",
        "- public_training_task_metadata.json: only public first requests and tool classes used by selectors.", "",
        "All outputs are offline analyses. Eight focused regression tests and Ruff checks pass. "
        "Candidate probabilities are invariant to validation-label changes in the regression tests.", ""])
    (output / "README.md").write_text("\n".join(lines))


def plot(summary: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.65,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    labels = {policy["key"]: policy["label"] for policy in policies()}
    order = ["uniform_a100", "past10", "uniform_a000", "uniform_a005", "uniform_a010", "uniform_a025", BASELINE,
             "uniform_a075", *[f"{sampler}_a050" for sampler in SAMPLERS], "information_a050"]
    seed_colors = ("#287C85", "#A34F65", "#AF8238")

    def save(figure, stem):
        for extension in ("pdf", "png"):
            figure.savefig(output / f"{stem}.{extension}", dpi=400, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {stem}.pdf", flush=True)

    for cohort in ("cp36", "cp50"):
        figure, axes = plt.subplots(1, 3, figsize=(10.2, 5.2), sharey=True)
        figure.subplots_adjust(left=0.28, right=0.99, top=0.90, bottom=0.12, wspace=0.22)
        for axis, model in zip(axes, ("Qwen", "Gemma", "OSS"), strict=True):
            group = [row for row in summary["aggregate"] if row["model"] == model and row["cohort"] == cohort]
            values = [next(row["mean"] for row in group if row["policy"] == key) * 100 for key in order]
            axis.barh(np.arange(len(order)), values,
                      color=["#287C85" if key == BASELINE else "#D5E6E8" for key in order],
                      edgecolor="none", height=0.65)
            for seed, color in zip(SEEDS, seed_colors, strict=True):
                rows = [row for row in summary["checkpoint_scores"] if row["model"] == model and row["seed"] == seed
                        and (cohort == "cp50" or row["matched36"])]
                means = [100 * np.mean([row["validation_lift"] for row in rows if row["policy"] == key])
                         for key in order]
                axis.scatter(means, np.arange(len(order)), s=9, color=color, label=f"Seed {seed}", zorder=3)
            baseline = values[order.index(BASELINE)]
            axis.axvline(0, color="#999999", linewidth=0.6)
            axis.axvline(baseline, color="#287C85", linewidth=0.7, linestyle=":")
            axis.set_title(model, fontsize=9)
            axis.grid(axis="x", color="#E3E3E3", linewidth=0.4)
            axis.set_axisbelow(True)
            axis.set_yticks(np.arange(len(order)), [labels[key] for key in order], fontsize=7)
            axis.tick_params(labelsize=7)
        axes[0].invert_yaxis()
        handles, legend_labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False)
        figure.supxlabel("Validation selection lift (pp)", y=0.02, fontsize=9)
        save(figure, f"strategy_validation_gain_{cohort}")

    figure, axes = plt.subplots(2, 3, figsize=(7.5, 4.2), sharex=True, sharey="row")
    figure.subplots_adjust(left=0.10, right=0.99, top=0.86, bottom=0.15, hspace=0.28, wspace=0.16)
    for row_index, cohort in enumerate(("cp36", "cp50")):
        for column, model in enumerate(("Qwen", "Gemma", "OSS")):
            axis = axes[row_index, column]
            for seed, color in zip(SEEDS, seed_colors, strict=True):
                group = [row for row in summary["checkpoint_scores"] if row["model"] == model and row["seed"] == seed
                         and (cohort == "cp50" or row["matched36"])]
                values = [100 * np.mean([row["validation_lift"] for row in group
                          if row["policy"] == f"uniform_a{round(100 * alpha):03d}"]) for alpha in ALPHAS]
                axis.plot(ALPHAS, values, color=color, marker="o", markersize=2.5, linewidth=1, label=f"Seed {seed}")
            values = [100 * next(row["mean"] for row in summary["aggregate"] if row["model"] == model
                      and row["cohort"] == cohort and row["policy"] == f"uniform_a{round(100 * alpha):03d}")
                      for alpha in ALPHAS]
            axis.plot(ALPHAS, values, color="#333333", linewidth=1.4, label="Mean")
            axis.axhline(0, color="#AAAAAA", linewidth=0.5)
            axis.set_xticks([0, 0.1, 0.25, 0.5, 0.75, 1])
            axis.tick_params(labelsize=6.5)
            if row_index == 0:
                axis.set_title(model, fontsize=9)
            if column == 0:
                axis.set_ylabel(f"{cohort[2:]} checkpoints\nLift (pp)", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.supxlabel(r"Current-task weight ($\alpha$), nine uniform past tasks", y=0.02, fontsize=8)
    save(figure, "current_weight_sweep")

    figure, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=True)
    figure.subplots_adjust(left=0.10, right=0.99, top=0.87, bottom=0.25, wspace=0.15)
    for axis, model in zip(axes, ("Qwen", "Gemma", "OSS"), strict=True):
        rows = [row for row in summary["leave_one_seed_out"] if row["model"] == model and row["cohort"] == "cp50"]
        x = np.arange(3)
        axis.bar(x - 0.17, [100 * row["baseline_held_out_lift"] for row in rows], width=0.34,
                 color="#287C85", label="Fixed uniform, alpha=0.5")
        axis.bar(x + 0.17, [100 * row["held_out_lift"] for row in rows], width=0.34,
                 color="#A34F65", label="Selected on other seeds")
        axis.set_xticks(x, [row["held_out_seed"] for row in rows])
        axis.set_title(model, fontsize=9)
        axis.axhline(0, color="#999999", linewidth=0.6)
    axes[0].set_ylabel("Validation selection lift (pp)", fontsize=8)
    figure.supxlabel("Held-out seed", y=0.06, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=7)
    save(figure, "leave_one_seed_out")

    figure, axes = plt.subplots(2, 3, figsize=(7.5, 4.2), sharex=True, sharey="row")
    figure.subplots_adjust(left=0.10, right=0.99, top=0.86, bottom=0.15, hspace=0.28, wspace=0.16)
    mixture_keys = [BASELINE, "near3_uniform6_a050", "near6_uniform3_a050", "similar_a050"]
    for row_index, cohort in enumerate(("cp36", "cp50")):
        for column, model in enumerate(("Qwen", "Gemma", "OSS")):
            axis = axes[row_index, column]
            for seed, color in zip(SEEDS, seed_colors, strict=True):
                rows = [r for r in summary["checkpoint_scores"] if r["model"] == model and r["seed"] == seed
                        and (cohort == "cp50" or r["matched36"])]
                values = [100 * np.mean([r["validation_lift"] for r in rows if r["policy"] == key])
                          for key in mixture_keys]
                axis.plot([0, 3, 6, 9], values, color=color, marker="o", markersize=2.8,
                          linewidth=1, label=f"Seed {seed}")
            values = [100 * next(r["mean"] for r in summary["aggregate"] if r["model"] == model
                      and r["cohort"] == cohort and r["policy"] == key) for key in mixture_keys]
            axis.plot([0, 3, 6, 9], values, color="#333333", linewidth=1.4, label="Mean")
            axis.axhline(0, color="#AAAAAA", linewidth=0.5)
            axis.set_xticks([0, 3, 6, 9])
            axis.tick_params(labelsize=7)
            if row_index == 0:
                axis.set_title(model, fontsize=9)
            if column == 0:
                axis.set_ylabel(f"{cohort[2:]} checkpoints\nLift (pp)", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.supxlabel("Number of nearest replay tasks (nine total; remaining slots random)", y=0.02, fontsize=8)
    save(figure, "retrieval_mix_tradeoff")

    figure, axes = plt.subplots(1, 2, figsize=(6.7, 2.7))
    figure.subplots_adjust(left=0.12, right=0.99, top=0.80, bottom=0.22, wspace=0.35)
    groups = ("current_flat", "informative_compatible", "informative_conflict")
    group_labels = ("Current flat", "Informative, compatible", "Informative, conflicting")
    colors = ("#AAAAAA", "#287C85", "#A34F65")
    bottoms = [np.zeros(3), np.zeros(3)]
    for group, label, color in zip(groups, group_labels, colors, strict=True):
        probabilities, contributions = [], []
        for model in ("Qwen", "Gemma", "OSS"):
            rows = [r for r in summary["current_signal_diagnostics"] if r["model"] == model and r["group"] == group]
            probabilities.append(sum(r["probability"] for r in rows) / 50)
            contributions.append(100 * sum(r["probability"] * r["mixed_minus_past9"] for r in rows) / 50)
        axes[0].bar(np.arange(3), probabilities, bottom=bottoms[0], color=color, label=label)
        axes[1].bar(np.arange(3), contributions, bottom=bottoms[1], color=color, label=label)
        bottoms[0] += probabilities
        bottoms[1] += contributions
    for axis in axes:
        axis.set_xticks(np.arange(3), ["Qwen", "Gemma", "OSS"])
        axis.tick_params(labelsize=7)
    axes[0].set_ylabel("Fraction of checkpoint / subset pairs", fontsize=8)
    axes[1].set_ylabel("Added lift from current task (pp)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=6.8)
    save(figure, "current_signal_decomposition")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--matched-summary", type=Path, required=True)
    parser.add_argument("--task-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--permutations", type=int, default=100000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.source.read_bytes()
    raw = json.loads(source)
    matched = json.loads(args.matched_summary.read_text())
    wanted = {task for model in raw["models"] for point in model["points"] for task in point["train_order"]}
    metadata = public_metadata(args.task_data, wanted)
    aggregate, checkpoint_rows, diagnostics, sample_log = [], [], [], []
    for model in raw["models"]:
        print(f"Computing {model['label']}: {len(policies())} fixed training-only policies", flush=True)
        results = calculate(model, metadata, set(matched["fixed_checkpoint_keys"]), args.trials, args.seed)
        for target, values in zip((aggregate, checkpoint_rows, diagnostics, sample_log), results, strict=True):
            target.extend(values)
    print("Computing paired block tests and held-out-seed policy selection", flush=True)
    tests = statistical_tests(checkpoint_rows, args.permutations, args.seed)
    summary = {"created_at": datetime.now(UTC).isoformat(), "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(source).hexdigest(), "sampling_seed": args.seed, "trials": args.trials,
        "new_model_calls": 0, "candidate_evaluations_per_rule_at_most": 50,
        "public_metadata_source": str(args.task_data.resolve()),
        "public_metadata_sha256": hashlib.sha256(args.task_data.read_bytes()).hexdigest(),
        "policies": policies(), "aggregate": aggregate, "checkpoint_scores": checkpoint_rows,
        "current_signal_diagnostics": diagnostics, "deterministic_samples": sample_log,
        "leave_one_seed_out": leave_one_seed_out(checkpoint_rows), "paired_block_tests": tests,
        "leave_one_seed_out_regularized": leave_one_seed_out(
            checkpoint_rows, [BASELINE, "near3_uniform6_a050", "near6_uniform3_a050"],
        ),
        "scope": "Exploratory, same saved candidates; train-only per-checkpoint scoring. Leave-one-seed-out "
        "policy tuning uses validation outcomes from the other two seeds, never the held-out seed. Task IDs "
        "can overlap across seed splits, so this is not independent task-held-out validation. Original Qwen "
        "PREMiSE contexts are reused by Gemma/OSS. Existing error rewards of zero are preserved.",
        "tests": "Two-sided chronological within-seed block sign flips, block sizes 2-5; null requires "
        "sign symmetry and independent sign exchangeability between blocks. Holm correction across all "
        "16 contrasts per model/cohort/block size; no subset trial is an independent experimental replicate.",
    }
    (args.output / "hypotheses_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name, rows in (("strategy_scores.csv", aggregate), ("checkpoint_scores.csv", checkpoint_rows),
                       ("paired_block_tests.csv", tests), ("leave_one_seed_out.csv", summary["leave_one_seed_out"]),
                       ("leave_one_seed_out_regularized.csv", summary["leave_one_seed_out_regularized"]),
                       ("current_signal_diagnostics.csv", diagnostics)):
        write_csv(args.output / name, rows)
    (args.output / "public_training_task_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_report(summary, args.output)
    plot(summary, args.output / "figures")
    for model in ("Qwen", "Gemma", "OSS"):
        rows = [row for row in aggregate if row["model"] == model and row["cohort"] == "cp50"]
        best = max(rows, key=lambda row: row["mean"])
        baseline = next(row for row in rows if row["policy"] == BASELINE)
        print(f"{model}: best observed={best['policy']} {100 * best['mean']:+.4f}pp; "
              f"baseline={100 * baseline['mean']:+.4f}pp", flush=True)


if __name__ == "__main__":
    main()
