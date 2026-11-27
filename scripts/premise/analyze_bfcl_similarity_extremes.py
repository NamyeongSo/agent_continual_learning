# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Mix current-similar and current-dissimilar past tasks at a fixed ten-task selection budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
from scripts.premise.analyze_bfcl_temporal_selection import correlations, describe, write_csv  # noqa: E402
from scripts.premise.analyze_bfcl_training_selection_hypotheses import (  # noqa: E402
    public_metadata,
    selection_probabilities,
    similarities,
)
from scripts.premise.plot_bfcl_sample_size_contours import ellipse_points  # noqa: E402

COUNTS = (0, 3, 4, 5, 6, 9)
MODES = (("task_mean", 0.1), ("period_mean", 0.5))


def ranked_draws(similarity: np.ndarray, trials: int, rng: np.random.Generator) -> np.ndarray:
    similarity = np.asarray(similarity, dtype=float)
    if similarity.ndim != 1 or len(similarity) < 9 or not np.isfinite(similarity).all():
        raise ValueError("At least nine finite past-task similarities are required")
    # Randomize exact ties without adding numerical jitter to unequal similarity values.
    primary = np.broadcast_to(-similarity, (trials, len(similarity)))
    return np.lexsort((rng.random(primary.shape), primary), axis=-1)


def mixed_indices(ranking: np.ndarray, near: int, random_priorities: np.ndarray | None = None) -> np.ndarray:
    if near not in COUNTS or ranking.ndim != 2 or ranking.shape[1] < 9:
        raise ValueError("Expected one supported near/far split and enough past tasks")
    far = 9 - near
    if random_priorities is not None:
        if random_priorities.shape != ranking.shape:
            raise ValueError("Random complement must share the same past-task pool")
        priorities = random_priorities.copy()
        np.put_along_axis(priorities, ranking[:, :near], np.inf, axis=1)
        complement = np.argsort(priorities, axis=1)[:, :far]
        return np.concatenate((ranking[:, :near], complement), axis=1)
    return np.concatenate((ranking[:, :near], ranking[:, -far:] if far else ranking[:, :0]), axis=1)


def calculate(model: dict, metadata: dict, matched: set[str], trials: int, seed: int,
              random_complement: bool = False) -> tuple:
    if trials < 1:
        raise ValueError("Trials must be positive")
    sums = {(cohort, mode, count): np.zeros(trials) for cohort in ("cp36", "cp50")
            for mode, _ in MODES for count in COUNTS}
    targets = ("full_train", "full_validation")
    rho_sums = {(cohort, mode, count, target): np.zeros(trials) for cohort, mode, count in sums
                for target in targets}
    rho_counts = {(cohort, target): np.zeros(trials) for cohort in ("cp36", "cp50") for target in targets}
    counts = {"cp36": 0, "cp50": 0}
    rows = []
    for point in model["points"]:
        t = point["checkpoint"]
        matrix = np.asarray(point["train_rewards"])
        sim = similarities(point["train_order"][:t + 1], metadata)
        ranking = ranked_draws(sim[-1, :-1], trials, rng_for(seed, f"near-far-ties:{point['key']}"))
        priorities = rng_for(seed, f"near-random-complement:{point['key']}").random(ranking.shape)
        current = matrix[:, t][None, :]
        target = np.asarray(point["scores"]["full_validation"])
        train = np.asarray(point["scores"]["full_train"])
        cohorts = ["cp50", "cp36"] if point["key"] in matched else ["cp50"]
        point_rhos = {}
        for cohort in cohorts:
            counts[cohort] += 1
        for near in COUNTS:
            indices = mixed_indices(ranking, near, priorities if random_complement else None)
            replay = matrix[:, indices].mean(axis=-1).T
            if replay.shape != (trials, 5):
                raise ValueError("A single shared task subset must score all five candidates")
            near_similarity = float(sim[-1, :-1][indices[:, :near]].mean()) if near else None
            far_similarity = float(sim[-1, :-1][indices[:, near:]].mean()) if near < 9 else None
            for mode, alpha in MODES:
                scores = alpha * current + (1 - alpha) * replay
                probabilities = selection_probabilities(scores)
                gain = probabilities @ target - target.mean()
                rho_fields = {}
                for name in targets:
                    rho = correlations(scores, np.asarray(point["scores"][name]))
                    point_rhos[mode, near, name] = rho
                    rho_fields[f"rho_{name}"] = describe(rho)["mean"]
                    rho_fields[f"rho_{name}_defined_fraction"] = float(np.isfinite(rho).mean())
                for cohort in cohorts:
                    sums[cohort, mode, near] += gain
                rows.append({"model": model["label"], "key": point["key"], "seed": point["seed"],
                             "checkpoint": t, "matched36": point["key"] in matched, "weight": mode,
                             "near": near, "far": 9 - near, "tasks": 10, "current_tasks": 1,
                             "validation_lift": float(gain.mean()), "mean_near_similarity": near_similarity,
                             "mean_far_similarity": far_similarity,
                             "train_parent_gain": float((probabilities @ train).mean())
                             - point["parent_scores"]["full_train"],
                             "train_proposal_lift": float((probabilities @ train).mean() - train.mean()),
                             **rho_fields,
                             **{f"candidate_{i + 1}_probability": float(probabilities[:, i].mean()) for i in range(5)}})
        # Each correlation comparison uses exactly the same CPs in a draw across all ratios and both weights.
        for name in targets:
            common = np.logical_and.reduce([np.isfinite(point_rhos[mode, count, name])
                                            for mode, _ in MODES for count in COUNTS])
            for cohort in cohorts:
                rho_counts[cohort, name] += common
                for mode, _ in MODES:
                    for count in COUNTS:
                        rho_sums[cohort, mode, count, name] += np.where(common, point_rhos[mode, count, name], 0)
    aggregate = [{"model": model["label"], "cohort": cohort, "weight": mode, "near": count, "far": 9 - count,
                  "checkpoints": counts[cohort], **describe(values / counts[cohort])}
                 for (cohort, mode, count), values in sums.items()]
    for row in aggregate:
        for name in targets:
            defined = rho_counts[row["cohort"], name]
            values = np.full(trials, np.nan)
            np.divide(rho_sums[row["cohort"], row["weight"], row["near"], name], defined,
                      out=values, where=defined > 0)
            row[f"rho_{name}"] = describe(values)
            row[f"rho_{name}_common_checkpoints"] = describe(defined)
    return aggregate, rows


def tests(rows: list[dict], previous: dict, permutations: int, seed: int) -> list[dict]:
    output = []
    for model in ("Qwen", "Gemma", "OSS"):
        for cohort in ("cp36", "cp50"):
            subset = [row for row in rows if row["model"] == model and (cohort == "cp50" or row["matched36"])]
            for block in (2, 3, 4, 5):
                family = []
                for mode, _ in MODES:
                    current = [row for row in subset if row["weight"] == mode]
                    alpha_key = "uniform_a010" if mode == "task_mean" else "uniform_a050"
                    uniform = [row for row in previous["checkpoint_scores"] if row["model"] == model
                               and row["policy"] == alpha_key and (cohort == "cp50" or row["matched36"])]
                    near_only = [row for row in current if row["near"] == 9]
                    for reference, baseline in (("uniform", uniform), ("near_only", near_only)):
                        points = sorted(baseline, key=lambda row: (row["seed"], row["checkpoint"]))
                        allowed = COUNTS if reference == "uniform" else (3, 4, 5, 6)
                        for near in allowed:
                            values = {row["key"]: row["validation_lift"] for row in current if row["near"] == near}
                            difference = np.asarray([values[row["key"]] - row["validation_lift"] for row in points])
                            contributions, _ = paired_block_contributions(points, points, difference, block)
                            key = f"nearfar:{model}:{cohort}:{mode}:{reference}:{near}:{block}"
                            test = sign_flip_test(contributions, permutations, rng_for(seed, key))
                            effect = test.pop("mean_rho_difference")
                            family.append({"model": model, "cohort": cohort, "weight": mode, "near": near,
                                           "reference": reference, "block_size": block, "difference": effect, **test})
                for row, adjusted in zip(family, holm_adjust([row["p_two_sided"] for row in family]), strict=True):
                    row["p_holm"] = adjusted
                    row["family_size"] = len(family)
                output.extend(family)
    return output


def plot(summary: dict, previous: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.65,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = ("#287C85", "#A34F65", "#AF8238")

    def save(figure, stem):
        for extension in ("pdf", "png"):
            figure.savefig(output / f"{stem}.{extension}", dpi=400, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {stem}.pdf", flush=True)

    for cohort in ("cp36", "cp50"):
        figure, axes = plt.subplots(2, 3, figsize=(7.5, 4.25), sharex=True, sharey="row")
        figure.subplots_adjust(left=0.10, right=0.99, top=0.86, bottom=0.15, hspace=0.27, wspace=0.17)
        for column, model in enumerate(("Qwen", "Gemma", "OSS")):
            for index, (mode, _) in enumerate(MODES):
                axis = axes[index, column]
                for seed, color in zip((42, 43, 44), colors, strict=True):
                    rows = [row for row in summary["checkpoint_scores"] if row["model"] == model
                            and row["seed"] == seed and row["weight"] == mode
                            and (cohort == "cp50" or row["matched36"])]
                    y = [100 * np.mean([row["validation_lift"] for row in rows if row["near"] == near])
                         for near in COUNTS]
                    axis.plot(COUNTS, y, color=color, linewidth=1, marker="o", markersize=2.4,
                              label=f"Seed {seed}")
                y = [100 * next(row["mean"] for row in summary["aggregate"] if row["model"] == model
                     and row["cohort"] == cohort and row["weight"] == mode and row["near"] == near) for near in COUNTS]
                axis.plot(COUNTS, y, color="#333333", linewidth=1.5, label="Mean")
                policy = "uniform_a010" if mode == "task_mean" else "uniform_a050"
                baseline = next(row["mean"] for row in previous["aggregate"] if row["model"] == model
                                and row["cohort"] == cohort and row["policy"] == policy)
                axis.axhline(100 * baseline, color="#777777", linestyle=":", linewidth=0.8, label="Uniform replay")
                axis.axhline(0, color="#AAAAAA", linewidth=0.5)
                axis.set_xticks(COUNTS)
                axis.tick_params(labelsize=7)
                if index == 0:
                    axis.set_title(model, fontsize=9)
                if column == 0:
                    axis.set_ylabel(("Task mean" if index == 0 else "Period mean") + "\nLift (pp)", fontsize=8)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=7)
        figure.supxlabel("Similar past tasks (nine total; remaining tasks dissimilar)", y=0.02, fontsize=8)
        save(figure, f"similarity_mix_{cohort}")

    figure, axes = plt.subplots(2, 3, figsize=(7.5, 4.25), sharex=True, sharey="row")
    figure.subplots_adjust(left=0.10, right=0.99, top=0.86, bottom=0.15, hspace=0.27, wspace=0.17)
    for row_index, cohort in enumerate(("cp36", "cp50")):
        for column, model in enumerate(("Qwen", "Gemma", "OSS")):
            axis = axes[row_index, column]
            near = (0, 3, 6, 9)
            random = [100 * next(row["mean"] for row in summary["random_complement_aggregate"]
                      if row["model"] == model and row["cohort"] == cohort and row["weight"] == "period_mean"
                      and row["near"] == count) for count in near]
            far = [100 * next(row["mean"] for row in summary["aggregate"] if row["model"] == model
                   and row["cohort"] == cohort and row["weight"] == "period_mean" and row["near"] == count)
                   for count in near]
            axis.plot(near, random, color="#287C85", marker="o", markersize=2.8, linewidth=1.2,
                      label="Similar + random")
            axis.plot(near, far, color="#A34F65", marker="s", markersize=2.8, linewidth=1.2,
                      label="Similar + dissimilar")
            axis.axhline(0, color="#AAAAAA", linewidth=0.5)
            axis.set_xticks(near)
            if row_index == 0:
                axis.set_title(model, fontsize=9)
            if column == 0:
                axis.set_ylabel(f"{cohort[2:]} checkpoints\nLift (pp)", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.supxlabel("Similar past tasks (nine total; current/replay weights 1:1)", y=0.02, fontsize=8)
    save(figure, "similar_vs_random_complement")

    mode_colors = ("#287C85", "#A34F65")
    mode_fills = ("#EAF4F5", "#F9EEF1")
    for target in ("full_train", "full_validation"):
        figure, axes = plt.subplots(1, 3, figsize=(7.3, 2.25), sharex=True, sharey=True)
        figure.subplots_adjust(left=0.075, right=0.99, top=0.80, bottom=0.25, wspace=0.15)
        for axis, model in zip(axes, ("Qwen", "Gemma", "OSS"), strict=True):
            for (mode, _), color in zip(MODES, mode_colors, strict=True):
                values = [next(row[f"rho_{target}"] for row in summary["aggregate"] if row["model"] == model
                               and row["cohort"] == "cp50" and row["weight"] == mode and row["near"] == near)
                          for near in COUNTS]
                axis.plot(COUNTS, [row["mean"] for row in values], marker="o", markersize=2.5, color=color,
                          linewidth=1.2, label="Task mean" if mode == "task_mean" else "Period mean")
            axis.axhline(0, color="#AAAAAA", linewidth=0.5)
            axis.set_xticks(COUNTS)
            axis.set_title(model, fontsize=9)
        axes[0].set_ylabel(r"$\rho$", fontsize=10)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        figure.supxlabel("Similar past tasks (nine total; remaining tasks dissimilar)", y=0.02, fontsize=8)
        save(figure, f"correlation_{target}")

        figure, axes = plt.subplots(3, 6, figsize=(9.3, 4.8), sharex=True, sharey=True)
        figure.subplots_adjust(left=0.09, right=0.99, bottom=0.10, top=0.87, wspace=0.20, hspace=0.16)
        for index, model in enumerate(("Qwen", "Gemma", "OSS")):
            for column, near in enumerate(COUNTS):
                axis = axes[index, column]
                for number, ((mode, _), color, fill) in enumerate(zip(MODES, mode_colors, mode_fills, strict=True)):
                    row = next(row for row in summary["aggregate"] if row["model"] == model
                               and row["cohort"] == "cp50" and row["weight"] == mode and row["near"] == near)
                    rho = row[f"rho_{target}"]["mean"]
                    subscript = "T" if mode == "task_mean" else "P"
                    label = rf"$\rho_{{{subscript}}}={rho:+.3f}$" if rho is not None else r"$\rho=\mathrm{N/A}$"
                    axis.text(0.03, 0.96 - number * 0.17, label, transform=axis.transAxes, color=color,
                              fontsize=6.4, va="top", zorder=4,
                              bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.1})
                    if rho is not None and abs(rho) < 1:
                        covariance = np.array([[1, rho], [rho, 1]])
                        inner, outer = (ellipse_points(covariance, mass, 121) for mass in (0.5, 0.9))
                        axis.fill(inner[:, 0], inner[:, 1], color=fill, linewidth=0)
                        axis.plot(inner[:, 0], inner[:, 1], color=color, linewidth=0.8)
                        axis.plot(outer[:, 0], outer[:, 1], color=color, linewidth=0.6, linestyle="--")
                axis.axhline(0, color="#DDDDDD", linewidth=0.4)
                axis.axvline(0, color="#DDDDDD", linewidth=0.4)
                axis.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), xticks=[-2, 0, 2], yticks=[-2, 0, 2])
                axis.set_aspect("equal")
                axis.tick_params(labelsize=6, length=2)
                if index == 0:
                    axis.set_title(f"{near} similar + {9 - near} dissimilar", fontsize=7)
                if column == 0:
                    axis.set_ylabel(model, fontsize=8)
        figure.legend(axes[0, 0].plot([], [], color=mode_colors[0], label="Task mean")
                      + axes[0, 0].plot([], [], color=mode_colors[1], label="Period mean"),
                      ["Task mean", "Period mean"], loc="upper center", ncol=2, frameon=False)
        xlabel = r"$z(\delta_t^i)$" if target == "full_train" else r"$z(s_{t,\mathrm{val50}}^i)$"
        figure.supxlabel(xlabel, y=0.01, fontsize=9)
        figure.supylabel(r"$z(s_{t,10}^i)$", x=0.015, fontsize=9)
        save(figure, f"bivariate_{target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    previous_raw = args.previous.read_bytes()
    previous = json.loads(previous_raw)
    raw_path = Path(previous["source"])
    raw = json.loads(raw_path.read_text())
    matched_path = raw_path.parent / "temporal_selection_summary.json"
    matched = set(json.loads(matched_path.read_text())["fixed_checkpoint_keys"])
    task_path = Path(previous["public_metadata_source"])
    wanted = {task for model in raw["models"] for point in model["points"] for task in point["train_order"]}
    metadata = public_metadata(task_path, wanted)
    aggregate, checkpoint_rows, control_aggregate, control_rows = [], [], [], []
    for model in raw["models"]:
        print(f"Computing {model['label']}: 6 near/far mixtures and both weighting schemes", flush=True)
        results, rows = calculate(model, metadata, matched, args.trials, args.seed)
        aggregate.extend(results)
        checkpoint_rows.extend(rows)
        results, rows = calculate(model, metadata, matched, args.trials, args.seed, random_complement=True)
        control_aggregate.extend(results)
        control_rows.extend(rows)
    print("Computing paired block tests", flush=True)
    summary = {"previous": str(args.previous.resolve()), "previous_sha256": hashlib.sha256(previous_raw).hexdigest(),
        "source": str(raw_path), "source_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "task_metadata_source": str(task_path),
        "task_metadata_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
        "seed": args.seed, "trials": args.trials, "new_model_calls": 0,
        "design": "Current 1 + past 9. Similar/dissimilar refer to ranking by similarity to the current first "
        "request and public tools, not pairwise diversity within the replay set. Exact similarity ties are "
        "randomized, with disjoint top/bottom prefixes of the same ranking. The shared subset scores all five "
        "candidates under both task-mean and period-mean weighting. All task IDs are current/past training only.",
        "baseline_difference": "Previous all-similar replay broke similarity ties chronologically. This "
        "analysis randomizes ties without changing unequal similarities; its all-similar endpoint can differ. "
        "The random-complement control is recomputed with the same randomized similar-task prefixes.",
        "correlation": "Within-CP Pearson across five candidates. Primary means use the same jointly defined "
        "CPs in each draw across all six ratios and both weights. Coverage is recorded for each target. "
        "Bivariate contours use zero-mean unit-variance Gaussian moments with that paired rho; normality "
        "is not assumed for inference or established empirically.",
        "uncertainty": "Sampling SD reflects similarity-tie randomization only. Paired block tests use CP "
        "means, never treat draws as independent replications, and require independent sign exchangeability "
        "between chronological blocks. Holm family has 20 comparisons per model/cohort/block setting.",
        "aggregate": aggregate, "checkpoint_scores": checkpoint_rows,
        "random_complement_aggregate": control_aggregate, "random_complement_checkpoint_scores": control_rows,
        "paired_block_tests": tests(checkpoint_rows, previous, 100000, args.seed)}
    (args.output / "similarity_mix_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_csv(args.output / "scores.csv", aggregate)
    write_csv(args.output / "checkpoint_scores.csv", checkpoint_rows)
    write_csv(args.output / "random_complement_scores.csv", control_aggregate)
    write_csv(args.output / "random_complement_checkpoint_scores.csv", control_rows)
    write_csv(args.output / "paired_block_tests.csv", summary["paired_block_tests"])
    plot(summary, previous, args.output / "figures")
    for mode, _ in MODES:
        rows = [row for row in aggregate if row["model"] == "Qwen"
                and row["cohort"] == "cp50" and row["weight"] == mode]
        print(mode, [(row["near"], round(100 * row["mean"], 5)) for row in rows], flush=True)


if __name__ == "__main__":
    main()
