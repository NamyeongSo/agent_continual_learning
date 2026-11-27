# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Plot separate, exploratory process-reward ablations from measured BFCL traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.analyze_bfcl_process_reward_ablations import COHORT, RECIPES  # noqa: E402
from scripts.premise.plot_bfcl_sample_size_contours import ellipse_points  # noqa: E402
from scripts.premise.plot_bfcl_trajectory_rewards import (  # noqa: E402
    MODELS,
    TARGET_STYLE,
    metric_value,
    standardize_bivariate,
)

LABELS = {
    "success_count": "Success",
    "all_trajectory_loop_penalty": "Avoidable error loops",
    "paired_success_efficiency": "Parent-relative efficiency",
}
MARKERS = ("o", "s", "^")
LINESTYLES = ("-", "--", "-.")


def plot(summary: dict, output: Path) -> None:
    """Keep post hoc diagnostics distinct from the six frozen reward comparisons."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if tuple(summary["recipes"]) != RECIPES:
        raise ValueError("Exploratory figure recipe order must match its separate analysis")
    rows = summary["aggregate"]
    if any(row["model"] not in MODELS or row["cohort"] != COHORT for row in rows):
        raise ValueError("Only Qwen/Gemma process-ablation records may be plotted here")
    indexed = {(row["model"], row["recipe"], row["n"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Repeated aggregate rows")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .65,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    sizes = np.arange(1, 11)
    diagnostics = {"status": "exploratory_post_hoc_process_ablations", "analysis_created_at": summary["created_at"],
                   "files": [], "bivariate": [], "coverage": summary["coverage"]}

    def handles():
        return [Line2D([], [], color="#444444", linewidth=1, marker=MARKERS[index],
                       linestyle=LINESTYLES[index], markersize=3, label=LABELS[recipe])
                for index, recipe in enumerate(RECIPES)]

    def save(figure, stem):
        for extension in ("pdf", "png"):
            figure.savefig(output / f"{stem}.{extension}", dpi=350, bbox_inches="tight")
        plt.close(figure)
        diagnostics["files"].append(stem)
        print(f"Saved {stem}.pdf", flush=True)

    def title(model):
        counts = [indexed[model, "success_count", int(n)]["checkpoints"] for n in sizes
                  if (model, "success_count", int(n)) in indexed]
        if not counts:
            return f"{model} (0 CPs)"
        count = str(min(counts)) if min(counts) == max(counts) else f"{min(counts)}-{max(counts)}"
        return f"{model} ({count} CPs)"

    def curves(metric, factor, ylabel, target, stem):
        color = TARGET_STYLE[target][0]
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), sharex=True, sharey=True)
        figure.subplots_adjust(left=.10, right=.985, bottom=.19, top=.83, wspace=.13)
        for column, (axis, model) in enumerate(zip(axes, MODELS, strict=True)):
            for index, recipe in enumerate(RECIPES):
                values = np.asarray([factor * metric_value(indexed.get((model, recipe, int(n)), {}), metric)
                                     for n in sizes])
                if not np.isfinite(values).any():
                    continue
                axis.plot(sizes, values, color=color, marker=MARKERS[index], linestyle=LINESTYLES[index],
                          linewidth=1.3 if index == 0 else 1, markersize=3.1,
                          markerfacecolor=color if index == 0 else "white", markeredgewidth=.7)
            axis.axhline(0, color="#B9B9B9", linewidth=.5, zorder=0)
            axis.set_title(title(model), fontsize=8.5, pad=4)
            axis.set(xlim=(.7, 10.3), xticks=(1, 3, 5, 7, 10))
            axis.set_xlabel(r"Selection sample size $n$", fontsize=8)
            axis.tick_params(labelsize=7, length=2.5)
            axis.grid(axis="y", color="#E9E9E9", linewidth=.4, zorder=0)
            if column == 0:
                axis.set_ylabel(ylabel, fontsize=8)
        figure.legend(handles=handles(), ncol=3, loc="upper center", frameon=False, fontsize=7,
                      columnspacing=1.1, handlelength=2.4)
        save(figure, stem)

    for target, gain_metric, gain_label in (
        ("full_validation", "validation_lift", "Validation lift (pp)"),
        ("full_train", "train_parent_gain", r"Training gain $\delta_t^{i^\star}$ (pp)"),
    ):
        curves(gain_metric, 100, gain_label, target, f"process_gain_{target}_n1_10")
        curves(f"rho_{target}", 1, r"Pearson $\rho$", target,
               f"process_correlation_pearson_{target}_n1_10")
        curves(f"spearman_{target}", 1, r"Spearman $\rho_{\mathrm{S}}$", target,
               f"process_correlation_spearman_{target}_n1_10")

    for target, (color, fill) in TARGET_STYLE.items():
        clouds, notes = standardize_bivariate(summary["checkpoint_scores"], target, RECIPES, cohort=COHORT)
        diagnostics["bivariate"].extend(notes)
        if not any(len(cloud) for cloud in clouds.values()):
            continue
        by_key = {(row["model"], row["recipe"]): row for row in notes}
        figure, axes = plt.subplots(2, 3, figsize=(7.3, 4.7), sharex=True, sharey=True)
        figure.subplots_adjust(left=.10, right=.99, bottom=.14, top=.91, hspace=.18, wspace=.13)
        for row_index, model in enumerate(MODELS):
            for column, recipe in enumerate(RECIPES):
                axis, cloud, note = axes[row_index, column], clouds[model, recipe], by_key[model, recipe]
                if len(cloud):
                    covariance = np.cov(cloud, rowvar=False, ddof=0)
                    axis.scatter(cloud[:, 0], cloud[:, 1], s=7, color=color, alpha=.23,
                                 edgecolors="none", rasterized=True, zorder=2)
                    if np.linalg.eigvalsh(covariance).min() > 1e-12:
                        inner, outer = (ellipse_points(covariance, mass, 181) for mass in (.5, .9))
                        axis.fill(inner[:, 0], inner[:, 1], color=fill, zorder=1)
                        axis.plot(inner[:, 0], inner[:, 1], color=color, linewidth=.9, zorder=3)
                        axis.plot(outer[:, 0], outer[:, 1], color=color, linewidth=.7, linestyle="--", zorder=3)
                    annotation = rf"$\rho={note['rho']:+.3f}$" + f"\n{note['common_defined_checkpoints']} CPs"
                else:
                    annotation = "N/A\n0 CPs"
                axis.text(.04, .96, annotation, transform=axis.transAxes, ha="left", va="top", fontsize=7,
                          color=color, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8})
                axis.axhline(0, color="#D6D6D6", linewidth=.4, zorder=0)
                axis.axvline(0, color="#D6D6D6", linewidth=.4, zorder=0)
                axis.set(xlim=(-3, 3), ylim=(-3, 3), xticks=(-2, 0, 2), yticks=(-2, 0, 2))
                axis.set_aspect("equal", adjustable="box")
                axis.tick_params(labelsize=7, length=2.5)
                if row_index == 0:
                    axis.set_title(LABELS[recipe], fontsize=8, pad=5)
                if column == 0:
                    label = "Full training score" if target == "full_train" else "Full validation score"
                    axis.set_ylabel(f"{model}\n{label} ($z$)", fontsize=8)
                if row_index == 1:
                    axis.set_xlabel("Standardized selection score", fontsize=8)
        save(figure, f"process_bivariate_{target}_n10")

    color = TARGET_STYLE["full_validation"][0]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), sharey=True)
    figure.subplots_adjust(left=.10, right=.985, bottom=.18, top=.83, wspace=.13)
    for column, (axis, model) in enumerate(zip(axes, MODELS, strict=True)):
        for index, recipe in enumerate(RECIPES):
            row = indexed.get((model, recipe, 10), {})
            x, y = metric_value(row, "selected_environment_steps"), 100 * metric_value(row, "validation_lift")
            if np.isfinite(x) and np.isfinite(y):
                axis.scatter(x, y, marker=MARKERS[index], s=31, color=color,
                             facecolors=color if index == 0 else "none", linewidths=.9)
        count = indexed.get((model, "success_count", 10), {}).get("checkpoints", 0)
        axis.set_title(f"{model} ({count} CPs)", fontsize=8.5)
        axis.set_xlabel("Environment steps / selected task", fontsize=8)
        axis.axhline(0, color="#B9B9B9", linewidth=.5)
        axis.tick_params(labelsize=7, length=2.5)
        if column == 0:
            axis.set_ylabel("Validation lift (pp)", fontsize=8)
    figure.legend(handles=handles(), ncol=3, loc="upper center", frameon=False, fontsize=7)
    save(figure, "process_execution_validation_tradeoff_n10")

    (output / "figure_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
    (output / "README.md").write_text(
        "# Exploratory process-reward ablations\n\n"
        "These two additional recipes were introduced after inspecting process-feature prevalence. "
        "They are separate from the six frozen comparisons. Both use only the same n sampled training "
        "tasks; the paired efficiency rule also uses their parent trajectories. No new inference was run.\n\n"
        "- `process_gain_*`: validation lift is selected full-validation score minus the five-proposal "
        "mean. Training gain is relative to the measured parent. Each model uses matched trace-complete "
        "checkpoints across all three recipes. Counts may vary with n.\n"
        "- `process_correlation_*`: mean within-checkpoint Pearson/Spearman correlation; constant "
        "candidate vectors remain undefined and are excluded from each metric's mean.\n"
        "- `process_bivariate_*`: actual n=10 five-candidate pairs, standardized within checkpoint. "
        "All panels for a model share jointly nonconstant checkpoints across the three recipes. "
        "The displayed rho equals their mean within-checkpoint Pearson correlation. Gaussian 50%/90% "
        "ellipses are descriptive fitted contours, not confidence regions.\n"
        "- `process_execution_validation_tradeoff_n10`: selected-task cost is a mechanism diagnostic "
        "on the optimized samples, rather than held-out efficiency.\n\n"
        "The n=1..10 curves use nested prefixes of one frozen uniform ten-task ordering per checkpoint. "
        "No uncertainty bands or artificial observations are added. Exact bivariate checkpoint identities "
        "and coverage are in `figure_diagnostics.json`.\n\n"
        + "\n".join(f"- [{stem}.pdf]({stem}.pdf) · [PNG]({stem}.png)" for stem in diagnostics["files"])
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot(json.loads(args.summary.read_text()), args.output or args.summary.parent / "figures")


if __name__ == "__main__":
    main()
