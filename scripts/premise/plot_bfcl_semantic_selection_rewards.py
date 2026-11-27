# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Plot training-only semantic selection rewards for the paired Qwen/Gemma study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.plot_bfcl_sample_size_contours import ellipse_points  # noqa: E402

MODELS = ("Qwen", "Gemma")
SEED_COLORS = ("#287C85", "#A34F65", "#AF8238")
TARGET_STYLE = {"full_train": ("#287C85", "#EAF4F5"), "full_validation": ("#A34F65", "#F9EEF1")}
BASELINE = "period_mean"
EQUIVALENT_CONTROLS = ("paired_linear", "log_mean")
LABELS = {
    "period_mean": "Period mean",
    "task_mean": "Task mean",
    "balanced_mean": "Balanced mean",
    "paired_linear": "Paired linear control",
    "log_mean": "Log control",
    "repair_aversion": "Repair / regression",
    "preserve_barrier": "Preservation barrier",
    "geometric_balance": "Geometric balance",
    "soft_worst": "Soft worst group",
    "conflict_aversion": "Conflict penalty",
    "paired_risk": "Paired-risk penalty",
    "repair_retention_harmonic": "Repair-retention harmonic",
    "transfer_breadth": "Transfer breadth",
}
CONTOUR_LABELS = {
    "period_mean": "Period\nmean",
    "repair_aversion": "Repair /\nregression",
    "preserve_barrier": "Preservation\nbarrier",
    "geometric_balance": "Geometric\nbalance",
    "soft_worst": "Soft worst\ngroup",
    "conflict_aversion": "Conflict\npenalty",
    "paired_risk": "Paired-risk\npenalty",
    "repair_retention_harmonic": "Repair-retention\nharmonic",
    "transfer_breadth": "Transfer\nbreadth",
}


def study_rewards(summary: dict, *, include_controls: bool = False) -> list[str]:
    """Keep the recorded, prespecified ordering rather than sorting by validation."""
    return [spec["reward"] for spec in summary["rewards"]
            if include_controls or spec["reward"] not in EQUIVALENT_CONTROLS]


def value(row: dict, metric: str) -> float:
    number = row[metric]
    if isinstance(number, dict):
        number = number["mean"]
    return np.nan if number is None else float(number)


def plot(summary: dict, output: Path) -> None:
    """Write PDFs/PNGs without treating task draws as population confidence intervals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    models = tuple(model for model in MODELS if any(row["model"] == model for row in summary["aggregate"]))
    if models != MODELS or any(row["model"] not in MODELS for row in summary["aggregate"]):
        raise ValueError("This study must contain Qwen and Gemma only")
    indexed = {(row["model"], row["cohort"], row["seed"], row["reward"]): row for row in summary["aggregate"]}
    names = study_rewards(summary)
    labels = [LABELS.get(name, name.replace("_", " ")) for name in names]
    position = np.arange(len(names))
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.65,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    seed_handles = [Line2D([], [], marker="o", linewidth=0, color=color, markersize=3.5, label=f"Seed {seed}")
                    for seed, color in zip((42, 43, 44), SEED_COLORS, strict=True)]
    mean_handle = Line2D([], [], marker="D", linewidth=0, color="#222222", markersize=3.8, label="CP mean")

    def select(model, cohort, seed, name):
        return indexed[model, cohort, seed, name]

    def save(figure, stem):
        for extension in ("pdf", "png"):
            figure.savefig(output / f"{stem}.{extension}", dpi=400, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {stem}.pdf", flush=True)

    def axis_style(axis, column, xlabel):
        axis.axvline(0, color="#B3B3B3", linewidth=0.55, zorder=0)
        axis.set_yticks(position, labels)
        axis.invert_yaxis()
        axis.tick_params(axis="both", labelsize=7, length=2.5)
        axis.tick_params(axis="y", labelleft=column == 0)
        axis.set_xlabel(xlabel, fontsize=8)
        axis.grid(axis="x", color="#E9E9E9", linewidth=0.45, zorder=0)

    def seed_points(axis, model, cohort, metric, factor=1):
        means = [factor * value(select(model, cohort, "all", name), metric) for name in names]
        axis.barh(position, means, height=0.55, color="#F4F4F4", edgecolor="#D8D8D8", linewidth=0.4,
                  zorder=1)
        for offset, seed, color in zip((-.16, 0, .16), (42, 43, 44), SEED_COLORS, strict=True):
            numbers = [factor * value(select(model, cohort, seed, name), metric) for name in names]
            axis.scatter(numbers, position + offset, s=12, color=color, alpha=.9, zorder=3,
                         linewidths=.3, edgecolors="white")
        axis.scatter(means, position, marker="D", s=17, color="#222222", zorder=4,
                     linewidths=.3, edgecolors="white")

    # Spearman is the primary rank diagnostic because nonlinear score transforms can change Pearson rho.
    for cohort in ("cp36", "cp50"):
        figure, axes = plt.subplots(2, 2, figsize=(8.1, 7.5), sharex="row", squeeze=False)
        figure.subplots_adjust(left=.245, right=.985, top=.92, bottom=.07, hspace=.24, wspace=.12)
        for column, model in enumerate(models):
            for row_index, (metric, xlabel, factor) in enumerate((
                ("validation_lift", "Validation lift (pp)", 100),
                ("spearman_full_validation", r"Spearman $\rho_{\mathrm{S}}$ · full validation", 1),
            )):
                axis = axes[row_index, column]
                seed_points(axis, model, cohort, metric, factor)
                baseline = factor * value(select(model, cohort, "all", BASELINE), metric)
                axis.axvline(baseline, color="#777777", linewidth=.75, linestyle=":", zorder=2)
                axis_style(axis, column, xlabel)
                if row_index == 0:
                    axis.set_title(model, fontsize=9, pad=5)
        figure.legend(handles=[*seed_handles, mean_handle], loc="upper center", ncol=4, frameon=False,
                      fontsize=7, bbox_to_anchor=(.61, .995))
        save(figure, f"reward_overview_{cohort}")

    # Parent-relative repairs and regressions are measured on the same ten selection tasks.
    figure, axes = plt.subplots(2, 2, figsize=(8.1, 7.5), sharex="row", squeeze=False)
    figure.subplots_adjust(left=.245, right=.985, top=.92, bottom=.07, hspace=.24, wspace=.12)
    for column, model in enumerate(models):
        for row_index, (metric, xlabel) in enumerate((
            ("selected_repaired_parent_failures", "Repaired parent failures / 10 tasks"),
            ("selected_broken_parent_successes", "Broken parent successes / 10 tasks"),
        )):
            axis = axes[row_index, column]
            seed_points(axis, model, "cp50", metric)
            reference = value(select(model, "cp50", "all", BASELINE), metric)
            axis.axvline(reference, color="#777777", linewidth=.75, linestyle=":", zorder=2)
            axis_style(axis, column, xlabel)
            if row_index == 0:
                axis.set_title(model, fontsize=9, pad=5)
    figure.legend(handles=[*seed_handles, mean_handle], loc="upper center", ncol=4, frameon=False,
                  fontsize=7, bbox_to_anchor=(.61, .995))
    save(figure, "parent_preservation_repair_cp50")

    remaining_metrics = (
        ("unselected_train_repaired_fraction", "Repairs on unselected 40 tasks (%)"),
        ("unselected_train_broken_fraction", "Regressions on unselected 40 tasks (%)"),
    )
    if all(metric in row for metric, _ in remaining_metrics for row in summary["aggregate"]):
        figure, axes = plt.subplots(2, 2, figsize=(8.1, 7.5), sharex="row", squeeze=False)
        figure.subplots_adjust(left=.245, right=.985, top=.92, bottom=.07, hspace=.24, wspace=.12)
        for column, model in enumerate(models):
            for row_index, (metric, xlabel) in enumerate(remaining_metrics):
                axis = axes[row_index, column]
                seed_points(axis, model, "cp50", metric, 100)
                reference = 100 * value(select(model, "cp50", "all", BASELINE), metric)
                axis.axvline(reference, color="#777777", linewidth=.75, linestyle=":", zorder=2)
                axis_style(axis, column, xlabel)
                if row_index == 0:
                    axis.set_title(model, fontsize=9, pad=5)
        figure.legend(handles=[*seed_handles, mean_handle], loc="upper center", ncol=4, frameon=False,
                      fontsize=7, bbox_to_anchor=(.61, .995))
        save(figure, "preservation_outside_selection_cp50")

    # Expose both targets and both correlation statistics, retaining the prespecified formula order.
    figure, axes = plt.subplots(2, 2, figsize=(8.1, 7.5), sharex="row", squeeze=False)
    figure.subplots_adjust(left=.245, right=.985, top=.92, bottom=.07, hspace=.24, wspace=.12)
    for column, model in enumerate(models):
        for row_index, (prefix, xlabel) in enumerate((("rho", r"Pearson $\rho$"),
                                                     ("spearman", r"Spearman $\rho_{\mathrm{S}}$"))):
            axis = axes[row_index, column]
            for offset, target in zip((-.13, .13), TARGET_STYLE, strict=True):
                color, _ = TARGET_STYLE[target]
                numbers = [value(select(model, "cp50", "all", name), f"{prefix}_{target}") for name in names]
                axis.plot(numbers, position + offset, marker="o", linestyle="none", markersize=3.3,
                          color=color, zorder=3)
            axis_style(axis, column, xlabel)
            if row_index == 0:
                axis.set_title(model, fontsize=9, pad=5)
    handles = [Line2D([], [], marker="o", linewidth=0, color=TARGET_STYLE[target][0], markersize=3.5,
                      label=label) for target, label in (("full_train", "Full train"),
                                                         ("full_validation", "Full validation"))]
    figure.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=7,
                  bbox_to_anchor=(.61, .995))
    save(figure, "correlation_targets_cp50")

    contour_names = [BASELINE] + [spec["reward"] for spec in summary["rewards"]
                                 if spec["reward"] not in (BASELINE, "task_mean", "balanced_mean",
                                                           *EQUIVALENT_CONTROLS)]
    for target, (color, fill) in TARGET_STYLE.items():
        figure, axes = plt.subplots(2, len(contour_names), figsize=(1.34 * len(contour_names) + .4, 3.35),
                                    sharex=True, sharey=True, squeeze=False)
        figure.subplots_adjust(left=.055, right=.995, top=.85, bottom=.15, hspace=.16, wspace=.12)
        for row_index, model in enumerate(models):
            for column, name in enumerate(contour_names):
                axis = axes[row_index, column]
                baseline = value(select(model, "cp50", "all", BASELINE), f"rho_{target}")
                rho = value(select(model, "cp50", "all", name), f"rho_{target}")
                contours = [(rho, color, "-")]
                if name != BASELINE:
                    contours.insert(0, (baseline, "#888888", ":"))
                for correlation, curve_color, style in contours:
                    if np.isfinite(correlation) and abs(correlation) < 1:
                        covariance = np.array([[1, correlation], [correlation, 1]])
                        inner, outer = (ellipse_points(covariance, mass, 121) for mass in (.5, .9))
                        if style == "-":
                            axis.fill(inner[:, 0], inner[:, 1], color=fill, linewidth=0, zorder=1)
                        axis.plot(inner[:, 0], inner[:, 1], color=curve_color, linewidth=.85,
                                  linestyle=style, zorder=2)
                        axis.plot(outer[:, 0], outer[:, 1], color=curve_color, linewidth=.6,
                                  linestyle="--" if style == "-" else ":", zorder=2)
                rho_text = rf"$\rho={rho:+.3f}$" if np.isfinite(rho) else r"$\rho=\mathrm{N/A}$"
                axis.text(.035, .97, rho_text, transform=axis.transAxes, fontsize=6.2, color=color, va="top",
                          zorder=4, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": .1})
                if name != BASELINE:
                    baseline_text = (rf"$\rho_{{\mathrm{{P}}}}={baseline:+.3f}$" if np.isfinite(baseline)
                                     else r"$\rho_{\mathrm{P}}=\mathrm{N/A}$")
                    axis.text(.035, .81, baseline_text, transform=axis.transAxes, fontsize=6.2,
                              color="#777777", va="top", zorder=4,
                              bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": .1})
                axis.axhline(0, color="#DDDDDD", linewidth=.4)
                axis.axvline(0, color="#DDDDDD", linewidth=.4)
                axis.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), xticks=[-2, 0, 2], yticks=[-2, 0, 2])
                axis.set_aspect("equal")
                axis.tick_params(labelsize=6, length=2)
                if row_index == 0:
                    axis.set_title(CONTOUR_LABELS.get(name, name), fontsize=6.5, pad=4)
                if column == 0:
                    axis.set_ylabel(model, fontsize=8)
        xlabel = r"$z(\delta_t^i)$" if target == "full_train" else r"$z(s_{t,\mathrm{val50}}^i)$"
        figure.supxlabel(xlabel, fontsize=9, y=.02)
        figure.supylabel(r"$z(s_t^i)$", fontsize=9, x=.005)
        handles = [Line2D([], [], color=color, linewidth=.9, label="Selection reward"),
                   Line2D([], [], color="#888888", linewidth=.9, linestyle=":", label="Period mean")]
        figure.legend(handles=handles, ncol=2, frameon=False, loc="upper center", fontsize=7,
                      bbox_to_anchor=(.5, 1.02))
        save(figure, f"bivariate_{target}")

    notes = {
        "models": list(models), "main_rewards": names, "equivalent_controls_omitted": EQUIVALENT_CONTROLS,
        "main_rank_statistic": "Within-checkpoint Spearman correlation, common-defined coverage from summary",
        "seed_points": "Means over checkpoints within each seed; black diamonds are CP-weighted means",
        "uncertainty": "No task-draw confidence interval is drawn; seed points are descriptive, not a CI",
        "parent_tradeoff": "Expected number of selected-candidate repairs/breaks on the same ten selection tasks",
        "outside_selection_tradeoff": "Post hoc diagnostic only: parent failures repaired and parent successes "
                                      "broken on the other 40 training tasks, each divided by 40 and expressed "
                                      "as percent. These targets are not used by selection formulas. The "
                                      "remaining tasks are not a new independent benchmark or split.",
        "contours": "Illustrative standardized Gaussian with covariance [[1,rho],[rho,1]], using summary "
                    "common-defined mean Pearson rho. Filled solid contour encloses 50%, dashed encloses 90%. "
                    "These are probability-mass contours, not confidence intervals or a normality test.",
        "contour_selection": "Prespecified period mean and all eight novel rewards; no validation rank sorting",
        "validation_lift": "Selected full-validation reward minus the five-proposal full-validation mean; "
                           "parent-relative validation gain is not measured",
        "colors": TARGET_STYLE,
    }
    (output / "figure_notes.json").write_text(json.dumps(notes, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plot(json.loads(args.source.read_text()), args.output)


if __name__ == "__main__":
    main()
