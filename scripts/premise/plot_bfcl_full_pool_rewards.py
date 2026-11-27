# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Plot full-training-pool BFCL sample/reward designs for Qwen and Gemma."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.bfcl_full_pool_rewards import REWARD_KEYS, reward_metadata  # noqa: E402
from scripts.premise.plot_bfcl_sample_size_contours import ellipse_points  # noqa: E402

MODELS = ("Qwen", "Gemma")
SAMPLER_COLORS = ("#287C85", "#A34F65", "#AF8238", "#6677A1")
TARGET_STYLE = {"full_train": ("#287C85", "#EAF4F5"), "full_validation": ("#A34F65", "#F9EEF1")}
SAMPLER_LABELS = {"uniform": "Uniform", "representative": "Coverage",
                  "cluster_stratified": "Stratified", "adaptive_disagreement": "Adaptive"}
LABELS = {
    "mean": "Task mean",
    "kernel_population_mean": "Kernel population mean",
    "cluster_population_mean": "Cluster population mean",
    "cluster_macro_mean": "Cluster macro mean",
    "repair_aversion": "Repair / regression",
    "preserve_barrier": "Preservation barrier",
    "paired_risk": "Paired-risk penalty",
    "repair_retention_harmonic": "Repair-retention harmonic",
    "cluster_geometric": "Geometric capability",
    "cluster_soft_worst": "Soft worst capability",
    "committee_rarity": "Committee rarity",
    "local_consistency": "Local consistency",
}
CONTOUR_LABELS = {
    "mean": "Task\nmean",
    "kernel_population_mean": "Kernel\npopulation",
    "cluster_population_mean": "Cluster\npopulation",
    "cluster_macro_mean": "Cluster\nmacro",
    "repair_aversion": "Repair /\nregression",
    "preserve_barrier": "Preservation\nbarrier",
    "paired_risk": "Paired-risk\npenalty",
    "repair_retention_harmonic": "Repair-retention\nharmonic",
    "cluster_geometric": "Geometric\ncapability",
    "cluster_soft_worst": "Soft worst\ncapability",
    "committee_rarity": "Committee\nrarity",
    "local_consistency": "Local\nconsistency",
}


def value(row: dict, metric: str) -> float:
    number = row[metric]
    if isinstance(number, dict):
        number = number["mean"]
    return np.nan if number is None else float(number)


def _sampler_metadata(summary: dict) -> tuple[list[str], dict[str, str]]:
    ids = list(dict.fromkeys(row["sampler"] for row in summary["aggregate"]))
    labels = {key: SAMPLER_LABELS.get(key, key.replace("_", " ").capitalize()) for key in ids}
    recorded = summary.get("samplers", [])
    if isinstance(recorded, list) and recorded and all(isinstance(row, dict) for row in recorded):
        keys = [row.get("sampler", row.get("key")) for row in recorded]
        if set(keys) == set(ids):
            ids = keys
            labels.update({row.get("sampler", row.get("key")): row.get("label", row.get("name", key))
                           for row, key in zip(recorded, keys, strict=True)})
    if len(ids) != 4:
        raise ValueError("Expected four recorded sampler designs")
    return ids, labels


def plot(summary: dict, output: Path) -> None:
    """Write paired gain plots and descriptive conditional-correlation displays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    actual_models = {row["model"] for row in summary["aggregate"]}
    if actual_models != set(MODELS):
        raise ValueError("Full-pool reward figures must include Qwen and Gemma only")
    samplers, sampler_labels = _sampler_metadata(summary)
    metadata = {row["reward"]: row for row in summary.get("rewards", reward_metadata())}
    if set(metadata) != set(REWARD_KEYS):
        raise ValueError("Expected exactly the twelve recorded reward rules")
    rows = {(row["model"], row["seed"], row["sampler"], row["reward"], row["n"]): row
            for row in summary["aggregate"]}
    sizes = np.arange(1, 11)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .65,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    sampler_handles = [Line2D([], [], color=color, linewidth=1.1, marker="o", markersize=2.8,
                              label=sampler_labels[sampler])
                       for sampler, color in zip(samplers, SAMPLER_COLORS, strict=True)]
    rose_cmap = LinearSegmentedColormap.from_list("validation_reward", ["#F9EEF1", "#A34F65"])

    def select(model, sampler, reward, n, seed="all"):
        return rows[model, seed, sampler, reward, n]

    def label(reward, contour=False):
        text = (CONTOUR_LABELS if contour else LABELS)[reward]
        return text + (r"$^{\dagger}$" if metadata[reward]["requires_parent"] else "")

    def save(figure, stem):
        for extension in ("pdf", "png"):
            figure.savefig(output / f"{stem}.{extension}", dpi=350, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {stem}.pdf", flush=True)

    for metric, factor, stem, colorbar_label in (
        ("validation_lift", 100, "gain_n10_heatmap", "Validation lift (pp)"),
        ("rho_full_validation", 1, "correlation_n10_heatmap", r"Pearson $\rho$"),
    ):
        matrices = [np.asarray([[factor * value(select(model, sampler, reward, 10), metric)
                                 for sampler in samplers] for reward in REWARD_KEYS]) for model in MODELS]
        finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
        minimum, maximum = (float(finite.min()), float(finite.max())) if finite.size else (0, 1)
        if minimum == maximum:
            minimum, maximum = minimum - .01, maximum + .01
        figure, axes = plt.subplots(1, 2, figsize=(7.7, 5.7), squeeze=False)
        figure.subplots_adjust(left=.255, right=.88, bottom=.19, top=.94, wspace=.12)
        for column, (model, matrix) in enumerate(zip(MODELS, matrices, strict=True)):
            axis = axes[0, column]
            mesh = axis.imshow(matrix, cmap=rose_cmap, vmin=minimum, vmax=maximum, aspect="auto")
            for row_index, reward in enumerate(REWARD_KEYS):
                for column_index, sampler in enumerate(samplers):
                    number = matrix[row_index, column_index]
                    annotation = f"{number:+.3f}" if np.isfinite(number) else "N/A"
                    if metric == "rho_full_validation":
                        count = value(select(model, sampler, reward, 10),
                                      "rho_full_validation_common_checkpoints")
                        annotation += f"\n{count:.1f}/50" if np.isfinite(count) else "\n0/50"
                    white = np.isfinite(number) and (number - minimum) / (maximum - minimum) > .66
                    axis.text(column_index, row_index, annotation, ha="center", va="center",
                              fontsize=6.4, color="white" if white else "#333333")
            axis.set_yticks(np.arange(len(REWARD_KEYS)), [label(key) for key in REWARD_KEYS])
            axis.set_xticks(np.arange(4), [sampler_labels[key] for key in samplers], rotation=35, ha="right")
            axis.tick_params(axis="both", length=0, labelsize=7)
            axis.tick_params(axis="y", labelleft=column == 0)
            axis.set_title(model, fontsize=9)
            for spine in axis.spines.values():
                spine.set_visible(False)
        colorbar_axis = figure.add_axes((.905, .19, .019, .75))
        colorbar = figure.colorbar(mesh, cax=colorbar_axis)
        colorbar.set_label(colorbar_label, fontsize=8)
        colorbar.ax.tick_params(labelsize=7, length=2)
        save(figure, stem)

    def curves(axis, model, reward, metric, factor):
        for sampler, color in zip(samplers, SAMPLER_COLORS, strict=True):
            y = [factor * value(select(model, sampler, reward, int(n)), metric) for n in sizes]
            axis.plot(sizes, y, color=color, linewidth=1.1, marker="o", markersize=2.5)
        axis.axhline(0, color="#BBBBBB", linewidth=.5, zorder=0)
        axis.set_xticks((1, 3, 5, 7, 10))
        axis.set_xlim(.7, 10.3)
        axis.tick_params(labelsize=7, length=2.5)
        axis.grid(axis="y", color="#E9E9E9", linewidth=.4, zorder=0)

    fixed_curves = ("mean", "paired_risk", "cluster_geometric")
    for metric, factor, stem, ylabel in (
        ("validation_lift", 100, "gain_curves_n1_10", "Validation lift (pp)"),
        ("rho_full_validation", 1, "correlation_curves_n1_10", r"Pearson $\rho$"),
    ):
        figure, axes = plt.subplots(2, 3, figsize=(8.0, 4.5), sharex=True, sharey="row", squeeze=False)
        figure.subplots_adjust(left=.10, right=.99, bottom=.13, top=.85, hspace=.22, wspace=.14)
        for row_index, model in enumerate(MODELS):
            for column, reward in enumerate(fixed_curves):
                axis = axes[row_index, column]
                curves(axis, model, reward, metric, factor)
                if row_index == 0:
                    axis.set_title(label(reward), fontsize=8)
                if column == 0:
                    axis.set_ylabel(f"{model}\n{ylabel}", fontsize=8)
                if row_index == 1:
                    axis.set_xlabel(r"Selection sample size $n$", fontsize=8)
        figure.legend(handles=sampler_handles, ncol=4, loc="upper center", frameon=False, fontsize=7)
        save(figure, stem)

    for model in MODELS:
        slug = model.lower()
        for metric, factor, metric_slug, ylabel in (
            ("validation_lift", 100, "gain", "Validation lift (pp)"),
            ("rho_full_validation", 1, "correlation", r"Pearson $\rho$"),
        ):
            figure, axes = plt.subplots(3, 4, figsize=(8.2, 5.5), sharex=True, sharey=True)
            figure.subplots_adjust(left=.085, right=.99, bottom=.12, top=.85, hspace=.38, wspace=.18)
            for axis, reward in zip(axes.flat, REWARD_KEYS, strict=True):
                curves(axis, model, reward, metric, factor)
                axis.set_title(label(reward), fontsize=7.2, pad=4)
            figure.legend(handles=sampler_handles, ncol=4, loc="upper center", frameon=False, fontsize=7)
            figure.supxlabel(r"Selection sample size $n$", fontsize=9, y=.02)
            figure.supylabel(ylabel, fontsize=9, x=.012)
            save(figure, f"{slug}_{metric_slug}_all_rewards_n1_10")

        for target, (color, fill) in TARGET_STYLE.items():
            figure, axes = plt.subplots(4, len(REWARD_KEYS), figsize=(15.0, 5.8), sharex=True, sharey=True,
                                        squeeze=False)
            figure.subplots_adjust(left=.09, right=.995, bottom=.09, top=.89, hspace=.16, wspace=.11)
            for row_index, sampler in enumerate(samplers):
                for column, reward in enumerate(REWARD_KEYS):
                    axis = axes[row_index, column]
                    rho = value(select(model, sampler, reward, 10), f"rho_{target}")
                    if np.isfinite(rho) and abs(rho) < 1:
                        covariance = np.array([[1, rho], [rho, 1]])
                        inner, outer = (ellipse_points(covariance, mass, 121) for mass in (.5, .9))
                        axis.fill(inner[:, 0], inner[:, 1], color=fill, linewidth=0, zorder=1)
                        axis.plot(inner[:, 0], inner[:, 1], color=color, linewidth=.8, zorder=2)
                        axis.plot(outer[:, 0], outer[:, 1], color=color, linewidth=.6, linestyle="--", zorder=2)
                    annotation = rf"$\rho={rho:+.3f}$" if np.isfinite(rho) else r"$\rho=\mathrm{N/A}$"
                    axis.text(.035, .97, annotation, transform=axis.transAxes, fontsize=5.8, color=color,
                              va="top", zorder=4,
                              bbox={"facecolor": "white", "edgecolor": "none", "alpha": .88, "pad": .1})
                    axis.axhline(0, color="#DDDDDD", linewidth=.4)
                    axis.axvline(0, color="#DDDDDD", linewidth=.4)
                    axis.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), xticks=[-2, 0, 2], yticks=[-2, 0, 2])
                    axis.set_aspect("equal")
                    axis.tick_params(labelsize=5.5, length=2)
                    if row_index == 0:
                        axis.set_title(label(reward, contour=True), fontsize=6, pad=4)
                    if column == 0:
                        axis.set_ylabel(sampler_labels[sampler].replace(" ", "\n", 1), fontsize=7)
            xlabel = r"$z(\delta_t^i)$" if target == "full_train" else r"$z(s_{t,\mathrm{val50}}^i)$"
            figure.supxlabel(xlabel, fontsize=9, y=.01)
            figure.supylabel(r"$z(s_t^i)$", fontsize=9, x=.012)
            save(figure, f"{slug}_bivariate_{target}_n10")

    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharex=True, sharey=True)
    figure.subplots_adjust(left=.09, right=.99, bottom=.23, top=.76, wspace=.14)
    for axis, model in zip(axes, MODELS, strict=True):
        for sampler, color in zip(samplers, SAMPLER_COLORS, strict=True):
            counts = [value(select(model, sampler, "mean", int(n)),
                            "rho_full_validation_common_checkpoints") for n in sizes]
            axis.plot(sizes, counts, color=color, marker="o", markersize=2.5, linewidth=1)
        axis.set_title(model, fontsize=9)
        axis.set_xlabel(r"Selection sample size $n$", fontsize=8)
        axis.set_xticks((1, 3, 5, 7, 10))
        axis.set_ylim(0, 50)
        axis.tick_params(labelsize=7)
    axes[0].set_ylabel("Common defined CPs / 50", fontsize=8)
    figure.legend(handles=sampler_handles, ncol=4, loc="upper center", frameon=False, fontsize=7)
    save(figure, "correlation_coverage_n1_10")

    notes = {
        "models": list(MODELS), "rewards": list(REWARD_KEYS), "samplers": samplers,
        "sampler_labels": sampler_labels,
        "main_curve_rules": list(fixed_curves), "rule_order": "Prespecified, never sorted by validation score",
        "parent_cost": "Dagger marks four parent-aware rewards. Five candidates require 5n task evaluations; "
                       "same-n parent outcomes cost n more if unavailable, or 6n total. Existing parent caches "
                       "make this offline study require zero new model calls.",
        "uncertainty": "Curves and heatmap cells are descriptive CP means. Monte Carlo task draws are not "
                       "independent experiments and are not shown as population confidence intervals.",
        "gain_coverage": "All 50 checkpoints contribute to validation lift. Seed means are kept in summary.",
        "correlation_coverage": "For each sampler/n/draw, all 12 rewards share common-defined checkpoints. "
                                "Different samplers/n can have different eligible checkpoints, so cross-sampler "
                                "rho comparisons also change coverage. Heatmap annotations give mean common "
                                "CP count / 50; correlation_coverage_n1_10 shows the full coverage curve.",
        "validation_lift": "Selected full-validation score minus the mean score of the same five proposals; "
                           "this is not parent-relative validation gain.",
        "contours": "Illustrative zero-mean/unit-variance Gaussian covariance [[1,rho],[rho,1]] using the "
                    "common-defined mean within-checkpoint Pearson rho. Solid filled contour encloses 50% "
                    "mass and dashed contour 90%; neither is a confidence interval or normality test.",
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
