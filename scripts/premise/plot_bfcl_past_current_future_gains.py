# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Plot seven archived success-mean selectors against two chronological targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.analyze_bfcl_past_current_future_gains import TARGETS  # noqa: E402
from scripts.premise.plot_bfcl_temporal_parent_preservation import (  # noqa: E402
    CASE_LABELS,
    CASES,
    COLORS,
    FILLS,
    MARKERS,
    MODELS,
    STYLES,
    WEIGHT_LABELS,
    WEIGHTS,
)

TARGET_LABELS = {"past_current": "Past + current", "future": "Future"}


def index_rows(summary: dict) -> dict[tuple, dict]:
    """Require one fixed cohort and valid target decomposition for all curves."""
    indexed = {}
    count = summary["checkpoints_per_model"]
    for row in summary["aggregate"]:
        if row["scope"] != "all":
            continue
        if row["checkpoints"] != count:
            raise ValueError("All samplers and both targets must use the same fixed checkpoint count")
        key = tuple(row[k] for k in ("model", "weight", "case", "n", "target"))
        if key in indexed:
            raise ValueError("Repeated curve coordinate")
        for suffix in ("", "_unqueried"):
            gain = row[f"selected_gain{suffix}"]["mean"]
            proposal = row[f"proposal_gain{suffix}"]["mean"]
            lift = row[f"selection_lift{suffix}"]["mean"]
            if not np.isclose(gain, proposal + lift, rtol=0, atol=1e-12):
                raise ValueError("Selected gain must decompose into proposal gain plus selection lift")
        indexed[key] = row
    return indexed


def _value(row: dict, metric: str, field: str = "mean") -> float:
    value = row[metric].get(field)
    return float(value) if value is not None else np.nan


def plot(summary: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    indexed = index_rows(summary)
    references = {(r["model"], r["target"]): r for r in summary["validation_reference_aggregate"]
                  if r["scope"] == "all"}
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
                         "axes.labelsize": 10, "axes.titlesize": 9.5, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "axes.linewidth": .65, "pdf.fonttype": 42, "ps.fonttype": 42})
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = {"files": [], "checkpoints_per_model": summary["checkpoints_per_model"],
                   "std_bands": "Monte Carlo task-sampling std of checkpoint-averaged values; not a seed CI",
                   "val10": "Only n=10 was evaluated; no interpolated validation curve"}

    def save(figure, stem):
        for extension in ("pdf", "png"):
            figure.savefig(output / f"{stem}.{extension}", dpi=260, bbox_inches="tight", pad_inches=.035)
        plt.close(figure)
        diagnostics["files"].append(stem)
        print(f"Saved {stem}.pdf", flush=True)

    def draw(axis, model, weight, target, metric, *, bands=True):
        is_rho = metric == "rho"
        scale = 1 if is_rho else 100
        for case in CASES:
            rows = [indexed[model, weight, case, n, target] for n in range(1, 11)
                    if (model, weight, case, n, target) in indexed]
            x = np.asarray([r["n"] for r in rows])
            y = scale * np.asarray([_value(r, metric) for r in rows])
            std = scale * np.asarray([_value(r, metric, "std") for r in rows])
            if bands:
                axis.fill_between(x, y - std, y + std, facecolor=FILLS[case], alpha=.45, linewidth=0)
            axis.plot(x, y, color=COLORS[case], linestyle=STYLES[case], linewidth=1.2,
                      marker=MARKERS[case], markersize=3.0, markeredgewidth=.55,
                      label=CASE_LABELS[case], zorder=5 if case == "current_past" else 3)
        if metric == "selected_gain":
            baseline = indexed[model, weight, "current", 10, target]["proposal_gain"]["mean"]
            axis.axhline(100 * baseline, color="#777777", linewidth=.85, linestyle=(0, (4, 3)))
        if metric != "rho" and metric != "selected_gain_unqueried":
            ref = references[model, target]
            axis.scatter([10], [100 * ref[metric]["mean"]], s=54, marker="*", color="#111111",
                         edgecolor="white", linewidth=.5, zorder=7, clip_on=False)
        elif metric == "selected_gain_unqueried":
            ref = references[model, target]
            axis.scatter([10], [100 * ref[metric]["mean"]], s=54, marker="*", color="#111111",
                         edgecolor="white", linewidth=.5, zorder=7, clip_on=False)
        axis.axhline(0, color="#B1B1B1", linewidth=.55, zorder=0)
        axis.set(xlim=(.85, 10.25), xticks=(1, 2, 4, 6, 8, 10))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", linewidth=.4, alpha=.2)
        axis.tick_params(length=2.5, pad=2)

    handles = [Line2D([0], [0], color=COLORS[c], linestyle=STYLES[c], marker=MARKERS[c],
                      markersize=3, linewidth=1.2, label=CASE_LABELS[c]) for c in CASES]
    val_handle = Line2D([0], [0], marker="*", linestyle="none", color="#111111", markersize=7,
                        label="Val 10")
    proposal_handle = Line2D([0], [0], color="#777777", linestyle=(0, (4, 3)), linewidth=.85,
                             label=r"Proposal mean $\mu_\delta$")
    metric_labels = {
        "selected_gain": r"$\delta_t^{i^\star}$ (pp)",
        "selection_lift": r"$\delta_t^{i^\star}-\mu_\delta$ (pp)",
        "selected_gain_unqueried": r"$\delta_t^{i^\star}$ (pp)",
        "selection_lift_unqueried": r"$\delta_t^{i^\star}-\mu_\delta$ (pp)",
        "rho": r"$\rho$",
    }
    for target in TARGETS:
        for metric in metric_labels:
            figure, axes = plt.subplots(2, 2, figsize=(7.2, 4.45), sharex=True, sharey="row", squeeze=False)
            figure.subplots_adjust(left=.09, right=.985, bottom=.11, top=.74, wspace=.27, hspace=.42)
            for row_index, model in enumerate(MODELS):
                for column, weight in enumerate(WEIGHTS):
                    axis = axes[row_index, column]
                    draw(axis, model, weight, target, metric)
                    axis.set_title(f"{model} · {WEIGHT_LABELS[weight]}", pad=4)
                    if column == 0:
                        axis.set_ylabel(metric_labels[metric])
                    if row_index == 1:
                        axis.set_xlabel(r"Sample size $n$", labelpad=3)
            extra = [proposal_handle, val_handle] if metric == "selected_gain" else (
                [] if metric == "rho" else [val_handle])
            figure.legend(handles=[*handles, *extra], loc="upper center", bbox_to_anchor=(.52, .94),
                          ncol=3, frameon=False, fontsize=7.6, handlelength=2.2, columnspacing=1.2,
                          labelspacing=.5)
            qualifier = " · Unqueried tasks" if "unqueried" in metric else ""
            figure.suptitle(TARGET_LABELS[target] + qualifier, fontsize=10, y=.995)
            save(figure, f"success_mean_{metric}_{target}_n1_10")

    # Compact direct comparison of the user's preferred current+past/task-mean rule.
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 4.1), sharex=True, sharey="row", squeeze=False)
    figure.subplots_adjust(left=.09, right=.985, bottom=.11, top=.85, wspace=.28, hspace=.4)
    for r, model in enumerate(MODELS):
        for c, target in enumerate(TARGETS):
            axis = axes[r, c]
            rows = [indexed[model, "task_mean", "current_past", n, target] for n in range(2, 11)]
            x = np.asarray([row["n"] for row in rows])
            for metric, label, color, fill, style in (
                ("selected_gain", "Full target", "#287C85", "#EAF4F5", "-"),
                ("selected_gain_unqueried", "Query excluded", "#A34F65", "#F9EEF1", "--"),
            ):
                y = 100 * np.asarray([_value(row, metric) for row in rows])
                std = 100 * np.asarray([_value(row, metric, "std") for row in rows])
                axis.fill_between(x, y - std, y + std, color=fill, linewidth=0)
                axis.plot(x, y, color=color, linestyle=style, marker="o", markersize=3,
                          linewidth=1.3, label=label)
            baseline = 100 * rows[0]["proposal_gain"]["mean"]
            axis.axhline(baseline, color="#777777", linestyle=(0, (4, 3)), linewidth=.85)
            axis.axhline(0, color="#B1B1B1", linewidth=.55)
            axis.scatter([10], [100 * references[model, target]["selected_gain"]["mean"]],
                         marker="*", color="#111111", s=50, clip_on=False, zorder=5)
            axis.set_title(f"{model} · {TARGET_LABELS[target]}", pad=4)
            axis.set(xlim=(1.7, 10.25), xticks=(2, 4, 6, 8, 10))
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", linewidth=.4, alpha=.2)
            if c == 0:
                axis.set_ylabel(metric_labels["selected_gain"])
            if r == 1:
                axis.set_xlabel(r"Sample size $n$")
    figure.legend(handles=[*axes[0, 0].get_legend_handles_labels()[0], proposal_handle, val_handle],
                  loc="upper center", bbox_to_anchor=(.52, 1.005), ncol=4, frameon=False,
                  fontsize=8, columnspacing=1.5)
    save(figure, "current_past_task_mean_two_targets")
    (output / "plot_manifest.json").write_text(json.dumps(diagnostics, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot(json.loads(args.input.read_text()), args.output or args.input.parent / "figures")


if __name__ == "__main__":
    main()
