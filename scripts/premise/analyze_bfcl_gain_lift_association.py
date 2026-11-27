# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Describe gain/lift association without treating checkpoints as independent runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.analyze_bfcl_temporal_selection import write_csv  # noqa: E402
from scripts.premise.plot_bfcl_checkpoint_distributions import validated_rows  # noqa: E402
from scripts.premise.plot_bfcl_temporal_parent_preservation import (  # noqa: E402
    CASE_LABELS,
    CASES,
    COLORS,
    MARKERS,
    MODELS,
    WEIGHT_LABELS,
    WEIGHTS,
)

SEEDS = (42, 43, 44)
SEED_COLORS = {42: "#287C85", 43: "#A34F65", 44: "#8A733F"}
SEED_MARKERS = {42: "o", 43: "^", 44: "s"}


def correlation(x, y, rank: bool = False) -> float | None:
    """Constant vectors remain undefined, including floating-point zero vectors."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2 or not np.isfinite([x, y]).all():
        raise ValueError("Need at least two paired finite outcomes")
    if np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return None
    if rank:
        x, y = rankdata(x), rankdata(y)
    return float(np.corrcoef(x, y)[0, 1])


def center_by_group(x, groups) -> np.ndarray:
    x, groups = np.asarray(x, dtype=float), np.asarray(groups)
    result = x.copy()
    for group in np.unique(groups):
        mask = groups == group
        result[mask] -= x[mask].mean()
    return result


def describe_group(rows: list[dict]) -> list[dict]:
    x = np.array([r["gain_past"] for r in rows])
    y = np.array([r["lift_validation40"] for r in rows])
    seeds = np.array([r["seed"] for r in rows])
    identity = {k: rows[0][k] for k in ("model", "weight", "n", "case")}
    result = []
    for scope in ("pooled", *SEEDS, "seed_centered"):
        mask = seeds == scope if isinstance(scope, int) else np.ones(len(rows), dtype=bool)
        a, b = x[mask], y[mask]
        if scope == "seed_centered":
            a, b = center_by_group(a, seeds), center_by_group(b, seeds)
        result.append({**identity, "scope": scope, "checkpoints": len(a), "pearson": correlation(a, b),
                       "spearman": correlation(a, b, rank=True), "p_value": None,
                       "gain_past_mean_pp": float(a.mean() * 100),
                       "validation_lift_mean_pp": float(b.mean() * 100)})
    return result


def label(value: float | None) -> str:
    return "undefined" if value is None else f"{value:+.2f}"


def setup_plots():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 10, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.linewidth": .6})
    return plt


def draw_scatter(ax, points: list[dict], stats: list[dict], *, detailed: bool = False):
    for seed in SEEDS:
        subset = [r for r in points if r["seed"] == seed]
        x = np.array([r["gain_past"] * 100 for r in subset])
        y = np.array([r["lift_validation40"] * 100 for r in subset])
        coefficient = next(r["pearson"] for r in stats if r["scope"] == seed)
        ax.scatter(x, y, color=SEED_COLORS[seed], marker=SEED_MARKERS[seed],
                   s=27 if detailed else 12, alpha=.85, edgecolors="white", linewidths=.35,
                   label=f"Seed {seed}: $r={label(coefficient)}$")
        if np.ptp(x) > 1e-10:
            slope, intercept = np.polyfit(x, y, 1)
            ends = np.array([x.min(), x.max()])
            ax.plot(ends, slope * ends + intercept, color=SEED_COLORS[seed], linewidth=.9, alpha=.55)
    ax.axhline(0, color="#BBBBBB", linewidth=.6, zorder=0)
    ax.axvline(0, color="#BBBBBB", linewidth=.6, zorder=0)
    ax.grid(alpha=.16, linewidth=.4)
    ax.spines[["top", "right"]].set_visible(False)
    pooled = next(r["pearson"] for r in stats if r["scope"] == "pooled")
    centered = next(r["pearson"] for r in stats if r["scope"] == "seed_centered")
    ax.text(.03, .97, f"Pooled $r={label(pooled)}$\nSeed-centered $r={label(centered)}$",
            va="top", transform=ax.transAxes, fontsize=8 if detailed else 6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1.5})
    if detailed:
        ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=.85, edgecolor="none")


def plot(rows: list[dict], stats: list[dict], output: Path):
    plt = setup_plots()
    groups, indexed = defaultdict(list), defaultdict(list)
    for r in rows:
        groups[r["model"], r["weight"], r["n"], r["case"]].append(r)
    for r in stats:
        indexed[r["model"], r["weight"], r["n"], r["case"]].append(r)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.99, bottom=.19, top=.84, wspace=.13)
    for model, ax in zip(MODELS, axes, strict=True):
        key = (model, "task_mean", 10, "current_past")
        draw_scatter(ax, groups[key], indexed[key], detailed=True)
        ax.set_title(model)
        ax.set_xlabel("Past gain relative to parent (pp)")
    axes[0].set_ylabel("Validation lift relative to proposal mean (pp)")
    fig.suptitle("Current + past selection · Task mean · $n=10$", fontsize=11, y=.98)
    fig.text(.08, .035, "Each point: one checkpoint (12 per seed). Lines: within-seed descriptive fits. "
             "Validation uses the common remaining 40 tasks.\n"
             "Seed-centered r removes seed-specific means; CPs along a training run remain dependent.", fontsize=7.3)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"past_gain_validation_lift_current_past_n10.{ext}", dpi=230)
    plt.close(fig)
    print("Primary scatter saved.", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.3), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.99, bottom=.12, top=.81, wspace=.17, hspace=.38)
    for i, model in enumerate(MODELS):
        for j, weight in enumerate(WEIGHTS):
            ax = axes[i, j]
            for case in CASES:
                values = [r for r in stats if r["model"] == model and r["weight"] == weight
                          and r["case"] == case and r["scope"] == "seed_centered"]
                values.sort(key=lambda r: r["n"])
                ax.plot([r["n"] for r in values], [np.nan if r["pearson"] is None else r["pearson"]
                                                  for r in values], color=COLORS[case], marker=MARKERS[case],
                        linewidth=1.2, markersize=3, label=CASE_LABELS[case])
            ax.set_title(f"{model} · {WEIGHT_LABELS[weight]}")
            ax.axhline(0, color="#BBBBBB", linewidth=.6)
            ax.set(ylim=(-1, 1), xticks=range(1, 11))
            ax.grid(alpha=.18, linewidth=.4)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 1:
                ax.set_xlabel("Sample size $n$")
            if j == 0:
                ax.set_ylabel("Seed-centered Pearson $r$")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=4, loc="upper center",
               bbox_to_anchor=(.53, .99), frameon=False, fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"gain_lift_correlation_n1_10.{ext}", dpi=220)
    plt.close(fig)
    for n in range(1, 11):
        fig, axes = plt.subplots(4, 7, figsize=(17.5, 10), sharex=True, sharey=True)
        fig.subplots_adjust(left=.065, right=.99, bottom=.10, top=.92, wspace=.10, hspace=.35)
        for i, (model, weight) in enumerate((m, w) for m in MODELS for w in WEIGHTS):
            for j, case in enumerate(CASES):
                ax = axes[i, j]
                key = (model, weight, n, case)
                if key not in groups:
                    ax.text(.5, .5, "Not feasible", ha="center", transform=ax.transAxes, color="#999999")
                else:
                    draw_scatter(ax, groups[key], indexed[key])
                if i == 0:
                    ax.set_title(CASE_LABELS[case].replace(" + ", " +\n"), fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"{model}\n{WEIGHT_LABELS[weight]}", fontsize=9)
                ax.tick_params(labelsize=6)
        fig.suptitle(f"Past gain vs validation lift · $n={n}$ · Seed 42/43/44: circle/triangle/square", y=.98)
        fig.supxlabel("Past gain relative to parent (pp)", y=.035)
        fig.supylabel("Validation40 lift relative to proposal mean (pp)", x=.008)
        for ext in ("pdf", "png"):
            fig.savefig(output / f"gain_lift_all_methods_n{n:02d}.{ext}", dpi=180)
        plt.close(fig)


def plot_baseline_control(rows: list[dict], adjusted: list[dict], output: Path):
    """Use both parent-relative gain and proposal-relative past lift, keeping their names distinct."""
    plt = setup_plots()
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 7.4), sharey=True)
    fig.subplots_adjust(left=.09, right=.99, bottom=.11, top=.90, wspace=.13, hspace=.38)
    for i, source in enumerate((rows, adjusted)):
        for j, model in enumerate(MODELS):
            group = [r for r in source if r["model"] == model and r["weight"] == "task_mean"
                     and r["case"] == "current_past" and r["n"] == 10]
            draw_scatter(axes[i, j], group, describe_group(group), detailed=True)
            axes[i, j].set_title(model)
            axes[i, j].set_xlabel("Past gain relative to parent (pp)" if i == 0 else
                                 "Past selection lift relative to proposal mean (pp)")
            if j == 0:
                axes[i, j].set_ylabel("Validation40 proposal-relative lift (pp)")
    fig.suptitle("Current + past · Task mean · $n=10$ · Baseline comparison", fontsize=11, y=.98)
    fig.text(.09, .025, "Bottom row also subtracts the five-proposal mean past gain within each checkpoint.\n"
             "This isolates selection lift from proposal quality; it is distinct from parent-relative past gain.",
             fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(output / f"past_gain_and_selection_lift_validation_n10.{ext}", dpi=230)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = validated_rows(json.loads(args.input.read_text()))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"], row["weight"], row["n"], row["case"]].append(row)
    stats = [r for group in grouped.values() for r in describe_group(group)]
    candidates = json.loads(args.candidates.read_text())
    cp = defaultdict(list)
    for row in candidates:
        cp[row["model"], row["key"]].append(row)
    within = []
    for (model, key), group in cp.items():
        if len(group) != 5:
            raise ValueError("Every parent requires the same five candidates")
        within.append({"model": model, "key": key, "seed": group[0]["seed"],
                       "pearson": correlation([r["gain_past"] for r in group],
                                              [r["validation40_lift"] for r in group]),
                       "spearman": correlation([r["gain_past"] for r in group],
                                               [r["validation40_lift"] for r in group], rank=True)})
    proposal_means = {key: np.mean([r["gain_past"] for r in group]) for key, group in cp.items()}
    adjusted = [{**r, "gain_past": float(r["gain_past"] - proposal_means[r["model"], r["key"]])} for r in rows]
    adjusted_groups = defaultdict(list)
    for r in adjusted:
        adjusted_groups[r["model"], r["weight"], r["n"], r["case"]].append(r)
    adjusted_stats = [{**r, "x_metric": "past_selection_lift_relative_to_proposal_mean"}
                      for group in adjusted_groups.values() for r in describe_group(group)]
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "gain_lift_correlations.csv", stats)
    write_csv(args.output / "past_selection_lift_correlations.csv", adjusted_stats)
    write_csv(args.output / "within_parent_candidate_correlations.csv", within)
    (args.output / "correlations.json").write_text(json.dumps({"checkpoint_correlations": stats,
        "candidate_within_parent": within, "past_selection_lift_correlations": adjusted_stats,
        "new_model_calls": 0, "new_environment_calls": 0,
        "inference": "Descriptive correlations only. Checkpoints within each training seed are dependent. "
                     "No iid checkpoint p-values or confidence intervals.",
        "seed_centering": "Subtract each seed's x and y means before Pearson correlation. "
                          "Spearman for this scope ranks residuals; per-seed Spearman is also saved.",
        "source_files": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in (args.input, args.candidates)]}, indent=2, allow_nan=False) + "\n")
    plot(rows, stats, args.output / "figures")
    plot_baseline_control(rows, adjusted, args.output / "figures")
    for r in stats:
        if r["case"] == "current_past" and r["n"] == 10 and r["weight"] == "task_mean":
            print(json.dumps(r), flush=True)
    print(f"Saved {len(stats)} correlations and {len(within)} within-parent correlations.", flush=True)


if __name__ == "__main__":
    main()
