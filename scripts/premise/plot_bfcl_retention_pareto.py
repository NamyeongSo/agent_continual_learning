# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Past-gain/validation-lift Pareto comparisons at sample sizes one through ten."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.analyze_bfcl_temporal_parent_preservation import allocation  # noqa: E402
from scripts.premise.analyze_bfcl_temporal_selection import write_csv  # noqa: E402
from scripts.premise.plot_bfcl_temporal_parent_preservation import (  # noqa: E402
    CASE_LABELS,
    CASES,
    COLORS,
    MARKERS,
    MODELS,
    WEIGHT_LABELS,
    WEIGHTS,
)


def pareto_mask(points: np.ndarray) -> np.ndarray:
    """Mark every non-dominated point when maximizing both coordinates.

    Tied coordinates are retained together. Rounding to twelve decimal places
    removes arithmetic noise before an otherwise exact partial-order comparison.
    This computes the full measured frontier, not a convex hull or weighted sum.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite (alternatives, 2) coordinates")
    values = np.round(points, 12)
    # [i,j] means point j weakly exceeds i on both axes, strictly on at least one.
    dominates = ((values[None, :, :] >= values[:, None, :]).all(axis=-1)
                 & (values[None, :, :] > values[:, None, :]).any(axis=-1))
    return ~dominates.any(axis=1)


def panel_points(summary: dict, model: str, weight: str, n: int) -> list[dict]:
    """All feasible temporal selectors and displayed reference choices, one cohort."""
    expected = summary["checkpoints_per_model"]
    selected = [r for r in summary["aggregate"] if r["scope"] == "all" and r["model"] == model
                and r["weight"] == weight and r["n"] == n]
    selected += [r for r in summary["proposal_reference"] if r["scope"] == "all" and r["model"] == model]
    if n == 10:
        selected += [r for r in summary["validation_reference"] if r["scope"] == "all" and r["model"] == model]
    rows, identities = [], set()
    for r in selected:
        case = r["case"]
        if case in identities or r["checkpoints"] != expected:
            raise ValueError("Pareto alternatives must be distinct and share the same checkpoint cohort")
        identities.add(case)
        metrics = [r[k] for k in ("gain_past", "lift_validation40")]
        if any(m["defined_checkpoints"] != expected or m["mean"] is None for m in metrics):
            raise ValueError("Each alternative needs both outcomes on the entire common cohort")
        rows.append({"model": model, "weight": weight, "n": n, "case": case,
                     "checkpoints": expected,
                     "actual_n": sum(allocation(case, n)) if case in CASES else 10 if case == "val10" else 0,
                     "gain_past": metrics[0]["mean"], "validation40_lift": metrics[1]["mean"],
                     "future_samples_used": case in ("future", "current_future", "past_future", "current_past_future")})
    feasible = {case for case in CASES if allocation(case, n) is not None}
    if identities != feasible | {"proposal"} | ({"val10"} if n == 10 else set()):
        raise ValueError("Missing or unexpected temporal selector/reference at this sample size")
    frontier = pareto_mask(np.asarray([[r["gain_past"], r["validation40_lift"]] for r in rows]))
    for row, is_pareto in zip(rows, frontier, strict=True):
        row["is_pareto"] = bool(is_pareto)
    return rows


def plot(source: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    summary = json.loads(source.read_text())
    panels = {(model, weight, n): panel_points(summary, model, weight, n)
              for model in MODELS for weight in WEIGHTS for n in range(1, 11)}
    all_rows = [r for rows in panels.values() for r in rows]
    coords = 100 * np.asarray([[r["gain_past"], r["validation40_lift"]] for r in all_rows])
    limits = []
    for axis in range(2):
        low, high = min(0., coords[:, axis].min()), max(0., coords[:, axis].max())
        padding = max((high - low) * .09, .025)
        limits.append((low - padding, high + padding))
    output = source.parent / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                         "axes.labelsize": 9.5, "axes.titlesize": 9, "axes.linewidth": .6,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    handles = [Line2D([0], [0], color=COLORS[c], marker=MARKERS[c], linestyle="none", markersize=4.5,
                      label=CASE_LABELS[c]) for c in CASES]
    handles += [Line2D([0], [0], color="#777777", marker="x", linestyle="none", markersize=5,
                       label="Proposal mean"),
                Line2D([0], [0], color="#333333", marker="*", linestyle="none", markersize=7,
                       label="Val 10 (n=10)"),
                Line2D([0], [0], color="#222222", marker="o", markerfacecolor="none",
                       linestyle="--", linewidth=.7, markersize=6, label="Pareto frontier")]
    files = []
    for model in MODELS:
        for weight in WEIGHTS:
            fig, axes = plt.subplots(2, 5, figsize=(11.8, 4.9), sharex=True, sharey=True, squeeze=False)
            fig.subplots_adjust(left=.075, right=.99, bottom=.135, top=.76, wspace=.12, hspace=.34)
            for n, ax in enumerate(axes.flat, 1):
                rows = panels[model, weight, n]
                frontier = np.asarray([[100 * r["gain_past"], 100 * r["validation40_lift"]]
                                       for r in rows if r["is_pareto"]])
                frontier = np.unique(frontier.round(10), axis=0)
                frontier = frontier[np.argsort(frontier[:, 0])]
                if len(frontier) > 1:
                    ax.plot(frontier[:, 0], frontier[:, 1], color="#333333", linestyle="--", linewidth=.8,
                            zorder=2)
                ax.scatter(frontier[:, 0], frontier[:, 1], s=80, facecolors="none", edgecolors="#222222",
                           linewidths=.8, zorder=5)
                for r in rows:
                    case = r["case"]
                    color = COLORS.get(case, "#777777" if case == "proposal" else "#333333")
                    marker = MARKERS.get(case, "x" if case == "proposal" else "*")
                    ax.scatter(100 * r["gain_past"], 100 * r["validation40_lift"], c=color, marker=marker,
                               s=32 if case in CASES else 35 if case == "proposal" else 65,
                               linewidths=.6, zorder=4)
                ax.axhline(0, color="#BBBBBB", linewidth=.55, zorder=0)
                ax.axvline(0, color="#BBBBBB", linewidth=.55, zorder=0)
                ax.set_title(f"$n={n}$", pad=4)
                ax.set(xlim=limits[0], ylim=limits[1])
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.grid(linewidth=.35, alpha=.18)
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(length=2.5, pad=2)
            fig.suptitle(f"{model} · {WEIGHT_LABELS[weight]}", fontsize=10, y=.995)
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .935), ncol=5,
                       frameon=False, fontsize=8, columnspacing=1.5, labelspacing=.7, handletextpad=.5)
            fig.supxlabel(r"Past gain $\delta_{t,\mathrm{past}}^{i^\star}$ (pp)", y=.025)
            fig.supylabel(r"Validation lift $s_{\rm val}^{i^\star}-\bar{s}_{\rm val}$ (pp)", x=.012)
            stem = f"pareto_past_gain_validation40_{model.lower()}_{weight}_n1_10"
            for extension in ("pdf", "png"):
                fig.savefig(output / f"{stem}.{extension}", dpi=280, bbox_inches="tight", pad_inches=.04)
            plt.close(fig)
            files.append(stem)
            print(f"Saved {stem}.pdf", flush=True)
    result = {
        "source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "new_model_calls": 0, "new_environment_calls": 0, "files": files,
        "frontier": "Maximize both past parent-relative gain and validation40 proposal-relative lift. "
                    "All feasible temporal selectors and displayed references enter dominance comparisons; "
                    "proposal mean at all n, archived val10 only at n=10. Tied alternatives remain co-optimal.",
        "interpretation": "Frontier is over estimated checkpoint means, not statistical dominance. "
                          "Dashed connecting segments guide the eye and do not assert achievable interpolation. "
                          "Future-containing selectors are retrospective conditions, "
                          "not observed-history-only choices.",
        "rounding_for_dominance": "12 decimal places in success-rate units",
        "checkpoints_per_model": summary["checkpoints_per_model"], "axes_limits_pp": limits,
        "points": all_rows,
    }
    (source.parent / "pareto_frontiers.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_csv(source.parent / "pareto_frontiers.csv", all_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    plot(args.input)


if __name__ == "__main__":
    main()
