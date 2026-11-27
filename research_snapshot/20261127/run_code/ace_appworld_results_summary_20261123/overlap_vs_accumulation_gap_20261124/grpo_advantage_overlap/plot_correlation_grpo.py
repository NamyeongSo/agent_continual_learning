# SPDX-License-Identifier: Apache-2.0
"""Change only the earlier correlation plot's x metric to GRPO-advantage overlap."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent
ROOT = PREVIOUS.parent
SPEC = importlib.util.spec_from_file_location("original_scatter", PREVIOUS / "plot_correlation.py")
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)
SOURCE.OUT = OUT
SOURCE.OVERLAP = (ROOT / "reflection_gain_with_oss120b_20261123/grpo_advantage_with_random_20261124/"
                  "overlap_by_model_mode_gt.csv")
NAME = "grpo_advantage_overlap_vs_best_minus_worst_accumulated_train_pct"


def draw(points, stats):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(14, 8.6))
    for point in points:
        color = SOURCE.COLORS[point["model"]]
        setting = point["mode_group"], point["arm"]
        ax.errorbar(point["overlap_percent"], point["gap_mean_pp"], yerr=point["gap_std_pp"],
                    fmt="none", color=color, alpha=0.38, elinewidth=1.3, capsize=3.2, capthick=1, zorder=2)
        ax.scatter(point["overlap_percent"], point["gap_mean_pp"], s=112, marker=SOURCE.MARKERS[setting],
                   color=color, edgecolor="white", linewidth=0.95, zorder=4)
    left = min(p["overlap_percent"] for p in points) - 3
    right = max(p["overlap_percent"] for p in points) + 3
    xx = np.linspace(left, right, 200)
    ax.plot(xx, stats["ols_intercept_gap_pp"] + stats["ols_slope_gap_pp_per_overlap_percentage_point"] * xx,
            color="#29384b", linewidth=1.9, linestyle=(0, (5, 3)), zorder=3)
    ax.axhline(0, color="#99a6b6", linewidth=1.05, zorder=1)
    lower = min(p["gap_mean_pp"] - p["gap_std_pp"] for p in points)
    upper = max(p["gap_mean_pp"] + p["gap_std_pp"] for p in points)
    pad = (upper - lower) * 0.12
    ax.set(xlim=(left, right), ylim=(lower - pad, upper + pad))
    ax.set_xlabel("GRPO advantage overlap of best-left + worst-right with Random (%)", fontsize=11.8, labelpad=12)
    ax.set_ylabel("Best - worst accumulated Train PCT (percentage points)", fontsize=12.5, labelpad=12)
    ax.grid(color="#e4e9f0", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c9d1dc")
    ax.tick_params(length=0, pad=7)
    ax.text(0.965, 0.965,
            f"Pearson r = {stats['pearson_r']:+.3f}\nSpearman $\\rho$ = {stats['spearman_rho']:+.3f}\n"
            "20 settings / 5 models", transform=ax.transAxes, ha="right", va="top", fontsize=12.5,
            bbox={"facecolor": "white", "edgecolor": "#dbe3ec", "boxstyle": "round,pad=0.65", "alpha": 0.95})
    model_handles = [Line2D([0], [0], marker="o", linestyle="none", color=SOURCE.COLORS[m], markersize=8,
                            label=SOURCE.LABELS[m]) for m in SOURCE.MODELS]
    setting_handles = [Line2D([0], [0], marker=SOURCE.MARKERS[s], linestyle="none", color="#4a586b", markersize=8,
                              label=SOURCE.SETTING_LABELS[s]) for s in SOURCE.SETTINGS]
    model_legend = ax.legend(handles=model_handles, title="Model", frameon=False, loc="upper left",
                             bbox_to_anchor=(1.025, 1.015), fontsize=10.5, title_fontsize=11.5,
                             borderaxespad=0, labelspacing=1.1)
    ax.add_artist(model_legend)
    settings_legend = ax.legend(handles=setting_handles, title="Setting", frameon=False, loc="upper left",
                                bbox_to_anchor=(1.025, 0.66), fontsize=10.1, title_fontsize=11.5,
                                borderaxespad=0, labelspacing=1.1)
    ax.add_artist(settings_legend)
    lines = [Line2D([0], [0], color="#29384b", linestyle="--", linewidth=1.9, label="OLS descriptive trend"),
             Line2D([0], [0], color="#99a6b6", linewidth=1.05, label="Best = worst")]
    ax.legend(handles=lines, loc="upper left", bbox_to_anchor=(1.025, 0.32), frameon=False,
              fontsize=10.2, borderaxespad=0, labelspacing=1.2)
    fig.suptitle("GRPO advantage overlap vs. accumulation selection gap", fontsize=20.5, fontweight="bold",
                 x=0.475, y=0.975, color="#233247")
    fig.text(0.475, 0.925, "One point per model, reasoning mode and GT setting · "
             "Original measured best / worst contexts", ha="center", fontsize=11.5, color="#526176")
    fig.text(0.5, 0.071,
             "Y unchanged: mean paired seed gap (42, 43, 44) ±1 sample SD. "
             "X: overlap of task-local G5 GRPO advantages; Gaussian KDE bandwidth 0.12; all-tied tasks included.",
             ha="center", fontsize=9.4, color="#526176")
    fig.text(0.5, 0.037,
             "Descriptive correlations. Four settings share each model; "
             "the 20 settings are not 20 independent models. No p-values or significance claims.",
             ha="center", fontsize=9.8, color="#526176")
    fig.subplots_adjust(left=0.085, right=0.765, top=0.87, bottom=0.18)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{NAME}.{extension}", dpi=220, facecolor="white")
    fig.savefig(OUT / f"{NAME}_preview.png", dpi=110, facecolor="white")
    plt.close(fig)


def main():
    previous_files = [PREVIOUS / ("overlap_vs_best_minus_worst_accumulated_train_pct." + ext)
                      for ext in ("png", "pdf", "svg")]
    unchanged = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in previous_files}
    points = SOURCE.collect()
    previous = {(r["model"], r["mode"], r["arm"]): r
                for r in SOURCE.read_csv(PREVIOUS / "scatter_points.csv")}
    for point in points:
        old = previous[point["model"], point["mode"], point["arm"]]
        for field in ("best_accumulated_train_pct_percent", "worst_accumulated_train_pct_percent",
                      "gap_mean_pp", "gap_std_pp"):
            assert float(old[field]) == point[field], (point, field)
        point["previous_pct_gain_overlap_percent"] = float(old["overlap_percent"])
    SOURCE.write_csv("scatter_points.csv", points)
    stats = SOURCE.correlation(points)
    draw(points, stats)
    for path, sha in unchanged.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
    result = {
        "generated_at": datetime.now(UTC).isoformat(), "overall": stats,
        "per_setting_descriptive": [
            {"mode_group": mode, "arm": arm,
             **SOURCE.correlation([r for r in points if r["mode_group"] == mode and r["arm"] == arm])}
            for mode, arm in SOURCE.SETTINGS
        ],
        "x_definition": "Geometric overlap of best-left + worst-right and Random GRPO-advantage KDEs / Random area",
        "advantage_definition": "Task-local G5: (3-rollout candidate mean PCT - group mean) / (population SD + 1e-8)",
        "kde_bandwidth_advantage": 0.12, "zero_signal_tasks": "Included with all five advantages equal to zero",
        "y_definition": "Original measured plain best minus worst accumulated Train PCT; 3 paired seed mean",
        "y_error": "Sample SD over seeds 42/43/44; matched original plot exactly",
        "inference": "Descriptive; settings reuse the same five models; no independent-sample p-values",
        "new_model_calls": 0,
        "original_images_unchanged": unchanged,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (SOURCE.OVERLAP, SOURCE.PER_SEED, PREVIOUS / "scatter_points.csv")},
        "helper_sha256": hashlib.sha256((PREVIOUS / "plot_correlation.py").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "correlation_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), **stats}))


if __name__ == "__main__":
    main()
