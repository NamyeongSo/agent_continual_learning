# SPDX-License-Identifier: Apache-2.0
"""Relate split-violin overlap to the measured best-minus-worst accumulation gap."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev

import matplotlib
import numpy as np
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
OVERLAP = ROOT / "reflection_gain_with_oss120b_20261123/overlap_with_random_20261124/overlap_by_model_mode_gt.csv"
PER_SEED = ROOT / "model_strategy_bars_20261124/metrics_by_seed.csv"
MODELS = ("qwen38", "qwen36", "qwen35", "oss120", "oss20")
LABELS = {"qwen38": "Qwen3.8-27B", "qwen36": "Qwen3.6-35B-A3B", "qwen35": "Qwen3.5-9B",
          "oss120": "GPT-OSS-120B", "oss20": "GPT-OSS-20B"}
COLORS = {"qwen38": "#3973ac", "qwen36": "#218a69", "qwen35": "#d98b2b",
          "oss120": "#8b67b1", "oss20": "#c95860"}
SETTINGS = (("thinking", "without_gt"), ("thinking", "with_gt"),
            ("nonthinking", "without_gt"), ("nonthinking", "with_gt"))
MARKERS = dict(zip(SETTINGS, ("o", "^", "s", "D"), strict=True))
SETTING_LABELS = {SETTINGS[0]: "Thinking / High, w/o GT", SETTINGS[1]: "Thinking / High, w/ GT",
                  SETTINGS[2]: "Non-thinking / Low, w/o GT", SETTINGS[3]: "Non-thinking / Low, w/ GT"}


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mode_group(mode):
    return "thinking" if mode in ("thinking", "high") else "nonthinking"


def collect():
    overlap = read_csv(OVERLAP)
    seed_rows = read_csv(PER_SEED)
    assert len(overlap) == len({(r["model"], r["mode"], r["arm"]) for r in overlap}) == 20
    by_seed = {(r["model"], r["mode"], r["arm"], r["strategy"], int(r["seed"])): r for r in seed_rows}
    points, differences = [], []
    for model in MODELS:
        for group, arm in SETTINGS:
            matches = [r for r in overlap if r["model"] == model and mode_group(r["mode"]) == group and r["arm"] == arm]
            assert len(matches) == 1
            source = matches[0]
            mode = source["mode"]
            paired = []
            for seed in (42, 43, 44):
                best = by_seed[model, mode, arm, "best", seed]
                worst = by_seed[model, mode, arm, "worst", seed]
                assert int(best["train_tasks"]) == int(worst["train_tasks"]) == 50
                assert int(best["learned_tasks"]) == int(worst["learned_tasks"]) == 50
                record = {"model": model, "mode": mode, "mode_group": group, "arm": arm, "seed": seed,
                          "overlap_percent": float(source["overlap_of_random_percent"]),
                          "best_accumulated_train_pct_percent": float(best["accumulated_pct"]) * 100,
                          "worst_accumulated_train_pct_percent": float(worst["accumulated_pct"]) * 100,
                          "gap_pp": (float(best["accumulated_pct"]) - float(worst["accumulated_pct"])) * 100}
                paired.append(record)
                differences.append(record)
            point = {"model": model, "model_label": LABELS[model], "mode": mode, "mode_group": group, "arm": arm,
                     "overlap_percent": float(source["overlap_of_random_percent"]),
                     "best_accumulated_train_pct_percent": mean(
                         r["best_accumulated_train_pct_percent"] for r in paired),
                     "worst_accumulated_train_pct_percent": mean(
                         r["worst_accumulated_train_pct_percent"] for r in paired),
                     "gap_mean_pp": mean(r["gap_pp"] for r in paired),
                     "gap_std_pp": stdev(r["gap_pp"] for r in paired), "seeds": "42,43,44", "seed_count": 3}
            assert abs(point["gap_mean_pp"] - point["best_accumulated_train_pct_percent"]
                       + point["worst_accumulated_train_pct_percent"]) < 1e-12
            points.append(point)
    assert len(points) == 20 and len(differences) == 60
    write_csv("scatter_points.csv", points)
    write_csv("paired_seed_gaps.csv", differences)
    return points


def correlation(points):
    x = np.array([r["overlap_percent"] for r in points])
    y = np.array([r["gap_mean_pp"] for r in points])
    slope, intercept = np.polyfit(x, y, 1)
    return {"settings": len(points), "models": len({r["model"] for r in points}),
            "pearson_r": float(pearsonr(x, y).statistic),
            "spearman_rho": float(spearmanr(x, y).statistic),
            "ols_slope_gap_pp_per_overlap_percentage_point": float(slope), "ols_intercept_gap_pp": float(intercept)}


def plot(points, stats):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(14, 8.6))
    for point in points:
        color = COLORS[point["model"]]
        key = point["mode_group"], point["arm"]
        ax.errorbar(point["overlap_percent"], point["gap_mean_pp"], yerr=point["gap_std_pp"],
                    fmt="none", color=color, alpha=0.38, elinewidth=1.3, capsize=3.2, capthick=1.0, zorder=2)
        ax.scatter(point["overlap_percent"], point["gap_mean_pp"], s=112, marker=MARKERS[key],
                   color=color, edgecolor="white", linewidth=0.95, zorder=4)
    left = min(r["overlap_percent"] for r in points) - 2.5
    right = max(r["overlap_percent"] for r in points) + 2.5
    xx = np.linspace(left, right, 200)
    ax.plot(xx, stats["ols_intercept_gap_pp"] + stats["ols_slope_gap_pp_per_overlap_percentage_point"] * xx,
            color="#29384b", linewidth=1.9, linestyle=(0, (5, 3)), zorder=3)
    ax.axhline(0, color="#99a6b6", linewidth=1.05, zorder=1)
    ax.set_xlim(left, right)
    lower = min(r["gap_mean_pp"] - r["gap_std_pp"] for r in points)
    upper = max(r["gap_mean_pp"] + r["gap_std_pp"] for r in points)
    pad = (upper - lower) * 0.12
    ax.set_ylim(lower - pad, upper + pad)
    ax.set_xlabel("Overlap of best-left + worst-right with Random (%)", fontsize=12.5, labelpad=12)
    ax.set_ylabel("Best - worst accumulated Train PCT (percentage points)", fontsize=12.5, labelpad=12)
    ax.grid(color="#e4e9f0", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c9d1dc")
    ax.tick_params(length=0, pad=7)
    ax.text(0.965, 0.965,
            f"Pearson r = {stats['pearson_r']:+.3f}\n"
            f"Spearman $\\rho$ = {stats['spearman_rho']:+.3f}\n"
            "20 settings / 5 models",
            transform=ax.transAxes, ha="right", va="top", fontsize=12.5,
            bbox={"facecolor": "white", "edgecolor": "#dbe3ec", "boxstyle": "round,pad=0.65", "alpha": 0.95})
    model_handles = [Line2D([0], [0], marker="o", linestyle="none", color=COLORS[m], markersize=8,
                            label=LABELS[m]) for m in MODELS]
    setting_handles = [Line2D([0], [0], marker=MARKERS[s], linestyle="none", color="#4a586b", markersize=8,
                              label=SETTING_LABELS[s]) for s in SETTINGS]
    model_legend = ax.legend(handles=model_handles, title="Model", frameon=False, loc="upper left",
                             bbox_to_anchor=(1.025, 1.015), fontsize=10.5, title_fontsize=11.5,
                             borderaxespad=0, labelspacing=1.1)
    ax.add_artist(model_legend)
    setting_legend = ax.legend(handles=setting_handles, title="Setting", frameon=False, loc="upper left",
                               bbox_to_anchor=(1.025, 0.66), fontsize=10.1, title_fontsize=11.5,
                               borderaxespad=0, labelspacing=1.1)
    ax.add_artist(setting_legend)
    line_handles = [Line2D([0], [0], color="#29384b", linestyle="--", linewidth=1.9, label="OLS descriptive trend"),
                    Line2D([0], [0], color="#99a6b6", linewidth=1.05, label="Best = worst")]
    ax.legend(handles=line_handles, loc="upper left", bbox_to_anchor=(1.025, 0.32), frameon=False,
              fontsize=10.2, borderaxespad=0, labelspacing=1.2)
    fig.suptitle("Reflection overlap vs. accumulation selection gap", fontsize=22, fontweight="bold",
                 x=0.475, y=0.975, color="#233247")
    fig.text(0.475, 0.925, "One point per model, reasoning mode and GT setting · "
             "Original measured best / worst contexts",
             ha="center", fontsize=11.5, color="#526176")
    fig.text(0.5, 0.071, "Y = mean of paired seed gaps (42, 43, 44); error bars = ±1 sample SD of those gaps. "
             "X = the annotated split-shape overlap normalized by Random area, with 7 pp KDE bandwidth.",
             ha="center", fontsize=9.8, color="#526176")
    fig.text(0.5, 0.037, "Correlations and trend are descriptive. Four settings share each model; "
             "the 20 settings are not 20 independent models. No p-values or significance claims are shown.",
             ha="center", fontsize=9.8, color="#526176")
    fig.subplots_adjust(left=0.085, right=0.765, top=0.87, bottom=0.18)
    name = "overlap_vs_best_minus_worst_accumulated_train_pct"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=220, facecolor="white")
    fig.savefig(OUT / f"{name}_preview.png", dpi=110, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    points = collect()
    stats = correlation(points)
    plot(points, stats)
    by_setting = [{"mode_group": group, "arm": arm,
                   **correlation([r for r in points if r["mode_group"] == group and r["arm"] == arm])}
                  for group, arm in SETTINGS]
    models = [{"model": model,
               "overlap_percent": mean(r["overlap_percent"] for r in points if r["model"] == model),
               "gap_mean_pp": mean(r["gap_mean_pp"] for r in points if r["model"] == model)} for model in MODELS]
    payload = {"generated_at": datetime.now(UTC).isoformat(), "overall": stats,
               "per_setting_descriptive": by_setting,
               "model_averaged_descriptive": correlation(models),
               "x_definition": "Geometric overlap of best-left + worst-right versus gray Random, "
                               "divided by Random area",
               "y_definition": "Mean over seeds 42/43/44 of measured accumulated best Train PCT minus worst Train PCT",
               "y_error": "Sample SD of the three paired seed-level gaps, ddof=1",
               "conditions": "Plain best/worst on Train50, original measured results, all five models",
               "inference": "Descriptive correlations only; no independent-sample p-value, "
                            "because settings share models",
               "new_model_calls": 0,
               "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (OVERLAP, PER_SEED)},
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "correlation_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("OUTPUT", OUT)


if __name__ == "__main__":
    main()
