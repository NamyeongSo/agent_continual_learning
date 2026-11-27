# SPDX-License-Identifier: Apache-2.0
"""Compare gain/GRPO overlap with paired accumulation gaps for Qwen and Gemma w/o GT."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev

import numpy as np

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent
SUMMARY = PREVIOUS.parent
ROOT = SUMMARY.parents[1]
OVERLAP = (SUMMARY / "reflection_gain_with_oss120b_20261123/"
           "qwen_gemma26b_wo_gt_advantage_gain_20261125/overlap_and_means.csv")
PER_SEED = SUMMARY / "model_strategy_bars_20261124/metrics_by_seed.csv"
GEMMA = ROOT / "outputs/ace_appworld_gemma4_26b_a4b_it_wo_gt_train_only_20261124"
MODELS = ("qwen38", "qwen36", "qwen35", "gemma26")
MODES = ("thinking", "nonthinking")
LABELS = {"qwen38": "Qwen3.8-27B", "qwen36": "Qwen3.6-35B-A3B", "qwen35": "Qwen3.5-9B",
          "gemma26": "Gemma4-26B-A4B"}
COLORS = {"qwen38": "#3973ac", "qwen36": "#218a69", "qwen35": "#d98b2b", "gemma26": "#8b67b1"}
MARKERS = {"thinking": "o", "nonthinking": "s"}
METRICS = {"gain": "PCT gain overlap", "advantage": "GRPO advantage overlap"}
SEEDS = (42, 43, 44)
SOURCES = {}
SPEC = importlib.util.spec_from_file_location("original_gap_scatter", PREVIOUS / "plot_correlation.py")
ORIGINAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORIGINAL)
plt, Line2D = ORIGINAL.plt, ORIGINAL.Line2D


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    SOURCES[str(path)] = digest(path)
    with path.open() as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    SOURCES[str(path)] = digest(path)
    return json.loads(path.read_bytes())


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect():
    overlaps = {(r["model"], r["mode"], r["metric"]): float(r["overlap_of_random_percent"])
                for r in read_csv(OVERLAP)}
    assert len(overlaps) == 16
    qwen = {(r["model"], r["mode"], r["strategy"], int(r["seed"])): r
            for r in read_csv(PER_SEED) if r["arm"] == "without_gt" and r["model"] in MODELS}
    gemma = {(r["mode"], r["strategy"], r["seed"]): r
             for r in read_json(GEMMA / "train_final_by_seed.json")}
    previous = {(r["model"], r["mode"]): r for r in read_csv(PREVIOUS / "scatter_points.csv")
                if r["arm"] == "without_gt"}
    points, paired_rows = [], []
    for model in MODELS:
        for mode in MODES:
            paired = []
            for seed in SEEDS:
                scores = {}
                for strategy in ("best", "worst"):
                    if model == "gemma26":
                        row = gemma[mode, strategy, seed]
                        assert row["tasks"] == 50
                        assert row["condition_id"] == f"s{seed}_without_gt_{strategy}"
                        checkpoint = read_json(GEMMA / mode / "accumulation/conditions" /
                                               row["condition_id"] / "checkpoints/050.json")
                        assert checkpoint["session_count"] == 50
                        scores[strategy] = row["pct"] * 100
                    else:
                        row = qwen[model, mode, strategy, seed]
                        assert int(row["train_tasks"]) == int(row["learned_tasks"]) == 50
                        scores[strategy] = float(row["accumulated_pct"]) * 100
                record = {"model": model, "mode": mode, "arm": "without_gt", "seed": seed,
                          "best_pct_percent": scores["best"], "worst_pct_percent": scores["worst"],
                          "gap_pp": scores["best"] - scores["worst"]}
                paired.append(record)
                paired_rows.append(record)
            base = {"model": model, "model_label": LABELS[model], "mode": mode, "arm": "without_gt",
                    "best_accumulated_train_pct_percent": mean(r["best_pct_percent"] for r in paired),
                    "worst_accumulated_train_pct_percent": mean(r["worst_pct_percent"] for r in paired),
                    "gap_mean_pp": mean(r["gap_pp"] for r in paired),
                    "gap_std_pp": stdev(r["gap_pp"] for r in paired), "seeds": "42,43,44", "seed_count": 3}
            if model != "gemma26":
                for field in ("gap_mean_pp", "gap_std_pp", "best_accumulated_train_pct_percent",
                              "worst_accumulated_train_pct_percent"):
                    assert np.isclose(base[field], float(previous[model, mode][field]), atol=1e-12, rtol=0)
            for metric in METRICS:
                points.append({**base, "overlap_metric": metric, "overlap_percent": overlaps[model, mode, metric]})
    assert len(points) == 16 and len(paired_rows) == 24
    write_csv("scatter_points.csv", points)
    write_csv("paired_seed_gaps.csv", paired_rows)
    return points


def axes_plot(ax, points, metric, limits):
    stats = ORIGINAL.correlation(points)
    for point in points:
        color = COLORS[point["model"]]
        ax.errorbar(point["overlap_percent"], point["gap_mean_pp"], yerr=point["gap_std_pp"],
                    fmt="none", color=color, alpha=0.4, elinewidth=1.4, capsize=3.5, zorder=2)
        ax.scatter(point["overlap_percent"], point["gap_mean_pp"], s=125, marker=MARKERS[point["mode"]],
                   color=color, edgecolor="white", linewidth=0.9, zorder=4)
    left, right = min(r["overlap_percent"] for r in points) - 3, max(r["overlap_percent"] for r in points) + 3
    xx = np.linspace(left, right, 200)
    ax.plot(xx, stats["ols_intercept_gap_pp"] + stats["ols_slope_gap_pp_per_overlap_percentage_point"] * xx,
            color="#29384b", linewidth=1.8, linestyle=(0, (5, 3)), zorder=3)
    ax.axhline(0, color="#99a6b6", linewidth=1.05, zorder=1)
    ax.set(xlim=(left, right), ylim=limits, xlabel=f"{METRICS[metric]} with Random (%)")
    ax.set_title(METRICS[metric], fontsize=15, pad=16)
    ax.grid(color="#e4e9f0", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c9d1dc")
    ax.tick_params(length=0, pad=7)
    ax.text(0.97, 0.97,
            f"Pearson r = {stats['pearson_r']:+.3f}\nSpearman $\\rho$ = {stats['spearman_rho']:+.3f}\n"
            "8 settings / 4 models", transform=ax.transAxes, ha="right", va="top", fontsize=11.5,
            bbox={"facecolor": "white", "edgecolor": "#dbe3ec", "boxstyle": "round,pad=0.55", "alpha": 0.95})


def draw(points, metrics, name):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    width = 17 if len(metrics) == 2 else 11
    fig, axes = plt.subplots(1, len(metrics), figsize=(width, 8.3), sharey=True, squeeze=False)
    lower = min(p["gap_mean_pp"] - p["gap_std_pp"] for p in points)
    upper = max(p["gap_mean_pp"] + p["gap_std_pp"] for p in points)
    padding = (upper - lower) * 0.15
    for metric, ax in zip(metrics, axes[0], strict=True):
        axes_plot(ax, [p for p in points if p["overlap_metric"] == metric], metric,
                  (lower - padding, upper + padding))
    axes[0, 0].set_ylabel("Best - Worst accumulated Train PCT (percentage points)", labelpad=12)
    model_handles = [Line2D([0], [0], marker="o", linestyle="none", color=COLORS[m], markersize=8,
                            label=LABELS[m]) for m in MODELS]
    mode_handles = [Line2D([0], [0], marker=MARKERS[m], linestyle="none", color="#4a586b", markersize=8,
                           label="Thinking" if m == "thinking" else "Non-thinking") for m in MODES]
    mode_handles += [Line2D([0], [0], color="#29384b", linestyle="--", label="OLS descriptive trend")]
    fig.legend(handles=model_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.112))
    fig.legend(handles=mode_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.070))
    fig.suptitle("Reflection overlap vs. accumulation gap · Qwen and Gemma · w/o GT",
                 fontsize=20 if len(metrics) == 2 else 16, weight="bold", y=0.976, color="#233247")
    fig.text(0.5, 0.918, "One point per model and reasoning mode · "
             "Same measured Best / Worst contexts on both x-axis versions",
             ha="center", fontsize=10.5, color="#526176")
    fig.text(0.5, 0.041, "Y = mean of paired seed gaps (42, 43, 44) ±1 sample SD · Train50 per seed · Plain Best/Worst",
             ha="center", fontsize=9.3, color="#526176")
    fig.text(0.5, 0.016, "Descriptive correlations: two settings share each model; "
             "these are not eight independent models.",
             ha="center", fontsize=9.3, color="#526176")
    fig.subplots_adjust(left=0.075 if len(metrics) == 2 else 0.11, right=0.985,
                        top=0.835, bottom=0.255, wspace=0.12)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=220, facecolor="white")
    fig.savefig(OUT / f"{name}_preview.png", dpi=110, facecolor="white")
    plt.close(fig)


def main():
    original_images = {p: digest(p) for directory in (PREVIOUS, PREVIOUS / "grpo_advantage_overlap")
                       for p in directory.iterdir() if p.suffix in {".png", ".pdf", ".svg"}}
    points = collect()
    draw(points, tuple(METRICS), "overlap_vs_accumulation_gap_gain_grpo_2panel")
    for metric in METRICS:
        draw(points, (metric,), f"{metric}_overlap_vs_accumulation_gap")
    for path, sha in original_images.items():
        assert digest(path) == sha
    statistics = {metric: ORIGINAL.correlation([p for p in points if p["overlap_metric"] == metric])
                  for metric in METRICS}
    manifest = {"generated_at": datetime.now(UTC).isoformat(), "models": list(MODELS), "modes": list(MODES),
                "arm": "without_gt", "points_per_metric": 8, "paired_seed_measurements": 24,
                "overall_descriptive": statistics,
                "per_mode_descriptive": {metric: {mode: ORIGINAL.correlation([
                    p for p in points if p["overlap_metric"] == metric and p["mode"] == mode])
                    for mode in MODES} for metric in METRICS},
                "x": "Displayed split-shape overlap (best-left + worst-right with Random) / Random area",
                "x_source": "Exact overlaps from the new Qwen/Gemma w/o-GT four-panel violin, not recomputed",
                "y": "Mean paired seed42/43/44 gap in measured plain Best minus Worst accumulated Train PCT",
                "error_bars": "Sample SD of the three paired gaps, ddof=1",
                "both_panels_share_identical_y": True, "original_qwen_y_matched": True,
                "new_inference_calls": 0, "source_sha256": SOURCES,
                "original_images_unchanged": {str(p): sha for p, sha in original_images.items()},
                "script_sha256": digest(Path(__file__))}
    (OUT / "correlation_summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "correlation": statistics,
                      "gemma": [p for p in points if p["model"] == "gemma26"]}))


if __name__ == "__main__":
    main()
