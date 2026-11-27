# SPDX-License-Identifier: Apache-2.0
"""Replace only accumulation PCT gaps with official TGC gaps; preserve both overlap x axes."""

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
SPEC = importlib.util.spec_from_file_location("qwen_gemma_pct_scatter", PREVIOUS / "plot_scatter.py")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
plt, Line2D = BASE.plt, BASE.Line2D


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect():
    previous = BASE.read_csv(PREVIOUS / "scatter_points.csv")
    qwen = {(r["model"], r["mode"], r["strategy"], int(r["seed"])): r
            for r in BASE.read_csv(BASE.PER_SEED)
            if r["model"] in BASE.MODELS and r["arm"] == "without_gt"}
    gemma = {(r["mode"], r["strategy"], r["seed"]): r
             for r in BASE.read_json(BASE.GEMMA / "train_final_by_seed.json")}
    by_setting, paired_rows = {}, []
    for model in BASE.MODELS:
        for mode in BASE.MODES:
            paired = []
            for seed in BASE.SEEDS:
                scores = {}
                for strategy in ("best", "worst"):
                    row = gemma[mode, strategy, seed] if model == "gemma26" else qwen[model, mode, strategy, seed]
                    assert int(row["tasks"] if model == "gemma26" else row["train_tasks"]) == 50
                    value = row["tgc"] if model == "gemma26" else float(row["accumulated_tgc"])
                    assert 0 <= value <= 1 and np.isclose(value * 50, round(value * 50))
                    scores[strategy] = value * 100
                row = {"model": model, "mode": mode, "arm": "without_gt", "seed": seed,
                       "best_tgc_percent": scores["best"], "worst_tgc_percent": scores["worst"],
                       "gap_pp": scores["best"] - scores["worst"]}
                paired.append(row)
                paired_rows.append(row)
            by_setting[model, mode] = {
                "model": model, "model_label": BASE.LABELS[model], "mode": mode, "arm": "without_gt",
                "best_accumulated_train_tgc_percent": mean(r["best_tgc_percent"] for r in paired),
                "worst_accumulated_train_tgc_percent": mean(r["worst_tgc_percent"] for r in paired),
                "gap_mean_pp": mean(r["gap_pp"] for r in paired),
                "gap_std_pp": stdev(r["gap_pp"] for r in paired), "seeds": "42,43,44", "seed_count": 3}
    points = [{**by_setting[r["model"], r["mode"]], "overlap_metric": r["overlap_metric"],
               "overlap_percent": float(r["overlap_percent"])} for r in previous]
    assert len(points) == 16 and len(paired_rows) == 24
    assert all(p["overlap_percent"] == float(old["overlap_percent"])
               for p, old in zip(points, previous, strict=True))
    write_csv("scatter_points.csv", points)
    write_csv("paired_seed_gaps.csv", paired_rows)
    return points


def draw(points, metrics, name):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, len(metrics), figsize=(17 if len(metrics) == 2 else 11, 8.3),
                             sharey=True, squeeze=False)
    lower = min(p["gap_mean_pp"] - p["gap_std_pp"] for p in points)
    upper = max(p["gap_mean_pp"] + p["gap_std_pp"] for p in points)
    padding = (upper - lower) * 0.15
    for metric, ax in zip(metrics, axes[0], strict=True):
        BASE.axes_plot(ax, [p for p in points if p["overlap_metric"] == metric], metric,
                       (lower - padding, upper + padding))
    axes[0, 0].set_ylabel("Best - Worst accumulated Train TGC (percentage points)", labelpad=12)
    model_handles = [Line2D([0], [0], marker="o", linestyle="none", color=BASE.COLORS[m], markersize=8,
                            label=BASE.LABELS[m]) for m in BASE.MODELS]
    mode_handles = [Line2D([0], [0], marker=BASE.MARKERS[m], linestyle="none", color="#4a586b", markersize=8,
                           label="Thinking" if m == "thinking" else "Non-thinking") for m in BASE.MODES]
    mode_handles += [Line2D([0], [0], color="#29384b", linestyle="--", label="OLS descriptive trend")]
    fig.legend(handles=model_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.112))
    fig.legend(handles=mode_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.070))
    fig.suptitle("Reflection overlap vs. accumulation TGC gap · Qwen and Gemma · w/o GT",
                 fontsize=20 if len(metrics) == 2 else 15.5, weight="bold", y=0.976, color="#233247")
    fig.text(0.5, 0.918, "One point per model and reasoning mode · "
             "X unchanged: PCT gain / GRPO overlap; Y uses recorded official task success",
             ha="center", fontsize=10.5, color="#526176")
    fig.text(0.5, 0.041, "Y = mean paired TGC gap (seeds 42, 43, 44) ±1 sample SD · "
             "Train50 per seed · Plain Best/Worst",
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
    original_files = {p: BASE.digest(p) for p in PREVIOUS.iterdir() if p.is_file()}
    points = collect()
    draw(points, tuple(BASE.METRICS), "overlap_vs_accumulation_gap_tgc_gain_grpo_2panel")
    draw(points, ("gain",), "gain_overlap_vs_accumulation_gap_tgc")
    draw(points, ("advantage",), "grpo_overlap_vs_accumulation_gap_tgc")
    assert all(BASE.digest(p) == sha for p, sha in original_files.items())
    statistics = {metric: BASE.ORIGINAL.correlation([p for p in points if p["overlap_metric"] == metric])
                  for metric in BASE.METRICS}
    result = {
        "generated_at": datetime.now(UTC).isoformat(), "models": list(BASE.MODELS), "modes": list(BASE.MODES),
        "arm": "without_gt", "points_per_metric": 8, "paired_seed_measurements": 24,
        "overall_descriptive": statistics,
        "x_unchanged": "Exact x values copied from the previous PCT-y scatter; PCT gain / GRPO overlap",
        "y": "Mean seed-paired Best minus Worst final accumulated Train TGC, in percentage points",
        "tgc": "Saved official boolean task-success aggregation; no PCT thresholding",
        "error_bars": "Sample SD of the three paired seed gaps, ddof=1",
        "both_panels_share_identical_y": True, "new_inference_calls": 0,
        "source_sha256": {**BASE.SOURCES, str(PREVIOUS / "plot_scatter.py"): BASE.digest(
            PREVIOUS / "plot_scatter.py")},
        "original_files_unchanged": {str(p): sha for p, sha in original_files.items()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "correlation_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "correlation": statistics,
                      "gemma": [p for p in points if p["model"] == "gemma26"]}))


if __name__ == "__main__":
    main()
