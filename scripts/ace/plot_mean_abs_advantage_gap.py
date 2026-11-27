# SPDX-License-Identifier: Apache-2.0
"""Relate raw candidate mean |GRPO advantage| to saved Best-Worst Train PCT gaps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev

import matplotlib
import numpy as np
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = (ROOT / "outputs/ace_appworld_results_summary_20261123"
                  "/qwen_gemma26b_31b_figures_20261127")
MODELS = ("qwen38", "qwen36", "qwen35", "gemma26", "gemma31")
MODES = ("thinking", "nonthinking")
LABELS = {"qwen38": "Qwen3.8-27B", "qwen36": "Qwen3.6-35B-A3B", "qwen35": "Qwen3.5-9B",
          "gemma26": "Gemma4-26B-A4B", "gemma31": "Gemma4-31B"}
COLORS = dict(zip(MODELS, ("#3973ac", "#218a69", "#d98b2b", "#8b67b1", "#b95c58"), strict=True))
MARKERS = {"thinking": "o", "nonthinking": "s"}
NAME = "mean_abs_grpo_advantage_vs_accumulation_pct_gap"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(bundle):
    candidate_path = bundle / "data/all_candidates/reflection_advantages.csv"
    gap_path = bundle / "data/overlap_scatter/pct_overlap_vs_accumulation_pct_gap_points.csv"
    seed_path = bundle / "data/accumulated/metrics_by_seed.csv"
    groups = defaultdict(list)
    for row in read_csv(candidate_path):
        if row["arm"] == "without_gt" and row["model"] in MODELS:
            groups[row["model"], row["mode"]].append(row)
    gaps = {(r["model"], r["mode"]): r for r in read_csv(gap_path) if r["overlap_metric"] == "gain"}
    seeds = {(r["model"], r["mode"], int(r["seed"]), r["strategy"]): r for r in read_csv(seed_path)
             if r["arm"] == "without_gt"}
    if len(groups) != 10:
        raise ValueError("Expected five models x two modes, w/o GT only")
    points, group_rows = [], []
    for model in MODELS:
        for mode in MODES:
            rows = groups[model, mode]
            if len(rows) != 250 or len({(r["task_id"], r["source_repeat"]) for r in rows}) != 250:
                raise ValueError("Every candidate must occur once; do not duplicate selection seeds")
            tasks = defaultdict(list)
            for row in rows:
                tasks[row["task_id"]].append(row)
            if len(tasks) != 50:
                raise ValueError("Expected 50 task groups")
            zero_groups = 0
            for task, trials in tasks.items():
                if len(trials) != 5 or {int(r["source_repeat"]) for r in trials} != set(range(5)):
                    raise ValueError("Expected G5 within every task")
                rewards = np.array([float(r["mean_pct"]) for r in trials])
                advantages = np.array([float(r["advantage"]) for r in trials])
                tied = bool(np.all(rewards == rewards[0]))
                expected = np.zeros(5) if tied else (rewards - rewards.mean()) / (rewards.std(ddof=0) + 1e-8)
                if not np.allclose(advantages, expected, atol=1e-9, rtol=0):
                    raise ValueError("Saved advantages do not match group z-normalization")
                if abs(advantages.mean()) > 1e-9:
                    raise ValueError("Group advantages must have mean zero")
                if any((r["zero_signal_group"] == "True") != tied for r in trials):
                    raise ValueError("All-tied group metadata differs")
                zero_groups += tied
                group_rows.append({"model": model, "mode": mode, "task_id": task,
                                   "mean_abs_advantage": float(np.abs(advantages).mean()),
                                   "zero_signal_group": tied})
            values = [float(r["advantage"]) for r in rows]
            distance = mean(abs(a) for a in values)
            if not np.isclose(distance, mean(r["mean_abs_advantage"] for r in group_rows
                                            if r["model"] == model and r["mode"] == mode), atol=1e-12):
                raise ValueError("Candidate mean and equally weighted task mean differ")
            paired = []
            for seed in (42, 43, 44):
                best = seeds[model, mode, seed, "best"]
                worst = seeds[model, mode, seed, "worst"]
                if any(int(r["train_tasks"]) != 50 for r in (best, worst)):
                    raise ValueError("Every paired seed must have all 50 Train outcomes")
                paired.append(100 * (float(best["accumulated_pct"]) - float(worst["accumulated_pct"])))
            gap = gaps[model, mode]
            y, error = mean(paired), stdev(paired)
            if abs(y - float(gap["gap_mean_pp"])) > 1e-10 or abs(error - float(gap["gap_std_pp"])) > 1e-10:
                raise ValueError("Y differs from the existing PCT overlap scatter")
            points.append({"model": model, "mode": mode, "arm": "without_gt", "tasks": 50,
                           "candidates": 250, "mean_abs_advantage": distance,
                           "exact_zero_candidate_fraction": sum(a == 0 for a in values) / 250,
                           "zero_signal_task_fraction": zero_groups / 50,
                           "gap_mean_pp": y, "gap_std_pp": error,
                           "candidate_reward": "three-rollout mean PCT", "seeds": "42,43,44"})
    pins = {str(path): digest(path) for path in (candidate_path, gap_path, seed_path)}
    return points, group_rows, pins


def draw(bundle, points, stats):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(13.5, 9.2))
    for point in points:
        color = COLORS[point["model"]]
        ax.errorbar(point["mean_abs_advantage"], point["gap_mean_pp"], yerr=point["gap_std_pp"],
                    fmt="none", color=color, alpha=.4, elinewidth=1.5, capsize=4, zorder=2)
        ax.scatter(point["mean_abs_advantage"], point["gap_mean_pp"], s=145, color=color,
                   marker=MARKERS[point["mode"]], edgecolor="white", linewidth=1, zorder=4)
    x = np.array([p["mean_abs_advantage"] for p in points])
    low = min(p["gap_mean_pp"] - p["gap_std_pp"] for p in points)
    high = max(p["gap_mean_pp"] + p["gap_std_pp"] for p in points)
    padding = .12 * (high - low)
    xs = np.linspace(x.min() - .04, x.max() + .04, 200)
    ax.plot(xs, stats["ols_intercept_gap_pp"] + stats["ols_slope_gap_pp_per_mean_abs_advantage"] * xs,
            color="#29384b", lw=1.8, linestyle=(0, (5, 3)), zorder=1)
    ax.axhline(0, color="#99a6b6", lw=1.05)
    ax.set(xlim=(xs[0], xs[-1]), ylim=(low - padding, high + padding),
           xlabel="Mean absolute GRPO advantage  (mean |A| across 250 candidates)",
           ylabel="Best - Worst accumulated Train PCT (percentage points)")
    ax.grid(color="#e4e9f0", lw=.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c9d1dc")
    ax.text(.025, .975, f"Pearson r = {stats['pearson_r']:+.3f}\n"
            f"Spearman $\\rho$ = {stats['spearman_rho']:+.3f}\n10 settings / 5 models",
            transform=ax.transAxes, ha="left", va="top", fontsize=11.5,
            bbox={"facecolor": "white", "edgecolor": "#dbe3ec", "boxstyle": "round,pad=.55"})
    handles = [Line2D([0], [0], marker="o", linestyle="none", color=COLORS[m], label=LABELS[m], markersize=8)
               for m in MODELS]
    modes = [Line2D([0], [0], marker=MARKERS[m], linestyle="none", color="#4a586b", markersize=8,
                    label="Thinking" if m == "thinking" else "Non-thinking") for m in MODES]
    modes.append(Line2D([0], [0], color="#29384b", linestyle="--", label="OLS descriptive trend"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10,
               bbox_to_anchor=(.5, .118))
    fig.legend(handles=modes, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .080))
    fig.suptitle("Mean absolute GRPO advantage vs. accumulation PCT gap · w/o GT",
                 fontsize=17.5, weight="bold", y=.975, color="#233247")
    fig.text(.5, .92, "Qwen3.8 / Qwen3.6 / Qwen3.5 / Gemma26 / Gemma31 · "
             "Original PCT-selected plain Best / Worst contexts", ha="center", fontsize=10.5, color="#526176")
    fig.text(.5, .054, "X: raw candidate advantages, including exact zeros; no KDE or selection-seed duplication. "
             "Y: paired seed gap mean ±1 sample SD (42, 43, 44).", ha="center", fontsize=9.1, color="#526176")
    provenance = bundle / "data/accumulated/provenance.json"
    reruns = (json.loads(provenance.read_text()).get("diagnostic_update", {}).get("rerun_counts", {})
              if provenance.exists() else {})
    if reruns.get("gemma26") == 115:
        note = ("Gemma26: 115 and Gemma31: 52 selected-case reruns; other outcomes retained. "
                "Selection bias applies. Same Y as the saved PCT overlap scatter.")
    else:
        note = ("Gemma31 Thinking includes 52 selected-case reruns; Gemma26 retains the earlier scores. "
                "Same Y as the saved PCT overlap scatter.")
    fig.text(.5, .030, note, ha="center", fontsize=9.1, color="#526176")
    fig.text(.5, .007, "Z-normalization removes within-task reward scale. "
             "Correlations are descriptive; the two modes of each model are not independent models.",
             ha="center", fontsize=9.1, color="#526176")
    fig.subplots_adjust(left=.10, right=.975, top=.855, bottom=.265)
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = bundle / f"{NAME}.{suffix}"
        fig.savefig(path, dpi=200, facecolor="white")
        paths.append(path)
    path = bundle / f"{NAME}_preview.png"
    fig.savefig(path, dpi=100, facecolor="white")
    paths.append(path)
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    old_figures = {str(p): digest(p) for p in bundle.iterdir() if p.suffix in {".png", ".pdf", ".svg"}
                   and not p.name.startswith(NAME)}
    points, groups, pins = collect(bundle)
    x, y = (np.array([p[field] for p in points]) for field in ("mean_abs_advantage", "gap_mean_pp"))
    slope, intercept = np.polyfit(x, y, 1)
    stats = {"pearson_r": float(pearsonr(x, y).statistic), "spearman_rho": float(spearmanr(x, y).statistic),
             "ols_slope_gap_pp_per_mean_abs_advantage": float(slope), "ols_intercept_gap_pp": float(intercept)}
    paths = draw(bundle, points, stats)
    folder = bundle / "data/mean_abs_advantage"
    folder.mkdir(exist_ok=True)
    write_csv(folder / "scatter_points.csv", points)
    write_csv(folder / "task_group_distances.csv", groups)
    if any(digest(Path(path)) != sha for path, sha in old_figures.items()):
        raise ValueError("An existing figure was modified")
    snapshot = bundle / "source_snapshot" / Path(__file__).name
    shutil.copy2(__file__, snapshot)
    report = {"generated_at": datetime.now(UTC).isoformat(), "points": 10, "models": 5,
              "candidates": 2500, "formula": "D = sum(abs(A_i) for i in 250 candidates) / 250",
              "all_tied_task_groups": "Retained as five exact zero advantages",
              "y": "Same PCT Best-Worst seed means and paired sample SD as existing scatter",
              "statistics_descriptive": stats, "source_sha256": pins,
              "script_sha256": digest(snapshot), "new_inference_calls": 0,
              "original_figures_preserved": old_figures,
              "generated_figure_sha256": {str(path): digest(path) for path in paths}}
    (folder / "provenance.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"figures": [str(p) for p in paths], "statistics": stats, "verified": True}, indent=2))


if __name__ == "__main__":
    main()
