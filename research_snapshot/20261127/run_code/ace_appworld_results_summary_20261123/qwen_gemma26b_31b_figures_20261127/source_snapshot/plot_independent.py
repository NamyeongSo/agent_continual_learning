# SPDX-License-Identifier: Apache-2.0
"""Plot measured Qwen/Gemma Train results with model-local initial best-of-five lines."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = Path('/home/thskadud/Sico/labs/AgentStream/exgentic/outputs/ace_appworld_results_summary_20261123/qwen_gemma26b_31b_figures_20261127/data/independent')
REFERENCE = Path('/home/thskadud/Sico/labs/AgentStream/exgentic/outputs/ace_appworld_results_summary_20261123/model_strategy_bars_20261124/plot_model_strategy_bars.py')
SPEC = importlib.util.spec_from_file_location("ace_original_strategy_bars", REFERENCE)
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
DATA = BASE.DATA
GEMMAS = {"gemma26": Path('/home/thskadud/Sico/labs/AgentStream/exgentic/outputs/ace_appworld_gemma4_26b_a4b_it_wo_gt_train_only_20261124'), "gemma31": Path('/home/thskadud/Sico/labs/AgentStream/exgentic/outputs/ace_appworld_gemma4_31b_it_wo_gt_train_only_20261124')}
for model, cohort in GEMMAS.items():
    DATA.MODEL_RUNS[model] = {m: cohort / m / "accumulation" for m in ("thinking", "nonthinking")}
    DATA.ADV_RUNS[model] = cohort / "advantage"
MODELS = ("qwen38", "qwen36", "qwen35", "gemma26", "gemma31")
MODES = ("thinking", "nonthinking")
ARMS = ("without_gt", "with_gt")
SEEDS = (42, 43, 44)
STRATEGIES = BASE.STRATEGIES
LABELS = {**BASE.LABELS, "gemma26": "Gemma4-26B-A4B", "gemma31": "Gemma4-31B"}
COLORS = BASE.COLORS
OFFSETS = (0.0, 1.35, 2.50, 3.65, 4.80, 5.95)
STRIDE = 7.45
WIDTH = 0.76
LINE_STYLE = (0, (4, 3))


def export_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bar(ax, x, value, color, std=0):
    ax.bar(x, value, width=WIDTH, color=color, edgecolor="white", linewidth=0.85, zorder=3)
    if std > 1e-10:
        ax.errorbar(x, value, yerr=std, fmt="none", ecolor="#263244", elinewidth=1.0,
                    capsize=2.4, capthick=0.9, zorder=5)
    ax.text(x, value + std + 1.5, f"{value:.1f}", ha="center", va="bottom",
            rotation=90, fontsize=9.4, color="#354052", zorder=6)


def draw(initial, summary, metric):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titleweight": "semibold", "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(40, 16), sharey=True)
    plotted, lines = [], []
    line_metric = "pass_at_5" if metric == "tgc" else "mean_task_max_pct_at_5"
    line_label = "Initial pass@5 (official TGC)" if metric == "tgc" else "Initial max PCT over 5 runs (task mean)"
    for row, mode in enumerate(MODES):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            ticks, tick_labels = [], []
            for index, model in enumerate(MODELS):
                start = index * STRIDE
                left, right = start - 0.60, start + 6.55
                if index % 2 == 0:
                    ax.axvspan(left, right, color="#f3f5f8", alpha=0.85, zorder=0)
                if index:
                    ax.axvline(start - 0.75, color="#d3dbe5", linewidth=0.9,
                               linestyle=(0, (3, 3)), zorder=1)
                baseline = initial[model, mode]
                value = baseline[metric] * 100
                bar(ax, start, value, COLORS["initial"])
                plotted.append({"metric": metric, "model": model, "mode": mode, "arm": arm,
                                "category": "initial", "variant": "baseline", "x": start,
                                "mean_percent": value, "std_percent": "", "n_training_seeds": ""})
                missing = model in GEMMAS and arm == "with_gt"
                for number, strategy in enumerate(STRATEGIES, start=1):
                    x = start + OFFSETS[number]
                    if missing:
                        continue
                    record = summary[model, mode, arm, strategy]
                    value = 100 * record[f"independent_{metric}_mean"]
                    std = 100 * record[f"independent_{metric}_std"]
                    bar(ax, x, value, COLORS[strategy], std)
                    plotted.append({"metric": metric, "model": model, "mode": mode, "arm": arm,
                                    "category": strategy, "variant": "independent", "x": x,
                                    "mean_percent": value, "std_percent": std, "n_training_seeds": 3})
                if missing:
                    ax.axvspan(start + 0.80, right, ymin=0, ymax=104 / 116, color="#eef1f5", alpha=0.85, zorder=1)
                    ax.text(start + 3.65, 53, "w/ GT\nnot run", ha="center", va="center",
                            fontsize=15, color="#7d8797", linespacing=1.6, zorder=6)
                line_value = baseline[line_metric] * 100
                ax.hlines(line_value, left + 0.04, right - 0.04, color=COLORS["pass_at_5"],
                          linewidth=1.8, linestyles=LINE_STYLE, zorder=4)
                ax.text(left + 0.10, line_value + 1.25, f"{line_value:.1f}%", ha="left", va="bottom",
                        fontsize=9.1, color="#876db2", zorder=7,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.8})
                lines.append({"metric": metric, "model": model, "mode": mode, "arm": arm,
                              "line_metric": line_metric, "mean_percent": line_value,
                              "left_x": left + 0.04, "right_x": right - 0.04,
                              "tasks": 50, "initial_rollouts_per_task": 5})
                ticks.extend(start + np.array(OFFSETS))
                tick_labels.extend((f"Initial\n{metric.upper()}", "Best", "Random", "Worst",
                                    "Best\nadv.", "Worst\nadv."))
                ax.text(start + 3.0, -0.18, LABELS[model], transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=13.5, fontweight="bold", color="#263244")
            ax.set_xticks(ticks, tick_labels, fontsize=9.8)
            ax.tick_params(axis="x", length=0, pad=9)
            ax.tick_params(axis="y", labelsize=11, length=0, pad=6, labelleft=True)
            ax.set_xlim(-0.65, (len(MODELS) - 1) * STRIDE + 6.60)
            ax.set_ylim(0, 116)
            ax.set_yticks(np.arange(0, 101, 20))
            ax.set_ylabel(f"Official {metric.upper()} (%)", fontsize=13, labelpad=10)
            ax.grid(axis="y", color="#dfe5ec", linewidth=0.8, zorder=0)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#c5ceda")
            arm_label = "w/o GT" if arm == "without_gt" else "w/ GT"
            mode_label = "Thinking" if mode == "thinking" else "Non-thinking"
            ax.set_title(f"{'ABCD'[row * 2 + col]}   {mode_label}  ·  {arm_label}",
                         loc="left", fontsize=19, pad=18)
    handles = [Patch(facecolor=COLORS["initial"], label=f"Initial {metric.upper()}"),
               Line2D([0], [0], color=COLORS["pass_at_5"], linewidth=1.8,
                      linestyle=LINE_STYLE, label=line_label),
               Patch(facecolor="#657488", label="Independent: single-reflection Context")]
    fig.suptitle(f"AppWorld Train · {metric.upper()} before learning and after one Reflection update",
                 y=0.978, fontsize=25, fontweight="bold", color="#1f2c3e")
    fig.text(0.5, 0.942,
             "50 matched training tasks  ·  Initial mean and five independent selection strategies per model",
             ha="center", fontsize=13, color="#536176")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3,
               frameon=False, fontsize=12, handlelength=3.0, columnspacing=3.0)
    fig.text(0.5, 0.089,
             "Strategy bars: mean ± 1 sample SD across seed-selected candidate-bank means (42, 43, 44); 50 tasks per seed. "
             "Initial bars: mean over 50 tasks x 5 runs.", ha="center", fontsize=11.5, color="#536176")
    definition = ("Dashed lines: fraction of tasks with at least one official success in 5 initial runs. "
                  "TGC always uses the recorded official boolean success."
                  if metric == "tgc" else
                  "Dashed lines: take the maximum official PCT among 5 initial runs for EACH task, "
                  "then average those 50 task maxima.")
    fig.text(0.5, 0.060, definition, ha="center", fontsize=11.5, color="#536176")
    fig.text(0.5, 0.032, 'Selected tasks: saved 3-rollout candidate mean; skipped zero-signal tasks: initial 5-rollout mean. Selection and plotted scores share the bank. Gemma w/ GT unmeasured.',
             ha="center", fontsize=11.5, color="#536176")
    fig.subplots_adjust(left=0.043, right=0.99, top=0.845, bottom=0.21, hspace=0.57, wspace=0.065)
    name = f"qwen_gemma26b_31b_independent_{metric}_4panel"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=200, facecolor="white")
    fig.savefig(OUT / f"{name}_preview.png", dpi=100, facecolor="white")
    plt.close(fig)
    assert len(plotted) == 100 and len(lines) == 20
    assert sum(r["variant"] == "independent" for r in plotted) == 80
    assert all(0 <= r["mean_percent"] <= 100 for r in plotted + lines)
    export_csv(f"plotted_bars_{metric}.csv", plotted)
    export_csv(f"reference_lines_{metric}.csv", lines)
    return str(OUT / f"{name}.png")
