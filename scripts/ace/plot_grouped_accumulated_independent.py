# SPDX-License-Identifier: Apache-2.0
"""Combine saved PCT charts: within each model, accumulated five then independent five."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from statistics import mean, stdev

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = (ROOT / "outputs/ace_appworld_results_summary_20261123"
                  "/qwen_gemma26b_31b_figures_20261127")
MODELS = ("qwen38", "qwen36", "qwen35", "gemma26", "gemma31")
MODES = ("thinking", "nonthinking")
ARMS = ("without_gt", "with_gt")
STRATEGIES = ("best", "random", "worst", "best_advantage", "worst_advantage")
VARIANTS = ("accumulated", "independent")
LABELS = {"qwen38": "Qwen3.8-27B", "qwen36": "Qwen3.6-35B-A3B", "qwen35": "Qwen3.5-9B",
          "gemma26": "Gemma4-26B-A4B", "gemma31": "Gemma4-31B"}
COLORS = {"initial": "#497fa9", "pass_at_5": "#a18ac4", "best": "#24864c", "random": "#87929e",
          "worst": "#ce4949", "best_advantage": "#65ad76", "worst_advantage": "#e68b83"}
OFFSETS = {"accumulated": np.array((1.4, 2.45, 3.5, 4.55, 5.6)),
           "independent": np.array((7.4, 8.45, 9.5, 10.55, 11.6))}
STRIDE, WIDTH = 13.25, .76
NAME = "qwen_gemma26b_31b_accumulated_then_independent_pct_4panel"


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


def key(row):
    return tuple(row[k] for k in ("model", "mode", "arm", "strategy"))


def collect(bundle):
    initial_path = bundle / "data/accumulated/initial_metrics.csv"
    initial = {(r["model"], r["mode"]): r for r in read_csv(initial_path)}
    summary, sources = {}, {str(initial_path): digest(initial_path)}
    for variant in VARIANTS:
        folder = bundle / "data" / variant
        path, seed_path = folder / "metrics_mean_std.csv", folder / "metrics_by_seed.csv"
        sources.update({str(p): digest(p) for p in (path, seed_path)})
        seeds = read_csv(seed_path)
        rows = read_csv(path)
        if len(rows) != 80 or len(seeds) != 240:
            raise ValueError("Expected 80 measured strategy groups and three seeds each")
        for row in rows:
            group = [r for r in seeds if key(r) == key(row)]
            if len(group) != 3 or {int(r["seed"]) for r in group} != {42, 43, 44}:
                raise ValueError("All three seeds must be present")
            values = [float(r[f"{variant}_pct"]) for r in group]
            for stat, actual in (("mean", mean(values)), ("std", stdev(values))):
                if abs(actual - float(row[f"{variant}_pct_{stat}"])) > 1e-12:
                    raise ValueError(f"Saved {variant} PCT {stat} does not match seed-level scores")
            summary[variant, *key(row)] = row
    if len(initial) != 10 or len(summary) != 160:
        raise ValueError("Missing or duplicate model/mode/strategy groups")
    return initial, summary, sources


def draw_bar(ax, x, value, std, color, variant):
    if variant == "independent":
        fill = tuple(.78 * c + .22 for c in to_rgb(color))
        edge, linewidth = color, 1.0
    else:
        fill, edge, linewidth = color, "white", .8
    ax.bar(x, value, width=WIDTH, facecolor=fill, edgecolor=edge, linewidth=linewidth,
           linestyle="solid", hatch="///" if variant == "independent" else None, zorder=3)
    if std > 1e-10:
        ax.errorbar(x, value, yerr=std, fmt="none", ecolor="#263244", elinewidth=.95,
                    capsize=2.1, capthick=.8, zorder=5)
    ax.text(x, value + std + 1.5, f"{value:.1f}", ha="center", va="bottom", rotation=90,
            fontsize=8.8, color="#354052", zorder=6)


def draw(bundle, initial, summary):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titleweight": "semibold", "svg.fonttype": "none", "pdf.fonttype": 42,
                         "hatch.linewidth": .7})
    fig, axes = plt.subplots(2, 2, figsize=(50, 17), sharey=True)
    plotted, lines = [], []
    for row, mode in enumerate(MODES):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            ticks, labels = [], []
            for index, model in enumerate(MODELS):
                start = index * STRIDE
                left, right = start - .60, start + 12.25
                if index % 2 == 0:
                    ax.axvspan(left, right, color="#f3f5f8", alpha=.85, zorder=0)
                if index:
                    ax.axvline(start - .75, color="#d3dbe5", lw=.9, linestyle=(0, (3, 3)), zorder=1)
                before = initial[model, mode]
                value = 100 * float(before["pct"])
                draw_bar(ax, start, value, 0, COLORS["initial"], "baseline")
                plotted.append({"metric": "pct", "model": model, "mode": mode, "arm": arm,
                                "category": "initial", "variant": "baseline", "x": start,
                                "mean_percent": value, "std_percent": "", "n_training_seeds": ""})
                ticks.append(start)
                labels.append("Initial\nPCT")
                missing = model.startswith("gemma") and arm == "with_gt"
                for variant in VARIANTS:
                    for strategy, offset in zip(STRATEGIES, OFFSETS[variant], strict=True):
                        x = start + offset
                        ticks.append(x)
                        labels.append({"best_advantage": "Best\nadv.", "worst_advantage": "Worst\nadv."}
                                      .get(strategy, strategy.title()))
                        if missing:
                            continue
                        record = summary[variant, model, mode, arm, strategy]
                        value = 100 * float(record[f"{variant}_pct_mean"])
                        std = 100 * float(record[f"{variant}_pct_std"])
                        draw_bar(ax, x, value, std, COLORS[strategy], variant)
                        plotted.append({"metric": "pct", "model": model, "mode": mode, "arm": arm,
                                        "category": strategy, "variant": variant, "x": float(x),
                                        "mean_percent": value, "std_percent": std, "n_training_seeds": 3})
                    center = start + float(OFFSETS[variant].mean())
                    ax.text(center, -.125, variant.title(), transform=ax.get_xaxis_transform(),
                            ha="center", va="top", fontsize=10.8, fontweight="semibold", color="#526176")
                if missing:
                    ax.axvspan(start + .85, right, ymin=0, ymax=104 / 116, color="#eef1f5", alpha=.85, zorder=1)
                    ax.text(start + 6.6, 53, "w/ GT\nnot run", ha="center", va="center",
                            fontsize=16, color="#7d8797", linespacing=1.6, zorder=6)
                else:
                    ax.axvline(start + 6.5, ymin=0, ymax=104 / 116, color="#cbd3de", lw=.7, zorder=1)
                reference = 100 * float(before["mean_task_max_pct_at_5"])
                ax.hlines(reference, left + .04, right - .04, color=COLORS["pass_at_5"], lw=1.8,
                          linestyles=(0, (4, 3)), zorder=4)
                ax.text(left + .10, reference + 1.25, f"{reference:.1f}%", fontsize=8.8, color="#876db2",
                        ha="left", va="bottom", zorder=7,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": .8})
                lines.append({"metric": "pct", "model": model, "mode": mode, "arm": arm,
                              "line_metric": "mean_task_max_pct_at_5", "mean_percent": reference,
                              "left_x": left + .04, "right_x": right - .04,
                              "tasks": 50, "initial_rollouts_per_task": 5})
                ax.text(start + 6, -.235, LABELS[model], transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=14, weight="bold", color="#263244")
            ax.set_xticks(ticks, labels, fontsize=8.9)
            ax.tick_params(axis="x", length=0, pad=9)
            ax.tick_params(axis="y", length=0, pad=6, labelleft=True)
            ax.set(xlim=(-.65, (len(MODELS) - 1) * STRIDE + 12.3), ylim=(0, 116),
                   ylabel="Official PCT (%)")
            ax.set_yticks(np.arange(0, 101, 20))
            ax.grid(axis="y", color="#dfe5ec", lw=.8, zorder=0)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#c5ceda")
            title = ("Thinking" if mode == "thinking" else "Non-thinking") + "  ·  "
            title += "w/o GT" if arm == "without_gt" else "w/ GT"
            ax.set_title(f"{'ABCD'[row * 2 + col]}   {title}", loc="left", fontsize=20, pad=18)
    handles = [Patch(facecolor=COLORS["initial"], label="Initial PCT"),
               Line2D([0], [0], color=COLORS["pass_at_5"], lw=1.8, linestyle=(0, (4, 3)),
                      label="Initial max PCT over 5 runs (task mean)"),
               Patch(facecolor="#657488", edgecolor="white", label="Accumulated: final Context"),
               Patch(facecolor="#b0b9c5", edgecolor="#657488", linestyle="solid", hatch="///",
                     label="Independent: single-reflection Context")]
    fig.suptitle("AppWorld Train · Accumulated and independent Reflection selection PCT",
                 y=.978, fontsize=27, weight="bold", color="#1f2c3e")
    fig.text(.5, .942, "Within every model: Initial → Accumulated (five strategies) → Independent (five strategies)",
             ha="center", fontsize=14, color="#536176")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .925), ncol=4,
               frameon=False, fontsize=12, handlelength=3, columnspacing=3)
    notes = [
        "Strategy bars: mean ±1 sample SD across seeds 42, 43, 44; all 50 Train tasks per seed. "
        "Accumulated uses solid bars; Independent uses a lighter fill with diagonal hatching.",
        "Initial bars and dashed reference values match the separate charts. "
        "Reference: mean of each task's maximum PCT over its five initial runs.",
        "Independent: saved 3-rollout candidate means; skipped zero-signal tasks retain initial means. "
        "Selection and plotted scores share the same candidate bank.",
        "Accumulated Gemma26/31 retain 115/52 selected-case reruns; other outcomes retained. "
        "Selection bias and resampling variation apply. Gemma w/ GT unmeasured.",
    ]
    for y, note in zip((.095, .070, .045, .020), notes, strict=True):
        fig.text(.5, y, note, ha="center", fontsize=11.2, color="#536176")
    fig.subplots_adjust(left=.035, right=.993, top=.845, bottom=.265, hspace=.75, wspace=.065)
    # Preserve upright labels while fitting dense within-model strategy blocks.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes.flat:
        positions = ax.transData.transform(np.column_stack((ax.get_xticks(), np.zeros(len(ax.get_xticks())))))[:, 0]
        for index, label in enumerate(ax.get_xticklabels()):
            neighbors = []
            if index:
                neighbors.append(positions[index] - positions[index - 1])
            if index + 1 < len(positions):
                neighbors.append(positions[index + 1] - positions[index])
            available = .92 * min(neighbors)
            while label.get_window_extent(renderer).width > available and label.get_fontsize() > 7:
                label.set_fontsize(max(7, label.get_fontsize() - .2))
    paths = []
    for extension in ("png", "pdf", "svg"):
        path = bundle / f"{NAME}.{extension}"
        fig.savefig(path, dpi=200, facecolor="white")
        paths.append(path)
    path = bundle / f"{NAME}_preview.png"
    fig.savefig(path, dpi=100, facecolor="white")
    paths.append(path)
    plt.close(fig)
    return paths, plotted, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    preserved = {p.name: digest(p) for p in bundle.iterdir() if p.suffix in {".png", ".pdf", ".svg"}
                 and not p.name.startswith(NAME)}
    initial, summary, sources = collect(bundle)
    paths, plotted, lines = draw(bundle, initial, summary)
    if len(plotted) != 180 or len(lines) != 20:
        raise ValueError("Expected 20 Initial and 80 strategy bars per variant")
    for variant in VARIANTS:
        reference = bundle / f"data/{variant}/plotted_bars_pct.csv"
        sources[str(reference)] = digest(reference)
        old = {(r["model"], r["mode"], r["arm"], r["category"]): r for r in read_csv(reference)}
        for row in plotted:
            if row["variant"] not in {variant, "baseline"}:
                continue
            before = old[row["model"], row["mode"], row["arm"], row["category"]]
            for field in ("mean_percent", "std_percent"):
                if str(row[field]) != before[field]:
                    raise ValueError("A combined bar differs from its source chart")
    line_path = bundle / "data/accumulated/reference_lines_pct.csv"
    sources[str(line_path)] = digest(line_path)
    for a, b in zip(read_csv(line_path), lines, strict=True):
        if a["mean_percent"] != str(b["mean_percent"]):
            raise ValueError("An initial reference-line value changed")
    for mode in MODES:
        for arm in ARMS:
            for model in MODELS:
                group = [r for r in plotted if (r["model"], r["mode"], r["arm"]) == (model, mode, arm)]
                variants = [r["variant"] for r in group]
                expected = ["baseline"] if model.startswith("gemma") and arm == "with_gt" else (
                    ["baseline"] + ["accumulated"] * 5 + ["independent"] * 5)
                if variants != expected or not all(a["x"] < b["x"] for a, b in pairwise(group)):
                    raise ValueError("Bars are not grouped accumulated first, then independent")
    if any(digest(bundle / name) != sha for name, sha in preserved.items()):
        raise ValueError("An existing figure changed")
    folder = bundle / "data/combined_accumulated_independent"
    folder.mkdir(exist_ok=True)
    write_csv(folder / "plotted_bars_pct.csv", plotted)
    write_csv(folder / "reference_lines_pct.csv", lines)
    shutil.copy2(__file__, bundle / "source_snapshot" / Path(__file__).name)
    record = {"generated_at": datetime.now(UTC).isoformat(), "figures": [str(p) for p in paths],
              "within_model_order": ["Initial", "Accumulated x5", "Independent x5"],
              "panels": "Same mode x GT four-panel layout as separate charts",
              "independent_style": "Lighter filled bars with diagonal /// hatching; separate block within each model",
              "seed_std": "Sample SD across seed42/43/44 Train50 means; source means/SDs exactly preserved",
              "new_inference_calls": 0, "source_sha256": sources, "original_figure_sha256": preserved,
              "script_sha256": digest(Path(__file__)),
              "generated_figure_sha256": {p.name: digest(p) for p in paths}}
    (folder / "provenance.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"figures": record["figures"], "bars": len(plotted), "verified": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
