# SPDX-License-Identifier: Apache-2.0
"""Plot initial official scores and reorder single-Reflection gain violins by each initial metric."""

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ace_plot_source", OUT.parent / "build_report.py")
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
source.OUT = OUT
METRICS = {"pct": "PCT", "tgc": "TGC", "pass_at_5": "Pass@5", "pass_power_5": "Pass⁵"}
MODE_LABELS = ("Thinking / High", "Non-thinking / Low")
STRATEGY_COLORS = {"best": "#17765c", "worst": "#b55643"}
GRID = np.linspace(-100, 100, 801)


def mode_for(model, row):
    return tuple(source.MODEL_RUNS[model])[row]


def initial_scores():
    records = {}
    for model, modes in source.MODEL_RUNS.items():
        for mode, run in modes.items():
            source.load_reflection(model, mode, run)
            tasks = source.INITIAL[model, mode]
            assert len(tasks) == 50 and len(source.BANKS[model, mode]) == 100
            success_counts = [sum(r["success"] for r in trials) for trials in tasks.values()]
            assert all(type(r["success"]) is bool for trials in tasks.values() for r in trials)
            rounds = [source.metrics([next(r for r in trials if r["source_repeat"] == repeat)
                                      for trials in tasks.values()]) for repeat in range(5)]
            record = {"model": model, "mode": mode, "tasks": 50, "rollouts": 250,
                      "pct": float(np.mean([r["pct"] for r in rounds])),
                      "tgc": float(np.mean([r["tgc"] for r in rounds])),
                      "pct_round_std": float(np.std([r["pct"] for r in rounds], ddof=1)),
                      "tgc_round_std": float(np.std([r["tgc"] for r in rounds], ddof=1)),
                      "pass_at_5": sum(n >= 1 for n in success_counts) / 50,
                      "pass_power_5": sum(n == 5 for n in success_counts) / 50}
            assert record["pass_power_5"] <= record["tgc"] <= record["pass_at_5"]
            records[model, mode] = record
    assert len(records) == 10
    assert len({tuple(sorted(tasks)) for tasks in source.INITIAL.values()}) == 1
    return records


def model_order(records, row, metric):
    # Python's stable sort retains the original model order for exact ties.
    return sorted(source.LABELS, key=lambda m: records[m, mode_for(m, row)][metric], reverse=True)


def plot_initial(records):
    fig, axes = plt.subplots(2, 4, figsize=(23, 10))
    for row in range(2):
        color = "#376d8b" if row == 0 else "#9a6b3d"
        for col, (metric, label) in enumerate(METRICS.items()):
            ax = axes[row, col]
            models = model_order(records, row, metric)
            y = np.arange(len(models))
            values = [records[m, mode_for(m, row)][metric] * 100 for m in models]
            ax.barh(y, values, height=0.55, color=color, alpha=0.84, zorder=3)
            ax.set_yticks(y, [source.LABELS[m].replace("Qwen3.6-35B-A3B", "Qwen3.6-35B") for m in models])
            ax.invert_yaxis()
            for position, value in enumerate(values):
                ax.text(value + 1.2, position, f"{value:.2f}%", va="center", fontsize=10)
            ax.set_xlim(0, 118)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xlabel("Score (%)")
            ax.set_title(f"{MODE_LABELS[row]}\n{label}", fontsize=13, weight="semibold", pad=14)
            ax.grid(axis="x", color="#e6ebf1", linewidth=0.8, zorder=0)
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
            ax.tick_params(length=0)
    fig.suptitle("AppWorld Train · Performance before context learning", y=0.98, fontsize=21, weight="bold")
    fig.text(0.5, 0.93, "50 matched tasks · 5 rollouts per task · Models ranked separately for each initial metric",
             ha="center", fontsize=12, color="#475363")
    fig.text(0.5, 0.025,
             "PCT = mean official test pass fraction · TGC = mean official success · "
             "Pass@5 = at least one success · Pass⁵ = all five successes", ha="center", fontsize=11, color="#475363")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.85, bottom=0.10, hspace=0.50, wspace=0.65)
    fig.savefig(OUT / "initial_performance_pct_tgc_pass_at_5_pass_power_5.png", dpi=180, facecolor="white")
    plt.close(fig)


def gain_data(filtered=False, advantage=False):
    result = {}
    for model, modes in source.MODEL_RUNS.items():
        for mode in modes:
            for arm in source.ARMS:
                points = source.gain_points(model, mode, arm, filtered=filtered, advantage=advantage)
                densities = {s: source.density(samples, GRID) for s, samples in points.items()}
                result[model, mode, arm] = points, densities
    return result


def plot_sorted_gain(records, data, metric, variant):
    scale = 0.43 / max(float(d.max()) for _, densities in data.values() for d in densities.values())
    fig, axes = plt.subplots(2, 2, figsize=(16.4, 11.3), sharey=True)
    advantage = variant == "advantage_sign"
    for row in range(2):
        models = model_order(records, row, metric)
        for col, arm in enumerate(source.ARMS):
            ax = axes[row, col]
            ticks = []
            for position, model in enumerate(models):
                mode = mode_for(model, row)
                points, densities = data[model, mode, arm]
                count = round(sum(weight for _, weight in points["best"]))
                baseline = records[model, mode][metric] * 100
                ticks.append(f"{source.SHORT[model]}\n{METRICS[metric]}={baseline:.2f}%\nn={count}/50")
                random_width = scale * densities["random"]
                ax.fill_betweenx(GRID, position - random_width, position + random_width, where=random_width > 0,
                                 facecolor="#dfe2e6", edgecolor="#aab1ba", alpha=0.72, linewidth=0.7, zorder=1)
                for strategy, side in (("best", -1), ("worst", 1)):
                    density = densities[strategy]
                    color = STRATEGY_COLORS[strategy]
                    ax.fill_betweenx(GRID, position, position + side * scale * density, where=density > 0,
                                     facecolor=to_rgba(color, 0.22), edgecolor=color, linewidth=0.85,
                                     hatch=source.HATCHES[model], zorder=3)
                    center = np.average([v for v, _ in points[strategy]], weights=[w for _, w in points[strategy]])
                    width = float(np.interp(center, GRID, density)) * scale
                    ax.plot([position, position + side * width], [center, center], color=color, linewidth=2.0, zorder=4)
            ax.axhline(0, color="#555e69", linestyle=(0, (4, 3)), linewidth=1.1, zorder=2)
            ax.set(title=f"{MODE_LABELS[row]} · {source.ARMS[arm]}", xticks=range(5), xticklabels=ticks,
                   xlim=(-0.65, 4.65), ylim=(-100, 100))
            ax.set_yticks(range(-100, 101, 25))
            ax.grid(axis="y", color="#e6ebf1", linewidth=0.8)
            ax.tick_params(length=0)
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
            if col == 0:
                ax.set_ylabel("PCT gain (percentage points)")
    fig.suptitle(f"Single reflection PCT gain · Ranked by initial {METRICS[metric]}", y=0.98,
                 fontsize=19, weight="bold")
    fig.text(0.5, 0.944, "Higher initial score on the left · Best left / Worst right · Random in gray · Area n / 50",
             ha="center", fontsize=11)
    handles = [Patch(facecolor=to_rgba(c, 0.22), edgecolor=c, label=s + ("_advantage" if advantage else ""))
               for s, c in STRATEGY_COLORS.items()]
    handles += [Patch(facecolor="#dfe2e6", edgecolor="#aab1ba", label="Random"),
                Line2D([0], [0], color="#555e69", linewidth=2, label="Mean"),
                Line2D([0], [0], color="#555e69", linestyle="--", label="Zero gain")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.040))
    cohort = "All 50 train tasks" if variant == "all_tasks" else "All-zero-advantage tasks excluded"
    fig.text(0.5, 0.018, f"{cohort} · Sorting uses all 50 initial tasks · Gain uses each task's initial mean · "
             "Shared Gaussian bandwidth 7 pp", ha="center", fontsize=9, color="#555e69")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.16, hspace=0.47, wspace=0.10)
    fig.savefig(OUT / f"gain_{variant}_sorted_by_initial_{metric}.png", dpi=190, facecolor="white")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "hatch.linewidth": 0.7})
    records = initial_scores()
    source.write_csv("initial_metrics.csv", list(records.values()))
    ranks = []
    for row in range(2):
        for metric in METRICS:
            for rank, model in enumerate(model_order(records, row, metric), 1):
                ranks.append({"mode_group": MODE_LABELS[row], "metric": metric, "rank": rank,
                              "model": model, "mode": mode_for(model, row),
                              "initial_score": records[model, mode_for(model, row)][metric]})
    source.write_csv("initial_metric_ranks.csv", ranks)
    plot_initial(records)
    for filtered, advantage, variant in [(False, False, "all_tasks"), (True, False, "nonzero_advantage"),
                                          (False, True, "advantage_sign")]:
        data = gain_data(filtered=filtered, advantage=advantage)
        for metric in METRICS:
            plot_sorted_gain(records, data, metric, variant)
    provenance = {"created_at_kst": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"),
                  "sort_order": "descending initial metric within each mode group; exact ties retain source model order",
                  "sort_population": "all initial train50 tasks, including when the gain distribution excludes zero-advantage tasks",
                  "initial_metrics": "PCT/TGC mean over 250 rollouts; pass@5/pass^5 over 50 groups of 5 recorded official success flags",
                  "mode_groups": MODE_LABELS, "gain_metric": "PCT minus task initial mean, percentage points",
                  "density": "same bandwidth 7pp and n/50 denominator as previous plots",
                  "source_sha256": source.SOURCES}
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(list(records.values()), ensure_ascii=False, indent=2))
    print("Generated one initial-performance figure and twelve sorted gain-violin figures")


if __name__ == "__main__":
    main()
