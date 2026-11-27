# SPDX-License-Identifier: Apache-2.0
"""Plot task-local GRPO advantages for best/worst/random and recompute split-shape overlap."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
REFERENCE = ROOT / "reflection_advantage_violin_20261124/plot_advantage_violin.py"
SPEC = importlib.util.spec_from_file_location("grpo_reference", REFERENCE)
GRPO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRPO)
REPORT = GRPO.REF
MODELS = tuple(REPORT.LABELS)
GRID = np.linspace(-2.0, 2.0, 1001)
BANDWIDTH = 0.12
NAME = "independent_grpo_advantage_split_violin_n_over_50_with_random_overlap_annotated"
COLORS = {"best": "#17765c", "worst": "#b55643"}


def export_csv(filename, rows):
    with (OUT / filename).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def area(values, grid=GRID):
    return float(np.trapz(values, grid))


def intersection(first, second):
    difference = first - second
    indices = np.flatnonzero(difference[:-1] * difference[1:] < 0)
    if not len(indices):
        return area(np.minimum(first, second))
    fraction = -difference[indices] / (difference[indices + 1] - difference[indices])
    crossings = GRID[indices] + fraction * (GRID[indices + 1] - GRID[indices])
    refined = np.sort(np.concatenate([GRID, crossings]))
    return area(np.minimum(np.interp(refined, GRID, first), np.interp(refined, GRID, second)), refined)


def density(points):
    values = np.array([v for v, _ in points])
    weights = np.array([w for _, w in points])
    kernels = np.zeros((len(GRID), len(values)))
    for reflected_values in (values, -4.0 - values, 4.0 - values):
        offsets = (GRID[:, None] - reflected_values[None, :]) / BANDWIDTH
        kernels += np.where(np.abs(offsets) <= 4, np.exp(-0.5 * offsets ** 2), 0)
    kernels /= np.trapz(kernels, GRID, axis=0)
    result = kernels @ (weights / 50)
    assert abs(area(result) - weights.sum() / 50) < 1e-12
    return result


def collect():
    _, candidates, task_groups, _, verified = GRPO.collect()
    indexed = {(r["model"], r["mode"], r["arm"], r["task_id"], r["source_repeat"]): r for r in candidates}
    assert len(indexed) == 5000
    cached, overlaps, selected = {}, [], []
    arrays = {"advantage_grid": GRID}
    for model in MODELS:
        for mode in REPORT.MODEL_RUNS[model]:
            for arm in REPORT.ARMS:
                points = {s: [] for s in ("best", "worst", "random")}
                tasks = sorted(task for group_arm, task in REPORT.BANKS[model, mode] if group_arm == arm)
                assert len(tasks) == 50
                zero_tasks = 0
                for task in tasks:
                    group = REPORT.BANKS[model, mode][arm, task]
                    rewards = [group[i]["exact_pct"] for i in range(5)]
                    zero_tasks += len(set(rewards)) == 1
                    pools = {
                        "best": [i for i, reward in enumerate(rewards) if reward == max(rewards)],
                        "worst": [i for i, reward in enumerate(rewards) if reward == min(rewards)],
                        "random": list(range(5)),
                    }
                    advantages = [indexed[model, mode, arm, task, i]["advantage"] for i in range(5)]
                    assert abs(sum(advantages)) < 1e-12
                    for strategy, pool in pools.items():
                        for i in pool:
                            candidate = indexed[model, mode, arm, task, i]
                            value = candidate["advantage"]
                            assert strategy != "best" or value >= 0
                            assert strategy != "worst" or value <= 0
                            points[strategy].append((value, 1 / len(pool)))
                            selected.append({**candidate, "strategy": strategy, "selection_weight": 1 / len(pool)})
                assert all(abs(sum(w for _, w in p) - 50) < 1e-10 for p in points.values())
                means = {s: float(np.average([v for v, _ in p], weights=[w for _, w in p]))
                         for s, p in points.items()}
                assert abs(means["random"]) < 1e-12
                densities = {s: density(p) for s, p in points.items()}
                best, worst, random = (densities[s] for s in ("best", "worst", "random"))
                left = intersection(best, random)
                right = intersection(worst, random)
                random_area = 2 * area(random)
                overlap = left + right
                record = {
                    "model": model, "mode": mode, "arm": arm, "tasks": 50,
                    "zero_signal_tasks": zero_tasks,
                    "best_mean_advantage": means["best"], "worst_mean_advantage": means["worst"],
                    "random_mean_advantage": means["random"],
                    "best_random_left_overlap_percent": 100 * left / area(random),
                    "worst_random_right_overlap_percent": 100 * right / area(random),
                    "overlap_of_random_percent": 100 * overlap / random_area,
                    "geometric_iou_percent": 100 * overlap / (area(best) + area(worst) + random_area - overlap),
                    "kde_bandwidth_advantage": BANDWIDTH, "population_std_ddof": 0,
                }
                assert 0 <= record["overlap_of_random_percent"] <= 100 + 1e-10
                overlaps.append(record)
                cached[model, mode, arm] = points, densities, record
                for strategy, values in densities.items():
                    arrays[f"{model}_{mode}_{arm}_{strategy}"] = values
    assert len(overlaps) == len(cached) == 20
    export_csv("reflection_advantages.csv", candidates)
    export_csv("task_group_statistics.csv", task_groups)
    export_csv("selection_weighted_advantages.csv", selected)
    export_csv("overlap_by_model_mode_gt.csv", overlaps)
    np.savez_compressed(OUT / "source_densities.npz", **arrays)
    return cached, overlaps, verified


def draw(cached):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "hatch.linewidth": 0.7,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    max_density = max(float(d.max()) for _, densities, _ in cached.values() for d in densities.values())
    scale = 0.43 / max_density
    fig, axes = plt.subplots(2, 2, figsize=(18, 12.4), sharey=True)
    for row in range(2):
        for col, (arm, arm_title) in enumerate(REPORT.ARMS.items()):
            ax = axes[row, col]
            ticks = []
            for position, model in enumerate(MODELS):
                mode = tuple(REPORT.MODEL_RUNS[model])[row]
                points, densities, stats = cached[model, mode, arm]
                ticks.append(REPORT.SHORT[model] + f"\nn=50/50\nOverlap {stats['overlap_of_random_percent']:.1f}%")
                random_width = scale * densities["random"]
                ax.fill_betweenx(GRID, position - random_width, position + random_width,
                                 where=random_width > 0,
                                 facecolor="#dfe2e6", edgecolor="#aab1ba", alpha=0.72, linewidth=0.7, zorder=1)
                for strategy, side in (("best", -1), ("worst", 1)):
                    d = densities[strategy]
                    color = COLORS[strategy]
                    ax.fill_betweenx(GRID, position, position + side * scale * d,
                                     where=d > 0,
                                     facecolor=to_rgba(color, 0.22), edgecolor=color, linewidth=0.85,
                                     hatch=REPORT.HATCHES[model], zorder=3)
                    center = np.average([v for v, _ in points[strategy]], weights=[w for _, w in points[strategy]])
                    width = float(np.interp(center, GRID, d)) * scale
                    ax.plot([position, position + side * width], [center, center],
                            color=color, linewidth=2, zorder=4)
            ax.axhline(0, color="#555e69", linestyle=(0, (4, 3)), linewidth=1.1, zorder=2)
            title = ("Thinking / High" if row == 0 else "Non-thinking / Low") + " · " + arm_title
            ax.set(title=title, xticks=range(5), xticklabels=ticks, xlim=(-0.65, 4.65), ylim=(-2, 2))
            ax.set_yticks(np.arange(-2, 2.01, 0.5))
            ax.tick_params(length=0, axis="both", pad=7)
            ax.grid(axis="y", color="#e6ebf1", linewidth=0.8)
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
            if col == 0:
                ax.set_ylabel("Task-local GRPO advantage")
    fig.suptitle("AppWorld Train · Single-reflection GRPO advantage", fontsize=22, weight="bold", y=0.982)
    fig.text(0.5, 0.947,
             "Best on the left · Worst on the right · Random in gray · Shared area proportional to n / 50",
             ha="center", fontsize=12)
    handles = [Patch(facecolor=to_rgba(c, 0.22), edgecolor=c, label=s) for s, c in COLORS.items()]
    handles += [Patch(facecolor="#dfe2e6", edgecolor="#aab1ba", label="Random"),
                Line2D([0], [0], color="#555e69", linewidth=2, label="Mean"),
                Line2D([0], [0], color="#555e69", linestyle="--", label="Zero advantage")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.079))
    fig.text(0.5, 0.061,
             "A = (candidate mean PCT over 3 evaluations - mean of 5 candidate rewards) / (population SD + 1e-8)",
             ha="center", fontsize=10, color="#44566d")
    fig.text(0.5, 0.039,
             "All train tasks · Uniform selection within tied best/worst pools; random over all 5 · "
             "All-tied task groups retained at A=0 · KDE bandwidth 0.12",
             ha="center", fontsize=9.5, color="#44566d")
    fig.text(0.5, 0.017,
             "Overlap = common area of best-left + worst-right and Random / total Random area. "
             "Smoothed tails can cross zero; observed best A >= 0 and worst A <= 0.",
             ha="center", fontsize=9.5, color="#44566d")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.213, hspace=0.53, wspace=0.10)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{NAME}.{extension}", dpi=210, facecolor="white")
    fig.savefig(OUT / f"{NAME}_preview.png", dpi=100, facecolor="white")
    plt.close(fig)


def main():
    cached, rows, verified = collect()
    draw(cached)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(), "model_order": list(MODELS),
        "groups": 20, "task_groups": 1000, "reflections": 5000,
        "reused_training_group_checks": verified, "candidate_rollouts": 3,
        "advantage": "(mean PCT over 3 evaluations - group mean of G5 candidate means) / (population SD + 1e-8)",
        "group_std_ddof": 0, "epsilon": 1e-8, "reward_scale": "PCT in [0,1]",
        "zero_signal": "All-five-tied groups retained at advantage=0",
        "selection": "Task-balanced, uniform over exact best/worst ties and over all 5 for Random",
        "density": "Gaussian KDE; bandwidth=0.12; 4-SD truncation; reflected at [-2,2]; "
                   "per-kernel area normalization; n/50",
        "overlap": "(integral min(best,random) + integral min(worst,random)) / (2*integral random)",
        "integration": "Exact piecewise-linear intersections on the displayed 0.004 advantage grid",
        "new_model_calls": 0, "source_sha256": REPORT.SOURCES,
        "helper_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "groups": len(rows), "training_group_checks": verified}))
    for row in rows:
        print(row["model"], row["mode"], row["arm"], f"overlap={row['overlap_of_random_percent']:.2f}%")


if __name__ == "__main__":
    main()
