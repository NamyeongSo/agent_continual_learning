# SPDX-License-Identifier: Apache-2.0
"""Recompute reflection rewards, gains, GRPO advantages and overlap from official TGC."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
SUMMARY = OUT.parents[1]
PCT_SOURCE = OUT.parent / "qwen_gemma26b_wo_gt_advantage_gain_20261125"
SCATTER_SOURCE = SUMMARY / "overlap_vs_accumulation_gap_20261124/qwen_gemma26b_wo_gt_20261125"
SCATTER_OUT = SCATTER_SOURCE / "tgc_reward_overlap"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


BASE = load_module("pct_reward_violin_reference", PCT_SOURCE / "plot_qwen_gemma.py")
SCATTER = load_module("tgc_gap_scatter_reference", SCATTER_SOURCE / "tgc/plot_scatter_tgc.py")
plt = BASE.plt
MODELS, MODES = BASE.MODELS, BASE.MODES
METRICS = {"gain": "TGC gain overlap", "advantage": "TGC-reward GRPO advantage overlap"}
SCATTER.BASE.METRICS = METRICS
SOURCES = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    SOURCES[str(path)] = digest(path)
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(folder, name, rows):
    with (folder / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect():
    candidates, groups, indexed, initial = [], [], {}, {}
    for model in MODELS:
        for mode in MODES:
            if model == "gemma26":
                BASE.load_gemma(mode)
            else:
                BASE.REF.load_reflection(model, mode, BASE.REF.MODEL_RUNS[model][mode])
            tasks = sorted(BASE.REF.INITIAL[model, mode])
            assert len(tasks) == 50 and tasks == sorted(BASE.REF.INITIAL["qwen38", "thinking"])
            for task in tasks:
                before = BASE.REF.INITIAL[model, mode][task]
                assert len(before) == 5 and all(type(r["success"]) is bool for r in before)
                initial_tgc = Fraction(sum(r["success"] for r in before), 5)
                initial[model, mode, task] = initial_tgc
                bank = BASE.REF.BANKS[model, mode]["without_gt", task]
                rewards = []
                for index in range(5):
                    rows = bank[index]["rows"]
                    assert len(rows) == 3 and all(type(r["success"]) is bool for r in rows)
                    assert {r["evaluation_repeat"] for r in rows} == {0, 1, 2}
                    rewards.append(Fraction(sum(r["success"] for r in rows), 3))
                center = sum(rewards, Fraction()) / 5
                sigma = math.sqrt(float(sum(((r - center) ** 2 for r in rewards), Fraction()) / 5))
                advantages = [float(r - center) / (sigma + 1e-8) for r in rewards]
                assert abs(sum(advantages)) < 1e-12
                assert max(abs(a) for a in advantages) <= 2 + 1e-12
                if sigma == 0:
                    assert all(a == 0 for a in advantages)
                groups.append({"model": model, "mode": mode, "task_id": task,
                               "initial_tgc": float(initial_tgc), "candidate_mean_tgc": float(center),
                               "candidate_population_std_tgc": sigma, "tgc_zero_signal_group": sigma == 0})
                for index in range(5):
                    rows = {r["evaluation_repeat"]: r for r in bank[index]["rows"]}
                    record = {"model": model, "mode": mode, "task_id": task, "source_repeat": index,
                              "candidate_mean_tgc": float(rewards[index]), "initial_tgc": float(initial_tgc),
                              "tgc_gain_pp": 100 * float(rewards[index] - initial_tgc),
                              "tgc_grpo_advantage": advantages[index], "tgc_zero_signal_group": sigma == 0,
                              **{f"success_run{j}": rows[j]["success"] for j in range(3)}}
                    indexed[model, mode, task, index] = record
                    candidates.append(record)
    assert len(candidates) == 2000 and len(groups) == 400
    selected = read_csv(PCT_SOURCE / "selection_weighted_values.csv")
    distributions = defaultdict(lambda: {s: [] for s in ("best", "worst", "random")})
    selection_rows, task_weights = [], defaultdict(float)
    for old in selected:
        model, mode, task = old["model"], old["mode"], old["task_id"]
        index, strategy, weight = int(old["source_repeat"]), old["strategy"], float(old["weight"])
        row = indexed[model, mode, task, index]
        selection_rows.append({**row, "strategy": strategy, "selection_weight": weight,
                               "selection_criterion": "PCT; exact original candidate identity and weight retained"})
        task_weights[model, mode, task, strategy] += weight
        distributions["gain", mode, model][strategy].append((row["tgc_gain_pp"], weight))
        distributions["advantage", mode, model][strategy].append((row["tgc_grpo_advantage"], weight))
    assert len(task_weights) == 1200 and all(np.isclose(w, 1) for w in task_weights.values())
    cached, summaries, arrays = {}, [], {"tgc_gain_grid_pp": BASE.OVERLAP.GRID,
                                       "tgc_advantage_grid": BASE.ADV.GRID}
    for (metric, mode, model), points in distributions.items():
        grid = BASE.ADV.GRID if metric == "advantage" else BASE.OVERLAP.GRID
        densities = {s: BASE.ADV.density(p) if metric == "advantage" else BASE.REF.density(p, grid)
                     for s, p in points.items()}
        assert all(np.isclose(sum(w for _, w in p), 50) for p in points.values())
        assert all(np.isclose(np.trapz(d, grid), 1, atol=1e-12) for d in densities.values())
        means = {s: float(np.average([v for v, _ in p], weights=[w for _, w in p])) for s, p in points.items()}
        if metric == "advantage":
            assert abs(means["random"]) < 1e-12
        best, worst, random = (densities[s] for s in ("best", "worst", "random"))
        overlap = 100 * (BASE.OVERLAP.intersection(best, random, grid)
                         + BASE.OVERLAP.intersection(worst, random, grid)) / (2 * BASE.OVERLAP.area(random, grid))
        summary = {"model": model, "mode": mode, "metric": metric, "reward_metric": "official TGC",
                   "selection_metric": "PCT (unchanged)", "tasks": 50,
                   "zero_signal_tasks": sum(r["tgc_zero_signal_group"] for r in groups
                                            if r["model"] == model and r["mode"] == mode),
                   "overlap_of_random_percent": overlap, **{f"{s}_mean": means[s] for s in means}}
        summaries.append(summary)
        cached[metric, mode, model] = points, densities, summary
        for strategy, values in densities.items():
            arrays[f"tgc_{metric}_{model}_{mode}_{strategy}"] = values
    assert len(cached) == 16
    write_csv(OUT, "candidate_tgc_rewards_advantages_gains.csv", candidates)
    write_csv(OUT, "task_tgc_group_statistics.csv", groups)
    write_csv(OUT, "pct_selected_candidates_with_tgc_values.csv", selection_rows)
    write_csv(OUT, "tgc_overlap_and_means.csv", summaries)
    np.savez_compressed(OUT / "source_densities.npz", **arrays)
    return cached, summaries


def violin(cached):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "hatch.linewidth": 0.7,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(17.2, 12.4), sharey="row")
    for row, metric in enumerate(("advantage", "gain")):
        grid = BASE.ADV.GRID if metric == "advantage" else BASE.OVERLAP.GRID
        scale = 0.43 / max(d.max() for (kind, _, _), (_, ds, _) in cached.items()
                           if kind == metric for d in ds.values())
        for col, mode in enumerate(MODES):
            ax, ticks = axes[row, col], []
            for x, model in enumerate(MODELS):
                _, densities, stats = cached[metric, mode, model]
                ticks.append(BASE.LABELS[model] + f"\nn=50/50\nOverlap {stats['overlap_of_random_percent']:.1f}%")
                width = scale * densities["random"]
                ax.fill_betweenx(grid, x - width, x + width, where=width > 0, facecolor="#dfe2e6",
                                 edgecolor="#aab1ba", alpha=0.72, linewidth=0.7, zorder=1)
                for strategy, side in (("best", -1), ("worst", 1)):
                    density, color = densities[strategy], BASE.COLORS[strategy]
                    ax.fill_betweenx(grid, x, x + side * scale * density, where=density > 0,
                                     facecolor=BASE.ADV.to_rgba(color, 0.22), edgecolor=color, linewidth=0.85,
                                     hatch=BASE.HATCHES[model], zorder=3)
                    center = stats[f"{strategy}_mean"]
                    ax.plot([x, x + side * scale * float(np.interp(center, grid, density))], [center, center],
                            color=color, linewidth=2, zorder=4)
            ax.axhline(0, color="#555e69", linestyle=(0, (4, 3)), linewidth=1.1, zorder=2)
            title = f"{'ABCD'[row * 2 + col]}   {'Thinking' if col == 0 else 'Non-thinking'} · "
            title += "GRPO advantage (TGC reward)" if metric == "advantage" else "TGC gain"
            ax.set(title=title, xticks=range(4), xticklabels=ticks, xlim=(-0.65, 3.65),
                   ylim=(-2, 2) if metric == "advantage" else (-100, 100))
            ax.set_yticks(np.arange(-2, 2.01, 0.5) if metric == "advantage" else np.arange(-100, 101, 25))
            ax.tick_params(length=0, axis="both", pad=7)
            ax.grid(axis="y", color="#e6ebf1", linewidth=0.8)
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
            if col == 0:
                ax.set_ylabel("Task-local GRPO advantage (TGC)" if metric == "advantage"
                              else "TGC gain (percentage points)")
    fig.suptitle("AppWorld Train · Qwen and Gemma · TGC reward · w/o GT", fontsize=22, weight="bold", y=0.982)
    fig.text(0.5, 0.947, "PCT selection held fixed · Best left / Worst right · Random in gray · "
             "All displayed rewards, gains and advantages use official TGC", ha="center", fontsize=10.7)
    handles = [BASE.ADV.Patch(facecolor=BASE.ADV.to_rgba(c, 0.22), edgecolor=c,
                             label=s.title() + " (PCT-selected)") for s, c in BASE.COLORS.items()]
    handles += [BASE.ADV.Patch(facecolor="#dfe2e6", edgecolor="#aab1ba", label="Random"),
                BASE.ADV.Line2D([0], [0], color="#555e69", linewidth=2, label="Mean"),
                BASE.ADV.Line2D([0], [0], color="#555e69", linestyle="--", label="Zero")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.077))
    notes = [
        "TGC reward = mean of 3 official success flags. GRPO = "
        "(candidate TGC - G5 mean TGC) / (G5 population SD + 1e-8).",
        "TGC gain = candidate 3-rollout TGC - task-specific initial 5-rollout TGC, in percentage points.",
        "All 50 tasks, including TGC-tied groups · Exact original PCT candidate pools/weights · "
        "Area n/50 · KDE: 0.12 advantage / 7 pp gain.",
        "Overlap = common area of best-left + worst-right with Random / total Random area. "
        "PCT-selected Best/Worst need not have positive/negative TGC advantage.",
    ]
    for y, note in zip((0.061, 0.044, 0.027, 0.010), notes, strict=True):
        fig.text(0.5, y, note, ha="center", fontsize=8.7, color="#44566d")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.895, bottom=0.205, hspace=0.50, wspace=0.13)
    save(fig, OUT, "qwen_gemma26b_wo_gt_tgc_grpo_advantage_and_tgc_gain_4panel")


def save(fig, folder, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(folder / f"{name}.{ext}", dpi=210, facecolor="white")
    fig.savefig(folder / f"{name}_preview.png", dpi=105, facecolor="white")
    plt.close(fig)


def scatter(summaries):
    previous = read_csv(SCATTER_SOURCE / "tgc/scatter_points.csv")
    overlaps = {(s["model"], s["mode"], s["metric"]): s["overlap_of_random_percent"] for s in summaries}
    points = [{**row, "overlap_percent": overlaps[row["model"], row["mode"], row["overlap_metric"]],
               **{k: float(row[k]) for k in ("gap_mean_pp", "gap_std_pp",
                                             "best_accumulated_train_tgc_percent",
                                             "worst_accumulated_train_tgc_percent")},
               "x_reward_metric": "TGC", "reflection_selection_metric": "PCT (unchanged)"} for row in previous]
    assert len(points) == 16
    for row, old in zip(points, previous, strict=True):
        assert row["gap_mean_pp"] == float(old["gap_mean_pp"])
        assert row["gap_std_pp"] == float(old["gap_std_pp"])
    write_csv(SCATTER_OUT, "scatter_points.csv", points)
    pairs = read_csv(SCATTER_SOURCE / "tgc/paired_seed_gaps.csv")
    write_csv(SCATTER_OUT, "paired_seed_gaps.csv", pairs)
    correlations = {metric: SCATTER.BASE.ORIGINAL.correlation([
        p for p in points if p["overlap_metric"] == metric]) for metric in METRICS}
    for metrics, name in ((tuple(METRICS), "tgc_reward_overlap_vs_accumulation_tgc_gap_2panel"),
                          (("gain",), "tgc_gain_overlap_vs_accumulation_tgc_gap"),
                          (("advantage",), "tgc_grpo_overlap_vs_accumulation_tgc_gap")):
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                             "pdf.fonttype": 42, "svg.fonttype": "none"})
        fig, axes = plt.subplots(1, len(metrics), figsize=(17 if len(metrics) == 2 else 11, 8.3),
                                 sharey=True, squeeze=False)
        lower = min(p["gap_mean_pp"] - p["gap_std_pp"] for p in points)
        upper = max(p["gap_mean_pp"] + p["gap_std_pp"] for p in points)
        padding = (upper - lower) * 0.15
        for metric, ax in zip(metrics, axes[0], strict=True):
            SCATTER.BASE.axes_plot(ax, [p for p in points if p["overlap_metric"] == metric], metric,
                                   (lower - padding, upper + padding))
        axes[0, 0].set_ylabel("Best - Worst accumulated Train TGC (percentage points)", labelpad=12)
        model_handles = [SCATTER.Line2D([0], [0], marker="o", linestyle="none", color=SCATTER.BASE.COLORS[m],
                                        markersize=8, label=SCATTER.BASE.LABELS[m]) for m in MODELS]
        mode_handles = [SCATTER.Line2D([0], [0], marker=SCATTER.BASE.MARKERS[m], linestyle="none",
                                       color="#4a586b", markersize=8,
                                       label="Thinking" if m == "thinking" else "Non-thinking") for m in MODES]
        mode_handles += [SCATTER.Line2D([0], [0], color="#29384b", linestyle="--", label="OLS descriptive trend")]
        fig.legend(handles=model_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.112))
        fig.legend(handles=mode_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.070))
        fig.suptitle("TGC reflection overlap vs. accumulation TGC gap · Qwen and Gemma · w/o GT",
                     fontsize=19 if len(metrics) == 2 else 15, weight="bold", y=0.976, color="#233247")
        fig.text(0.5, 0.918, "X: TGC-based gain / GRPO overlap · "
                 "Y: official Train TGC of existing PCT-selected Best/Worst contexts",
                 ha="center", fontsize=10.2, color="#526176")
        fig.text(0.5, 0.041, "PCT selection held fixed · 50 tasks · "
                 "Y = mean paired seed gap (42, 43, 44) ±1 sample SD · No new inference calls",
                 ha="center", fontsize=9, color="#526176")
        fig.text(0.5, 0.016, "Descriptive correlations: two settings share each model; "
                 "these are not eight independent models.", ha="center", fontsize=9, color="#526176")
        fig.subplots_adjust(left=0.075 if len(metrics) == 2 else 0.11, right=0.985,
                            top=0.835, bottom=0.255, wspace=0.12)
        save(fig, SCATTER_OUT, name)
    return correlations


def main():
    original_files = {p: digest(p) for folder in (PCT_SOURCE, SCATTER_SOURCE, SCATTER_SOURCE / "tgc")
                      for p in folder.iterdir() if p.is_file()}
    cached, summaries = collect()
    violin(cached)
    correlations = scatter(summaries)
    assert all(digest(p) == sha for p, sha in original_files.items())
    provenance = {
        "generated_at": datetime.now(UTC).isoformat(), "models": list(MODELS), "modes": list(MODES),
        "arm": "without_gt", "selection_metric": "PCT; original candidate identities and tie weights unchanged",
        "reward": "Official success booleans, mean over 3 independent candidate evaluations",
        "initial_reward": "Official success booleans, mean over 5 independent initial evaluations per task",
        "gain": "100 * (candidate mean TGC - task initial mean TGC)",
        "advantage": "(candidate mean TGC - G5 mean TGC) / (G5 population SD + 1e-8)",
        "zero_signal_groups": "All retained; every advantage exactly zero when G5 TGC rewards are all tied",
        "candidate_reflections": 2000, "task_groups": 400, "tasks_per_model_mode": 50,
        "weighting": "Original exact PCT best/worst pools, uniform ties; Random uniform over all5; task mass1/50",
        "density": "Original estimators: gain bandwidth7pp, GRPO bandwidth0.12, per-kernel area normalization",
        "overlap": "(integral min(best,random) + integral min(worst,random)) / (2 * integral random)",
        "scatter_y": "Existing accumulated plain PCT-selected Best minus Worst official Train TGC",
        "scatter_y_unchanged_from_previous_tgc_gap": True, "scatter_x_recomputed_from_tgc": True,
        "correlations_descriptive": correlations, "new_inference_calls": 0,
        "source_sha256": {**BASE.REF.SOURCES, **BASE.SOURCES, **SOURCES},
        "unchanged_previous_files": {str(p): sha for p, sha in original_files.items()},
        "script_sha256": digest(Path(__file__)),
    }
    for folder in (OUT, SCATTER_OUT):
        (folder / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"violin_directory": str(OUT), "scatter_directory": str(SCATTER_OUT),
                      "correlations": correlations,
                      "overlaps": [{k: s[k] for k in ("model", "mode", "metric", "zero_signal_tasks",
                                                       "overlap_of_random_percent")} for s in summaries]}))


if __name__ == "__main__":
    main()
