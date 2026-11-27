# SPDX-License-Identifier: Apache-2.0
"""Combine w/o-GT Qwen/Gemma single-reflection GRPO advantage and PCT gain violins."""

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

OUT = Path('/home/thskadud/Sico/labs/AgentStream/exgentic/outputs/ace_appworld_results_summary_20261123/qwen_gemma26b_31b_figures_20261127/data/selected_reflection')
SOURCE = Path('/home/thskadud/Sico/labs/AgentStream/exgentic/outputs/ace_appworld_results_summary_20261123/reflection_gain_with_oss120b_20261123')
ROOT = Path('/home/thskadud/Sico/labs/AgentStream/exgentic')
GEMMAS = {"gemma26": Path('/home/thskadud/Sico/labs/AgentStream/exgentic/outputs/ace_appworld_gemma4_26b_a4b_it_wo_gt_train_only_20261124'), "gemma31": Path('/home/thskadud/Sico/labs/AgentStream/exgentic/outputs/ace_appworld_gemma4_31b_it_wo_gt_train_only_20261124')}
MODELS = ("qwen38", "qwen36", "qwen35", "gemma26", "gemma31")
MODES = ("thinking", "nonthinking")
LABELS = {"qwen38": "Qwen3.8\n27B", "qwen36": "Qwen3.6\n35B-A3B", "qwen35": "Qwen3.5\n9B",
          "gemma26": "Gemma4\n26B-A4B", "gemma31": "Gemma4\n31B"}
HATCHES = {"qwen38": "", "qwen36": "///", "qwen35": "...", "gemma26": "++", "gemma31": "xx"}
COLORS = {"best": "#17765c", "worst": "#b55643"}
NAME = "qwen_gemma26b_31b_wo_gt_grpo_advantage_and_pct_gain_4panel"
SOURCES = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    SOURCES[str(path)] = digest(path)
    return json.loads(path.read_bytes())


def module(name, path):
    SOURCES[str(path)] = digest(path)
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


OVERLAP = module("saved_gain_overlap", SOURCE / "overlap_with_random_20261124/plot_overlap.py")
ADV = module("saved_grpo_overlap", SOURCE / "grpo_advantage_with_random_20261124/plot_grpo_advantage.py")
REF = OVERLAP.REPORT
plt = ADV.plt


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_gemma(model, mode):
    folder = GEMMAS[model] / mode / "reflection"
    protocol = read(folder / "protocol.json")
    assert protocol["model"] == {"gemma26": "openai/google/gemma-4-26B-A4B-it", "gemma31": "openai/google/gemma-4-31B-it"}[model]
    assert protocol["arms"] == ["without_gt"]
    assert protocol["history_policy_version"] == "gemma-tool-reasoning-history-v1"
    assert protocol["turn_limit_scoring"] == "official_pct_at_limit"
    initial_rows, candidate_rows = (read(folder / f"stage{stage}_results.json") for stage in (1, 3))
    assert len(initial_rows) == 250 and len(candidate_rows) == 750
    REF.validate(initial_rows)
    REF.validate(candidate_rows)
    tasks = set(protocol["task_ids"])
    assert len(tasks) == 50
    initial, candidates = defaultdict(list), defaultdict(dict)
    initial_sha = hashlib.sha256(protocol["initial_playbook"].encode()).hexdigest()
    for row in initial_rows:
        assert row["initial_context_sha256"] == initial_sha
        initial[row["task_id"]].append(row)
    for row in candidate_rows:
        assert row["arm"] == "without_gt" and row["task_id"] in tasks
        key = row["arm"], row["task_id"], row["source_repeat"]
        assert row["evaluation_repeat"] not in candidates[key]
        candidates[key][row["evaluation_repeat"]] = row
    assert set(initial) == tasks
    assert all(len(group) == 5 and {r["source_repeat"] for r in group} == set(range(5))
               for group in initial.values())
    banks = defaultdict(dict)
    for (arm, task, repeat), evaluations in candidates.items():
        assert set(evaluations) == {0, 1, 2}
        rows = list(evaluations.values())
        banks[arm, task][repeat] = {**REF.metrics(rows), "rows": rows,
                                  "exact_pct": sum((REF.exact_reward(r) for r in rows), Fraction()) / 3}
    assert len(banks) == 50 and all(set(group) == set(range(5)) for group in banks.values())
    REF.INITIAL[model, mode], REF.BANKS[model, mode] = initial, banks


def collect():
    old = {}
    for metric, directory in (("gain", "overlap_with_random_20261124"),
                              ("advantage", "grpo_advantage_with_random_20261124")):
        path = SOURCE / directory / "source_densities.npz"
        SOURCES[str(path)] = digest(path)
        with np.load(path) as data:
            old[metric] = {key: data[key].copy() for key in data.files}
    cached, candidates_out, groups_out, selected_out, summaries = {}, [], [], [], []
    old_checks = 0
    for model in MODELS:
        for mode in MODES:
            if model in GEMMAS:
                load_gemma(model, mode)
                reference = read(GEMMAS[model] / "advantage" / f"{mode}_advantage_groups.json")["without_gt"]
            else:
                REF.load_reflection(model, mode, REF.MODEL_RUNS[model][mode])
                reference = None
            tasks = sorted(REF.INITIAL[model, mode])
            assert len(tasks) == 50 and set(tasks) == set(REF.INITIAL["qwen38", "thinking"])
            gain_points = REF.gain_points(model, mode, "without_gt")
            advantage_points = {s: [] for s in ("best", "worst", "random")}
            zero_tasks = 0
            for task in tasks:
                group = REF.BANKS[model, mode]["without_gt", task]
                rewards = [group[i]["exact_pct"] for i in range(5)]
                center = sum(rewards, Fraction()) / 5
                sigma = math.sqrt(float(sum(((r - center) ** 2 for r in rewards), Fraction()) / 5))
                advantages = [float(r - center) / (sigma + 1e-8) for r in rewards]
                zero_tasks += sigma == 0
                assert abs(sum(advantages)) < 1e-12 and max(abs(a) for a in advantages) <= 2 + 1e-12
                if reference is not None:
                    assert reference[task]["zero_signal_group"] == (sigma == 0)
                    assert np.allclose(reference[task]["advantages"], advantages, rtol=0, atol=1e-12)
                groups_out.append({"model": model, "mode": mode, "task_id": task,
                                   "mean_candidate_pct": float(center), "population_std_pct": sigma,
                                   "zero_signal": sigma == 0})
                for i in range(5):
                    rows = {r["evaluation_repeat"]: r for r in group[i]["rows"]}
                    candidates_out.append({"model": model, "mode": mode, "task_id": task, "source_repeat": i,
                                           "mean_pct": float(rewards[i]), "advantage": advantages[i],
                                           **{f"pct_run{j}": float(REF.exact_reward(rows[j])) for j in range(3)}})
                pools = {"best": [i for i, r in enumerate(rewards) if r == max(rewards)],
                         "worst": [i for i, r in enumerate(rewards) if r == min(rewards)], "random": list(range(5))}
                baseline = REF.metrics(REF.INITIAL[model, mode][task])["pct"]
                for strategy, pool in pools.items():
                    for i in pool:
                        advantage_points[strategy].append((advantages[i], 1 / len(pool)))
                        selected_out.append({"model": model, "mode": mode, "task_id": task,
                                             "strategy": strategy, "source_repeat": i, "weight": 1 / len(pool),
                                             "advantage": advantages[i],
                                             "pct_gain_pp": (group[i]["pct"] - baseline) * 100})
            if model == "gemma26":
                assert zero_tasks == (11 if mode == "thinking" else 4)
            for metric, points, grid in (("advantage", advantage_points, ADV.GRID),
                                         ("gain", gain_points, OVERLAP.GRID)):
                densities = {s: ADV.density(p) if metric == "advantage" else REF.density(p, grid)
                             for s, p in points.items()}
                for strategy, values in points.items():
                    assert np.isclose(sum(w for _, w in values), 50)
                    assert np.isclose(np.trapz(densities[strategy], grid), 1)
                    if model.startswith("qwen"):
                        key = f"{model}_{mode}_without_gt_{strategy}"
                        assert np.allclose(densities[strategy], old[metric][key], rtol=0, atol=1e-12), key
                        old_checks += 1
                best, worst, random = (densities[s] for s in ("best", "worst", "random"))
                overlap = 100 * (OVERLAP.intersection(best, random, grid)
                                 + OVERLAP.intersection(worst, random, grid)) / (2 * OVERLAP.area(random, grid))
                means = {s: float(np.average([v for v, _ in p], weights=[w for _, w in p]))
                         for s, p in points.items()}
                if metric == "advantage":
                    assert abs(means["random"]) < 1e-12
                summary = {"model": model, "mode": mode, "metric": metric, "tasks": 50,
                           "zero_signal_tasks": zero_tasks, "overlap_of_random_percent": overlap,
                           **{f"{s}_mean": means[s] for s in means}}
                summaries.append(summary)
                cached[metric, mode, model] = points, densities, summary
    assert old_checks == 36 and len(candidates_out) == 2500
    write_csv("candidate_scores_and_advantages.csv", candidates_out)
    write_csv("task_group_statistics.csv", groups_out)
    write_csv("selection_weighted_values.csv", selected_out)
    write_csv("overlap_and_means.csv", summaries)
    arrays = {"advantage_grid": ADV.GRID, "pct_gain_grid_pp": OVERLAP.GRID}
    for (metric, mode, model), (_, densities, _) in cached.items():
        for strategy, values in densities.items():
            arrays[f"{metric}_{model}_{mode}_{strategy}"] = values
    np.savez_compressed(OUT / "source_densities.npz", **arrays)
    return cached, old_checks


def draw(cached):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "hatch.linewidth": 0.7, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(21.5, 12.4), sharey="row")
    for row, metric in enumerate(("advantage", "gain")):
        grid = ADV.GRID if metric == "advantage" else OVERLAP.GRID
        maximum = max(d.max() for (kind, _, _), (_, ds, _) in cached.items() if kind == metric for d in ds.values())
        scale = 0.43 / maximum
        for col, mode in enumerate(MODES):
            ax = axes[row, col]
            ticks = []
            for x, model in enumerate(MODELS):
                _, densities, stats = cached[metric, mode, model]
                ticks.append(LABELS[model] + f"\nn=50/50\nOverlap {stats['overlap_of_random_percent']:.1f}%")
                width = scale * densities["random"]
                ax.fill_betweenx(grid, x - width, x + width, where=width > 0, facecolor="#dfe2e6",
                                 edgecolor="#aab1ba", alpha=0.72, linewidth=0.7, zorder=1)
                for strategy, side in (("best", -1), ("worst", 1)):
                    d, color = densities[strategy], COLORS[strategy]
                    ax.fill_betweenx(grid, x, x + side * scale * d, where=d > 0,
                                     facecolor=ADV.to_rgba(color, 0.22), edgecolor=color, linewidth=0.85,
                                     hatch=HATCHES[model], zorder=3)
                    center = stats[f"{strategy}_mean"]
                    width = float(np.interp(center, grid, d)) * scale
                    ax.plot([x, x + side * width], [center, center], color=color, linewidth=2, zorder=4)
            ax.axhline(0, color="#555e69", linestyle=(0, (4, 3)), linewidth=1.1, zorder=2)
            title = f"{'ABCD'[row * 2 + col]}   {'Thinking' if col == 0 else 'Non-thinking'} · "
            title += "GRPO advantage" if metric == "advantage" else "PCT gain"
            ax.set(title=title, xticks=range(len(MODELS)), xticklabels=ticks, xlim=(-0.65, len(MODELS) - 0.35),
                   ylim=(-2, 2) if metric == "advantage" else (-100, 100))
            ax.set_yticks(np.arange(-2, 2.01, 0.5) if metric == "advantage" else np.arange(-100, 101, 25))
            ax.tick_params(length=0, axis="both", pad=7)
            ax.grid(axis="y", color="#e6ebf1", linewidth=0.8)
            ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
            if col == 0:
                ax.set_ylabel("Task-local GRPO advantage" if metric == "advantage" else "PCT gain (percentage points)")
    fig.suptitle("AppWorld Train · Qwen and Gemma · w/o GT", fontsize=23, weight="bold", y=0.982)
    fig.text(0.5, 0.947, "Single-reflection evaluation · Best on the left · Worst on the right · Random in gray",
             ha="center", fontsize=12)
    handles = [ADV.Patch(facecolor=ADV.to_rgba(c, 0.22), edgecolor=c, label=s.title()) for s, c in COLORS.items()]
    handles += [ADV.Patch(facecolor="#dfe2e6", edgecolor="#aab1ba", label="Random"),
                ADV.Line2D([0], [0], color="#555e69", linewidth=2, label="Mean"),
                ADV.Line2D([0], [0], color="#555e69", linestyle="--", label="Zero")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.077))
    notes = [
        "Advantage = (candidate 3-rollout mean PCT - task-group mean) / "
        "(task-group population SD + 1e-8). All-tied groups stay at zero.",
        "Gain = candidate 3-rollout mean PCT - task-specific initial 5-rollout mean PCT, in percentage points.",
        "All 50 tasks · Uniform best/worst ties; Random over all 5 · Area proportional to n/50 within each row · "
        "KDE: 0.12 advantage / 7 pp gain.",
        "Overlap = common area of the best-left + worst-right shape and Random / total Random area. "
        "No new inference calls.",
    ]
    for y, note in zip((0.061, 0.044, 0.027, 0.010), notes, strict=True):
        fig.text(0.5, y, note, ha="center", fontsize=9.1, color="#44566d")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.895, bottom=0.205, hspace=0.50, wspace=0.13)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{NAME}.{ext}", dpi=210, facecolor="white")
    fig.savefig(OUT / f"{NAME}_preview.png", dpi=105, facecolor="white")
    plt.close(fig)


def main():
    originals = {p: digest(p) for name in ("overlap_with_random_20261124", "grpo_advantage_with_random_20261124")
                 for p in (SOURCE / name).iterdir() if p.is_file()}
    cached, verified = collect()
    draw(cached)
    assert all(digest(p) == sha for p, sha in originals.items())
    manifest = {"generated_at": datetime.now(UTC).isoformat(), "models": list(MODELS), "arm": "without_gt",
                "columns": list(MODES), "rows": ["GRPO advantage", "PCT gain in percentage points"],
                "groups": 20, "tasks_per_model_mode": 50, "candidate_reflections": 2500,
                "candidates_per_task": 5, "rollouts_per_candidate": 3, "initial_rollouts_per_task": 5,
                "task_weighting": "One unit mass per task, uniformly split across exact best/worst ties or all5 Random",
                "zero_signal": "All tasks included, including tied groups. No advantage-filtered task subset.",
                "selection_note": "Best/worst based on the same 3-rollout bank used for scoring; "
                                  "selection bias applies",
                "density_scaling": "One density-to-width scale per metric row across both mode panels; n/50",
                "old_qwen_density_arrays_matched": verified, "gemma_training_advantages_verified": 200,
                "original_files_unchanged": {str(p): sha for p, sha in originals.items()},
                "source_sha256": {**REF.SOURCES, **SOURCES}, "new_inference_calls": 0,
                "script_sha256": digest(Path(__file__))}
    (OUT / "provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    gemma = [stats for (_, _, model), (_, _, stats) in cached.items() if model == "gemma26"]
    print(json.dumps({"output": str(OUT / f"{NAME}.png"), "old_density_checks": verified, "gemma": gemma}))


if __name__ == "__main__":
    main()
