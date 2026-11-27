# SPDX-License-Identifier: Apache-2.0
"""Compare task-local GRPO advantages of all generated Reflection candidates."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from statistics import mean, median, pstdev

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ace_reflection_advantage_violin_mpl")

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

OUT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("reflection_violin_reference", OUT.parent / "build_report.py")
REF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REF)
MODELS = ("qwen38", "qwen36", "qwen35", "oss120", "oss20")
ARMS = ("without_gt", "with_gt")
MODE_LABELS = ("Thinking / High", "Non-thinking / Low")
EPSILON = 1e-8
BANDWIDTH = 0.12
GRID = np.linspace(-2.0, 2.0, 1001)
COLOR = "#487f9b"


def export_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect():
    candidates, task_groups, summaries = [], [], []
    panels = {}
    matched_tasks = None
    verified = 0
    for model in MODELS:
        for mode, directory in REF.MODEL_RUNS[model].items():
            protocol, bank = REF.load_reflection(model, mode, directory)
            tasks = protocol["task_orders"]["42"]["train"]
            if matched_tasks is None:
                matched_tasks = set(tasks)
            assert len(tasks) == 50 and set(tasks) == matched_tasks
            assert len(REF.BANKS[model, mode]) == 100
            advantage_run = (REF.ADV_RUNS[model] if model != "oss120" else
                             REF.BASE / "ace_appworld_gpt_oss120b_add_only_20261121/advantage")
            previous_path = advantage_run / f"{mode}_advantage_groups.json"
            previous = REF.read(previous_path) if previous_path.exists() else None
            for arm in ARMS:
                values = []
                zero_groups = 0
                for task in tasks:
                    rewards = [sum((REF.exact_reward(r) for r in bank[arm, task, index].values()), Fraction()) / 3
                               for index in range(5)]
                    group_mean = sum(rewards, Fraction()) / 5
                    variance = sum(((r - group_mean) ** 2 for r in rewards), Fraction()) / 5
                    sigma = math.sqrt(float(variance))
                    advantages = [float(r - group_mean) / (sigma + EPSILON) for r in rewards]
                    assert abs(sum(advantages)) < 1e-12 and max(abs(v) for v in advantages) <= 2 + 1e-12
                    assert abs(pstdev(advantages) - sigma / (sigma + EPSILON)) < 1e-12
                    zero_groups += sigma == 0
                    if previous:
                        reference = previous[arm][task]
                        assert reference["zero_signal_group"] == (sigma == 0)
                        assert np.allclose(advantages, reference["advantages"], atol=1e-12, rtol=0)
                        verified += 1
                    tag = {"model": model, "mode": mode, "arm": arm, "task_id": task}
                    task_groups.append({**tag, "mean_candidate_pct": float(group_mean),
                                        "candidate_population_std_pct": sigma, "zero_signal_group": sigma == 0})
                    for index, (reward, advantage) in enumerate(zip(rewards, advantages, strict=True)):
                        candidate = bank[arm, task, index]
                        assert set(candidate) == {0, 1, 2}
                        candidates.append({**tag, "source_repeat": index, "mean_pct": float(reward),
                                           **{f"pct_run{repeat}": float(REF.exact_reward(candidate[repeat]))
                                              for repeat in range(3)},
                                           "advantage": advantage, "zero_signal_group": sigma == 0})
                        values.append(advantage)
                assert len(values) == 250 and abs(mean(values)) < 1e-12
                panels[model, mode, arm] = np.array(values)
                summaries.append({"model": model, "mode": mode, "arm": arm, "tasks": 50,
                                  "reflections": 250, "mean_advantage": mean(values),
                                  "median_advantage": median(values), "population_std_advantage": pstdev(values),
                                  "min_advantage": min(values), "max_advantage": max(values),
                                  "negative_count": sum(v < 0 for v in values),
                                  "exact_zero_count": sum(v == 0 for v in values),
                                  "positive_count": sum(v > 0 for v in values),
                                  "nonnegative_count": sum(v >= 0 for v in values),
                                  "nonnegative_fraction": sum(v >= 0 for v in values) / len(values),
                                  "zero_signal_task_groups": zero_groups})
    assert len(candidates) == 5000 and len(task_groups) == 1000 and len(panels) == 20
    return panels, candidates, task_groups, summaries, verified


def density(values):
    """Fixed-bandwidth Gaussian KDE, reflected at the mathematical G5 support bounds."""
    kernel = np.zeros_like(GRID)
    for locations in (values, -4.0 - values, 4.0 - values):
        z = (GRID[:, None] - locations[None, :]) / BANDWIDTH
        kernel += np.exp(-0.5 * z * z).sum(axis=1)
    result = kernel / (len(values) * BANDWIDTH * math.sqrt(2 * math.pi))
    area = np.sum((result[1:] + result[:-1]) * np.diff(GRID) / 2)
    result /= area
    assert abs(np.sum((result[1:] + result[:-1]) * np.diff(GRID) / 2) - 1) < 1e-12
    return result


def draw(panels, summaries):
    densities = {key: density(values) for key, values in panels.items()}
    # One common scale preserves equal violin area for the same 250-candidate denominator.
    scale = 0.43 / max(d.max() for d in densities.values())
    fig, axes = plt.subplots(2, 2, figsize=(18, 11.5), sharey=True)
    positions = np.arange(5) * 1.35
    details = {(r["model"], r["mode"], r["arm"]): r for r in summaries}
    for row in range(2):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            ax.axhline(0, color="#a77d40", linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
            for x, model in zip(positions, MODELS, strict=True):
                mode = tuple(REF.MODEL_RUNS[model])[row]
                key = model, mode, arm
                width = densities[key] * scale
                ax.fill_betweenx(GRID, x - width, x + width, facecolor=COLOR, edgecolor="#2d536b",
                                 alpha=0.84, linewidth=0.9, zorder=3)
                zero_percent = details[key]["exact_zero_count"] / 250 * 100
                nonnegative_percent = details[key]["nonnegative_fraction"] * 100
                ax.text(x, 2.35, f"A ≥ 0: {nonnegative_percent:.1f}%", ha="center", va="center",
                        fontsize=9.6, fontweight="semibold", color="#285e4a")
                ax.text(x, 2.13, f"A = 0: {zero_percent:.1f}%", ha="center", va="center",
                        fontsize=9.1, color="#795b2e")
            ax.set_xlim(positions[0] - 0.65, positions[-1] + 0.65)
            ax.set_ylim(-2.15, 2.55)
            ax.set_yticks(np.arange(-2, 2.01, 0.5))
            ax.set_xticks(positions, [REF.SHORT[model] for model in MODELS], fontsize=11)
            ax.tick_params(length=0, labelleft=True, pad=7)
            ax.set_ylabel("GRPO advantage", fontsize=12, labelpad=8)
            ax.grid(axis="y", color="#e3e9ef", linewidth=0.75, zorder=0)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#c9d3dd")
            gt = "w/o GT" if arm == "without_gt" else "w/ GT"
            ax.set_title(f"{'ABCD'[row * 2 + col]}   {MODE_LABELS[row]} · {gt}",
                         fontsize=15, fontweight="semibold", loc="left", pad=14)
    fig.suptitle("AppWorld · Advantage distribution of generated Reflections",
                 fontsize=23, fontweight="bold", y=0.98, color="#233348")
    fig.text(0.5, 0.937, "All 250 Reflections per model / panel · 50 matched tasks, 5 candidates each · "
             "Each candidate reward is its mean PCT across 3 saved rollouts",
             ha="center", fontsize=11.5, color="#526176")
    fig.text(0.5, 0.080, "A = (candidate reward - task-group mean) / (task-group population SD + 1e-8). "
             "All-tied groups contribute five exact zeros; "
             "no candidate is filtered or duplicated across training seeds.",
             ha="center", fontsize=10.2, color="#526176")
    fig.text(0.5, 0.052, "All violins use one density-to-width scale and the same 250-candidate denominator. "
             "Gaussian KDE bandwidth = 0.12 advantage units; boundary correction on the G5 support [-2, 2].",
             ha="center", fontsize=10.2, color="#526176")
    fig.text(0.5, 0.024, "The dashed line is advantage zero. "
             "Labels show exact A ≥ 0 and A = 0 fractions before smoothing. "
             "Advantages are centered within each task, so the shapes describe relative candidate scores.",
             ha="center", fontsize=10.2, color="#526176")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.855, bottom=0.16, hspace=0.40, wspace=0.19)
    name = "reflection_advantage_all_candidates_violin_4panel"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=200, facecolor="white")
    fig.savefig(OUT / f"{name}_preview.png", dpi=110, facecolor="white")
    plt.close(fig)
    export_csv("violin_density_grid.csv", [
        {"model": key[0], "mode": key[1], "arm": key[2], "advantage": float(y),
         "density": float(d), "half_width": float(d * scale)}
        for key, values in densities.items() for y, d in zip(GRID, values, strict=True)
    ])
    return scale


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
    panels, candidates, groups, summaries, verified = collect()
    export_csv("reflection_advantages.csv", candidates)
    export_csv("task_group_statistics.csv", groups)
    export_csv("distribution_summary.csv", summaries)
    scale = draw(panels, summaries)
    provenance = {"generated_at": datetime.now(UTC).isoformat(), "model_order": list(MODELS),
                  "panel_rows": list(MODE_LABELS), "panel_columns": list(ARMS),
                  "reflections": len(candidates), "task_groups": len(groups), "violins": len(panels),
                  "reflections_per_violin": 250, "candidate_rollouts_per_reflection": 3,
                  "reference_training_groups_verified": verified, "epsilon": EPSILON,
                  "advantage": "(mean PCT over 3 evaluations - mean over 5 candidate rewards) / "
                               "(population SD of the five rewards + epsilon)",
                  "all_tied_groups": "Retained, all five advantages exactly zero",
                  "annotations": {"nonnegative_fraction": "count(A >= 0) / 250, including exact zeros",
                                  "exact_zero_fraction": "count(A == 0) / 250"},
                  "kde": {"absolute_bandwidth": BANDWIDTH, "support": [-2, 2],
                          "boundary_method": "reflection", "normalization": "unit integral for every violin",
                          "common_half_width_scale": scale},
                  "new_inference_calls": 0, "source_sha256": REF.SOURCES,
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in provenance.items() if k != "source_sha256"}, indent=2))
    print("OUTPUT", OUT)


if __name__ == "__main__":
    main()
