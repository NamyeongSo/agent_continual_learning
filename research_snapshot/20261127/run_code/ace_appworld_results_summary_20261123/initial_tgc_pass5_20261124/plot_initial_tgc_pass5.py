# SPDX-License-Identifier: Apache-2.0
"""Plot official initial TGC and empirical pass@5 for every model/mode/GT setting."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("ace_report_data", OUT.parent / "build_report.py")
DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATA)

MODELS = ("qwen38", "qwen36", "qwen35", "oss120", "oss20")
LABELS = ("Qwen3.8\n27B", "Qwen3.6\n35B-A3B", "Qwen3.5\n9B", "GPT-OSS\n120B", "GPT-OSS\n20B")
MODE_TITLES = ("Thinking / High", "Non-thinking / Low")
ARMS = {"without_gt": "w/o GT", "with_gt": "w/ GT"}
COLORS = {"tgc_percent": "#3979AD", "pass_at_5_percent": "#9D70B7"}
SOURCES = {}


def read_json(path):
    raw = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def write_csv(filename, rows):
    with (OUT / filename).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect():
    metrics, task_metrics = {}, []
    task_sets = []
    for model in MODELS:
        for mode, run in DATA.MODEL_RUNS[model].items():
            protocol = read_json(run / "protocol.json")
            assert protocol["appworld_policy_sha256"] == DATA.POLICY_SHA
            tasks = set(protocol["task_orders"]["42"]["train"])
            assert len(tasks) == 50
            task_sets.append(tasks)
            bank = Path(protocol["source_reflection_run"])
            rows = read_json(bank / "stage1_results.json")
            assert len(rows) == len({r["id"] for r in rows}) == 250
            DATA.validate(rows)
            context_sha = hashlib.sha256(protocol["initial_playbook"].encode()).hexdigest()
            assert {r["initial_context_sha256"] for r in rows} == {context_sha}
            groups = defaultdict(list)
            for row in rows:
                assert row["stage"] == 1 and type(row["success"]) is bool
                groups[row["task_id"]].append(row)
            assert set(groups) == tasks
            for task, group in sorted(groups.items()):
                assert len(group) == 5 and {r["source_repeat"] for r in group} == set(range(5))
                task_metrics.append({
                    "model": model, "mode": mode, "task_id": task,
                    "successful_rollouts": sum(r["success"] for r in group),
                    "rollouts": 5, "any_success": any(r["success"] for r in group),
                })
            total_success = sum(r["success"] for r in rows)
            any_success = sum(any(r["success"] for r in group) for group in groups.values())
            metrics[model, mode] = {
                "model": model, "mode": mode, "tasks": 50, "rollouts": 250,
                "successful_rollouts": total_success, "tasks_with_any_success": any_success,
                "tgc_percent": 100 * mean(r["success"] for r in rows),
                "pass_at_5_percent": 100 * any_success / 50,
                "gt_baseline_shared": True,
            }
    assert len(metrics) == 10 and len(task_metrics) == 500
    assert all(tasks == task_sets[0] for tasks in task_sets)

    cached_path = OUT.parent / "model_strategy_bars_20261124/initial_metrics.csv"
    raw = cached_path.read_bytes()
    SOURCES[str(cached_path)] = hashlib.sha256(raw).hexdigest()
    with cached_path.open(newline="") as stream:
        cached = list(csv.DictReader(stream))
    assert len(cached) == 10
    for row in cached:
        actual = metrics[row["model"], row["mode"]]
        assert abs(actual["tgc_percent"] - 100 * float(row["tgc"])) < 1e-10
        assert abs(actual["pass_at_5_percent"] - 100 * float(row["pass_at_5"])) < 1e-10
    write_csv("initial_metrics.csv", list(metrics.values()))
    write_csv("initial_task_success_counts.csv", task_metrics)
    return metrics


def draw(metrics):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "axes.titleweight": "semibold",
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(18.5, 10.5), sharey=True)
    plotted = []
    x = np.arange(len(MODELS))
    width = 0.34
    for row in range(2):
        for col, (arm, arm_title) in enumerate(ARMS.items()):
            ax = axes[row, col]
            for metric_index, (metric, color) in enumerate(COLORS.items()):
                values = []
                for model in MODELS:
                    mode = tuple(DATA.MODEL_RUNS[model])[row]
                    record = metrics[model, mode]
                    values.append(record[metric])
                    plotted.append({"arm": arm, "metric": metric, **record})
                positions = x + (metric_index - 0.5) * width
                bars = ax.bar(positions, values, width=width, color=color, zorder=3)
                ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=4, fontsize=11)
            ax.set_title(f"{MODE_TITLES[row]} · {arm_title}", fontsize=16, pad=13)
            ax.set_xticks(x, LABELS, fontsize=12)
            ax.set_ylim(0, 104)
            ax.set_yticks(np.arange(0, 101, 20))
            ax.tick_params(axis="x", length=0, pad=9)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="y", color="#E1E7EE", linewidth=0.8, zorder=0)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#CDD5DF")
            if col == 0:
                ax.set_ylabel("Score (%)", labelpad=10)

    fig.suptitle("AppWorld Train · Initial TGC and pass@5", fontsize=25, fontweight="bold", y=0.98)
    fig.text(0.5, 0.936, "Before reflection or accumulation · 50 tasks with 5 independent rollouts per model/mode",
             ha="center", fontsize=13, color="#4D5B6B")
    fig.legend(handles=[Patch(facecolor=COLORS["tgc_percent"], label="TGC"),
                        Patch(facecolor=COLORS["pass_at_5_percent"], label="pass@5")],
               loc="upper center", bbox_to_anchor=(0.5, 0.919), ncol=2, frameon=False, fontsize=13)
    fig.text(0.5, 0.067,
             "GT is supplied only during reflection; w/o GT and w/ GT share the same initial rollouts and scores.",
             ha="center", fontsize=11.5, color="#4D5B6B")
    fig.text(0.5, 0.035,
             "TGC = official successful rollouts / 250.  pass@5 = tasks with ≥1 official success in 5 rollouts / 50.",
             ha="center", fontsize=11.5, color="#4D5B6B")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.845, bottom=0.17, wspace=0.12, hspace=0.37)
    stem = OUT / "initial_tgc_pass_at_5_by_model_mode_gt_4panel"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix("." + suffix), dpi=220, facecolor="white")
    fig.savefig(OUT / (stem.name + "_preview.png"), dpi=100, facecolor="white")
    plt.close(fig)
    assert len(plotted) == 40
    write_csv("plotted_bars.csv", plotted)


def main():
    metrics = collect()
    draw(metrics)
    provenance = {
        "generated_at": datetime.now(UTC).isoformat(),
        "models": list(MODELS), "unique_model_modes": 10, "displayed_gt_settings": 20,
        "tasks_per_model_mode": 50, "rollouts_per_model_mode": 250,
        "total_unique_initial_rollouts": 2500, "bars": 40,
        "tgc_definition": "Mean of the 250 recorded official boolean success outcomes",
        "pass_at_5_definition": "Fraction of 50 tasks with any official success among the 5 initial rollouts",
        "gt": "GT is used during reflection only; both arms reuse exactly the same initial rollout bank",
        "uncertainty": "Aggregate observed baseline scores; no across-training-seed SD is implied",
        "validation": "Recomputed from all 2500 initial rows; matched every cached baseline TGC/pass@5",
        "appworld_policy_sha256": DATA.POLICY_SHA,
        "new_model_calls": 0, "source_sha256": SOURCES,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "unique_initial_rollouts": 2500, "bars": 40}))


if __name__ == "__main__":
    main()
