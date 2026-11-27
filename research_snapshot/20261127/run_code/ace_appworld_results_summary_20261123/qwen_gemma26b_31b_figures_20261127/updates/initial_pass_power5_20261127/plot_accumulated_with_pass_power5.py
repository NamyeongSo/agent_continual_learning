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

OUT = Path('/home/thskadud/Sico/labs/AgentStream/exgentic/outputs/ace_appworld_results_summary_20261123/qwen_gemma26b_31b_figures_20261127/data/accumulated')
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
COLORS = {**BASE.COLORS, "pass_power_5": "#b69742"}
OFFSETS = (0.0, 1.10, 2.55, 3.90, 5.25, 6.60, 7.95)
STRIDE = 9.45
WIDTH = 0.76
LINE_STYLE = (0, (4, 3))


def export_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect():
    initial, by_seed, summary, task_baselines = {}, [], {}, []
    matched_tasks = None
    for model in MODELS:
        for mode in MODES:
            run = DATA.MODEL_RUNS[model][mode]
            protocol = DATA.read(run / "protocol.json")
            assert protocol["playbook_edit_mode"] == "add_only"
            assert protocol["appworld_policy_sha256"] == DATA.POLICY_SHA
            tasks = set(protocol["task_orders"]["42"]["train"])
            assert len(tasks) == 50
            if matched_tasks is None:
                matched_tasks = tasks
            assert tasks == matched_tasks
            trials = DATA.read(Path(protocol["source_reflection_run"]) / "stage1_results.json")
            DATA.validate(trials)
            assert len(trials) == len({(r["task_id"], r["source_repeat"]) for r in trials}) == 250
            groups = defaultdict(list)
            for trial in trials:
                groups[trial["task_id"]].append(trial)
            assert set(groups) == tasks
            for task, group in sorted(groups.items()):
                assert {r["source_repeat"] for r in group} == set(range(5))
                values = [float(DATA.exact_reward(r)) for r in group]
                task_baselines.append({
                    "model": model, "mode": mode, "task_id": task,
                    "initial_mean_pct": mean(values), "initial_max_pct": max(values),
                    "initial_mean_tgc": mean(int(r["success"]) for r in group),
                    "initial_pass_at_5": int(any(r["success"] for r in group)),
                })
            baseline = DATA.metrics(trials)
            initial[model, mode] = {
                "model": model, "mode": mode, **baseline,
                "pass_at_5": mean(any(r["success"] for r in group) for group in groups.values()),
                "mean_task_max_pct_at_5": mean(
                    max(float(DATA.exact_reward(r)) for r in group) for group in groups.values()
                ),
                "tasks": 50, "rollouts": 250,
            }
            assert initial[model, mode]["mean_task_max_pct_at_5"] >= baseline["pct"]
            for arm in ARMS:
                if model in GEMMAS and arm == "with_gt":
                    continue
                for strategy in STRATEGIES:
                    cohort, advantage = BASE.accumulation_source(model, mode, arm, strategy)
                    cp = DATA.read(cohort / "protocol.json")
                    for field in ("model", "model_settings", "appworld_policy_sha256", "initial_playbook",
                                  "playbook_edit_mode", "actor_limits", "turn_limit_scoring"):
                        assert cp[field] == protocol[field], (cohort, field)
                    scores = []
                    for seed in SEEDS:
                        order = cp["task_orders"][str(seed)]["train"]
                        assert len(order) == len(set(order)) and set(order) <= tasks
                        cid = f"s{seed}_{arm}_{strategy}"
                        selected = DATA.read(cohort / "conditions" / cid / "selections.json")
                        assert [r["candidate"]["task_id"] for r in selected] == order
                        rows = BASE.rows_for_context(cohort, advantage, mode, seed, cid)
                        assert {r["task_id"] for r in rows} == tasks
                        assert {r["arm"] for r in rows} == {arm}
                        assert {r["phase"] for r in rows} == {"train_final"}
                        checkpoint = DATA.read(cohort / "conditions" / cid / "checkpoints" / f"{len(order):03d}.json")
                        assert checkpoint["session_count"] == len(order)
                        assert checkpoint["condition"]["id"] == cid
                        context_sha = hashlib.sha256(checkpoint["playbook"].encode()).hexdigest()
                        assert {r["initial_context_sha256"] for r in rows} == {context_sha}
                        result = DATA.metrics(rows)
                        record = {
                            "model": model, "mode": mode, "arm": arm, "strategy": strategy, "seed": seed,
                            "learned_tasks": len(order), "train_tasks": 50,
                            "accumulated_pct": result["pct"], "accumulated_tgc": result["tgc"],
                        }
                        by_seed.append(record)
                        scores.append(record)
                    record = {
                        "model": model, "mode": mode, "arm": arm, "strategy": strategy,
                        "learned_tasks": scores[0]["learned_tasks"], "train_tasks": 50, "seeds": "42,43,44",
                    }
                    assert len({r["learned_tasks"] for r in scores}) == 1
                    for metric in ("pct", "tgc"):
                        values = [r[f"accumulated_{metric}"] for r in scores]
                        record[f"accumulated_{metric}_mean"] = mean(values)
                        record[f"accumulated_{metric}_std"] = stdev(values)
                    summary[model, mode, arm, strategy] = record
    assert len(initial) == 10 and len(by_seed) == 240 and len(summary) == 80
    assert len(task_baselines) == 500
    return initial, by_seed, summary, task_baselines


def verify_cached(initial, by_seed, summary):
    with Path('/home/thskadud/Sico/labs/AgentStream/exgentic/outputs/ace_appworld_results_summary_20261123/qwen_gemma_accumulated_20261125/metrics_by_seed.csv').open(newline="") as stream:
        old = {(r["model"], r["mode"], r["arm"], r["strategy"], int(r["seed"])): r
               for r in csv.DictReader(stream)}
    indexed = {(r["model"], r["mode"], r["arm"], r["strategy"], r["seed"]): r for r in by_seed}
    for key, row in old.items():
        for metric in ("pct", "tgc"):
            assert abs(indexed[key][f"accumulated_{metric}"] - float(row[f"accumulated_{metric}"])) < 1e-12
    for model, cohort in GEMMAS.items():
        for row in DATA.read(cohort / "train_final_by_seed.json"):
            actual = indexed[model, row["mode"], "without_gt", row["strategy"], row["seed"]]
            for metric in ("pct", "tgc"):
                assert abs(actual[f"accumulated_{metric}"] - row[metric]) < 1e-12
        for row in DATA.read(cohort / "train_final_mean_std.json"):
            actual = summary[model, row["mode"], "without_gt", row["strategy"]]
            for metric in ("pct", "tgc"):
                for stat in ("mean", "std"):
                    assert abs(actual[f"accumulated_{metric}_{stat}"] - row[f"{metric}_{stat}"] ) < 1e-12
    with Path('/home/thskadud/Sico/labs/AgentStream/exgentic/outputs/ace_appworld_results_summary_20261123/qwen_gemma_accumulated_20261125/initial_metrics.csv').open(newline="") as stream:
        for row in csv.DictReader(stream):
            for metric in ("pct", "tgc", "pass_at_5", "mean_task_max_pct_at_5"):
                assert abs(initial[row["model"], row["mode"]][metric] - float(row[metric])) < 1e-12


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
    fig, axes = plt.subplots(2, 2, figsize=(46, 16), sharey=True)
    plotted, lines = [], []
    line_metric = "pass_at_5" if metric == "tgc" else "mean_task_max_pct_at_5"
    line_label = "Initial pass@5 (official TGC)" if metric == "tgc" else "Initial max PCT over 5 runs (task mean)"
    for row, mode in enumerate(MODES):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            ticks, tick_labels = [], []
            for index, model in enumerate(MODELS):
                start = index * STRIDE
                left, right = start - 0.60, start + 8.55
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
                all_five = baseline["pass_power_5"] * 100
                pass_x = start + OFFSETS[1]
                bar(ax, pass_x, all_five, COLORS["pass_power_5"])
                plotted.append({"metric": metric, "model": model, "mode": mode, "arm": arm,
                                "category": "pass_power_5", "variant": "baseline", "x": pass_x,
                                "mean_percent": all_five, "std_percent": "", "n_training_seeds": ""})
                missing = model in GEMMAS and arm == "with_gt"
                for number, strategy in enumerate(STRATEGIES, start=2):
                    x = start + OFFSETS[number]
                    if missing:
                        continue
                    record = summary[model, mode, arm, strategy]
                    value = 100 * record[f"accumulated_{metric}_mean"]
                    std = 100 * record[f"accumulated_{metric}_std"]
                    bar(ax, x, value, COLORS[strategy], std)
                    plotted.append({"metric": metric, "model": model, "mode": mode, "arm": arm,
                                    "category": strategy, "variant": "accumulated", "x": x,
                                    "mean_percent": value, "std_percent": std, "n_training_seeds": 3})
                if missing:
                    ax.axvspan(start + 1.95, right, ymin=0, ymax=104 / 116, color="#eef1f5", alpha=0.85, zorder=1)
                    ax.text(start + 5.15, 53, "w/ GT\nnot run", ha="center", va="center",
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
                tick_labels.extend((f"Initial\n{metric.upper()}", "Pass$^5$", "Best", "Random", "Worst",
                                    "Best\nadv.", "Worst\nadv."))
                ax.text(start + 4.0, -0.18, LABELS[model], transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=13.5, fontweight="bold", color="#263244")
            ax.set_xticks(ticks, tick_labels, fontsize=9.8)
            ax.tick_params(axis="x", length=0, pad=9)
            ax.tick_params(axis="y", labelsize=11, length=0, pad=6, labelleft=True)
            ax.set_xlim(-0.65, (len(MODELS) - 1) * STRIDE + 8.60)
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
               Patch(facecolor=COLORS["pass_power_5"],
                     label="Initial pass$^5$ (all 5 official successes)"),
               Line2D([0], [0], color=COLORS["pass_at_5"], linewidth=1.8,
                      linestyle=LINE_STYLE, label=line_label),
               Patch(facecolor="#657488", label="Accumulated: final Context")]
    fig.suptitle(f"AppWorld Train · {metric.upper()} before learning and after Context accumulation",
                 y=0.978, fontsize=25, fontweight="bold", color="#1f2c3e")
    fig.text(0.5, 0.942,
             "50 matched training tasks  ·  Initial mean, initial pass$^5$, and five accumulated selection strategies per model",
             ha="center", fontsize=13, color="#536176")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=4,
               frameon=False, fontsize=12, handlelength=3.0, columnspacing=3.0)
    fig.text(0.5, 0.089,
             "Strategy bars: mean ± 1 sample SD across seed-level Train means (42, 43, 44); 50 tasks per seed. "
             "Initial: mean over 50 tasks x 5 runs. pass$^5$: fraction of tasks with all five official successes.", ha="center", fontsize=11.5, color="#536176")
    definition = ("Dashed lines: fraction of tasks with at least one official success in 5 initial runs. "
                  "TGC always uses the recorded official boolean success."
                  if metric == "tgc" else
                  "Dashed lines: take the maximum official PCT among 5 initial runs for EACH task, "
                  "then average those 50 task maxima.")
    fig.text(0.5, 0.060, definition, ha="center", fontsize=11.5, color="#536176")
    fig.text(0.5, 0.032,
             'Gemma31 Thinking: 52 prior-format-error cases replaced by regex-fallback reruns; other outcomes retained. Selected-case diagnostic, not an unbiased full rerun. Gemma w/ GT unmeasured.',
             ha="center", fontsize=11.5, color="#536176")
    fig.subplots_adjust(left=0.043, right=0.99, top=0.845, bottom=0.21, hspace=0.57, wspace=0.065)
    name = f"qwen_gemma26b_31b_accumulated_{metric}_4panel"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=200, facecolor="white")
    fig.savefig(OUT / f"{name}_preview.png", dpi=100, facecolor="white")
    plt.close(fig)
    assert len(plotted) == 120 and len(lines) == 20
    assert sum(r["variant"] == "accumulated" for r in plotted) == 80
    assert all(0 <= r["mean_percent"] <= 100 for r in plotted + lines)
    export_csv(f"plotted_bars_{metric}.csv", plotted)
    export_csv(f"reference_lines_{metric}.csv", lines)
    return str(OUT / f"{name}.png")


def main():
    original_image = (REFERENCE.parent / "tgc" /
                      "model_strategy_initial_pass5_independent_accumulated_tgc_4panel_preview.png")
    original_sha = hashlib.sha256(original_image.read_bytes()).hexdigest()
    initial, by_seed, summary, task_baselines = collect()
    verify_cached(initial, by_seed, summary)
    export_csv("initial_metrics.csv", list(initial.values()))
    export_csv("initial_task_metrics.csv", task_baselines)
    export_csv("metrics_by_seed.csv", by_seed)
    export_csv("metrics_mean_std.csv", list(summary.values()))
    figures = [draw(initial, summary, metric) for metric in ("tgc", "pct")]
    assert hashlib.sha256(original_image.read_bytes()).hexdigest() == original_sha
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(), "model_order": list(MODELS),
        "model_labels": {model: LABELS[model] for model in MODELS},
        "strategy_order": list(STRATEGIES), "panel_rows": list(MODES), "panel_columns": list(ARMS),
        "initial_banks": 10, "initial_rollouts": 2500, "complete_seed_contexts": 240,
        "train_rollouts": 12000, "strategy_groups": 80, "bars_per_figure": 100,
        "model_local_reference_lines_per_figure": 20,
        "initial": "Mean official metric over 50 matched Train tasks, 5 initial repeats per task",
        "tgc": "Recorded official boolean success, never inferred by rounding or thresholding PCT",
        "tgc_dashed_line": "Mean over tasks of any official success among that task's 5 initial runs (pass@5)",
        "pct_dashed_line": "Mean over tasks of max(official PCT) across that task's 5 initial runs",
        "strategy": "One saved final-Context evaluation per task per seed; no independent bars",
        "std": "Sample SD of seed42/43/44 Train50 means, ddof=1",
        "selection": "Original saved PCT-based selections; no relabeling using TGC",
        "missing_conditions": "Both Gemma w/ GT cohorts were not run. Shared initial bars/lines only; no imputed strategy scores",
        "counterfactual": False, "new_model_calls": 0,
        "validation": "All 240 seed scores recalculated from official outcomes, checkpoint-context SHA validated, "
                      "all seed scores cross-checked with original Qwen and completed Gemma reports",
        "original_image_sha256": {str(original_image): original_sha},
        "source_sha256": DATA.SOURCES,
        "helper_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        "data_helper_sha256": hashlib.sha256(Path('/home/thskadud/Sico/labs/AgentStream/exgentic/outputs/ace_appworld_results_summary_20261123/build_report.py').read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "figures": figures,
    }
    (OUT / "provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"figures": figures, "seed_contexts": len(by_seed), "bars_per_figure": 100}))


if __name__ == "__main__":
    main()
