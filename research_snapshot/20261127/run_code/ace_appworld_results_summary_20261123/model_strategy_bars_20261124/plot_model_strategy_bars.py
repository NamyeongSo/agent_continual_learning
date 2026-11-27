# SPDX-License-Identifier: Apache-2.0
"""Four panels of ordered model blocks: initial PCT, pass@5, and paired strategy PCT."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from statistics import mean, stdev

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("ace_report_data", OUT.parent / "build_report.py")
DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATA)

MODELS = ("qwen38", "qwen36", "qwen35", "oss120", "oss20")
STRATEGIES = ("best", "random", "worst", "best_advantage", "worst_advantage")
SEEDS = (42, 43, 44)
ARMS = ("without_gt", "with_gt")
MODE_TITLES = ("Thinking / High", "Non-thinking / Low")
COLORS = {"initial": "#497fa9", "pass_at_5": "#a18ac4", "best": "#24864c", "random": "#87929e",
          "worst": "#ce4949", "best_advantage": "#65ad76", "worst_advantage": "#e68b83"}
LABELS = {"qwen38": "Qwen3.8-27B", "qwen36": "Qwen3.6-35B-A3B", "qwen35": "Qwen3.5-9B",
          "oss120": "GPT-OSS-120B", "oss20": "GPT-OSS-20B"}
TICK_LABELS = ("Initial\nPCT", "Pass\n@5", "Best", "Random", "Worst", "Best\nadv.", "Worst\nadv.")
TICK_OFFSETS = (0.0, 0.95, 2.40, 4.00, 5.60, 7.20, 8.80)
WIDTH = 0.61
STRIDE = 10.55


def export_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def accumulation_source(model, mode, arm, strategy):
    if not strategy.endswith("_advantage"):
        return DATA.MODEL_RUNS[model][mode], None
    run = DATA.ADV_RUNS.get(model)
    if model == "oss120":
        run = DATA.BASE / "ace_appworld_gpt_oss120b_add_only_20261121/advantage"
    return run / mode / arm, run


def rows_for_context(run, advantage, mode, seed, cid):
    if advantage is None:
        path = run / f"seed{seed}/train_final_results.json"
    else:
        path = advantage / "evaluations" / mode / cid / "train_final_results.json"
    if not path.exists():
        raise ValueError(f"Complete per-context Train evaluation missing: {path}")
    rows = DATA.read(path)
    rows = [r for r in rows if r["condition_id"] == cid and r["seed"] == seed]
    assert len(rows) == len({r["task_id"] for r in rows}) == 50, path
    DATA.validate(rows)
    return rows


def collect():
    initial, by_seed, summary = {}, [], {}
    for model in MODELS:
        for mode, run in DATA.MODEL_RUNS[model].items():
            protocol, candidates = DATA.load_reflection(model, mode, run)
            trials = DATA.INITIAL[model, mode]
            tasks = set(protocol["task_orders"]["42"]["train"])
            assert len(tasks) == len(trials) == 50
            all_initial = [r for group in trials.values() for r in group]
            assert len(all_initial) == 250
            initial[model, mode] = {
                "model": model, "mode": mode, "pct": DATA.metrics(all_initial)["pct"],
                "tgc": DATA.metrics(all_initial)["tgc"],
                "pass_at_5": mean(any(r["success"] for r in group) for group in trials.values()),
                "tasks": 50, "rollouts": 250,
            }
            for arm in ARMS:
                for strategy in STRATEGIES:
                    cohort, advantage = accumulation_source(model, mode, arm, strategy)
                    cp = DATA.read(cohort / "protocol.json")
                    for field in ("model", "model_settings", "appworld_policy_sha256", "initial_playbook",
                                  "playbook_edit_mode"):
                        assert cp[field] == protocol[field], (cohort, field)
                    scores = []
                    for seed in SEEDS:
                        cid = f"s{seed}_{arm}_{strategy}"
                        selected = DATA.read(cohort / "conditions" / cid / "selections.json")
                        order = cp["task_orders"][str(seed)]["train"]
                        assert [r["candidate"]["task_id"] for r in selected] == order
                        assert set(order) <= tasks
                        for item in selected:
                            candidate = item["candidate"]
                            rows = list(candidates[arm, candidate["task_id"], candidate["source_repeat"]].values())
                            assert sum(DATA.exact_reward(r) for r in rows) / 3 == Fraction(
                                candidate["score_numerator"], candidate["score_denominator"]
                            )
                        expectation = DATA.independent(model, mode, seed, arm, strategy, selected, candidates)
                        rows = rows_for_context(cohort, advantage, mode, seed, cid)
                        assert {r["task_id"] for r in rows} == tasks
                        final = cohort / "conditions" / cid / "checkpoints" / f"{len(order):03d}.json"
                        checkpoint = DATA.read(final)
                        assert checkpoint["session_count"] == len(order)
                        assert checkpoint["condition"]["id"] == cid
                        context_hash = hashlib.sha256(checkpoint["playbook"].encode()).hexdigest()
                        assert {r["initial_context_sha256"] for r in rows} == {context_hash}, final
                        accumulated = DATA.metrics(rows)
                        record = {**expectation, "accumulated_pct": accumulated["pct"],
                                  "accumulated_tgc": accumulated["tgc"], "train_tasks": 50}
                        by_seed.append(record)
                        scores.append(record)
                    record = {"model": model, "mode": mode, "arm": arm, "strategy": strategy,
                              "learned_tasks": scores[0]["learned_tasks"], "train_tasks": 50,
                              "seeds": "42,43,44"}
                    assert len({r["learned_tasks"] for r in scores}) == 1
                    for name in ("independent_pct", "accumulated_pct", "independent_tgc", "accumulated_tgc"):
                        values = [r[name] for r in scores]
                        record[name + "_mean"] = mean(values)
                        record[name + "_std"] = stdev(values)
                    summary[model, mode, arm, strategy] = record
    assert len(initial) == 10 and len(summary) == 100 and len(by_seed) == 300
    assert len({tuple(sorted(t)) for t in DATA.INITIAL.values()}) == 1
    return initial, by_seed, summary


def tinted(color, fraction=0.25):
    return tuple(1 - fraction + fraction * value for value in to_rgb(color))


def bar(ax, x, value, color, *, std=None, independent=False):
    ax.bar(x, value, width=WIDTH, color=tinted(color) if independent else color,
           edgecolor=color if independent else "white", linewidth=0.85,
           hatch="///" if independent else None, zorder=3)
    if std is not None and std > 1e-10:
        ax.errorbar(x, value, yerr=std, fmt="none", ecolor="#263244", elinewidth=0.9,
                    capsize=2.0, capthick=0.8, zorder=5)
    ax.text(x, value + (std or 0) + 1.25, f"{value:.1f}", ha="center", va="bottom",
            rotation=90, fontsize=7.8, color="#354052", zorder=6)


def draw(initial, summary):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titleweight": "semibold", "hatch.linewidth": 0.7,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(43, 17.5), sharey=True)
    plotted = []
    for row in range(2):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            ticks, labels = [], []
            for index, model in enumerate(MODELS):
                mode = tuple(DATA.MODEL_RUNS[model])[row]
                start = index * STRIDE
                left, right = start - 0.65, start + 9.50
                if index % 2 == 0:
                    ax.axvspan(left, right, color="#f3f5f8", alpha=0.8, zorder=0)
                if index:
                    ax.axvline(start - 0.83, color="#d3dbe5", linewidth=0.85, linestyle=(0, (3, 3)), zorder=1)
                baseline = initial[model, mode]
                for category, offset in (("initial", TICK_OFFSETS[0]), ("pass_at_5", TICK_OFFSETS[1])):
                    metric = "pct" if category == "initial" else "pass_at_5"
                    value = baseline[metric] * 100
                    bar(ax, start + offset, value, COLORS[category])
                    plotted.append({"row": row, "arm": arm, "model": model, "mode": mode,
                                    "category": category, "variant": "baseline", "x": start + offset,
                                    "mean_percent": value, "std_percent": "", "n": 5 if metric == "pct" else 1})
                for number, strategy in enumerate(STRATEGIES):
                    record = summary[model, mode, arm, strategy]
                    for variant, delta in (("independent", -WIDTH / 2), ("accumulated", WIDTH / 2)):
                        x = start + TICK_OFFSETS[number + 2] + delta
                        value, std = (100 * record[f"{variant}_pct_{name}"] for name in ("mean", "std"))
                        bar(ax, x, value, COLORS[strategy], std=std, independent=variant == "independent")
                        plotted.append({"row": row, "arm": arm, "model": model, "mode": mode,
                                        "category": strategy, "variant": variant, "x": x,
                                        "mean_percent": value, "std_percent": std, "n": 3})
                ticks.extend(start + np.array(TICK_OFFSETS))
                labels.extend(TICK_LABELS)
                ax.text(start + 4.40, -0.18, LABELS[model], transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=14, fontweight="bold", color="#263244")
            ax.set_xticks(ticks, labels, fontsize=8.7)
            ax.tick_params(axis="x", length=0, pad=8)
            ax.tick_params(axis="y", labelsize=11, length=0, pad=6, labelleft=True)
            ax.set_xlim(-0.70, (len(MODELS) - 1) * STRIDE + 9.55)
            ax.set_ylim(0, 114)
            ax.set_yticks(np.arange(0, 101, 20))
            ax.set_ylabel("Score (%)", fontsize=13, labelpad=10)
            ax.grid(axis="y", color="#dfe5ec", linewidth=0.8, zorder=0)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#c5ceda")
            label = "w/o GT" if arm == "without_gt" else "w/ GT"
            letter = "ABCD"[row * 2 + col]
            ax.set_title(f"{letter}   {MODE_TITLES[row]}  ·  {label}", loc="left", fontsize=19, pad=19)
    handles = [Patch(facecolor=COLORS["initial"], label="Initial PCT"),
               Patch(facecolor=COLORS["pass_at_5"], label="Initial pass@5 (official TGC)"),
               Patch(facecolor=tinted("#657488"), edgecolor="#657488", hatch="///",
                     label="Independent: selected Reflection"),
               Patch(facecolor="#657488", label="Accumulated: final Context")]
    fig.suptitle("AppWorld Train · Initial performance and Reflection selection strategies",
                 y=0.982, fontsize=26, fontweight="bold", color="#1f2c3e")
    fig.text(0.5, 0.951, "50 matched training tasks  ·  Each model: initial PCT, pass@5, "
             "then five independent / accumulated pairs",
             ha="center", fontsize=14, color="#536176")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.933), ncol=4,
               frameon=False, fontsize=12, handlelength=2.6, columnspacing=3.0)
    fig.text(0.5, 0.078,
             "Strategy bars: mean ± 1 sample SD across seeds 42, 43, 44; every seed uses all 50 Train tasks. "
             "Initial PCT: 5-rollout mean. Initial pass@5: fraction of tasks with ≥1 official success in 5 rollouts.",
             ha="center", fontsize=12, color="#536176")
    fig.text(0.5, 0.055,
             "Independent: the actual seed-selected Reflection, measured on its saved 3-rollout bank. "
             "Accumulated: one final-Context rollout per task per seed. "
             "Advantage strategies: skipped zero-signal tasks retain their initial PCT in the independent expectation.",
             ha="center", fontsize=11.5, color="#536176")
    fig.text(0.5, 0.033, "Initial and pass@5 are shared by both GT panels. "
             "No error bar is assigned to these two baseline bars. "
             "Independent scores use the same candidate bank as selection.  adv. = advantage.",
             ha="center", fontsize=11.5, color="#536176")
    fig.subplots_adjust(left=0.035, right=0.99, top=0.855, bottom=0.20, hspace=0.55, wspace=0.055)
    name = "model_strategy_initial_pass5_independent_accumulated_pct_4panel"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=200, facecolor="white")
    fig.savefig(OUT / f"{name}_preview.png", dpi=90, facecolor="white")
    plt.close(fig)
    assert len(plotted) == 240
    for row in range(2):
        for arm in ARMS:
            subset = [r for r in plotted if r["row"] == row and r["arm"] == arm]
            assert len(subset) == 60
            assert [r["model"] for r in subset[::12]] == list(MODELS)
            assert all(0 <= r["mean_percent"] <= 100 + 1e-9 for r in subset)
    export_csv("plotted_bars.csv", plotted)


def main():
    OUT.mkdir(exist_ok=True)
    initial, by_seed, summary = collect()
    export_csv("initial_metrics.csv", list(initial.values()))
    export_csv("metrics_by_seed.csv", by_seed)
    export_csv("metrics_mean_std.csv", list(summary.values()))
    draw(initial, summary)
    manifest = {"generated_at": datetime.now(UTC).isoformat(), "model_order": list(MODELS),
                "strategy_order": list(STRATEGIES), "rows": list(MODE_TITLES), "columns": list(ARMS),
                "initial_banks": len(initial), "complete_seed_contexts": len(by_seed),
                "seed_contexts_per_model": 60, "strategy_groups": len(summary), "plotted_bars": 240,
                "initial_metric": "mean official PCT over 50 tasks x 5 rollouts",
                "pass_at_5": "mean over 50 tasks of any official boolean success in the five initial rollouts",
                "strategy_metric": "PCT; task macro mean over Train50, then mean across seeds 42/43/44",
                "error_bars": "Strategy bars only: sample SD of the three seed-level means (ddof=1)",
                "independent_expectation": "Actual selected Reflection per seed; three saved candidate evaluations. "
                                           "Skipped zero-advantage groups use the task's initial five-rollout mean.",
                "accumulated": "Final Context, one saved official Train evaluation per task and seed",
                "known_limit": "Selection and independent measurement share the three-repeat bank.",
                "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "report_helper_sha256": hashlib.sha256((OUT.parent / "build_report.py").read_bytes()).hexdigest(),
                "source_sha256": DATA.SOURCES}
    (OUT / "provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "source_sha256"}, indent=2))
    print("OUTPUT", OUT)


if __name__ == "__main__":
    main()
