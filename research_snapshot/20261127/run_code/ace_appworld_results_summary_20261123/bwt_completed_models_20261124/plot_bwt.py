# SPDX-License-Identifier: Apache-2.0
"""Plot measured BWT for completed ACE ADD-only cohorts, excluding GPT-OSS-120B."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from collections import defaultdict
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from statistics import mean, stdev

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

OUT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("bwt_reference", OUT.parent / "build_report.py")
REF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REF)
MODELS = ("qwen38", "qwen36", "qwen35", "oss20")
STRATEGIES = ("best", "random", "worst", "best_advantage", "worst_advantage")
SEEDS = (42, 43, 44)
ARMS = ("without_gt", "with_gt")
MODE_LABELS = ("Thinking / High", "Non-thinking / Low")
COLORS = {"best": "#24864c", "random": "#87929e", "worst": "#ce4949",
          "best_advantage": "#65ad76", "worst_advantage": "#e68b83"}
NAMES = {"best": "Best", "random": "Random", "worst": "Worst",
         "best_advantage": "Best advantage", "worst_advantage": "Worst advantage"}
TICKS = ("Best", "Random", "Worst", "Best\nadv.", "Worst\nadv.")
MARKERS = dict(zip(STRATEGIES, ("o", "s", "^", "D", "v"), strict=True))
SOURCES = {}
ROW_CACHE = {}


def read(path, *, cache=False):
    path = Path(path)
    if cache and str(path) in ROW_CACHE:
        return ROW_CACHE[str(path)]
    raw = path.read_bytes()
    SOURCES[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
    value = json.loads(raw)
    if cache:
        ROW_CACHE[str(path)] = value
    return value


def export_csv(name, records):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def cohort(model, mode, arm, strategy):
    if strategy.endswith("_advantage"):
        advantage = REF.ADV_RUNS[model]
        return advantage / mode / arm, advantage
    return REF.MODEL_RUNS[model][mode], None


def evaluations(directory, advantage, mode, seed, cid, phase):
    path = (directory / f"seed{seed}/{phase}_results.json" if advantage is None else
            advantage / "evaluations" / mode / cid / f"{phase}_results.json")
    if not path.exists():
        raise ValueError(f"Incomplete evaluation phase: {path}")
    rows = [r for r in read(path, cache=True) if r["condition_id"] == cid and r["seed"] == seed]
    assert rows and len({r["task_id"] for r in rows}) == len(rows), path
    REF.validate(rows)
    for row in rows:
        assert row["phase"] == phase and row["evaluation_repeat"] == 0
    return {r["task_id"]: r for r in rows}


def validate_parent(directory, cid, row, step, task=None):
    assert row["checkpoint_step"] == step
    checkpoint = read(directory / "conditions" / cid / "checkpoints" / f"{step:03d}.json")
    assert checkpoint["condition"]["id"] == cid and checkpoint["session_count"] == step
    assert checkpoint["context_sha256"] == row["initial_context_sha256"]
    assert hashlib.sha256(checkpoint["playbook"].encode()).hexdigest() == row["initial_context_sha256"]
    if task is not None:
        assert checkpoint["task_id"] == task
    return checkpoint


def collect():
    pairs, seed_rows = [], []
    task_set = None
    for model in MODELS:
        for mode, source in REF.MODEL_RUNS[model].items():
            original_protocol = read(source / "protocol.json", cache=True)
            for arm in ARMS:
                for strategy in STRATEGIES:
                    directory, advantage = cohort(model, mode, arm, strategy)
                    protocol = read(directory / "protocol.json", cache=True)
                    assert protocol["playbook_edit_mode"] == "add_only"
                    for key in ("model", "model_settings", "appworld_policy_sha256", "initial_playbook"):
                        assert protocol[key] == original_protocol[key], (directory, key)
                    for seed in SEEDS:
                        cid = f"s{seed}_{arm}_{strategy}"
                        original_order = original_protocol["task_orders"][str(seed)]["train"]
                        learned = protocol["task_orders"][str(seed)]["train"]
                        if task_set is None:
                            task_set = set(original_order)
                        assert len(original_order) == 50 and set(original_order) == task_set
                        assert learned == [t for t in original_order if t in set(learned)]
                        assert set(learned).isdisjoint(protocol["task_orders"][str(seed)]["val"])
                        assert set(learned).isdisjoint(protocol["task_orders"][str(seed)]["test"])
                        choices = read(directory / "conditions" / cid / "selections.json")
                        assert [c["candidate"]["task_id"] for c in choices] == learned
                        final = evaluations(directory, advantage, mode, seed, cid, "train_final")
                        after = evaluations(directory, advantage, mode, seed, cid, "bwt_checkpoint")
                        assert set(final) == set(original_order) and set(after) == set(learned)
                        assert len(final) == 50 and len(after) == len(learned)
                        first_final = final[learned[0]]
                        validate_parent(directory, cid, first_final, len(learned))
                        assert {r["initial_context_sha256"] for r in final.values()} == {
                            first_final["initial_context_sha256"]
                        }
                        tag = {"model": model, "mode": mode, "arm": arm, "strategy": strategy, "seed": seed}
                        batch = []
                        for step, task in enumerate(learned, 1):
                            immediate, end = after[task], final[task]
                            validate_parent(directory, cid, immediate, step, task)
                            assert immediate["appworld_policy_sha256"] == end["appworld_policy_sha256"]
                            row = {**tag, "condition_id": cid, "task_id": task, "step": step,
                                   "original_stream_step": original_order.index(task) + 1,
                                   "learned_tasks": len(learned),
                                   "post_learning_pct": float(REF.exact_reward(immediate)),
                                   "final_pct": float(REF.exact_reward(end)),
                                   "bwt_pct_pp": 100 * float(REF.exact_reward(end) - REF.exact_reward(immediate)),
                                   "post_learning_tgc": int(immediate["success"]), "final_tgc": int(end["success"]),
                                   "bwt_tgc_pp": 100 * (int(end["success"]) - int(immediate["success"]))}
                            pairs.append(row)
                            batch.append(row)
                        seed_rows.append({**tag, "learned_tasks": len(learned),
                                          "bwt_pct_pp": mean(r["bwt_pct_pp"] for r in batch),
                                          "bwt_tgc_pp": mean(r["bwt_tgc_pp"] for r in batch)})
    assert len(seed_rows) == 240
    groups = defaultdict(list)
    for row in seed_rows:
        groups[row["model"], row["mode"], row["arm"], row["strategy"]].append(row)
    summary = {}
    for key, values in groups.items():
        assert len(values) == 3 and {r["seed"] for r in values} == set(SEEDS)
        assert len({r["learned_tasks"] for r in values}) == 1
        summary[key] = {"model": key[0], "mode": key[1], "arm": key[2], "strategy": key[3],
                        "seeds": "42,43,44", "learned_tasks": values[0]["learned_tasks"],
                        **{f"bwt_{metric}_{stat}_pp": func(r[f"bwt_{metric}_pp"] for r in values)
                           for metric in ("pct", "tgc") for stat, func in (("mean", mean), ("std", stdev))}}
    assert len(summary) == 80 and all(r["model"] != "oss120" for r in pairs)
    return pairs, seed_rows, summary


def save_figure(fig, name):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=200, facecolor="white")
    fig.savefig(OUT / f"{name}_preview.png", dpi=100, facecolor="white")
    plt.close(fig)


def style_axis(ax, limit, metric):
    ax.set_ylim(-limit, limit)
    ax.axhline(0, color="#344256", linewidth=1.2, zorder=2)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(length=0)
    ax.set_ylabel(f"BWT in {metric.upper()} (percentage points)", fontsize=11)


def plot_bars(summary, metric):
    fig, axes = plt.subplots(2, 2, figsize=(23, 12), sharey=True)
    bound = max(abs(r[f"bwt_{metric}_mean_pp"]) + r[f"bwt_{metric}_std_pp"] for r in summary.values())
    limit = max(10, 5 * ceil((bound + 4) / 5))
    for row in range(2):
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            ticks, labels = [], []
            for index, model in enumerate(MODELS):
                mode = tuple(REF.MODEL_RUNS[model])[row]
                start = 6.25 * index
                if index % 2 == 0:
                    ax.axvspan(start - 0.6, start + 4.6, color="#f4f6f9", zorder=0)
                for offset, strategy in enumerate(STRATEGIES):
                    item = summary[model, mode, arm, strategy]
                    value, std = (item[f"bwt_{metric}_{name}_pp"] for name in ("mean", "std"))
                    x = start + offset
                    ax.bar(x, value, width=0.70, color=COLORS[strategy], zorder=3)
                    ax.errorbar(x, value, yerr=std, fmt="none", ecolor="#29384c", capsize=2.8,
                                elinewidth=1.0, capthick=1.0, zorder=4)
                    direction = 1 if value >= 0 else -1
                    ax.text(x, value + direction * (std + limit * 0.04), f"{value:+.1f}",
                            ha="center", va="bottom" if direction > 0 else "top", fontsize=8.8, color="#334155")
                    ticks.append(x)
                    labels.append(TICKS[offset])
                n = summary[model, mode, arm, "best_advantage"]["learned_tasks"]
                assert summary[model, mode, arm, "worst_advantage"]["learned_tasks"] == n
                ax.text(start + 2, -0.20, f"{REF.LABELS[model]}\nN(full / adv.) = 50 / {n}",
                        transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11.5,
                        fontweight="semibold", color="#26364b", linespacing=1.6)
            style_axis(ax, limit, metric)
            ax.yaxis.set_major_locator(MultipleLocator(5 if limit <= 30 else 10))
            ax.set_xticks(ticks, labels, fontsize=8.5)
            ax.tick_params(axis="x", pad=8)
            ax.tick_params(axis="y", labelleft=True)
            ax.set_xlim(-0.70, 6.25 * (len(MODELS) - 1) + 4.70)
            gt = "w/o GT" if arm == "without_gt" else "w/ GT"
            ax.set_title(f"{'ABCD'[row * 2 + col]}   {MODE_LABELS[row]} · {gt}",
                         loc="left", fontsize=16, fontweight="semibold", pad=14)
    fig.suptitle(f"AppWorld Train · Backward transfer in {metric.upper()}", fontsize=23,
                 fontweight="bold", y=0.98, color="#233247")
    fig.text(0.5, 0.938, "BWT = final-Context score - score immediately after learning the same task",
             ha="center", fontsize=12.5, color="#526176")
    fig.text(0.5, 0.069, "Bars: mean ± 1 sample SD of the three seed-level BWT values (42, 43, 44). "
             "Positive = improvement; negative = forgetting.", ha="center", fontsize=11.5, color="#526176")
    fig.text(0.5, 0.039, "Average over actually learned tasks, including the final task. "
             "Advantage strategies exclude skipped zero-signal tasks. "
             "Each task/checkpoint has one independent evaluation; differences include rollout variability.",
             ha="center", fontsize=10.5, color="#526176")
    fig.subplots_adjust(left=0.055, right=0.99, top=0.875, bottom=0.22, wspace=0.16, hspace=0.70)
    save_figure(fig, f"bwt_{metric}_by_model_strategy_4panel")


def order_data(pairs):
    groups = defaultdict(list)
    for row in pairs:
        groups[row["model"], row["mode"], row["arm"], row["strategy"], row["seed"]].append(row)
    binned, point_rows = [], []
    for model in MODELS:
        for mode in REF.MODEL_RUNS[model]:
            for arm in ARMS:
                for strategy in STRATEGIES:
                    runs = [sorted(groups[model, mode, arm, strategy, seed], key=lambda r: r["step"]) for seed in SEEDS]
                    n = len(runs[0])
                    assert all([r["step"] for r in run] == list(range(1, n + 1)) for run in runs)
                    tag = {"model": model, "mode": mode, "arm": arm, "strategy": strategy}
                    for index in range(n):
                        point_rows.append({**tag, "step": index + 1,
                                           **{f"bwt_{m}_mean_pp": mean(run[index][f"bwt_{m}_pp"] for run in runs)
                                              for m in ("pct", "tgc")}})
                    for start in range(0, n, 5):
                        stop = min(n, start + 5)
                        record = {**tag, "first_step": start + 1, "last_step": stop,
                                  "center_step": (start + 1 + stop) / 2, "tasks_per_seed": stop - start}
                        for metric in ("pct", "tgc"):
                            values = [mean(r[f"bwt_{metric}_pp"] for r in run[start:stop]) for run in runs]
                            record[f"bwt_{metric}_mean_pp"] = mean(values)
                            record[f"bwt_{metric}_std_pp"] = stdev(values)
                        binned.append(record)
    return binned, point_rows


def plot_order(model, metric, binned, points):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), sharex=True, sharey=True)
    values = [abs(r[f"bwt_{metric}_mean_pp"]) for r in points if r["model"] == model]
    values += [abs(r[f"bwt_{metric}_mean_pp"]) + r[f"bwt_{metric}_std_pp"]
               for r in binned if r["model"] == model]
    limit = max(20, 10 * ceil((max(values) + 3) / 10))
    for row in range(2):
        mode = tuple(REF.MODEL_RUNS[model])[row]
        for col, arm in enumerate(ARMS):
            ax = axes[row, col]
            for strategy in STRATEGIES:
                target = model, mode, arm, strategy
                raw = sorted([r for r in points if (r["model"], r["mode"], r["arm"], r["strategy"]) == target],
                             key=lambda r: r["step"])
                chunks = sorted([r for r in binned if (r["model"], r["mode"], r["arm"], r["strategy"]) == target],
                                key=lambda r: r["center_step"])
                color = COLORS[strategy]
                ax.scatter([r["step"] for r in raw], [r[f"bwt_{metric}_mean_pp"] for r in raw],
                           color=color, s=11, alpha=0.20, marker=MARKERS[strategy], zorder=2)
                x = np.array([r["center_step"] for r in chunks])
                y = np.array([r[f"bwt_{metric}_mean_pp"] for r in chunks])
                sd = np.array([r[f"bwt_{metric}_std_pp"] for r in chunks])
                ax.plot(x, y, color=color, linewidth=1.9, marker=MARKERS[strategy], markersize=4.5,
                        linestyle="--" if strategy.endswith("_advantage") else "-", zorder=4)
                ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.085, linewidth=0, zorder=1)
            style_axis(ax, limit, metric)
            ax.set_xlim(0, 51)
            ax.set_xticks([1, 10, 20, 30, 40, 50])
            ax.tick_params(labelbottom=True, labelleft=True)
            ax.set_xlabel("Task learning order t (Curator steps)", fontsize=10.5)
            gt = "w/o GT" if arm == "without_gt" else "w/ GT"
            ax.set_title(f"{MODE_LABELS[row]} · {gt}", loc="left", fontsize=13.5, pad=12)
    handles = [Line2D([0], [0], color=COLORS[s], marker=MARKERS[s], linewidth=2,
                      linestyle="--" if s.endswith("_advantage") else "-", label=NAMES[s]) for s in STRATEGIES]
    fig.suptitle(f"{REF.LABELS[model]} · Task-wise {metric.upper()} BWT by learning order",
                 fontsize=19, fontweight="bold", y=0.985, color="#233247")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.944), ncol=5,
               frameon=False, fontsize=10.5)
    fig.text(0.5, 0.075, "Faint dots: 3-seed mean at each task position. "
             "Lines: non-overlapping 5-task-bin means; bands: ±1 SD across the three seed-level bin means.",
             ha="center", fontsize=10, color="#526176")
    fig.text(0.5, 0.046, "Tasks at the same position differ under the three seed shuffles. "
             "Filtered strategies end at their learned-task count; skipped tasks are not assigned zero.",
             ha="center", fontsize=10, color="#526176")
    fig.text(0.5, 0.018, "Each point compares final vs post-learning scores for that task. "
             "The x-axis is acquisition order, not a measured time course of benchmark-average BWT.",
             ha="center", fontsize=9.5, color="#526176")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.87, bottom=0.17, wspace=0.20, hspace=0.40)
    save_figure(fig, f"{model}_bwt_{metric}_by_learning_order_4panel")


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    pairs, seeds, summary = collect()
    bins, points = order_data(pairs)
    export_csv("bwt_task_pairs.csv", pairs)
    export_csv("bwt_by_seed.csv", seeds)
    export_csv("bwt_mean_std.csv", list(summary.values()))
    export_csv("bwt_order_bins.csv", bins)
    export_csv("bwt_order_points.csv", points)
    for metric in ("pct", "tgc"):
        plot_bars(summary, metric)
        for model in MODELS:
            plot_order(model, metric, bins, points)
    manifest = {"generated_at": datetime.now(UTC).isoformat(), "models": list(MODELS),
                "excluded_model": "oss120", "seeds": list(SEEDS), "condition_seed_records": len(seeds),
                "condition_groups": len(summary), "paired_task_evaluations": len(pairs),
                "metrics": ["official PCT", "official boolean TGC"], "new_inference_calls": 0,
                "bwt_definition": "Task-macro final score minus score at the task's own post-learning Context. "
                                  "Average over actually learned tasks, including the last learned task.",
                "bar_uncertainty": "Sample SD across the three seed-level BWT values, ddof=1",
                "order_axis": "Curator learning-step index in each strategy, not original unfiltered stream index",
                "order_trend": "Non-overlapping 5-task bins computed separately per seed, then seed mean and SD",
                "data_kind": "Recorded original evaluations; no completion counterfactual applied",
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_sha256": SOURCES}
    (OUT / "provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "source_sha256"}, indent=2))
    print("OUTPUT", OUT)


if __name__ == "__main__":
    main()
