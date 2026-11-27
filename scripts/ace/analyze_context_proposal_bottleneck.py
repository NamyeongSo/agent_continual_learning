# SPDX-License-Identifier: Apache-2.0
"""Analyze the complete matched ACE reflection bank without new model calls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from scripts.ace.run_reflection_accumulation import exact_reward

ARMS = ("without_gt", "with_gt")


def average(values):
    return sum(values, Fraction(0)) / len(values)


def direction(value):
    return "improved" if value > 0 else "degraded" if value < 0 else "unchanged"


def bootstrap_interval(values, samples=10000, seed=42):
    """Resample task-level effects; a paired contrast must be formed before this call."""
    values = np.asarray(values, dtype=float)
    indices = np.random.default_rng(seed).integers(0, len(values), (samples, len(values)))
    return np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()


def build_tables(tasks, initial, evaluated, policy_hash):
    baselines, scores = defaultdict(dict), defaultdict(dict)
    task_set = set(tasks)
    for row in initial:
        task, repeat = row["task_id"], row["source_repeat"]
        if task not in task_set or repeat not in range(5) or repeat in baselines[task]:
            raise ValueError("Invalid initial membership/repeat")
        if row["appworld_policy_sha256"] != policy_hash:
            raise ValueError("Initial policy mismatch")
        baselines[task][repeat] = row
    if set(baselines) != task_set or any(set(v) != set(range(5)) for v in baselines.values()):
        raise ValueError("Incomplete initial bank")
    expected = {(arm, task, r) for arm in ARMS for task in tasks for r in range(5)}
    for row in evaluated:
        key = row["arm"], row["task_id"], row["source_repeat"]
        if (key not in expected or row["evaluation_repeat"] not in range(3)
                or row["evaluation_repeat"] in scores[key]):
            raise ValueError("Invalid or duplicate evaluation")
        if row["appworld_policy_sha256"] != policy_hash:
            raise ValueError("Evaluation policy mismatch")
        scores[key][row["evaluation_repeat"]] = row
    if set(scores) != expected or any(set(v) != {0, 1, 2} for v in scores.values()):
        raise ValueError("Require five contexts per arm/task and three evaluations per context")
    task_rows, candidate_rows = [], []
    for task in tasks:
        initial_scores = [exact_reward(baselines[task][i]) for i in range(5)]
        baseline = average(initial_scores)
        row = {"task_id": task, "initial_mean_pct": float(baseline),
               "initial_success_count": sum(v["success"] for v in baselines[task].values()),
               "initial_sample_std": statistics.stdev(map(float, initial_scores))}
        for arm in ARMS:
            candidates = []
            for repeat in range(5):
                rs = list(scores[(arm, task, repeat)].values())
                values = [exact_reward(r) for r in rs]
                score = average(values)
                candidates.append(score)
                candidate_rows.append({
                    "task_id": task, "arm": arm, "source_repeat": repeat,
                    "initial_mean_pct": float(baseline), "source_trajectory_pct": float(initial_scores[repeat]),
                    "mean_pct": float(score), "delta_pct": float(score - baseline),
                    "direction": direction(score - baseline), "score_numerator": score.numerator,
                    "score_denominator": score.denominator,
                    "evaluation_sample_std": statistics.stdev(map(float, values)),
                    "official_success_count": sum(r["success"] for r in rs),
                    "turn_cap_count": sum(bool(r.get("turn_limit_reached")) for r in rs),
                    "other_budget_failures": sum(bool(r.get("budget_failure")) for r in rs),
                })
            mean = average(candidates)
            row.update({f"{arm}_mean_pct": float(mean), f"{arm}_delta_pct": float(mean - baseline),
                        f"{arm}_direction": direction(mean - baseline),
                        f"{arm}_candidate_sample_std": statistics.stdev(map(float, candidates)),
                        f"{arm}_best_pct": float(max(candidates)), f"{arm}_worst_pct": float(min(candidates)),
                        f"{arm}_best_tie_count": candidates.count(max(candidates)),
                        f"{arm}_worst_tie_count": candidates.count(min(candidates))})
        row["gt_difference_pct"] = row["with_gt_mean_pct"] - row["without_gt_mean_pct"]
        task_rows.append(row)
    return task_rows, candidate_rows


def summarize(task_rows, candidate_rows, initial, evaluated):
    baseline = statistics.mean(r["initial_mean_pct"] for r in task_rows)
    summary = {"tasks": len(task_rows), "contexts": len(candidate_rows), "initial_rollouts": len(initial),
               "updated_rollouts": len(evaluated), "initial_mean_pct": baseline,
               "bootstrap": {"unit": "task", "samples": 10000, "seed": 42,
                             "interpretation": "Task-resampling interval; not independent validation of selected best"},
               "arms": {}}
    for arm in ARMS:
        cs = [r for r in candidate_rows if r["arm"] == arm]
        rs = [r for r in evaluated if r["arm"] == arm]
        deltas = [r[f"{arm}_delta_pct"] for r in task_rows]
        correlation = spearmanr([r["initial_mean_pct"] for r in task_rows], deltas)
        summary["arms"][arm] = {
            "mean_pct": statistics.mean(r[f"{arm}_mean_pct"] for r in task_rows),
            "delta_pct": statistics.mean(deltas), "delta_task_bootstrap_ci95": bootstrap_interval(deltas),
            "official_success_rate": sum(r["success"] for r in rs) / len(rs),
            "candidate_directions": dict(Counter(r["direction"] for r in cs)),
            "task_mean_directions": dict(Counter(r[f"{arm}_direction"] for r in task_rows)),
            "observed_best_of5_pct": statistics.mean(r[f"{arm}_best_pct"] for r in task_rows),
            "observed_worst_of5_pct": statistics.mean(r[f"{arm}_worst_pct"] for r in task_rows),
            "tasks_with_best_ties": sum(r[f"{arm}_best_tie_count"] > 1 for r in task_rows),
            "tasks_with_worst_ties": sum(r[f"{arm}_worst_tie_count"] > 1 for r in task_rows),
            "mean_observed_candidate_std": statistics.mean(r[f"{arm}_candidate_sample_std"] for r in task_rows),
            "mean_within_candidate_evaluation_std": statistics.mean(r["evaluation_sample_std"] for r in cs),
            "turn_cap_rollouts": sum(bool(r.get("turn_limit_reached")) for r in rs),
            "other_budget_failures": dict(Counter(r["budget_failure"]["reason"] for r in rs
                                                  if r.get("budget_failure"))),
            "initial_vs_delta_spearman_descriptive": float(correlation.statistic),
        }
    gt = [r["gt_difference_pct"] for r in task_rows]
    summary["paired_gt_contrast"] = {"mean_difference_pct": statistics.mean(gt),
                                     "task_bootstrap_ci95": bootstrap_interval(gt)}
    summary["subgroups"] = {}
    groups = {"initial_ceiling": lambda r: r["initial_mean_pct"] == 1,
              "initial_below_ceiling": lambda r: r["initial_mean_pct"] < 1,
              "initial_zero_official_success": lambda r: r["initial_success_count"] == 0}
    for name, predicate in groups.items():
        rows = [r for r in task_rows if predicate(r)]
        if not rows:
            summary["subgroups"][name] = {"tasks": 0}
            continue
        group = {"tasks": len(rows), "initial_mean_pct": statistics.mean(r["initial_mean_pct"] for r in rows)}
        for arm in ARMS:
            group[arm] = {"mean_pct": statistics.mean(r[f"{arm}_mean_pct"] for r in rows),
                          "delta_pct": statistics.mean(r[f"{arm}_delta_pct"] for r in rows),
                          "degraded_tasks": sum(r[f"{arm}_direction"] == "degraded" for r in rows),
                          "degraded_candidates": sum(r["direction"] == "degraded" for r in candidate_rows
                                                     if r["arm"] == arm and r["task_id"] in
                                                     {t["task_id"] for t in rows})}
        group["gt_difference_pct"] = statistics.mean(r["gt_difference_pct"] for r in rows)
        summary["subgroups"][name] = group
    summary["initial_zero_success_task_details"] = [r for r in task_rows if r["initial_success_count"] == 0]
    summary["largest_gt_differences"] = sorted(task_rows, key=lambda r: r["gt_difference_pct"])
    return summary


def plot(output, task_rows, summary):
    colors = {"without_gt": "#3977ba", "with_gt": "#d78837"}
    labels = {"without_gt": "ACE without GT", "with_gt": "ACE with GT"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    all_deltas = [r[f"{arm}_delta_pct"] * 100 for r in task_rows for arm in ARMS]
    y_limits = (min(-5, min(all_deltas) - 2), max(55, max(all_deltas) + 2))
    lowest_initial = min(r["initial_mean_pct"] * 100 for r in task_rows)
    for ax, arm in zip(axes, ARMS, strict=True):
        x = [r["initial_mean_pct"] * 100 for r in task_rows]
        y = [r[f"{arm}_delta_pct"] * 100 for r in task_rows]
        ax.scatter(x, y, c=colors[arm], s=35, alpha=0.75, edgecolor="white")
        ax.axhline(0, color="#64748b", linestyle="--", linewidth=1)
        ax.set(xlabel="Initial mean PCT (%)", ylabel="Mean candidate PCT - initial PCT (pp)",
               title=labels[arm], xlim=(min(20, lowest_initial - 3), 103), ylim=y_limits)
        ax.grid(alpha=0.2)
    fig.suptitle("Reflection effect by initial performance | 50 tasks, 5 contexts x 3 evaluations")
    fig.savefig(output / "initial_vs_reflection_gain.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    x = np.arange(3)
    lowest_score = min(summary["initial_mean_pct"], *(summary["arms"][arm]["observed_worst_of5_pct"] for arm in ARMS))
    for offset, arm in zip([-0.18, 0.18], ARMS, strict=True):
        s = summary["arms"][arm]
        values = [s["observed_worst_of5_pct"], s["mean_pct"], s["observed_best_of5_pct"]]
        axes[0].bar(x + offset, np.array(values) * 100, width=0.34, color=colors[arm], label=labels[arm])
        for i, value in enumerate(values):
            axes[0].text(i + offset, value * 100 + 0.5, f"{value * 100:.2f}", ha="center", fontsize=8)
    axes[0].axhline(summary["initial_mean_pct"] * 100, color="#64748b", linestyle="--", label="Initial mean")
    axes[0].set(xticks=x, xticklabels=["Observed worst", "Uniform random\nexpectation", "Observed best"],
                ylabel="PCT (%)", ylim=(min(75, lowest_score * 100 - 3), 103),
                title="Selection on the same bank (best is optimistic)")
    axes[0].legend(loc="lower right", fontsize=8)
    names = ["initial_ceiling", "initial_below_ceiling", "initial_zero_official_success"]
    groups = [summary["subgroups"][name] for name in names]
    for offset, arm in zip([-0.18, 0.18], ARMS, strict=True):
        values = [group[arm]["delta_pct"] * 100 if group["tasks"] else np.nan for group in groups]
        axes[1].bar(x + offset, values, width=0.34, color=colors[arm], label=labels[arm])
    axes[1].axhline(0, color="#64748b", linewidth=1)
    labels_with_counts = [
        f"{label}\nn={group['tasks']}" for label, group in zip(
            ("Initial PCT=100%", "Initial PCT<100%", "Initial success=0/5"), groups, strict=True
        )
    ]
    axes[1].set(xticks=x, xticklabels=labels_with_counts,
                ylabel="Mean PCT change (pp)", title="Ceiling and zero-success subgroups")
    fig.savefig(output / "selection_and_subgroups.png", dpi=200)
    plt.close(fig)
    rows = sorted(task_rows, key=lambda r: r["gt_difference_pct"])
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    values = np.array([r["gt_difference_pct"] for r in rows]) * 100
    ax.bar(np.arange(len(rows)), values, color=np.where(values >= 0, "#d78837", "#3977ba"))
    ax.axhline(0, color="#64748b", linewidth=1)
    ax.set(xticks=np.arange(len(rows)), xticklabels=[r["task_id"] for r in rows],
           ylabel="With GT - without GT (pp)", title="Paired task-level GT effect | all 50 train tasks")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.savefig(output / "paired_gt_task_effects.png", dpi=200)
    plt.close(fig)


def analyze(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((run / "protocol.json").read_text())
    if len(protocol["task_ids"]) != 50 or len(set(protocol["task_ids"])) != 50:
        raise ValueError("Require the pinned 50-task training cohort")
    if json.loads((run / "status.json").read_text())["state"] != "completed":
        raise ValueError("Only analyze the complete bank")
    sources = {}

    def load(path):
        raw = path.read_bytes()
        sources[str(path.relative_to(run))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    initial = [load(p) for p in sorted((run / "stage1").glob("*/result.json"))]
    evaluated = [load(p) for p in sorted((run / "stage3").glob("*/result.json"))]
    tasks, candidates = build_tables(protocol["task_ids"], initial, evaluated, protocol["appworld_policy_sha256"])
    summary = summarize(tasks, candidates, initial, evaluated)
    summary.update({"source_run": str(run), "primary_metric": "exact official PCT, task-macro mean",
                    "selection_warning": "Best/worst are selected and measured on the same three-repeat bank",
                    "variance_warning": "Across-context score variation includes three-repeat evaluation noise",
                    "source_sha256": sources,
                    "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    for name, rows in [("task_effects", tasks), ("candidate_effects", candidates)]:
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    plot(output, tasks, summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in
                      {"source_sha256", "largest_gt_differences", "initial_zero_success_task_details"}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.run, args.output)
