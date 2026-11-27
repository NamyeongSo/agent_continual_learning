# SPDX-License-Identifier: Apache-2.0
"""Plot task-level mean PCT and repeat-score variability from saved initial rollouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator


def analyze(run_dir, output_dir):
    protocol = json.loads((run_dir / "protocol.json").read_text())
    grouped = defaultdict(dict)
    sources = {}
    for path in sorted((run_dir / "stage1").glob("*/result.json")):
        raw = path.read_bytes()
        result = json.loads(raw)
        task, repeat = result["task_id"], result["source_repeat"]
        if result["appworld_policy_sha256"] != protocol["appworld_policy_sha256"]:
            raise ValueError(f"Policy mismatch: {task}")
        if repeat in grouped[task] or repeat not in range(5):
            raise ValueError(f"Duplicate or invalid repeat: {task}/{repeat}")
        score = result["pct"]
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid PCT: {task}/{repeat}")
        if result.get("budget_failure"):
            if score != 0 or result["success"] is not False:
                raise ValueError("Budget failures must remain measured zeroes")
        elif not math.isclose(score, result["passed_tests"] / result["total_tests"], abs_tol=1e-12):
            raise ValueError("Stored PCT differs from exact official test counts")
        grouped[task][repeat] = result
        sources[str(path.relative_to(run_dir))] = hashlib.sha256(raw).hexdigest()
    tasks = protocol["task_ids"]
    if set(grouped) != set(tasks) or len(tasks) != 50:
        raise ValueError("Saved initial results must exactly cover the pinned train50 membership")
    rows = []
    for task in tasks:
        repeats = grouped[task]
        if set(repeats) != set(range(5)):
            raise ValueError(f"Incomplete task: {task}")
        scores = [repeats[i]["pct"] for i in range(5)]
        row = {
            "task_id": task, **{f"pct_r{i}": value for i, value in enumerate(scores)},
            "mean_pct": statistics.mean(scores), "sample_std_pct": statistics.stdev(scores),
            "population_std_pct": statistics.pstdev(scores), "min_pct": min(scores), "max_pct": max(scores),
            "budget_failures": sum(bool(r.get("budget_failure")) for r in repeats.values()),
            "turn_limit_rollouts": sum(bool(r.get("turn_limit_reached")) or
                                       (r.get("budget_failure") or {}).get("reason") == "max_turns"
                                       for r in repeats.values()),
        }
        assert math.isclose(row["sample_std_pct"], float(np.std(scores, ddof=1)), abs_tol=1e-12)
        rows.append(row)
    prior = json.loads((run_dir / "stage1_summary.json").read_text())
    assert all(math.isclose(r["mean_pct"], prior[r["task_id"]]["mean_pct"], abs_tol=1e-12) for r in rows)
    means = np.array([row["mean_pct"] for row in rows]) * 100
    deviations = np.array([row["sample_std_pct"] for row in rows]) * 100
    mean_bins = np.arange(0, 101, 10)
    sd_max = max(5, 5 * math.ceil(float(deviations.max()) / 5))
    sd_bins = np.arange(0, sd_max + 1, 5)
    mean_counts, _ = np.histogram(means, bins=mean_bins)
    sd_counts, _ = np.histogram(deviations, bins=sd_bins)
    assert sum(mean_counts) == sum(sd_counts) == 50
    budget_count = sum(r["budget_failures"] for r in rows)
    summary = {
        "metric": "exact official individual-test pass fraction (PCT)",
        "tasks": 50, "rollouts_per_task": 5, "total_rollouts": sum(len(v) for v in grouped.values()),
        "std_definition": "Within-task sample standard deviation of five PCT scores; ddof=1",
        "plot_units": {"mean_pct": "percent", "sample_std": "percentage points"},
        "mean_task_pct_percent": float(means.mean()), "median_task_pct_percent": float(np.median(means)),
        "tasks_mean_zero": int(sum(means == 0)), "tasks_mean_100_percent": int(sum(means == 100)),
        "mean_within_task_sample_std_pp": float(deviations.mean()),
        "median_within_task_sample_std_pp": float(np.median(deviations)),
        "max_within_task_sample_std_pp": float(deviations.max()),
        "tasks_observed_std_zero": int(sum(deviations == 0)),
        "budget_failure_rollouts": budget_count,
        "tasks_with_budget_failure": sum(r["budget_failures"] > 0 for r in rows),
        "turn_limit_rollouts": sum(r["turn_limit_rollouts"] for r in rows),
        "turn_limit_scoring": protocol.get("turn_limit_scoring", "zero"),
        "mean_pct_histogram": {"edges_percent": mean_bins.tolist(), "task_counts": mean_counts.tolist()},
        "sample_std_histogram": {"edges_pp": sd_bins.tolist(), "task_counts": sd_counts.tolist()},
        "budget_failure_reasons": dict(Counter(
            r["budget_failure"]["reason"] for repeats in grouped.values() for r in repeats.values()
            if r.get("budget_failure")
        )),
        "source_sha256": sources,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "task_initial_pct_statistics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "initial_pct_distribution.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    plot(output_dir, means, deviations, mean_bins, sd_bins, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "source_sha256"}, ensure_ascii=False, indent=2))


def plot(output_dir, means, deviations, mean_bins, sd_bins, summary):
    font = Path("/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=str(font)).get_name(),
        "axes.unicode_minus": False, "font.size": 11, "axes.titlesize": 14,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.6))
    fig.subplots_adjust(left=0.065, right=0.98, top=0.77, bottom=0.23, wspace=0.23)
    fig.suptitle("AppWorld 초기 Rollout: Task별 성능과 반복 변동성", x=0.065, ha="left", y=0.97,
                 fontsize=19, fontweight="bold")
    fig.text(0.065, 0.885, "Train 50 Tasks · Task당 독립 Rollout 5회 · 두 패널 모두 Task 50개의 분포",
             color="#475569", fontsize=11)
    for ax, values, bins, color, title, xlabel in zip(
        axes, [means, deviations], [mean_bins, sd_bins], ["#3977ba", "#d78837"],
        ["Task별 평균 PCT", "Task 내 반복 평가의 표준편차"],
        ["5회 평균 PCT (%)", "5회 PCT의 표본 표준편차 (pp)"], strict=True,
    ):
        counts, _, bars = ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=1.2)
        ax.set(title=title, xlabel=xlabel, ylabel="Task 수", xlim=(bins[0], bins[-1]))
        ax.set_xticks(bins)
        ax.set_ylim(0, max(counts) * 1.3 + 1)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        for bar, count in zip(bars, counts, strict=True):
            if count:
                ax.text(bar.get_x() + bar.get_width() / 2, count + 0.5, str(int(count)), ha="center", fontsize=11)
    axes[0].axvline(means.mean(), color="#1e3a5f", linestyle="--", linewidth=1.5,
                   label=f"평균 {means.mean():.1f}%")
    axes[0].legend(loc="upper left", frameon=False)
    axes[0].text(0.03, 0.82, f"평균 PCT 100%: {summary['tasks_mean_100_percent']} Tasks\n"
                 f"평균 PCT 0%: {summary['tasks_mean_zero']} Task", transform=axes[0].transAxes,
                 fontsize=10, color="#334155", va="top")
    axes[1].axvline(deviations.mean(), color="#80501e", linestyle="--", linewidth=1.5,
                   label=f"평균 {deviations.mean():.1f}pp")
    axes[1].legend(loc="upper right", frameon=False)
    axes[1].text(0.97, 0.82, f"중앙값 {np.median(deviations):.1f}pp\n"
                 f"5회 점수 동일: {summary['tasks_observed_std_zero']} Tasks", transform=axes[1].transAxes,
                 fontsize=10, color="#334155", ha="right", va="top")
    fig.text(0.065, 0.105, "표준편차: Task별 PCT 5개의 표본 표준편차 (ddof=1).  pp는 percentage points.",
             color="#475569", fontsize=10)
    if summary["turn_limit_scoring"] == "official_pct_at_limit":
        note = f"40턴 종료 {summary['turn_limit_rollouts']}/250회는 종료 시점의 공식 PCT로 채점했습니다."
    else:
        note = f"40턴 종료 {summary['turn_limit_rollouts']}/250회는 기존 규칙에 따라 0점으로 포함했습니다."
    fig.text(0.065, 0.05, note, color="#475569", fontsize=10)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"initial_rollouts_histograms.{extension}", dpi=220, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    analyze(args.run_dir, args.output_dir or args.run_dir / "analysis/initial_rollouts")


if __name__ == "__main__":
    main()
