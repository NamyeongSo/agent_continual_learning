# SPDX-License-Identifier: Apache-2.0
"""Plot saved initial-rollout means, sample SDs, official pass@5, and task candles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


def load_rows(run_dir, analysis_dir):
    protocol = json.loads((run_dir / "protocol.json").read_text())
    original = json.loads((analysis_dir / "initial_pct_distribution.json").read_text())
    grouped = defaultdict(dict)
    for relative, expected_hash in original["source_sha256"].items():
        raw = (run_dir / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError(f"Initial result changed: {relative}")
        result = json.loads(raw)
        if result["appworld_policy_sha256"] != protocol["appworld_policy_sha256"]:
            raise ValueError("Environment policy mismatch")
        task, repeat = result["task_id"], result["source_repeat"]
        if repeat in grouped[task] or type(result["success"]) is not bool:
            raise ValueError("Duplicate repeat or missing official success flag")
        grouped[task][repeat] = result
    if set(grouped) != set(protocol["task_ids"]) or len(grouped) != 50:
        raise ValueError("Expected the original 50 training tasks")
    rows = []
    for task in protocol["task_ids"]:
        repeats = grouped[task]
        if set(repeats) != set(range(5)):
            raise ValueError(f"Expected five complete independent repeats: {task}")
        scores = [repeats[i]["pct"] for i in range(5)]
        successes = sum(repeats[i]["success"] for i in range(5))
        rows.append({
            "task_id": task, **{f"pct_r{i}": score for i, score in enumerate(scores)},
            "mean_pct": statistics.mean(scores), "sample_std_pct": statistics.stdev(scores),
            "min_pct": min(scores), "q25_pct": float(np.percentile(scores, 25)),
            "median_pct": statistics.median(scores), "q75_pct": float(np.percentile(scores, 75)),
            "max_pct": max(scores), "official_successes_of_5": successes,
            "observed_official_pass_at_5": int(successes > 0),
            "budget_failures": sum(bool(r.get("budget_failure")) for r in repeats.values()),
            "turn_limit_rollouts": sum(bool(r.get("turn_limit_reached")) or
                                       (r.get("budget_failure") or {}).get("reason") == "max_turns"
                                       for r in repeats.values()),
        })
    return rows, original


def style():
    font = Path("/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=str(font)).get_name(),
        "axes.unicode_minus": False, "font.size": 11, "axes.titlesize": 14,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    })


def scatter(ax, rows):
    x = [r["mean_pct"] * 100 for r in rows]
    y = [r["sample_std_pct"] * 100 for r in rows]
    ax.scatter(x, y, s=65, color="#3977ba", alpha=0.65, edgecolors="white", linewidths=0.7, zorder=3)
    for (mean, sd), count in Counter(zip(x, y, strict=True)).items():
        if count > 1:
            offset = (-12, 14) if mean == 100 else (9, 10)
            ax.annotate(f"\u00d7{count}", (mean, sd), xytext=offset, textcoords="offset points",
                        ha="right" if mean == 100 else "left", fontsize=10, color="#1e3a5f",
                        arrowprops={"arrowstyle": "-", "color": "#94a3b8", "lw": 0.7})
    ax.set(title="평균 PCT vs. 반복 표준편차", xlabel="Task별 5회 평균 PCT (%)",
           ylabel="표본 표준편차 (pp, ddof=1)", xlim=(-4, 104), ylim=(-3, 61))
    ax.set_xticks(range(0, 101, 20))
    ax.set_yticks(range(0, 61, 10))
    ax.grid(color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)


def pass_histogram(ax, rows):
    scores = [r["observed_official_pass_at_5"] for r in rows]
    counts, _, bars = ax.hist(scores, bins=[-0.5, 0.5, 1.5], rwidth=0.65, edgecolor="white", linewidth=1.2)
    for bar, count, color in zip(bars, counts, ["#94a3b8", "#3977ba"], strict=True):
        bar.set_facecolor(color)
        ax.text(bar.get_x() + bar.get_width() / 2, count + 1.2,
                f"{int(count)} Tasks ({100 * count / len(rows):.0f}%)", ha="center", fontsize=12)
    ax.set(title="Task별 pass@5 분포", ylabel="Task 수", xlabel="관측 pass@5", ylim=(0, 54), xlim=(-0.5, 1.5))
    ax.set_xticks([0, 1], ["0\n5회 모두 완전 성공 없음", "1\n1회 이상 공식 완전 성공"])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)


def turn_limit_note(rows, policy):
    count = sum(r["turn_limit_rollouts"] for r in rows)
    if policy == "official_pct_at_limit":
        return f"40턴 종료 {count}/250회는 종료 시점의 공식 PCT로 채점했습니다."
    return f"40턴 종료 {count}/250회는 기존 규칙에 따라 0점으로 포함했습니다."


def candles(rows, policy):
    ordered = sorted(rows, key=lambda r: (r["mean_pct"], r["sample_std_pct"], r["task_id"]))
    fig, ax = plt.subplots(figsize=(18.5, 6.5))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.74, bottom=0.29)
    fig.suptitle("AppWorld 초기 평가: Task 50개의 PCT 캔들 분포", x=0.055, ha="left", y=0.97,
                 fontsize=20, fontweight="bold")
    fig.text(0.055, 0.895, "Task당 독립 Rollout 5회 · 평균 PCT 오름차순 정렬", fontsize=12, color="#475569")
    stats = [{"whislo": r["min_pct"] * 100, "q1": r["q25_pct"] * 100, "med": r["median_pct"] * 100,
              "q3": r["q75_pct"] * 100, "whishi": r["max_pct"] * 100, "mean": r["mean_pct"] * 100}
             for r in ordered]
    ax.bxp(stats, positions=np.arange(len(rows)), widths=0.55, patch_artist=True, showfliers=False,
           showmeans=True, boxprops={"facecolor": "#c7dcf0", "edgecolor": "#3977ba", "linewidth": 1.1},
           whiskerprops={"color": "#3977ba", "linewidth": 1.2},
           capprops={"color": "#3977ba", "linewidth": 1.1}, medianprops={"color": "#172b43", "linewidth": 1.6},
           meanprops={"marker": "o", "markerfacecolor": "#d67527", "markeredgecolor": "white", "markersize": 4.5})
    ax.set_xticks(np.arange(len(rows)), [r["task_id"] for r in ordered], rotation=90, fontsize=8.5)
    ax.set(xlim=(-0.7, 49.7), ylim=(-4, 105), ylabel="PCT (%)", xlabel="Task ID (평균 PCT 오름차순)")
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.legend(handles=[
        Patch(facecolor="#c7dcf0", edgecolor="#3977ba", label="몸통: 25~75% 구간"),
        Line2D([], [], color="#3977ba", label="심지: 최솟값~최댓값"),
        Line2D([], [], color="#172b43", linewidth=1.6, label="가로선: 중앙값"),
        Line2D([], [], color="none", marker="o", markerfacecolor="#d67527", label="주황점: 평균"),
    ], loc="upper left", bbox_to_anchor=(0.05, 0.855), ncol=4, frameon=False, fontsize=11)
    fig.text(0.055, 0.035, "각 캔들은 Task 하나의 5회 점수 분포입니다. " + turn_limit_note(rows, policy),
             fontsize=10, color="#475569")
    return fig, ordered


def analyze(run_dir):
    output = run_dir / "analysis/initial_rollouts"
    rows, original = load_rows(run_dir, output)
    style()
    combined, axes = plt.subplots(1, 2, figsize=(12.5, 5.9))
    combined.subplots_adjust(left=0.075, right=0.98, top=0.76, bottom=0.27, wspace=0.3)
    combined.suptitle("AppWorld 초기 평가: 평균·변동성과 pass@5", x=0.075, ha="left", y=0.97,
                      fontsize=19, fontweight="bold")
    combined.text(0.075, 0.88, "Train 50 Tasks · Task당 독립 Rollout 5회 · 산점도의 \u00d7N은 같은 좌표에 겹친 Task 수",
                  color="#475569", fontsize=11)
    scatter(axes[0], rows)
    pass_histogram(axes[1], rows)
    combined.text(0.075, 0.09, "pass@5는 5회 중 공식 완전 성공이 1회 이상이면 1, 아니면 0입니다. "
                  "평균 PCT와 별도 지표입니다.",
                  color="#475569", fontsize=10)
    policy = original.get("turn_limit_scoring", "zero")
    combined.text(0.075, 0.035, "표준편차는 ddof=1, 단위 pp. " + turn_limit_note(rows, policy),
                  color="#475569", fontsize=10)
    combined.savefig(output / "initial_scatter_and_pass5.png", dpi=220, facecolor="white")
    scatter_fig, scatter_ax = plt.subplots(figsize=(8, 5.8))
    scatter(scatter_ax, rows)
    scatter_fig.tight_layout(pad=2)
    scatter_fig.savefig(output / "initial_mean_pct_std_scatter.png", dpi=220, facecolor="white")
    pass_fig, pass_ax = plt.subplots(figsize=(8, 5.8))
    pass_histogram(pass_ax, rows)
    pass_fig.tight_layout(pad=2)
    pass_fig.savefig(output / "initial_pass_at_5_histogram.png", dpi=220, facecolor="white")
    candle_fig, ordered = candles(rows, policy)
    candle_fig.savefig(output / "initial_task50_candles.png", dpi=220, facecolor="white")
    with PdfPages(output / "initial_scatter_pass5_candles.pdf") as pdf:
        for fig in (combined, scatter_fig, pass_fig, candle_fig):
            pdf.savefig(fig, facecolor="white")
    for fig in (combined, scatter_fig, pass_fig, candle_fig):
        plt.close(fig)
    ranks = {r["task_id"]: i + 1 for i, r in enumerate(ordered)}
    for row in rows:
        row["candle_rank"] = ranks[row["task_id"]]
    with (output / "initial_task50_scatter_pass5_candles.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    passed = sum(r["observed_official_pass_at_5"] for r in rows)
    summary = {
        "tasks": 50, "rollouts_per_task": 5, "observed_official_pass_at_5_percent": 100 * passed / 50,
        "pass_at_5_task_counts": {"0": 50 - passed, "1": passed},
        "pass_at_5_definition": "I(at least one official success among the five saved independent rollouts)",
        "mean_best_pct_of_5_percent": 100 * statistics.mean(r["max_pct"] for r in rows),
        "official_success_count_distribution": dict(sorted(
            Counter(r["official_successes_of_5"] for r in rows).items()
        )),
        "scatter_x": "mean of five PCT scores (percent)", "scatter_y": "sample standard deviation (ddof=1, pp)",
        "scatter_unique_coordinates": len({(r["mean_pct"], r["sample_std_pct"]) for r in rows}),
        "candle_definition": {"wick": "min/max", "body": "25th/75th percentile", "line": "median", "dot": "mean"},
        "candle_task_order": [r["task_id"] for r in ordered],
        "budget_failure_rollouts": sum(r["budget_failures"] for r in rows),
        "turn_limit_rollouts": sum(r["turn_limit_rollouts"] for r in rows),
        "turn_limit_scoring": policy,
        "source_sha256": original["source_sha256"],
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "initial_scatter_pass5_candles_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in summary.items() if k not in {"source_sha256", "candle_task_order"}},
                     ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    analyze(parser.parse_args().run_dir)


if __name__ == "__main__":
    main()
