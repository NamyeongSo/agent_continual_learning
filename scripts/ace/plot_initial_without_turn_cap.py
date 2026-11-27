# SPDX-License-Identifier: Apache-2.0
"""Render diagnostic views that exclude capped rollouts, without changing the experiment."""

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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator


def capped(row):
    return bool(row.get("turn_limit_reached")) or (row.get("budget_failure") or {}).get("reason") == "max_turns"


def stats_rows(groups):
    result = []
    for task, items in groups.items():
        scores = [r["pct"] for r in items]
        n = len(scores)
        if not n:
            continue
        row = {"task_id": task, "n_rollouts": n,
               **{f"pct_r{i}": next((r["pct"] for r in items if r["source_repeat"] == i), None) for i in range(5)},
               **{f"official_success_r{i}": next((r["success"] for r in items if r["source_repeat"] == i), None)
                  for i in range(5)},
               "mean_pct": statistics.mean(scores), "sample_std_pct": statistics.stdev(scores) if n > 1 else None,
               "min_pct": min(scores), "q25_pct": float(np.percentile(scores, 25)),
               "median_pct": statistics.median(scores), "q75_pct": float(np.percentile(scores, 75)),
               "max_pct": max(scores), "official_successes": sum(r["success"] for r in items),
               "pass_at_5": int(any(r["success"] for r in items)) if n == 5 else None,
               "pass_and_5": int(all(r["success"] for r in items)) if n == 5 else None}
        result.append(row)
    return sorted(result, key=lambda r: (r["mean_pct"], r["task_id"]))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pass_hist(ax, rows, metric):
    eligible = [r for r in rows if r["n_rollouts"] == 5]
    scores = [r[metric] for r in eligible]
    total = len(eligible)
    counts, _, bars = ax.hist(scores, bins=[-0.5, 0.5, 1.5], rwidth=0.62, edgecolor="white")
    for bar, count, color in zip(bars, counts, ["#94a3b8", "#3977ba"], strict=True):
        bar.set_facecolor(color)
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.7,
                f"{int(count)} Tasks ({100 * count / total:.1f}%)", ha="center", fontsize=11)
    is_all = metric == "pass_and_5"
    title = "pass-and-5: 5회 모두 공식 완전 성공" if is_all else "pass@5: 1회 이상 공식 완전 성공"
    labels = (["0\n5회 모두 성공하지는 못함", "1\n5회 모두 성공"] if is_all
              else ["0\n5회 모두 성공 없음", "1\n1회 이상 성공"])
    ax.set(title=f"{title}\n5회가 모두 남은 {total}개 Task만 집계", xlabel="Task별 관측값", ylabel="Task 수",
           xlim=(-0.5, 1.5), ylim=(0, max(counts) * 1.3 + 2))
    ax.set_xticks([0, 1], labels)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.grid(axis="y", color="#e2e8f0")
    ax.set_axisbelow(True)


def draw_view(output, rows, scope):
    output.mkdir(parents=True, exist_ok=True)
    means = np.array([r["mean_pct"] for r in rows]) * 100
    deviations = np.array([r["sample_std_pct"] for r in rows if r["sample_std_pct"] is not None]) * 100
    n_tasks, n_rollouts = len(rows), sum(r["n_rollouts"] for r in rows)
    counts_n = dict(sorted(Counter(r["n_rollouts"] for r in rows).items()))
    subtitle = f"{n_tasks} Tasks · {n_rollouts} Rollouts · Task별 반복수 {counts_n}"
    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.21, wspace=0.25)
    fig.suptitle(scope, x=0.07, ha="left", y=0.97, fontsize=18, fontweight="bold")
    fig.text(0.07, 0.88, subtitle, color="#475569", fontsize=11)
    bins_sd = np.arange(0, max(5, 5 * math.ceil(float(deviations.max()) / 5)) + 1, 5)
    for ax, values, bins, color, title, xlabel in zip(
        axes, [means, deviations], [np.arange(0, 101, 10), bins_sd], ["#3977ba", "#d78837"],
        ["Task별 평균 PCT", "Task별 반복 표준편차"], ["평균 PCT (%)", "표본 표준편차 (pp, ddof=1)"], strict=True,
    ):
        counts, _, bars = ax.hist(values, bins=bins, color=color, edgecolor="white")
        for bar, count in zip(bars, counts, strict=True):
            if count:
                ax.text(bar.get_x() + bar.get_width() / 2, count + 0.5, str(int(count)), ha="center")
        ax.set(title=title, xlabel=xlabel, ylabel="Task 수", xlim=(bins[0], bins[-1]))
        ax.set_ylim(0, max(counts) * 1.3 + 1)
        ax.axvline(values.mean(), color="#334155", linestyle="--", lw=1.3,
                   label=f"평균 {values.mean():.1f}{'%' if ax is axes[0] else 'pp'}")
        ax.legend(frameon=False)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.grid(axis="y", color="#e2e8f0")
        ax.set_axisbelow(True)
    fig.text(0.07, 0.06, "40턴 종료로 표시된 Rollout은 집계에서 제외했습니다. "
             "남은 점수가 없는 Task는 포함하지 않았습니다.",
             fontsize=10, color="#475569")
    figures.append(("histograms", fig))

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for n in counts_n:
        subset = [r for r in rows if r["n_rollouts"] == n and r["sample_std_pct"] is not None]
        ax.scatter([r["mean_pct"] * 100 for r in subset], [r["sample_std_pct"] * 100 for r in subset],
                   s=60, alpha=0.65, label=f"n={n}: {len(subset)} Tasks", zorder=3)
    coordinates = Counter((r["mean_pct"] * 100, r["sample_std_pct"] * 100)
                          for r in rows if r["sample_std_pct"] is not None)
    for (x, y), count in coordinates.items():
        if count > 1:
            ax.annotate(f"\u00d7{count}", (x, y), xytext=(-10 if x == 100 else 8, 10),
                        textcoords="offset points", ha="right" if x == 100 else "left", fontsize=10)
    ax.set(title=scope + "\n" + subtitle, xlabel="Task별 평균 PCT (%)", ylabel="표본 표준편차 (pp, ddof=1)",
           xlim=(-3, 104), ylim=(-1, max(5, 5 * math.ceil(float(deviations.max()) / 5)) + 3))
    ax.legend(frameon=False, loc="upper left")
    ax.grid(color="#e2e8f0")
    ax.set_axisbelow(True)
    fig.tight_layout(pad=2)
    figures.append(("scatter", fig))

    fig, ax = plt.subplots(figsize=(max(13, n_tasks * 0.37), 6.5))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.76, bottom=0.32)
    fig.suptitle(scope + ": Task별 캔들", x=0.06, ha="left", y=0.97, fontsize=18, fontweight="bold")
    fig.text(0.06, 0.88, subtitle + " · 평균 PCT 오름차순", color="#475569", fontsize=11)
    stats = [{"whislo": r["min_pct"] * 100, "q1": r["q25_pct"] * 100, "med": r["median_pct"] * 100,
              "q3": r["q75_pct"] * 100, "whishi": r["max_pct"] * 100, "mean": r["mean_pct"] * 100} for r in rows]
    ax.bxp(stats, positions=np.arange(n_tasks), widths=0.55, patch_artist=True, showfliers=False, showmeans=True,
           boxprops={"facecolor": "#c7dcf0", "edgecolor": "#3977ba"},
           whiskerprops={"color": "#3977ba"}, capprops={"color": "#3977ba"},
           medianprops={"color": "#172b43", "linewidth": 1.5},
           meanprops={"marker": "o", "markerfacecolor": "#d67527", "markeredgecolor": "white", "markersize": 4})
    ax.set_xticks(np.arange(n_tasks), [f"{r['task_id']} [n={r['n_rollouts']}]" for r in rows], rotation=90, fontsize=8)
    ax.set(ylabel="PCT (%)", xlabel="Task ID [남은 Rollout 수]", ylim=(-3, 105), xlim=(-0.7, n_tasks - 0.3))
    ax.grid(axis="y", color="#e2e8f0")
    ax.set_axisbelow(True)
    fig.text(0.06, 0.035, "몸통: 25~75% 구간 · 심지: 최솟값~최댓값 · 가로선: 중앙값 · 주황점: 평균",
             fontsize=10, color="#475569")
    figures.append(("task_candles", fig))
    for metric, name in [("pass_at_5", "pass_at_5_histogram"), ("pass_and_5", "pass_and_5_histogram")]:
        fig, ax = plt.subplots(figsize=(8, 5.8))
        pass_hist(ax, rows, metric)
        fig.tight_layout(pad=2)
        figures.append((name, fig))
    with PdfPages(output / "figures.pdf") as pdf:
        for name, fig in figures:
            fig.savefig(output / f"{name}.png", dpi=220, facecolor="white")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
    eligible = [r for r in rows if r["n_rollouts"] == 5]
    summary = {
        "scope": scope, "tasks": n_tasks, "rollouts": n_rollouts, "repeat_count_distribution": counts_n,
        "mean_task_pct_percent": float(means.mean()), "mean_sample_std_pp": float(deviations.mean()),
        "median_sample_std_pp": float(np.median(deviations)), "max_sample_std_pp": float(deviations.max()),
        "pass_metric_eligible_tasks": len(eligible),
        "observed_pass_at_5_success_tasks": sum(r["pass_at_5"] for r in eligible),
        "observed_pass_at_5_percent": 100 * sum(r["pass_at_5"] for r in eligible) / len(eligible),
        "observed_pass_and_5_success_tasks": sum(r["pass_and_5"] for r in eligible),
        "observed_pass_and_5_percent": 100 * sum(r["pass_and_5"] for r in eligible) / len(eligible),
        "pass_metric_definition": "Official success flags; only tasks retaining all five observations. "
                                  "Missing is unmeasured.",
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task_ids": [r["task_id"] for r in rows],
    }
    write_csv(output / "task_statistics.csv", rows)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    run_dir = parser.parse_args().run_dir
    base = run_dir / "analysis/initial_rollouts"
    output = base / "without_turn_cap"
    output.mkdir(parents=True, exist_ok=True)
    font = Path("/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                         "axes.unicode_minus": False, "font.size": 11, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    groups, removed, source_hashes = defaultdict(list), [], {}
    for path in sorted((run_dir / "stage1").glob("*/result.json")):
        raw = path.read_bytes()
        row = json.loads(raw)
        groups[row["task_id"]].append(row)
        source_hashes[str(path.relative_to(run_dir))] = hashlib.sha256(raw).hexdigest()
        if capped(row):
            removed.append(row)
    assert len(groups) == 50 and all(len(rows) == 5 for rows in groups.values())
    assert len(removed) == 20
    all_rows = stats_rows(groups)
    fig, ax = plt.subplots(figsize=(8, 5.8))
    pass_hist(ax, all_rows, "pass_and_5")
    fig.tight_layout(pad=2)
    for extension in ("png", "pdf"):
        fig.savefig(base / f"pass_and_5_histogram.{extension}", dpi=220, facecolor="white")
    plt.close(fig)
    write_csv(base / "pass_and_5_tasks.csv", all_rows)
    full_summary = {"tasks": 50, "pass_and_5_success_tasks": sum(r["pass_and_5"] for r in all_rows),
                    "pass_and_5_percent": 100 * sum(r["pass_and_5"] for r in all_rows) / 50,
                    "definition": "All five official success flags must be true", "source_sha256": source_hashes}
    (base / "pass_and_5_summary.json").write_text(json.dumps(full_summary, ensure_ascii=False, indent=2) + "\n")
    filtered = {t: [r for r in items if not capped(r)] for t, items in groups.items()}
    rows = stats_rows(filtered)
    complete = stats_rows({t: items for t, items in filtered.items() if len(items) == 5})
    assert len(rows) == 49 and sum(r["n_rollouts"] for r in rows) == 230
    assert len(complete) == 38 and sum(r["n_rollouts"] for r in complete) == 190
    primary = draw_view(output, rows, "40턴 종료 Rollout 제외")
    strict = draw_view(output / "complete_five_tasks", complete, "40턴 종료가 있었던 Task 전체 제외")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.9))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.74, bottom=0.23, wspace=0.25)
    fig.suptitle("pass-and-5: 5회 모두 공식 완전 성공한 Task 비율", x=0.07, ha="left", y=0.97,
                 fontsize=18, fontweight="bold")
    for ax, values, label in zip(axes, [all_rows, rows], ["전체", "40턴 종료 제외"], strict=True):
        pass_hist(ax, values, "pass_and_5")
        eligible = sum(r["n_rollouts"] == 5 for r in values)
        ax.set_title(f"{label}: 5회가 모두 남은 {eligible}개 Task")
        ax.set_ylim(0, 40)
    fig.text(0.07, 0.06, "제외 후 n=3~4인 11개 Task는 pass-and-5가 미측정이므로 오른쪽 분모에서 제외했습니다.",
             fontsize=10, color="#475569")
    for extension in ("png", "pdf"):
        fig.savefig(base / f"pass_and_5_comparison.{extension}", dpi=220, facecolor="white")
    plt.close(fig)
    write_csv(output / "excluded_turn_cap_rollouts.csv", [
        {k: r[k] for k in ("task_id", "source_repeat", "source_id", "pct", "success")} for r in removed
    ])
    (output / "source_hashes.json").write_text(json.dumps(source_hashes, ensure_ascii=False, indent=2) + "\n")
    (output / "README.md").write_text(
        "# 40턴 종료 제외 진단 그래프\n\n"
        "이 폴더: 종료 Rollout 20개만 제외. 49 Tasks/230 Rollouts, Task별 n=3~5. "
        "988af8e_3는 5회 모두 제외되어 평균/표준편차를 계산하지 않았다.\n\n"
        "complete_five_tasks/: 종료가 한 번이라도 있었던 Task 12개 전체 제외. 38 Tasks/190 Rollouts, n=5.\n\n"
        "각 폴더: histograms.png, scatter.png, task_candles.png, pass_at_5_histogram.png, "
        "pass_and_5_histogram.png, figures.pdf, task_statistics.csv, summary.json.\n\n"
        "pass@5와 pass-and-5는 5회가 모두 남은 38개 Task만 분모로 사용한다. "
        "나머지 11개 Task의 관측값은 미측정으로 두며 실패로 치지 않았다. "
        "pass@5=34/38, pass-and-5=22/38. 원래 전체 폴더의 pass-and-5는22/50이다.\n\n"
        "분포는 남은 Rollout으로 계산한 Task별 평균과 표본 SD(ddof=1)이다. "
        "캔들은 Q1/Q3 몸통, 최소/최대 심지, 중앙값 가로선, 평균 주황점이다. "
        "이는 필터링된 조건부 통계이며 전체 실험 점수를 대체하지 않는다.\n"
    )
    print(json.dumps({"full_pass_and_5": full_summary["pass_and_5_percent"],
                      "rollout_filtered": primary, "task_filtered": strict}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
