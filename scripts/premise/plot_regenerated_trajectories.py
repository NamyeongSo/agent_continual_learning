# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Render benchmark-wise PRF/BLEU/METEOR curves from completed trajectory metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCHMARKS = (("bfcl", "BFCL"), ("appworld", "AppWorld"), ("terminalbench2", "Terminal-Bench"))
COLORS = {"strict": "#235789", "loose": "#c55a11"}
LABELS = {"precision": "Precision", "recall": "Recall", "f1": "F1", "bleu1": "BLEU-1",
          "bleu4": "BLEU-4 (effective order)", "meteor": "METEOR"}
GROUPS = (("prf", ("precision", "recall", "f1")), ("bleu_meteor", ("bleu1", "bleu4", "meteor")))


def save_figure(fig, output, name, files):
    for suffix in ("png", "pdf"):
        path = output / f"{name}.{suffix}"
        fig.savefig(path, dpi=200)
        files.append(path.name)
    plt.close(fig)


def draw_panels(summary, output, mode=None, fixed=False, complete_sequences=False):
    files = []
    for group, metrics in GROUPS:
        fig, axes = plt.subplots(3, 3, figsize=(13.5, 9.5), squeeze=False)
        source = (summary["cp0_complete_sequences"] if complete_sequences else
                  summary["cp0_reference"] if fixed else summary["distances"])
        coverage = summary["complete_sequence_coverage"] if complete_sequences else summary["coverage"]
        for i, (benchmark, title) in enumerate(BENCHMARKS):
            for j, metric in enumerate(metrics):
                ax = axes[i, j]
                available = []
                for arm in ((mode,) if mode else ("strict", "loose")):
                    rows = sorted((r for r in source if r["benchmark"] == benchmark and r["mode"] == arm),
                                  key=lambda r: r["distance"])
                    x = [r["distance"] for r in rows]
                    available.extend(x)
                    color = COLORS[arm]
                    ax.plot(x, [r[metric] for r in rows], "o-", color=color, markersize=4,
                            label=arm if fixed else "Regenerated sequence agreement")
                    control = f"{arm} CP0 self control" if complete_sequences else f"{arm} self control"
                    ax.plot(x, [r[f"self_{metric}"] for r in rows], ":", color=color if fixed else "#777777",
                            linewidth=1.5, label=control if fixed else "Mean within-context control")
                ax.set(title=f"{title} | {LABELS[metric]}", xlabel="Update distance", ylim=(-.03, 1.03))
                if available:
                    ax.set_xticks(sorted(set(available)))
                else:
                    maximum = 30 if benchmark == "terminalbench2" else 50
                    ax.set_xticks(list(range(0, maximum + 1, 10)))
                    ax.set_xlim(-1, maximum + 1)
                    ax.text(.5, .5, "No complete trajectories", ha="center", transform=ax.transAxes)
                ax.grid(alpha=.2)
                ax.spines[["top", "right"]].set_visible(False)
                if j == 0:
                    c = coverage[benchmark]
                    ax.set_ylabel(f"Mean score\n{c['complete_tasks']}/{c['planned_tasks']} complete tasks")
        heading = ("CP0 context bank | available complete sequences" if complete_sequences else
                   "Independent regenerated CP0 reference" if fixed else f"{mode.upper()} | all checkpoint pairs")
        fig.suptitle(f"{heading} | concatenated action sequences | Qwen3.8, T = {summary['temperature']:g}",
                     fontsize=15)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .948), ncol=4 if fixed else 2,
                   frameon=False, fontsize=9)
        fig.text(.5, .033, "Both sides are independently regenerated. No historical actions or shared history "
                 "are copied into the scored sequences.", ha="center", fontsize=9)
        footer = ("Only whole sequences; constant task membership across distances; valid sample-pair counts vary. "
                  "Conditional on valid reconstruction." if complete_sequences else
                  "Two reconstructions/context; same original prefixes. Descriptive means. "
                  "Dotted self controls are not confidence intervals.")
        fig.text(.5, .014, footer, ha="center", fontsize=9)
        fig.tight_layout(rect=(.005, .055, .995, .928), h_pad=1.8, w_pad=1.8)
        view = "cp0_complete_sequences" if complete_sequences else "cp0_reference" if fixed else mode
        save_figure(fig, output, f"{view}_trajectory_{group}_distance", files)
    return files


def draw_concat_comparison(summary, output):
    files = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), squeeze=False)
    for i, (benchmark, title) in enumerate(BENCHMARKS):
        ax = axes[0, i]
        for mode, color in COLORS.items():
            rows = sorted((r for r in summary["distances"] if r["benchmark"] == benchmark and r["mode"] == mode),
                          key=lambda r: r["distance"])
            x = [r["distance"] for r in rows]
            ax.plot(x, [r["f1"] for r in rows], "o-", color=color, label=f"{mode}: concatenate first")
            ax.plot(x, [r["prefix_f1"] for r in rows], "o:", color=color, label=f"{mode}: prefix mean")
            ax.set_xticks(x)
        ax.set(title=title, xlabel="Update distance", ylabel="F1", ylim=(-.03, 1.03))
        ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .93), ncol=4, frameon=False, fontsize=8)
    fig.suptitle(f"Effect of scoring unit on identical regenerated actions | T = {summary['temperature']:g}",
                 fontsize=14)
    fig.text(.5, .018, "Whole-sequence PRF matches action multisets across prefix positions; "
             "it does not enforce action order.", ha="center", fontsize=9)
    fig.tight_layout(rect=(.005, .065, .995, .86))
    save_figure(fig, output, "concat_vs_prefix_f1", files)
    return files


def write_report(summary, output):
    lines = ["# 양쪽 context 재생성 후 전체 action sequence 비교", "",
             f"분석 상태: {summary['analysis_status']}. 모델: {summary['model']}, "
             f"temperature={summary['temperature']:g}. "
             "전체 sequence를 context마다2회 생성하고 CP0 독립reference를1회 추가했다.", "",
             "원본 trajectory는 동일한 history prefixes와 수집 context(CP0)를 복원하는 데 사용한다. "
             "점수의 양쪽 sequence는 모두 Qwen3.8의 같은 생성 설정으로 별도 재생성한 action만 포함한다. "
             "History/observation/원본 action을 비교 sequence에 붙이지 않는다. 생성 action은 환경에서 실행하지 "
             "않으며 이후 prefix도 원본 그대로다. 따라서 조건부 action 재구성의 유사도 평가이다.", "",
             "모든 checkpoint pair를 거리0·10·20…으로 묶는다. 거리0은 같은CP의 서로 다른 repeat를 양방향 비교한다. "
             "양수 거리는 뒤CP를 prediction, 앞CP를 reference로 한다. 먼저 전체concat pair별 점수를 계산하고 "
             "task를 균등 평균한 뒤, 같은 거리의 checkpoint pair를 균등 평균한다. 모든CP/repeat/reference에서 "
             "완성된 trajectory의 공통 교집합만 사용하며, 누락/오류를0으로 처리하지 않는다.", "",
             "별도로 original collection context인CP0에서 재생성한 독립reference와 각CP의 재생성sequence를 비교한다. "
             "이 표의 거리0 역시 별도의 독립 생성물이므로 자기비교가 아니다.", "",
             "| Benchmark | completed/planned tasks | complete/planned acting prefixes |",
             "|---|---:|---:|"]
    for benchmark, title in BENCHMARKS:
        c = summary["coverage"][benchmark]
        lines.append(f"| {title} | {c['complete_tasks']}/{c['planned_tasks']} | "
                     f"{c['complete_prefixes']}/{c['planned_prefixes']} |")
    for title, key in (("모든 checkpoint pair의 거리별 결과", "distances"),
                       ("독립 재생성 CP0 reference 대비 결과", "cp0_reference"),
                       ("보조 결과: 완성된 전체sequence의 CP0-context bank 대비 비교", "cp0_complete_sequences")):
        lines += ["", f"## {title}", "",
                  "| benchmark | mode | Δupdate | P | R | F1 | BLEU-1 | BLEU-4 | METEOR | prefix F1 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in summary[key]:
            values = " | ".join(f"{row[m]:.4f}" for m in (*LABELS, "prefix_f1"))
            lines.append(f"| {row['benchmark']} | {row['mode']} | {row['distance']} | {values} |")
    lines += ["", "## 보조 결과의 coverage와 해석", "",
              summary["complete_sequence_definition"], "",
              "위 보조 결과는 CP0 완성sequence가2개 이상이고 모든후속CP에 완성sequence가1개 이상인 task의 "
              "공통교집합을 사용한다. CP0에서 생성한 전체sequence bank와 각CP의 완성sequence끼리 모든쌍을 "
              "비교한다. 거리0에서는 같은repeat ID를 제외한다. 일부prefix만 있는 생성물은 사용하지 않는다. "
              "task당 점수를 먼저 평균하므로 유효생성본이 많은task에 더 큰가중치를 주지 않는다. "
              "점선은 CP0 독립반복 기준선이며, 앞의 전CP·전repeat완료 결과의 두context 평균기준선과 다르다. "
              "유효생성물의조건부 유사도이고, context에따라유효생성확률이달라질 수 있다. 실패율/가용표본수와 "
              "함께해석해야한다. CSV의min/max_sample_pairs_per_task가각거리의가용쌍수를표시한다.", "",
              "보조 coverage: `" + json.dumps(summary["complete_sequence_coverage"], ensure_ascii=False) + "`", ""]
    lines += ["", "## 지표 정의와 해석 범위", "",
              "Strict PRF는 함수명과 전체 parsed arguments가 정확히 같은 action의 multiset overlap이며 "
              "중복 개수를 보존한다. Loose PRF는 함수/action 이름만 비교한다. Whole-sequence PRF는 서로 다른 "
              "시간 위치의 action도 매칭하므로 순서를 검증하지 않는다. Prefix F1은 동일한 재생성action을 "
              "원래 위치끼리 비교한 진단값이며, 두 단위의 차이를 같은 samples에서 측정한다.", ""]
    for name, text in summary["metric_definitions"].items():
        if name != "sources":
            lines.append(f"- **{name}**: {text}")
    lines += ["", "BLEU/METEOR의 strict·loose는 입력 action 표현의 차이다. NLP metric을 새로 정의한 것이 아니다. "
              "문자열/코드에 쓰는 lexical surrogate이며 task 성공, 코드 의미 일치 또는 importance ratio를 "
              "측정하지 않는다. Shared syntax나 동일 함수 이름만으로도 점수가 높아질 수 있다. "
              "METEOR는 기본 lowercase·stemming·WordNet 매칭을 사용하므로 코드 식별자의 의미 보존을 "
              "보장하지 않는다. 단일token exact METEOR도 fragmentation penalty 때문에0.5이다.", "",
              "[NLTK BLEU documentation](https://www.nltk.org/api/nltk.translate.bleu_score.html), "
              "[NLTK METEOR documentation](https://www.nltk.org/api/nltk.translate.meteor_score.html).", "",
              "Benchmark당 최대4task·CP당2회 재구성 pilot이다. 이전 partial-prefix audit의10회 반복과 "
              "직접 동일한 추정량으로 취급하지 않는다. Whole대prefix 비교는 이번 동일 sample/동일 coverage에서 "
              "비교한다. 모든pair가 생성물을 공유하므로 pair 수를 독립 표본 수로 쓴 CI/유의성을 제시하지 않는다. "
              "점선은 두context의 독립 self-score 평균이며 기술적인 sampling control이다.", "",
              "실제 request 설정/비용: `" + json.dumps(summary["request_audit"], ensure_ascii=False) + "`", "",
              "Coverage/error: `" + json.dumps(summary["coverage"], ensure_ascii=False) + "`", ""]
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    output = args.run.resolve() / "analysis"
    summary = json.loads((output / "summary.json").read_text())
    files = []
    for mode in COLORS:
        files.extend(draw_panels(summary, output, mode))
    files.extend(draw_panels(summary, output, fixed=True))
    files.extend(draw_panels(summary, output, fixed=True, complete_sequences=True))
    files.extend(draw_concat_comparison(summary, output))
    write_report(summary, output)
    print(json.dumps({"figures": files, "report": str(output / "report.md")}), flush=True)


if __name__ == "__main__":
    main()
