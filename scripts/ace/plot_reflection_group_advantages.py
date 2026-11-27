# SPDX-License-Identifier: Apache-2.0
"""Analyze task-local G5 advantages of saved ACE Reflection selections without inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from scripts.ace.plot_worst_reflection_repeats import ARMS, ROOT, RUNS, SEEDS, mode_label
from scripts.ace.run_reflection_accumulation import exact_reward

EPSILON = 1e-8
METHODS = ("worst", "best", "random")


def summarize(rows):
    values = [r["advantage"] for r in rows]
    return {
        "tasks": len(rows),
        "mean_advantage": statistics.mean(values) if values else None,
        "median_advantage": statistics.median(values) if values else None,
        "population_std_advantage": statistics.pstdev(values) if values else None,
        "min_advantage": min(values) if values else None,
        "max_advantage": max(values) if values else None,
        "negative": sum(v < 0 for v in values),
        "zero": sum(v == 0 for v in values),
        "positive": sum(v > 0 for v in values),
        "zero_signal_groups": sum(r["zero_signal_group"] for r in rows),
        "mean_pct": statistics.mean(r["reward_pct"] for r in rows) if rows else None,
    }


def analyze(output, runs=RUNS):
    output.mkdir(parents=True, exist_ok=True)
    hashes, candidate_rows, selected_rows, groups = {}, [], [], {}
    aud_verification = {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    for mode, run in runs.items():
        protocol = read(run / "protocol.json")
        source = Path(protocol["source_reflection_run"])
        source_protocol = read(source / "protocol.json")
        assert source_protocol["stage3_repeats_per_context"] == 3
        tasks = set(protocol["task_orders"]["42"]["train"])
        assert len(tasks) == 50
        bank = defaultdict(dict)
        for path in sorted((source / "stage3").glob("*/result.json")):
            r = read(path)
            assert r["appworld_policy_sha256"] == protocol["appworld_policy_sha256"]
            key = r["arm"], r["task_id"], r["source_repeat"]
            assert r["evaluation_repeat"] not in bank[key]
            bank[key][r["evaluation_repeat"]] = exact_reward(r)
        assert set(bank) == {(arm, task, i) for arm in ARMS for task in tasks for i in range(5)}
        assert all(set(v) == {0, 1, 2} for v in bank.values())
        means = {key: sum(values.values(), Fraction()) / 3 for key, values in bank.items()}
        candidates = {}
        for arm in ARMS:
            group_details = []
            for task in sorted(tasks):
                rewards = [means[(arm, task, i)] for i in range(5)]
                group_mean = sum(rewards, Fraction()) / 5
                variance = sum(((r - group_mean) ** 2 for r in rewards), Fraction()) / 5
                std = math.sqrt(float(variance))
                advantages = [float(r - group_mean) / (std + EPSILON) for r in rewards] if std else [0.0] * 5
                assert math.isclose(sum(advantages), 0, abs_tol=1e-12)
                assert max(abs(v) for v in advantages) <= 2 + 1e-12
                if std:
                    assert math.isclose(statistics.pstdev(advantages), std / (std + EPSILON), abs_tol=1e-12)
                for i, (reward, advantage) in enumerate(zip(rewards, advantages, strict=True)):
                    scores = bank[(arm, task, i)]
                    row = {
                        "mode": mode,
                        "arm": arm,
                        "task_id": task,
                        "source_repeat": i,
                        "source_id": f"{task}_r{i}",
                        **{f"pct_run{j + 1}": float(scores[j]) for j in range(3)},
                        "reward_pct": float(reward),
                        "reward_numerator": reward.numerator,
                        "reward_denominator": reward.denominator,
                        "group_mean_pct": float(group_mean),
                        "group_population_std_pct": std,
                        "reward_minus_group_mean": float(reward - group_mean),
                        "advantage": advantage,
                        "zero_signal_group": std == 0,
                    }
                    candidates[(arm, task, i)] = row
                    candidate_rows.append(row)
                group_details.append(
                    {
                        "task_id": task,
                        "rewards_pct": list(map(float, rewards)),
                        "advantages": advantages,
                        "group_mean_pct": float(group_mean),
                        "group_population_std_pct": std,
                        "reward_range_pct": float(max(rewards) - min(rewards)),
                        "zero_signal_group": std == 0,
                    }
                )
            groups[f"{mode}/{arm}"] = group_details
        for seed in SEEDS:
            assert set(protocol["task_orders"][str(seed)]["train"]) == tasks
            for arm in ARMS:
                for method in METHODS:
                    relative = f"conditions/s{seed}_{arm}_{method}/selections.json"
                    path = run / relative
                    choices = read(path)
                    assert len(choices) == 50
                    assert {c["candidate"]["task_id"] for c in choices} == tasks
                    aud_path = ROOT / f"outputs/ace_appworld_aud_accumulation_qwen38_20261117/{mode}" / relative
                    if runs is RUNS and aud_path.exists():
                        assert hashlib.sha256(aud_path.read_bytes()).hexdigest() == hashes[str(path)]
                        aud_verification[str(aud_path)] = "byte-identical selections"
                    else:
                        aud_verification[str(aud_path)] = "no matching AUD selection checked"
                    for choice in choices:
                        c = choice["candidate"]
                        assert choice["mode"] == method
                        key = arm, c["task_id"], c["source_repeat"]
                        r = candidates[key]
                        reward = Fraction(r["reward_numerator"], r["reward_denominator"])
                        assert reward == Fraction(c["score_numerator"], c["score_denominator"])
                        rewards = [means[(arm, c["task_id"], i)] for i in range(5)]
                        target = (min if method == "worst" else max)(rewards) if method != "random" else None
                        ties = [i for i in range(5) if target is None or rewards[i] == target]
                        assert choice["tie_indices"] == ties and c["source_repeat"] in ties
                        assert choice["candidate_index"] == c["source_repeat"]
                        assert method != "worst" or r["advantage"] <= 0
                        assert method != "best" or r["advantage"] >= 0
                        selected_rows.append(
                            {**r, "selection_seed": seed, "method": method, "selection_tie_count": len(ties)}
                        )
    summary = {}
    for mode in runs:
        for arm in ARMS:
            for seed in SEEDS:
                for method in METHODS:
                    rows = [
                        r
                        for r in selected_rows
                        if (r["mode"], r["arm"], r["selection_seed"], r["method"]) == (mode, arm, seed, method)
                    ]
                    summary[f"{mode}/{arm}/seed{seed}/{method}"] = {
                        "all_tasks": summarize(rows),
                        "nonzero_signal_tasks": summarize([r for r in rows if not r["zero_signal_group"]]),
                    }
    for mode in runs:
        for arm in ARMS:
            for method in ("worst", "best"):
                reference = None
                for seed in SEEDS:
                    values = {
                        r["task_id"]: r["advantage"]
                        for r in selected_rows
                        if (r["mode"], r["arm"], r["selection_seed"], r["method"]) == (mode, arm, seed, method)
                    }
                    assert reference is None or values == reference
                    reference = values
    result = {
        "definition": {
            "group": "Five Reflection candidates of the SAME task, sampler and GT arm; G=5.",
            "reward": "Exact arithmetic mean of the candidate's three saved official independent PCT evaluations.",
            "group_mean": "sum(r_i)/5",
            "group_std": "sqrt(sum((r_i - mean)^2)/5), population SD (ddof=0)",
            "advantage": "(r_i - mean) / (population_std + 1e-8); all zero if group_std=0",
            "zero_comparison": "Exact Fraction rewards determine ties and zero differences before float normalization.",
            "epsilon_reward_scale": "Rewards in [0, 1], epsilon=1e-8; advantages are dimensionless.",
        },
        "interpretation": [
            "Best>=0 and worst<=0 follow from selection by the same PCT, not an independent quality test.",
            "All-five-tied tasks contribute A=0 for all selections, including all-perfect ceiling groups.",
            "Uniform-random selection has expected A=0 within each group; realized seed samples need not average zero.",
            "Seed-specific selections reuse one frozen bank; do not pool them as fresh independent evaluations.",
            "Normalized advantage measures relative rank/spread, not absolute PCT gain or accumulated performance.",
            "GT arms and samplers are normalized separately; held-out data never enter these groups.",
        ],
        "summary": summary,
        "groups": groups,
        "aud_selection_verification": aud_verification,
        "source_sha256": hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "repository_normalization_reference": str(ROOT / "src/exgentic/agents/premise/group_relative.py"),
    }
    (output / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for filename, rows in (
        ("all_candidate_advantages.csv", candidate_rows),
        ("selected_reflection_advantages.csv", selected_rows),
    ):
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for seed in SEEDS:
        plot_histograms(selected_rows, seed, output, nonzero_only=False, modes=tuple(runs))
    plot_histograms(selected_rows, 42, output, nonzero_only=True, modes=tuple(runs))
    render(result, output, modes=tuple(runs))
    return result


def plot_histograms(rows, seed, output, *, nonzero_only, modes=tuple(RUNS)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    centers = np.linspace(-2, 2, 17)
    bins = np.linspace(-2.125, 2.125, 18)
    colors = {"worst": "#c44e52", "best": "#4c72b0", "random": "#55a868"}
    fig, axes = plt.subplots(len(modes), len(ARMS), figsize=(17, 10), sharex=True, sharey=True,
                             layout="constrained", squeeze=False)
    for i, mode in enumerate(modes):
        for j, arm in enumerate(ARMS):
            ax = axes[i, j]
            selected = [r for r in rows if (r["mode"], r["arm"], r["selection_seed"]) == (mode, arm, seed)]
            n_zero = sum(r["zero_signal_group"] for r in selected if r["method"] == "worst")
            for k, method in enumerate(METHODS):
                values = [
                    r["advantage"]
                    for r in selected
                    if r["method"] == method and (not nonzero_only or not r["zero_signal_group"])
                ]
                counts, _ = np.histogram(values, bins=bins)
                assert sum(counts) == len(values)
                mean = statistics.mean(values) if values else 0
                label = f"{method}: mean {mean:+.3f}, A=0: {values.count(0)}"
                ax.bar(centers + (k - 1) * 0.062, counts, width=0.059, color=colors[method], alpha=0.85, label=label)
                ax.axvline(mean, color=colors[method], linestyle="--", linewidth=1, alpha=0.8)
            ax.axvline(0, color="#333333", linewidth=0.7, zorder=0)
            ax.set_title(
                f"{mode_label(mode)} | "
                f"{'w/o GT' if arm == 'without_gt' else 'w/ GT'} | "
                f"n={50 - n_zero if nonzero_only else 50} tasks/selection\n"
                f"All-five-tied groups: {n_zero}/50 "
                f"({'excluded here' if nonzero_only else 'included as A=0'})",
                fontsize=11,
            )
            ax.set_xlabel("GRPO-style advantage (population SD, epsilon=1e-8)")
            ax.set_ylabel("Number of tasks")
            ax.set_xlim(-2.17, 2.17)
            ax.set_xticks(np.arange(-2, 2.01, 0.5))
            ax.tick_params(axis="x", labelbottom=True)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(
        f"Selected Reflection advantage distributions | selection seed {seed}\n"
        "Group: 5 candidates per task; reward: mean of 3 saved PCT runs; "
        "dashed lines: selection means",
        fontsize=14,
    )
    name = f"selected_advantages_seed{seed}" + ("_nonzero_groups" if nonzero_only else "")
    fig.savefig(output / f"{name}.png", dpi=180)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def render(result, output, *, modes=tuple(RUNS)):
    lines = [
        "# Reflection 후보의 GRPO식 Advantage 분포",
        "",
        "각 task·모델 설정·GT 조건별 Reflection5개를 한 그룹으로 묶었습니다. "
        "각 후보의 보상은 누적 전 독립 평가3회의 정확한 평균 PCT입니다.",
        "",
        "A_i = (r_i - mean(r_1,...,r_5)) / (population_std(r_1,...,r_5) + 1e-8). "
        "모표준편차(ddof=0)를 사용하고, 그룹 표준편차가0이면 모두 A=0입니다. "
        "정규화 전 PCT는0~1, Advantage는 단위가 없습니다.",
        "",
        "## seed42 실제 선정 결과 (task50개/조건)",
        "",
        "| 모델 | GT | 최저·최고 모두 동점인 task | worst 평균 A | best 평균 A | random 평균 A |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for mode in modes:
        for arm in ARMS:
            values = [result["summary"][f"{mode}/{arm}/seed42/{m}"]["all_tasks"] for m in METHODS]
            lines.append(
                f"| {mode} | {arm} | {values[0]['zero_signal_groups']}/50 | "
                + " | ".join(f"{v['mean_advantage']:+.4f}" for v in values)
                + " |"
            )
    lines += [
        "",
        "![전체 task 분포](selected_advantages_seed42.png)",
        "",
        "## 점수가 다른 후보가 있는 그룹만 (표준편차>0)",
        "",
        "| 모델 | GT | task 수 | worst 평균 A | best 평균 A | random 평균 A |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for mode in modes:
        for arm in ARMS:
            values = [result["summary"][f"{mode}/{arm}/seed42/{m}"]["nonzero_signal_tasks"] for m in METHODS]
            lines.append(
                f"| {mode} | {arm} | {values[0]['tasks']} | "
                + " | ".join(f"{v['mean_advantage']:+.4f}" for v in values)
                + " |"
            )
    lines += [
        "",
        "![동점 그룹 제외 분포](selected_advantages_seed42_nonzero_groups.png)",
        "",
        "## 해석 및 집계 기준",
        "",
        "- 5개 후보의 평균 점수가 모두 같은 task는 best·worst·random 모두 Advantage0입니다.",
        "- best의 양수·worst의 음수는 같은 PCT로 선정했기 때문에 생기는 관계입니다. "
        "독립적인 품질 검증이나 누적 효과의 증거가 아닙니다.",
        "- random의 그룹 내 기대 Advantage는0이지만, seed별50개 실제 선택 평균은0과 다를 수 있습니다.",
        "- 각 후보의3회 점수를 먼저 평균내고, 그5개 평균 사이에서 정규화했습니다. "
        "15개 개별 rollout이나 서로 다른 GT 조건을 한 그룹으로 합치지 않았습니다.",
        "- 원래 저장된 선정 파일을 사용했습니다. best·worst는 동점 후보가 달라도 Advantage가 "
        "세 seed에서 동일하며, random은 달라질 수 있습니다. 같은 bank를 새 독립 평가처럼 중복집계하지 않습니다.",
        "- 전체50task를 포함한 그림이 기본이며, 표준편차0인 task를 제외한 그림은 보조 분석입니다.",
        "- 40턴 종료는 기존 공식 최종상태 PCT, 나머지 승인된 예산 종료는 기존0점을 유지합니다. "
        "PCT 기반이며 TGC 기반 Advantage는 계산하지 않았습니다.",
        "- CSV는 모든 후보1000개와 실제 선정1800건을 보존합니다. 선정 건수에는 "
        "같은 후보의 여러 seed·전략 재선택이 포함되며 새 평가1800회를 뜻하지 않습니다.",
        "- 저장 결과만 CPU로 처리했으며 추가 학습·평가·모델 호출이 없습니다.",
        "",
        "## seed별 random 평균 A (전체50task)",
        "",
        "| 모델 | GT | seed42 | seed43 | seed44 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for mode in modes:
        for arm in ARMS:
            values = [result["summary"][f"{mode}/{arm}/seed{s}/random"]["all_tasks"]["mean_advantage"] for s in SEEDS]
            lines.append(f"| {mode} | {arm} | " + " | ".join(f"{v:+.4f}" for v in values) + " |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.output.resolve())
    print(json.dumps({key: value for key, value in result["summary"].items() if "/seed42/" in key}, indent=2))
