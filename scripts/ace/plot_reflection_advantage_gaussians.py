# SPDX-License-Identifier: Apache-2.0
"""Fit descriptive Gaussian curves to existing selected-Reflection advantage samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from scripts.ace.plot_reflection_group_advantages import ARMS, METHODS, RUNS, SEEDS


def analyze(source, output):
    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source / "analysis.json").read_text())
    with (source / "selected_reflection_advantages.csv").open() as stream:
        raw_rows = list(csv.DictReader(stream))
    rows = [
        {
            **r,
            "selection_seed": int(r["selection_seed"]),
            "advantage": float(r["advantage"]),
            "zero_signal_group": r["zero_signal_group"] == "True",
        }
        for r in raw_rows
    ]
    assert len(rows) == 1800
    assert len({(r["mode"], r["arm"], r["selection_seed"], r["method"], r["task_id"]) for r in rows}) == 1800
    fits = {}
    for seed in SEEDS:
        for mode in RUNS:
            for arm in ARMS:
                for method in METHODS:
                    selected = [
                        r
                        for r in rows
                        if (r["selection_seed"], r["mode"], r["arm"], r["method"]) == (seed, mode, arm, method)
                    ]
                    assert len(selected) == 50
                    key = f"{mode}/{arm}/seed{seed}/{method}"
                    for subset, selected_subset in (
                        ("all_tasks", selected),
                        ("nonzero_signal_tasks", [r for r in selected if not r["zero_signal_group"]]),
                    ):
                        values = [r["advantage"] for r in selected_subset]
                        mean = statistics.mean(values) if values else None
                        std = statistics.pstdev(values) if values else None
                        previous = metadata["summary"][key][subset]
                        assert len(values) == previous["tasks"]
                        assert mean == previous["mean_advantage"]
                        assert std == previous["population_std_advantage"]
                        fits[f"{key}/{subset}"] = {
                            "n_tasks": len(values),
                            "mu": mean,
                            "sigma_mle": std,
                            "empirical_zero_count": values.count(0),
                            "zero_signal_groups": sum(r["zero_signal_group"] for r in selected_subset),
                            "zero_signal_groups_excluded": 50 - len(values),
                            "empirical_min": min(values) if values else None,
                            "empirical_max": max(values) if values else None,
                        }
    result = {
        "purpose": "Descriptive Gaussian approximation requested by the user; not a normality assertion or test.",
        "fit": "Normal(mu, sigma^2); mu=mean of selected A, sigma=sqrt(mean((A-mu)^2)), MLE/population SD.",
        "advantage_definition": metadata["definition"],
        "source": str(source),
        "fits": fits,
        "limitations": [
            "Empirical distributions contain point masses, especially A=0; their shape is not generally Gaussian.",
            "Best A>=0 and worst A<=0 empirically; Gaussian fits can put artificial density across the opposite sign.",
            "Empirical G5 standardized advantages lie within [-2,2]; Gaussian tails outside are not observations.",
            "Filtered fits remove whole zero-signal groups, not all observations whose A happens to equal zero.",
            "No inference, new evaluation, reselection or change to the underlying Advantage values.",
        ],
        "source_sha256": {
            str(source / name): hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("analysis.json", "selected_reflection_advantages.csv")
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "gaussian_fits.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for seed in SEEDS:
        plot(fits, seed, output, subset="all_tasks")
    plot(fits, 42, output, subset="nonzero_signal_tasks")
    render(result, output)
    return result


def plot(fits, seed, output, *, subset):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"worst": "#c44e52", "best": "#4c72b0", "random": "#55a868"}
    x = np.linspace(-4, 3, 1800)
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), sharex=True, sharey=True, layout="constrained")
    for i, mode in enumerate(RUNS):
        for j, arm in enumerate(ARMS):
            ax = axes[i, j]
            n = None
            for method in METHODS:
                fit = fits[f"{mode}/{arm}/seed{seed}/{method}/{subset}"]
                n = fit["n_tasks"]
                mu, std = fit["mu"], fit["sigma_mle"]
                if mu is None:
                    continue
                if not std:
                    ax.axvline(mu, color=colors[method], label=f"{method}: point mass at {mu:.3f}")
                    continue
                y = np.exp(-0.5 * ((x - mu) / std) ** 2) / (std * math.sqrt(2 * math.pi))
                ax.plot(x, y, linewidth=2.1, color=colors[method], label=f"{method}: mu={mu:+.3f}, sigma={std:.3f}")
                ax.fill_between(x, y, alpha=0.07, color=colors[method])
                ax.axvline(mu, color=colors[method], linestyle="--", linewidth=0.9, alpha=0.7)
            ax.axvspan(-4, -2, color="#999999", alpha=0.07)
            ax.axvspan(2, 3, color="#999999", alpha=0.07)
            ax.axvline(-2, color="#999999", linewidth=0.8, linestyle=":")
            ax.axvline(2, color="#999999", linewidth=0.8, linestyle=":")
            ax.axvline(0, color="#777777", linewidth=0.8)
            subset_label = (
                "All 50 tasks, including zero-signal groups" if subset == "all_tasks" else "Zero-signal groups excluded"
            )
            ax.set_title(
                f"{'Thinking' if mode == 'thinking' else 'Non-thinking'} | "
                f"{'w/o GT' if arm == 'without_gt' else 'w/ GT'} | n={n} tasks/selection\n"
                f"{subset_label}",
                fontsize=11,
            )
            ax.set_xlabel("GRPO-style advantage")
            ax.set_ylabel("Fitted Gaussian probability density")
            ax.set_xlim(-4, 3)
            ax.set_ylim(bottom=0)
            ax.set_xticks(range(-4, 4))
            ax.tick_params(axis="x", labelbottom=True)
            ax.grid(alpha=0.15)
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"Gaussian approximation of selected Reflection advantages | selection seed {seed}\n"
        "Descriptive fit to empirical mean/SD; zero-point masses and bounded support are not Gaussian.\n"
        "Dashed lines: means. Shaded outer regions: beyond actual G5 advantage bounds [-2, 2].",
        fontsize=13,
    )
    filename = f"advantage_gaussian_seed{seed}" + ("_nonzero_groups" if subset != "all_tasks" else "")
    fig.savefig(output / f"{filename}.png", dpi=180)
    fig.savefig(output / f"{filename}.pdf")
    plt.close(fig)


def render(result, output):
    lines = [
        "# Reflection Advantage의 정규분포 근사",
        "",
        "기존 Advantage 값은 유지하고, 실제 선정된50개 task의 평균과 모표준편차로 "
        "각 선택 전략의 정규분포 N(mu, sigma²)를 적합했습니다. sigma는50개 Advantage 사이의 "
        "표준편차이며, Advantage 계산에 사용한 task 내5개 PCT의 표준편차와는 구분됩니다.",
        "",
        "## seed42 · 전체50task",
        "",
        "| 모델 | GT | worst mu / sigma | best mu / sigma | random mu / sigma |",
        "| --- | --- | --- | --- | --- |",
    ]
    for mode in RUNS:
        for arm in ARMS:
            fits = [result["fits"][f"{mode}/{arm}/seed42/{method}/all_tasks"] for method in METHODS]
            lines.append(
                f"| {mode} | {arm} | " + " | ".join(f"{v['mu']:+.4f} / {v['sigma_mle']:.4f}" for v in fits) + " |"
            )
    lines += [
        "",
        "![전체50task 정규분포 근사](advantage_gaussian_seed42.png)",
        "",
        "## 후보 점수가 다른 task만: 보조 그림",
        "",
        "![표준편차0그룹 제외 정규분포 근사](advantage_gaussian_seed42_nonzero_groups.png)",
        "",
        "실제 데이터는0에 질량이 몰리고, best는0이상·worst는0이하이며, "
        "G5 Advantage 범위는 [-2,2]입니다. 정규곡선이 반대 부호나 범위 밖에 부여한 밀도는 "
        "원자료에서 관측한 값이 아닙니다. 이 그림은 정규성 검정이나 실제 분포가 정규분포라는 주장이 아닙니다.",
        "",
        "원래 히스토그램과 함께 해석합니다. 기본 그림은50task를 모두 포함하며, 보조 그림은 "
        "task 내5개 후보 점수가 모두 같은 그룹을 제외했습니다. Advantage0인 점을 일괄 제거한 것이 아닙니다. "
        "동일 bank의 seed별 재선택을 새 독립 평가처럼 합산하지 않습니다.",
        "",
        "seed43·44 그림, 정규분포 모수 JSON, 입력 CSV·분석 JSON의 SHA256을 함께 보존합니다. 추가 모델 호출은 없습니다.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.source, args.output)
    print(json.dumps({"fits": len(result["fits"]), "output": str(args.output)}))
