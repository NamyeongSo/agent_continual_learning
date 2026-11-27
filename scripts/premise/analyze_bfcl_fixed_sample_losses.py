# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Compare training loss designs on exactly the same current/near/far task subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.analyze_bfcl_alignment_significance import (  # noqa: E402
    holm_adjust,
    paired_block_contributions,
    rng_for,
    sign_flip_test,
)
from scripts.premise.analyze_bfcl_similarity_extremes import mixed_indices, ranked_draws  # noqa: E402
from scripts.premise.analyze_bfcl_temporal_selection import correlations, describe, write_csv  # noqa: E402
from scripts.premise.analyze_bfcl_training_selection_hypotheses import (  # noqa: E402
    public_metadata,
    selection_probabilities,
    similarities,
)
from scripts.premise.plot_bfcl_sample_size_contours import ellipse_points  # noqa: E402

BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
BASELINE = "period_mean"
LOSS_NAMES = ("task_mean", "period_mean", "beta_000", "beta_025", "beta_050", "beta_075", "beta_100",
              "equal_three", "robust_replay")
LABELS = ("Task mean", "Period mean", r"$\beta=0$", r"$\beta=.25$", r"$\beta=.5$", r"$\beta=.75$",
          r"$\beta=1$", "Equal groups", "Robust replay")
TARGETS = ("full_train", "full_validation")
METRICS = ("validation_lift", "validation_score", "train_parent_gain", "train_proposal_lift")


def loss_metadata() -> list[dict]:
    linear = [("task_mean", (0.1, 0.5, 0.4)), ("period_mean", (0.5, 5 / 18, 2 / 9))]
    linear += [(f"beta_{int(100 * beta):03d}", (0.5, 0.5 * beta, 0.5 * (1 - beta))) for beta in BETAS]
    linear += [("equal_three", (1 / 3, 1 / 3, 1 / 3))]
    rows = [{"loss": name, "current_weight": weights[0], "near_group_weight": weights[1],
             "far_group_weight": weights[2], "evaluated_tasks": 10,
             "fixed_nonzero_weight_tasks": int(weights[0] > 0) + 5 * int(weights[1] > 0)
             + 4 * int(weights[2] > 0),
             "formula": f"{weights[0]:.16g}*C + {weights[1]:.16g}*N + {weights[2]:.16g}*F"}
            for name, weights in linear]
    rows.append({"loss": "robust_replay", "current_weight": 0.5, "near_group_weight": None,
                 "far_group_weight": None, "evaluated_tasks": 10, "fixed_nonzero_weight_tasks": None,
                 "formula": "0.5*C + 0.5*min(N,F)",
                 "nonlinear_task_usage": "All 10 tasks determine the minimum group. The active branch uses "
                 "current1+near5 or current1+far4; exact ties do not have unique linear weights."})
    return rows


def loss_scores(current: np.ndarray, near: np.ndarray, far: np.ndarray) -> dict[str, np.ndarray]:
    current, near, far = np.broadcast_arrays(current, near, far)
    if current.ndim != 2 or not all(np.isfinite(values).all() for values in (current, near, far)):
        raise ValueError("Scores must be finite draw-by-candidate matrices")
    replay = (5 * near + 4 * far) / 9
    result = {"task_mean": 0.1 * current + 0.9 * replay, "period_mean": 0.5 * current + 0.5 * replay}
    for beta in BETAS:
        result[f"beta_{int(100 * beta):03d}"] = 0.5 * current + 0.5 * (beta * near + (1 - beta) * far)
    result["equal_three"] = (current + near + far) / 3
    result["robust_replay"] = 0.5 * current + 0.5 * np.minimum(near, far)
    return result


def fixed_groups(point: dict, metadata: dict, trials: int, seed: int) -> tuple:
    if trials < 1:
        raise ValueError("Trials must be positive")
    t = point["checkpoint"]
    matrix = np.asarray(point["train_rewards"], dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 5 or matrix.shape[1] <= t:
        raise ValueError("Expected five candidates with complete current and past rewards")
    sim = similarities(point["train_order"][:t + 1], metadata)
    ranking = ranked_draws(sim[-1, :-1], trials, rng_for(seed, f"near-far-ties:{point['key']}"))
    indices = mixed_indices(ranking, 5)
    current = matrix[:, t][None, :]
    near = matrix[:, indices[:, :5]].mean(axis=-1).T
    far = matrix[:, indices[:, 5:]].mean(axis=-1).T
    digest = hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
    return current, near, far, {"key": point["key"], "checkpoint": t, "draws": trials,
                               "near_tasks": 5, "far_tasks": 4, "current_tasks": 1,
                               "past_index_draws_sha256": digest}


def calculate(model: dict, metadata: dict, matched: set[str], trials: int, seed: int) -> tuple:
    groups = ("all", 42, 43, 44)
    counts = {(cohort, group): 0 for cohort in ("cp36", "cp50") for group in groups}
    sums = {(cohort, group, name, metric): np.zeros(trials) for cohort, group in counts
            for name in LOSS_NAMES for metric in METRICS}
    rho_sums = {(cohort, group, name, target): np.zeros(trials) for cohort, group in counts
                for name in LOSS_NAMES for target in TARGETS}
    coverage = {(cohort, group, target): np.zeros(trials) for cohort, group in counts for target in TARGETS}
    checkpoint_rows, samples = [], []
    for point in model["points"]:
        current, near, far, sample = fixed_groups(point, metadata, trials, seed)
        samples.append({"model": model["label"], **sample})
        scores = loss_scores(current, near, far)
        probabilities = {name: selection_probabilities(values) for name, values in scores.items()}
        validation = np.asarray(point["scores"]["full_validation"])
        train = np.asarray(point["scores"]["full_train"])
        cohorts = ["cp50", "cp36"] if point["key"] in matched else ["cp50"]
        point_rhos = {(name, target): correlations(values, np.asarray(point["scores"][target]))
                      for name, values in scores.items() for target in TARGETS}
        for cohort in cohorts:
            for group in ("all", point["seed"]):
                counts[cohort, group] += 1
        for name, probability in probabilities.items():
            val_score, train_score = probability @ validation, probability @ train
            values = {"validation_lift": val_score - validation.mean(), "validation_score": val_score,
                      "train_parent_gain": train_score - point["parent_scores"]["full_train"],
                      "train_proposal_lift": train_score - train.mean()}
            total_variation = 0.5 * np.abs(probability - probabilities[BASELINE]).sum(axis=1)
            row = {"model": model["label"], "key": point["key"], "seed": point["seed"],
                   "checkpoint": point["checkpoint"], "matched36": point["key"] in matched, "loss": name,
                   "evaluated_tasks": 10, **{metric: float(value.mean()) for metric, value in values.items()},
                   "selection_total_variation_vs_period": float(total_variation.mean()),
                   "current_binary": bool(np.isin(current, [0, 1]).all()),
                   "current_informative": bool(np.ptp(current) > 1e-12),
                   "selected_current_reward": float((probability * current).sum(axis=1).mean()),
                   "selected_near_reward": float((probability * near).sum(axis=1).mean()),
                   "selected_far_reward": float((probability * far).sum(axis=1).mean()),
                   "validation_proposal_mean": float(validation.mean()),
                   "train_proposal_parent_gain": float(train.mean() - point["parent_scores"]["full_train"]),
                   **{f"candidate_{i + 1}_probability": float(probability[:, i].mean()) for i in range(5)}}
            for target in TARGETS:
                row[f"rho_{target}"] = describe(point_rhos[name, target])["mean"]
                row[f"rho_{target}_defined_fraction"] = float(np.isfinite(point_rhos[name, target]).mean())
            checkpoint_rows.append(row)
            for cohort in cohorts:
                for group in ("all", point["seed"]):
                    for metric, value in values.items():
                        sums[cohort, group, name, metric] += value
        for target in TARGETS:
            common = np.logical_and.reduce([np.isfinite(point_rhos[name, target]) for name in LOSS_NAMES])
            for cohort in cohorts:
                for group in ("all", point["seed"]):
                    coverage[cohort, group, target] += common
                    for name in LOSS_NAMES:
                        rho_sums[cohort, group, name, target] += np.where(common, point_rhos[name, target], 0)
    aggregate = []
    for (cohort, group), count in counts.items():
        if not count:
            continue
        for name in LOSS_NAMES:
            row = {"model": model["label"], "cohort": cohort, "seed": group, "loss": name, "checkpoints": count}
            row.update({metric: describe(sums[cohort, group, name, metric] / count) for metric in METRICS})
            for target in TARGETS:
                defined = coverage[cohort, group, target]
                values = np.full(trials, np.nan)
                np.divide(rho_sums[cohort, group, name, target], defined, out=values, where=defined > 0)
                row[f"rho_{target}"] = describe(values)
                row[f"rho_{target}_common_checkpoints"] = describe(defined)
            aggregate.append(row)
    return aggregate, checkpoint_rows, samples


def tests(rows: list[dict], permutations: int, seed: int) -> list[dict]:
    results = []
    for model in sorted({row["model"] for row in rows}):
        for cohort in ("cp36", "cp50"):
            subset = [row for row in rows if row["model"] == model and (cohort == "cp50" or row["matched36"])]
            baseline = sorted([row for row in subset if row["loss"] == BASELINE],
                              key=lambda row: (row["seed"], row["checkpoint"]))
            if not baseline:
                continue
            for block_size in (2, 3, 4, 5):
                family = []
                for name in LOSS_NAMES:
                    if name == BASELINE:
                        continue
                    compared = {row["key"]: row for row in subset if row["loss"] == name}
                    differences = np.asarray([compared[row["key"]]["validation_lift"] - row["validation_lift"]
                                              for row in baseline])
                    contributions, _ = paired_block_contributions(baseline, baseline, differences, block_size)
                    result = sign_flip_test(contributions, permutations,
                                            rng_for(seed, f"fixed-loss:{model}:{cohort}:{name}:{block_size}"))
                    difference = result.pop("mean_rho_difference")
                    family.append({"model": model, "cohort": cohort, "loss": name, "reference": BASELINE,
                                   "block_size": block_size, "difference": difference, **result})
                for row, adjusted in zip(family, holm_adjust([row["p_two_sided"] for row in family]), strict=True):
                    row["p_holm"] = adjusted
                    row["family_size"] = len(family)
                results.extend(family)
    return results


def flat_rows(rows: list[dict]) -> list[dict]:
    return [{new_key: new_value for key, value in row.items()
             for new_key, new_value in (([(f"{key}_{stat}", number) for stat, number in value.items()])
                                       if isinstance(value, dict) else [(key, value)])} for row in rows]


def plot(summary: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.65,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    model_names = ("Qwen", "Gemma", "OSS")
    colors = ("#287C85", "#A34F65", "#AF8238")

    def select(model, cohort, group, name):
        return next(row for row in summary["aggregate"] if (row["model"], row["cohort"], row["seed"], row["loss"])
                    == (model, cohort, group, name))

    def save(figure, stem):
        for extension in ("pdf", "png"):
            figure.savefig(output / f"{stem}.{extension}", dpi=400, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {stem}.pdf", flush=True)

    positions = np.arange(len(LOSS_NAMES))
    for cohort in ("cp36", "cp50"):
        figure, axes = plt.subplots(2, 3, figsize=(8.2, 4.5), sharex=True, sharey="row")
        figure.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.23, hspace=0.17, wspace=0.12)
        for column, model in enumerate(model_names):
            for index, metric in enumerate(("rho_full_validation", "validation_lift")):
                axis = axes[index, column]
                factor = 100 if metric == "validation_lift" else 1
                for group, color in zip((42, 43, 44), colors, strict=True):
                    y = [select(model, cohort, group, name)[metric]["mean"] for name in LOSS_NAMES]
                    y = np.asarray([np.nan if value is None else factor * value for value in y])
                    axis.plot(positions, y, color=color, marker="o", markersize=2.4, linewidth=0.8,
                              label=f"Seed {group}")
                y = [factor * select(model, cohort, "all", name)[metric]["mean"] for name in LOSS_NAMES]
                axis.plot(positions, y, color="#333333", linewidth=1.5, label="Mean")
                axis.axhline(0, color="#AAAAAA", linewidth=0.5)
                axis.axvline(1, color="#BBBBBB", linewidth=0.6, linestyle=":")
                axis.set_xticks(positions, LABELS, rotation=65, ha="right", fontsize=6.5)
                axis.tick_params(labelsize=7)
                if index == 0:
                    axis.set_title(model, fontsize=9)
                if column == 0:
                    axis.set_ylabel(r"$\rho$" if index == 0 else "Validation lift (pp)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, ncol=4, loc="upper center", frameon=False, fontsize=7)
        save(figure, f"fixed_loss_overview_{cohort}")

    figure, axes = plt.subplots(2, 3, figsize=(7.5, 4.2), sharex=True, sharey="row")
    figure.subplots_adjust(left=0.10, right=0.99, top=0.83, bottom=0.15, hspace=0.16, wspace=0.16)
    for column, model in enumerate(model_names):
        for index, metric in enumerate(("validation_lift", "rho_full_validation")):
            axis = axes[index, column]
            factor = 100 if metric == "validation_lift" else 1
            for group, color in zip((42, 43, 44), colors, strict=True):
                y = [select(model, "cp50", group, f"beta_{int(100 * beta):03d}")[metric]["mean"]
                     for beta in BETAS]
                axis.plot(BETAS, [np.nan if v is None else factor * v for v in y], color=color,
                          linewidth=0.9, marker="o", markersize=2.6, label=f"Seed {group}")
            y = [factor * select(model, "cp50", "all", f"beta_{int(100 * beta):03d}")[metric]["mean"]
                 for beta in BETAS]
            axis.plot(BETAS, y, color="#333333", linewidth=1.5, label="Mean")
            baseline = factor * select(model, "cp50", "all", BASELINE)[metric]["mean"]
            axis.axhline(baseline, color="#777777", linewidth=0.8, linestyle=":", label="Period mean")
            axis.axvline(5 / 9, color="#BBBBBB", linewidth=0.5, linestyle=":")
            axis.axhline(0, color="#AAAAAA", linewidth=0.5)
            axis.set_xticks(BETAS)
            axis.tick_params(labelsize=7)
            if index == 0:
                axis.set_title(model, fontsize=9)
            if column == 0:
                axis.set_ylabel("Validation lift (pp)" if index == 0 else r"$\rho$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=5, loc="upper center", frameon=False, fontsize=7)
    figure.supxlabel(r"Similar-group weight $\beta$ (current/replay weights $1/2,1/2$)", y=0.02, fontsize=8)
    save(figure, "beta_sweep_cp50")

    figure, axes = plt.subplots(1, 3, figsize=(8.2, 2.3), sharey=True)
    figure.subplots_adjust(left=0.08, right=0.99, top=0.87, bottom=0.39, wspace=0.17)
    for column, model in enumerate(model_names):
        axis = axes[column]
        for target, color in zip(TARGETS, colors, strict=False):
            y = [select(model, "cp50", "all", name)[f"rho_{target}"]["mean"] for name in LOSS_NAMES]
            axis.plot(positions, y, color=color, marker="o", markersize=2.5, linewidth=1,
                      label="Full train" if target == "full_train" else "Full validation")
        axis.axhline(0, color="#AAAAAA", linewidth=0.5)
        axis.set_xticks(positions, LABELS, rotation=65, ha="right", fontsize=6.5)
        axis.set_title(model, fontsize=9)
    axes[0].set_ylabel(r"$\rho$")
    axes[1].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.5), ncol=2)
    save(figure, "correlation_full_train_and_validation")

    for target in TARGETS:
        target_color, target_fill = ("#287C85", "#EAF4F5") if target == "full_train" else ("#A34F65", "#F9EEF1")
        figure, axes = plt.subplots(3, 5, figsize=(8.1, 4.8), sharex=True, sharey=True)
        figure.subplots_adjust(left=0.09, right=0.99, bottom=0.10, top=0.87, wspace=0.20, hspace=0.16)
        for index, model in enumerate(model_names):
            for column, beta in enumerate(BETAS):
                axis = axes[index, column]
                for number, (name, color) in enumerate(((BASELINE, "#888888"),
                                                        (f"beta_{int(100 * beta):03d}", target_color))):
                    rho = select(model, "cp50", "all", name)[f"rho_{target}"]["mean"]
                    subscript = "P" if name == BASELINE else r"\beta"
                    label = rf"$\rho_{{{subscript}}}={rho:+.3f}$" if rho is not None else r"$\rho=\mathrm{N/A}$"
                    axis.text(0.03, 0.96 - (1 - number) * 0.17, label, transform=axis.transAxes,
                              color=color, fontsize=6.4, va="top", zorder=4,
                              bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.1})
                    if rho is not None and abs(rho) < 1:
                        covariance = np.array([[1, rho], [rho, 1]])
                        inner, outer = (ellipse_points(covariance, mass, 121) for mass in (0.5, 0.9))
                        if name != BASELINE:
                            axis.fill(inner[:, 0], inner[:, 1], color=target_fill, linewidth=0, zorder=1)
                        style = ":" if name == BASELINE else "-"
                        axis.plot(inner[:, 0], inner[:, 1], color=color, linewidth=0.8, linestyle=style, zorder=2)
                        axis.plot(outer[:, 0], outer[:, 1], color=color, linewidth=0.6,
                                  linestyle=":" if name == BASELINE else "--", zorder=2)
                axis.axhline(0, color="#DDDDDD", linewidth=0.4)
                axis.axvline(0, color="#DDDDDD", linewidth=0.4)
                axis.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), xticks=[-2, 0, 2], yticks=[-2, 0, 2])
                axis.set_aspect("equal")
                axis.tick_params(labelsize=6, length=2)
                if index == 0:
                    axis.set_title(rf"$\beta={beta:g}$", fontsize=8)
                if column == 0:
                    axis.set_ylabel(model, fontsize=8)
        handles = axes[0, 0].plot([], [], color=target_color, label=r"$\beta$ loss")
        handles += axes[0, 0].plot([], [], color="#888888", linestyle=":", label="Period mean")
        figure.legend(handles, [r"$\beta$ loss", "Period mean"], loc="upper center", ncol=2, frameon=False)
        xlabel = r"$z(\delta_t^i)$" if target == "full_train" else r"$z(s_{t,\mathrm{val50}}^i)$"
        figure.supxlabel(xlabel, y=0.01, fontsize=9)
        figure.supylabel(r"$z(s_t^i)$", x=0.015, fontsize=9)
        save(figure, f"bivariate_{target}")


def write_report(summary: dict, output: Path) -> None:
    def row(model, name, cohort="cp50", group="all"):
        return next(x for x in summary["aggregate"]
                    if (x["model"], x["cohort"], x["seed"], x["loss"]) == (model, cohort, group, name))

    lines = ["# 동일한 task에서 loss 구성만 바꾼 selection 분석", "",
             "현재 task 1개 + 유사 과거 5개 + 비유사 과거 4개를 모든 조건에 동일하게 고정하고, "
             "학습 목적함수에 해당하는 score 집계 방식만 비교했다. 새 모델 호출은 없다. "
             "가중 reward의 최대화는 그 음수를 training loss로 두었을 때의 최소화와 동치다.", "",
             "## 설계", "", "`C`, `N`, `F`는 각각 현재 task reward, 유사 과거 5개의 평균, "
             "비유사 과거 4개의 평균이다. 모든 후보와 loss는 같은 draw의 task 10개를 공유한다. "
             "유사도는 현재/과거 first request의 TF-IDF cosine과 공개 도구 class의 Jaccard 평균이다. "
             "정확한 유사도 동점만 무작위화하며 seed=20261021, 10,000 draw를 사용했다. "
             "미래/validation task의 정보로 sample이나 weight를 정하지 않았다.", "",
             "| Loss | 선택 score | 평가한 task | 고정 비영 가중치 task |", "| --- | --- | ---: | ---: |"]
    for spec in summary["losses"]:
        count = spec["fixed_nonzero_weight_tasks"]
        lines.append(f"| {spec['loss']} | `{spec['formula']}` | 10 | {count if count is not None else '비선형'} |")
    lines += ["", "β=0/1은 task를 평가한 budget은 같지만 실제 score에서 한 집단을 무시하므로 "
              "유효 task 수는 5/6개다. Robust replay는 두 과거 집단 중 낮은 평균을 채택한다. "
              "최솟값을 판단하기 위해 10개가 필요하며, 활성 branch와 동점 때문에 고정 선형 weight로 "
              "표현할 수 없다. Beta sweep에서 유사 task 한 개의 weight는 beta/10, 비유사 task "
              "한 개의 weight는 (1-beta)/8이다. Beta=5/9가 period mean과 같은 sample 평균이며, "
              "beta=1/2는 과거 두 집단의 합산 weight를 같게 만드는 loss다.", "",
              "Robust score의 음수는 0.5*L_current + 0.5*max(L_near,L_far)이다. 즉 두 과거 집단의 "
              "평균 loss 중 더 나쁜 쪽을 줄이는 목적함수다. 이 오프라인 분석은 기존 후보들 사이에서 "
              "어떤 경험적 목적함수로 선택하느냐를 바꾸며, sample 선택과 loss weighting을 분리한다.",
              "", "## 전체 50 checkpoint 결과", "",
              "Validation lift는 선택 후보의 full validation 50-task 점수에서 5개 proposal의 "
              "평균 validation 점수를 뺀 값이다. 부모 대비 validation gain이 아니다. "
              "Train gain은 부모의 full train 점수 대비 실제 변화다. 단위는 pp다.", "",
              "| Loss | Qwen val lift | Gemma val lift | OSS val lift | Qwen train gain | Gemma train gain | "
              "OSS train gain |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in LOSS_NAMES:
        numbers = [100 * row(model, name)[metric]["mean"] for metric in ("validation_lift", "train_parent_gain")
                   for model in ("Qwen", "Gemma", "OSS")]
        lines.append(f"| {name} | " + " | ".join(f"{x:+.3f}" for x in numbers) + " |")
    lines += ["", "Qwen은 period mean 대비 beta_075의 validation lift 증가가 약 +0.018pp, "
              "Gemma는 beta_025의 증가가 약 +0.014pp로 작다. OSS는 동일 sample을 고정해도 "
              "유사 집단 가중치를 낮추면 lift 부호가 바뀐다. Period mean의 -0.038pp에서 beta_025는 "
              "+0.129pp, beta_000은 +0.269pp다. 이는 sampling 외에도 loss 집계가 결과를 좌우함을 "
              "보이지만 validation 최댓값을 본 뒤 보편적인 최적 weight로 채택할 근거는 아니다. "
              "CP36에서는 OSS beta_000도 +0.056pp이고 다른 여덟 loss는 음수여서 cohort 의존성이 "
              "크다. Robust replay는 Qwen/Gemma의 period mean보다 낮았고 OSS에서는 작게 높았다.",
              "", "### 현재 task의 가중치는 사실상 gate가 될 수 있음", "",
              "현재 reward가 binary이고 current/replay 가중치가 1/2:1/2이면, 현재 성공 후보의 "
              "score는 최소 0.5이고 실패 후보의 score는 최대 0.5다. 현재 성공+replay 0과 현재 "
              "실패+replay 1의 극단적 동점 외에는 현재 성공을 우선한다. 따라서 beta sweep과 "
              "robust replay는 주로 이 gate 안에서 과거 집단의 우선순위를 바꾸며, 현재 task의 "
              "과적합을 일반적으로 해소하지 못한다. Task mean의 current weight=0.1, equal-three의 "
              "current weight=1/3은 다른 절충을 허용한다. 다만 현재 점수가 후보 사이에서 같으면 "
              "현재항은 순위에 영향을 주지 않는다.", ""]
    flat_counts = []
    for model in ("Qwen", "Gemma", "OSS"):
        points = [x for x in summary["checkpoint_scores"] if x["model"] == model and x["loss"] == BASELINE]
        flat_counts.append(f"{model} {sum(not x['current_informative'] for x in points)}/{len(points)}")
    lines += ["현재 reward가 후보 사이에서 동일한 checkpoint 수: " + ", ".join(flat_counts) + ". "
              "Gate가 있는 loss들에서도 모델별 차이가 날 수 있는 이유를 현재 신호와 과거 신호로 "
              "구분해서 봐야 한다."]
    lines += ["", "### Seed별 validation lift", "", "| Model | Loss | Seed42 | Seed43 | Seed44 |",
              "| --- | --- | ---: | ---: | ---: |"]
    for model in ("Qwen", "Gemma", "OSS"):
        for name in LOSS_NAMES:
            numbers = [100 * row(model, name, group=seed)["validation_lift"]["mean"] for seed in (42, 43, 44)]
            lines.append(f"| {model} | {name} | " + " | ".join(f"{x:+.3f}" for x in numbers) + " |")
    lines += ["", "## Coverage와 해석", "",
              "모델마다 원래 CP50개를 사용했고 seed별 CP 수는 17/17/16개다. CP36은 앞선 시점 비교의 "
              "공통 CP(각 seed 12개)를 유지한 보조 집합이다. CP 평균은 CP별 동일 가중치다. "
              "Correlation은 CP 내 후보 5개의 Pearson $\\rho$이며, 매 draw마다 모든 9개 loss에서 "
              "동시에 정의되는 CP만 평균했다. 따라서 lift의 CP 집합보다 작다.", "",
              "| Model | cp36 공통 rho CP | cp50 공통 rho CP |", "| --- | ---: | ---: |"]
    for model in ("Qwen", "Gemma", "OSS"):
        values = [row(model, BASELINE, cohort)["rho_full_validation_common_checkpoints"]["mean"]
                  for cohort in ("cp36", "cp50")]
        lines.append(f"| {model} | {values[0]:.2f}/36 | {values[1]:.2f}/50 |")
    minimum = min(test["p_holm"] for test in summary["paired_block_tests"])
    significant = [test for test in summary["paired_block_tests"] if test["p_holm"] < 0.05]
    lines += ["", "Sampling SD는 similarity 동점 무작위화에 따른 변동이며 seed/CP 불확실성의 confidence "
              "interval이 아니다. Validation 결과를 보기 전에 이번 loss 목록과 비교를 plan에 고정했지만, "
              "기존 결과를 이미 본 데이터의 후속 탐색이며 새 held-out 확인은 아니다. 기존 모델 오류와 "
              "오류의 0점 처리를 그대로 유지했다.", "", "## 검정", "",
              "기준 loss `period_mean`과 나머지 8개를 CP 평균 lift로 paired 비교했다. Seed 안의 연속 "
              "CP block(크기 2/3/4/5)을 뒤집는 양측 검정이며, 각 model/cohort/block 설정에 Holm8을 "
              "적용했다. Draw를 독립 실험으로 세지 않았다. Block 사이 부호의 독립적 교환가능성을 "
              "가정한다.", "", f"전체 보정 p 최솟값은 {minimum:.4g}, 보정 p<0.05인 비교는 "
              f"{len(significant)}개다. 네 block 설정 전체를 함께 확인해야 하며 하나만 골라 "
              "유의성을 주장하면 안 된다.", "", "## 파일", "",
              "- [CP50 gain/correlation](figures/fixed_loss_overview_cp50.pdf), "
              "[CP36 gain/correlation](figures/fixed_loss_overview_cp36.pdf)",
              "- [β sweep와 seed별 결과](figures/beta_sweep_cp50.pdf)",
              "- [Full train·validation correlation](figures/correlation_full_train_and_validation.pdf)",
              "- [Full train bivariate](figures/bivariate_full_train.pdf), "
              "[full validation bivariate](figures/bivariate_full_validation.pdf): 5개 beta loss를 "
              "period mean과 비교. 각 cell의 공통-coverage 평균 rho를 표기했다.",
              "- [summary](fixed_loss_summary.json), [집계 CSV](scores.csv), "
              "[CP CSV](checkpoint_scores.csv), [paired 검정](paired_block_tests.csv)",
              "- [고정 subset draw digest](sample_draw_digests.csv), [사전 plan](analysis_plan.json)", "",
              "모든 그림은 같은 이름의 PNG도 저장했다. 입력 SHA256, 정확한 loss 계수와 sampling "
              "digest를 summary에 기록했다. Bivariate 그림은 평균 rho를 영평균·단위분산 Gaussian의 "
              "공분산 [[1,rho],[rho,1]]에 대입한 50%/90% 등고선이다. 관측 데이터의 정규성을 "
              "확인한 결과가 아니며 이 가정으로 검정하지 않았다."]
    (output / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    plan = args.output / "analysis_plan.json"
    if not plan.is_file():
        raise ValueError("Write the analysis plan before running loss comparisons")
    raw = json.loads(args.source.read_text())
    matched = set(json.loads(args.matched.read_text())["fixed_checkpoint_keys"])
    wanted = {task for model in raw["models"] for point in model["points"] for task in point["train_order"]}
    metadata = public_metadata(args.metadata, wanted)
    aggregate, checkpoint_rows, samples = [], [], []
    for model in raw["models"]:
        print(f"Computing {model['label']}: fixed current1/near5/far4, nine losses", flush=True)
        scores, rows, draws = calculate(model, metadata, matched, args.trials, args.seed)
        aggregate.extend(scores)
        checkpoint_rows.extend(rows)
        samples.extend(draws)
    # Every model uses the identical chronological task stream and identical sampled inputs.
    expected = {}
    for sample in samples:
        previous = expected.setdefault(sample["key"], sample["past_index_draws_sha256"])
        if previous != sample["past_index_draws_sha256"]:
            raise ValueError(f"Unpaired task draws across models: {sample['key']}")
    print("Computing Holm8 paired block contrasts", flush=True)
    summary = {"source": str(args.source.resolve()),
               "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
               "task_metadata_source": str(args.metadata.resolve()),
               "task_metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
               "matched_source": str(args.matched.resolve()),
               "matched_sha256": hashlib.sha256(args.matched.read_bytes()).hexdigest(),
               "analysis_plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
               "seed": args.seed, "trials": args.trials, "new_model_calls": 0, "losses": loss_metadata(),
               "cohort_keys": {"cp36": sorted(matched), "cp50": sorted(expected)},
               "original_model_errors": {model["label"]: model["original_errors"] for model in raw["models"]},
               "uncertainty": "Draw SD reflects exact similarity-tie randomization only, not population CI. "
               "Tests use CP means with chronological within-seed sign-flipped blocks and Holm8 per model, "
               "cohort, block size. Previously inspected validation; exploratory, not new heldout confirmation.",
               "aggregate": aggregate, "checkpoint_scores": checkpoint_rows, "sample_draw_digests": samples,
               "paired_block_tests": tests(checkpoint_rows, 100000, args.seed)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "fixed_loss_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_csv(args.output / "scores.csv", flat_rows(aggregate))
    write_csv(args.output / "checkpoint_scores.csv", checkpoint_rows)
    write_csv(args.output / "sample_draw_digests.csv", samples)
    write_csv(args.output / "paired_block_tests.csv", summary["paired_block_tests"])
    plot(summary, args.output / "figures")
    write_report(summary, args.output)
    for model in ("Qwen", "Gemma", "OSS"):
        values = [(row["loss"], round(100 * row["validation_lift"]["mean"], 5)) for row in aggregate
                  if row["model"] == model and row["cohort"] == "cp50" and row["seed"] == "all"]
        print(model, values, flush=True)


if __name__ == "__main__":
    main()
