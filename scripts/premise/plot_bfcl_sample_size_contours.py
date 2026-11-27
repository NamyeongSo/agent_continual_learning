# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Show each n-of-10 BFCL alignment as a moment-matched bivariate Gaussian."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.plot_bfcl_alignment_sample_sizes import COLORS, SELECTORS, SHADE_COLORS, TARGETS  # noqa: E402

MASS_LEVELS = (0.5, 0.9)
X_LABELS = {"full_train": r"$z(\delta_t^i)$", "full_validation": r"$z(s_{t,\mathrm{val50}}^i)$"}
Y_LABEL = r"$z(s_{t,n}^i)$"


def gaussian_moments(rho: float | None) -> dict:
    if rho is None:
        return {"defined": False, "reason": "No paired-defined nonempty trials"}
    if not np.isfinite(rho) or abs(rho) > 1:
        raise ValueError("Gaussian correlation must be finite and between -1 and 1")
    covariance = np.array([[1.0, rho], [rho, 1.0]])
    if abs(rho) == 1:
        return {"defined": False, "reason": "Singular covariance; closed two-dimensional contours undefined",
                "rho": rho, "mean": [0.0, 0.0], "covariance": covariance.tolist()}
    return {"defined": True, "rho": rho, "mean": [0.0, 0.0], "covariance": covariance.tolist()}


def ellipse_points(covariance: np.ndarray, mass: float, count: int = 361) -> np.ndarray:
    """A two-dimensional Gaussian has chi-square(2) squared Mahalanobis distances."""
    if covariance.shape != (2, 2) or not np.allclose(covariance, covariance.T):
        raise ValueError("Expected a symmetric 2x2 covariance")
    eigenvalues, vectors = np.linalg.eigh(covariance)
    if eigenvalues.min() <= 0 or not 0 < mass < 1:
        raise ValueError("Closed Gaussian contours require positive eigenvalues and 0 < mass < 1")
    angles = np.linspace(0, 2 * np.pi, count)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    return circle @ np.diag(np.sqrt(eigenvalues)) @ vectors.T * np.sqrt(-2 * np.log1p(-mass))


def fit_models(summary: dict) -> list[dict]:
    models = []
    for model in summary["models"]:
        if not model["n10_matches_existing_summary"]:
            raise ValueError(f"Source n=10 was not validated: {model['label']}")
        sizes = []
        if [entry["n"] for entry in model["sizes"]] != list(range(1, 11)):
            raise ValueError("Expected all ten evaluation sample sizes")
        for entry in model["sizes"]:
            targets = {}
            for target in TARGETS:
                values = entry["targets"][target]
                targets[target] = {
                    "paired_checkpoint_count_mean": values["paired_checkpoint_count_mean"],
                    "defined_trials": values["defined_trials"], "total_trials": values["total_trials"],
                    **{selector: gaussian_moments(values[selector]["mean_pearson"]) for selector in SELECTORS},
                }
            sizes.append({"n": entry["n"], "targets": targets})
        models.append({"label": model["label"], "errors": model["errors"], "evaluations": model["evaluations"],
                       "checkpoint_count": len(model["checkpoints"]), "sizes": sizes})
    return models


def draw_contours(axis, moments: dict) -> None:
    for index, (selector, color, shade_color) in enumerate(zip(SELECTORS, COLORS, SHADE_COLORS, strict=True)):
        fit = moments[selector]
        rho_label = rf"$\rho={fit['rho']:+.3f}$" if fit.get("rho") is not None else r"$\rho=\mathrm{N/A}$"
        axis.text(0.04, 0.97 - 0.14 * index, rho_label, transform=axis.transAxes, color=color,
                  fontsize=6, ha="left", va="top", zorder=4,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.3})
        if not fit["defined"]:
            continue
        covariance = np.asarray(fit["covariance"])
        inner = ellipse_points(covariance, MASS_LEVELS[0])
        outer = ellipse_points(covariance, MASS_LEVELS[1])
        axis.fill(inner[:, 0], inner[:, 1], color=shade_color, linewidth=0, zorder=1)
        axis.plot(inner[:, 0], inner[:, 1], color=color, linewidth=1.1, zorder=3)
        axis.plot(outer[:, 0], outer[:, 1], color=color, linewidth=0.9, linestyle="--", zorder=3)
    axis.axhline(0, color="#D6D6D6", linewidth=0.5, zorder=2)
    axis.axvline(0, color="#D6D6D6", linewidth=0.5, zorder=2)
    axis.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), xticks=[-2, 0, 2], yticks=[-2, 0, 2])
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(length=2, pad=2, labelsize=7)
    for spine in axis.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#A0A0A0")


def plot(models: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    handles = [Line2D([0], [0], color=color, linewidth=1.2, label=label)
               for color, label in zip(COLORS, ("Train", "Validation"), strict=True)]

    def save(figure, stem):
        for suffix in ("pdf", "png"):
            figure.savefig(output / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(figure)

    for n in range(1, 11):
        figure, axes = plt.subplots(len(models), 2, figsize=(6.7, 2.15 * len(models) + 0.65),
                                    sharex=True, sharey=True, squeeze=False)
        figure.subplots_adjust(left=0.08, right=0.99, top=0.875, bottom=0.075, hspace=0.27, wspace=0.06)
        for row, model in enumerate(models):
            for column, target in enumerate(TARGETS):
                axis = axes[row, column]
                draw_contours(axis, model["sizes"][n - 1]["targets"][target])
                target_label = "Full train" if target == "full_train" else "Full validation"
                axis.set_title(f"{model['label']} · {target_label}", fontsize=8.5, pad=5)
                if column == 0:
                    axis.set_ylabel(Y_LABEL, fontsize=9, labelpad=2)
                if row == len(models) - 1:
                    axis.set_xlabel(X_LABELS[target], fontsize=9, labelpad=2)
        figure.suptitle(rf"$n={n}$", fontsize=10, y=0.985)
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False)
        save(figure, f"bivariate_n{n:02d}")

    for model in models:
        slug = "".join(char.lower() if char.isalnum() else "_" for char in model["label"]).strip("_")
        for target in TARGETS:
            figure, axes = plt.subplots(2, 5, figsize=(7.1, 3.7), sharex=True, sharey=True)
            figure.subplots_adjust(left=0.075, right=0.99, top=0.81, bottom=0.15, wspace=0.18, hspace=0.34)
            for axis, size in zip(axes.flat, model["sizes"], strict=True):
                draw_contours(axis, size["targets"][target])
                axis.set_title(rf"$n={size['n']}$", fontsize=8, pad=3)
            figure.suptitle(model["label"], fontsize=10, y=1.005)
            figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=2, frameon=False)
            figure.supxlabel(X_LABELS[target], fontsize=9, y=0.03)
            figure.supylabel(Y_LABEL, fontsize=9, x=0.012)
            save(figure, f"{slug}_bivariate_{target}")

    for target in TARGETS:
        figure, axes = plt.subplots(2 * len(models), 5, figsize=(7.1, 2.85 * len(models) + 0.7),
                                    sharex=True, sharey=True, squeeze=False)
        figure.subplots_adjust(left=0.075, right=0.99, top=0.94, bottom=0.06, wspace=0.18, hspace=0.43)
        for index, model in enumerate(models):
            for number, size in enumerate(model["sizes"]):
                axis = axes[2 * index + number // 5, number % 5]
                draw_contours(axis, size["targets"][target])
                title = rf"$n={size['n']}$"
                if number == 0:
                    title = model["label"] + " · " + title
                axis.set_title(title, fontsize=8, pad=3)
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False)
        figure.supxlabel(X_LABELS[target], fontsize=9, y=0.025)
        figure.supylabel(Y_LABEL, fontsize=9, x=0.012)
        save(figure, f"bivariate_overview_{target}")


def run(source: Path, output: Path) -> None:
    raw = source.read_bytes()
    summary = json.loads(raw)
    models = fit_models(summary)
    output.mkdir(parents=True, exist_ok=True)
    plot(models, output)
    metadata = {
        "source": str(source.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "new_model_calls": 0,
        "method": "Moment-matched bivariate Gaussian of within-checkpoint standardized candidate pairs. "
                  "For each resampling trial, assign equal mass to paired-defined checkpoints and their "
                  "five candidates; average over nonempty trials. Means are zero, marginal variances one, "
                  "and covariance is the existing mean within-checkpoint Pearson correlation.",
        "standardization": "Within each checkpoint and each sampled n-task pool; population standard deviation.",
        "normality_tested": False,
        "interpretation": "Descriptive Gaussian approximation of alignment, not a test of empirical normality. "
                          "Standardization removes proposal mean gain and scale. These are pre-selection candidates.",
        "contours": {"solid_and_fill": MASS_LEVELS[0], "dashed": MASS_LEVELS[1],
                     "meaning": "Enclosed probability mass under the fitted 2D Gaussian, not uncertainty bands"},
        "axes": {"x_train": "Within-CP z of full50 train gain relative to shared parent",
                 "x_validation": "Within-CP z of full50 validation reward; parent gain is unmeasured",
                 "y": "Within-CP z of evaluation score from the sampled n tasks"},
        "color_scheme": {"train": COLORS[0], "validation": COLORS[1],
                         "train_fill": SHADE_COLORS[0], "validation_fill": SHADE_COLORS[1]},
        "rho_annotations": "Each panel labels Train and Validation rho in the matching curve color, to three decimals.",
        "errors": "Original zero-scored execution errors retained; checkpoint eligibility matches source rho curves.",
        "models": models,
    }
    (output / "bivariate_sample_size_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    notes = ["# n별 이변량 정규분포 등고선", "",
             "- `bivariate_n01.pdf/png` … `bivariate_n10.pdf/png`: 각 n에서 세 모델과 두 target 비교.",
             "- `qwen_bivariate_full_train.*`, `qwen_bivariate_full_validation.*`: Qwen의 n=1…10 비교.",
             "- `oss_bivariate_full_train.*`, `oss_bivariate_full_validation.*`: OSS의 n=1…10 비교.",
             "- `gemma_bivariate_full_train.*`, `gemma_bivariate_full_validation.*`: Gemma의 n=1…10 비교.",
             "- `bivariate_overview_full_train.*`, `bivariate_overview_full_validation.*`: 세 모델의 전체 n 비교.",
             "- `bivariate_sample_size_summary.json`: 적합 평균·공분산·rho·유효 CP/반복 수·원본 해시.", "",
             "각 CP의 후보 5개에서 full target과 n-task 평가 점수를 각각 표준화합니다.",
             "CP별 평균 0, 분산 1이며 부모 점수를 빼도 train 표준화 값은 동일합니다.",
             "각 반복의 유효 CP에 동일한 가중치를 부여하고 비어 있지 않은 반복을 평균한 분포의 모멘트를 사용합니다.",
             "정규근사의 평균은 (0,0), 공분산은 [[1,rho],[rho,1]]입니다.",
             "rho는 기존 n별 correlation 곡선과 정확히 같은 값입니다.",
             "각 패널의 두 rho 숫자는 Train과 Validation의 곡선 색으로 소수점 셋째 자리까지 표시합니다.",
             "실선과 채움은 정규근사 확률질량 50%, 점선은 90% 등고선입니다. STD band나 신뢰구간이 아닙니다.",
             "실제 표본이 정규분포라는 가설을 검증한 그림이 아니며, 표준화 때문에 raw 평균 gain은 나타내지 않습니다.",
             "실제 선택 후 후보의 분포가 아니라 선택 전 후보 5개의 alignment입니다.",
             "Validation x축은 full50 점수이며 측정되지 않은 validation delta를 사용하지 않습니다.",
             "작은 n의 유효 CP 부족 및 기존 실행 오류는 원래 correlation 분석과 동일합니다.", ""]
    (output / "bivariate_README.md").write_text("\n".join(notes))
    print(f"Saved n=1..10 contours for {len(models)} models to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.summary.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
