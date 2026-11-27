# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Visualize measured within-checkpoint alignment as bivariate Gaussian contours."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

TARGETS = (("full_train", "(a) Full train"), ("full_validation", "(b) Full validation"))
SELECTORS = (("train10", "Train 10", "#138A72"), ("val10", "Val 10", "#DE8F05"))


def standardized_pairs(summary: dict, target: str, selector: str) -> tuple[np.ndarray, float]:
    keys = set(summary["aggregate"][target]["paired_checkpoint_keys"])
    pairs = []
    for point in summary["checkpoints"]:
        if point["key"] not in keys:
            continue
        values = np.asarray([point["scores"][target], point["scores"][selector]], dtype=float).T
        deviations = values.std(axis=0, ddof=0)
        if values.shape != (5, 2) or np.any(deviations <= 1e-12):
            raise ValueError(f"Expected five nonconstant candidate pairs: {point['key']}")
        pairs.append((values - values.mean(axis=0)) / deviations)
    if not pairs:
        raise ValueError(f"No paired-defined checkpoints for {target}")
    values = np.concatenate(pairs)
    rho = float(np.corrcoef(values.T)[0, 1])
    expected = summary["aggregate"][target][selector]["mean_pearson"]
    if not np.isclose(rho, expected, atol=1e-12, rtol=0):
        raise ValueError("Standardized pooled correlation differs from the within-checkpoint mean")
    return values, rho


def selected_pairs(summary: dict, target: str, selector: str) -> tuple[np.ndarray, np.ndarray]:
    values, _ = standardized_pairs(summary, target, selector)
    keys = set(summary["aggregate"][target]["paired_checkpoint_keys"])
    indices, weights = [], []
    offset = 0
    for point in summary["checkpoints"]:
        if point["key"] not in keys:
            continue
        scores = np.asarray(point["scores"][selector])
        winners = np.flatnonzero(scores == scores.max())
        indices.extend(offset + winners)
        weights.extend([1 / len(winners)] * len(winners))
        offset += 5
    if not np.isclose(sum(weights), len(keys), atol=1e-12):
        raise ValueError("Each checkpoint must have unit total selection weight")
    return values[indices], np.asarray(weights)


def selected_records(summary: dict) -> list[dict]:
    rows = []
    for point in summary["checkpoints"]:
        for selector, _, _ in SELECTORS:
            scores = np.asarray(point["scores"][selector])
            winners = np.flatnonzero(scores == scores.max())
            for index in winners:
                train = point["scores"]["full_train"][index]
                val = point["scores"]["full_validation"][index]
                rows.append({
                    "checkpoint": point["key"], "seed": point["seed"], "selector": selector,
                    "candidate": f"candidate_{index + 1:02d}", "selection_weight": 1 / len(winners),
                    "tie_count": len(winners), "evaluation_score": float(scores[index]),
                    "full_train_performance": train, "full_validation_performance": val,
                    "train_gain_over_parent_pp": (train - point["parent_scores"]["full_train"]) * 100,
                    "validation_lift_over_proposal_mean_pp": (val - np.mean(point["scores"]["full_validation"])) * 100,
                })
    for selector, _, _ in SELECTORS:
        chosen = [row for row in rows if row["selector"] == selector]
        weights = [row["selection_weight"] for row in chosen]
        for target, metric, column in (
            ("full_train", "mean_selected_gain", "train_gain_over_parent_pp"),
            ("full_validation", "mean_selection_gain", "validation_lift_over_proposal_mean_pp"),
        ):
            measured = np.average([row[column] for row in chosen], weights=weights)
            expected = summary["aggregate"][target][selector][metric] * 100
            if not np.isclose(measured, expected, atol=1e-12, rtol=0):
                raise ValueError("Reconstructed argmax gains differ from the original analysis")
    return rows


def plot(source: Path, label: str, output: Path, *, selected: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    raw = source.read_bytes()
    summary = json.loads(raw)
    if "partial_progress" in summary or not all(p["evaluation_complete"] for p in summary["checkpoints"]):
        raise ValueError("This plot requires a completed experiment summary")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    bound = 2.8 if selected else 2.6
    axis_values = np.linspace(-bound, bound, 360)
    x, y = np.meshgrid(axis_values, axis_values)
    probabilities = np.asarray([0.5, 0.9])
    # For two Gaussian dimensions, squared Mahalanobis distance is chi-square(2).
    # These enclose probability mass, not confidence intervals for estimated parameters.
    levels = -2 * np.log(1 - probabilities)
    models = {}
    for scatter in ((True,) if selected else (False, True)):
        figure, axes = plt.subplots(1, 2, figsize=(3.4, 2.9 if selected else 2.6), sharex=True, sharey=True)
        figure.subplots_adjust(left=0.15, right=0.98, bottom=0.39 if selected else 0.35,
                               top=0.80 if selected else 0.78, wspace=0.24)
        for axis, (target, title) in zip(axes, TARGETS, strict=True):
            handles = []
            for selector, name, color in SELECTORS:
                values, rho = standardized_pairs(summary, target, selector)
                if abs(rho) >= 1:
                    raise ValueError("A singular Gaussian cannot be shown as closed density contours")
                mean = np.zeros(2)
                covariance = np.asarray([[1, rho], [rho, 1]])
                if selected:
                    values, weights = selected_pairs(summary, target, selector)
                    mean = np.average(values, weights=weights, axis=0)
                    centered = values - mean
                    covariance = (centered * weights[:, None]).T @ centered / weights.sum()
                if np.linalg.eigvalsh(covariance).min() <= 1e-12:
                    raise ValueError("Gaussian covariance must be positive definite")
                centered_grid = np.stack([x - mean[0], y - mean[1]], axis=-1)
                squared_distance = np.einsum("...i,ij,...j->...", centered_grid,
                                             np.linalg.inv(covariance), centered_grid)
                axis.contour(x, y, squared_distance, levels=levels, colors=[color], linewidths=[1.3, 0.9],
                             linestyles=["solid", "dashed"], alpha=0.95, zorder=3)
                if scatter:
                    axis.scatter(values[:, 0], values[:, 1], s=12 * weights if selected else 4,
                                 color=color, alpha=0.32 if selected else 0.16,
                                 edgecolors="none", rasterized=True, zorder=2)
                legend = rf"{name}: $\mu_x={mean[0]:+.2f}$" if selected else rf"{name}: $r={rho:+.3f}$"
                handles.append(Line2D([0], [0], color=color, lw=1.3, label=legend))
                if selected:
                    axis.scatter(*mean, marker="+", s=42, color=color, linewidths=1.6, zorder=4)
                models.setdefault(target, {})[selector] = {
                    "proposal_correlation": rho, "mean": mean.tolist(), "covariance": covariance.tolist(),
                    "checkpoints": summary["aggregate"][target]["paired_checkpoints"],
                    "candidate_pairs": len(values), "selection_weight": float(weights.sum()) if selected else None,
                    "paired_checkpoint_keys": summary["aggregate"][target]["paired_checkpoint_keys"],
                }
            axis.set_title(title, pad=7)
            axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                        frameon=False, handlelength=1.4, fontsize=6, borderpad=0, labelspacing=0.4)
            axis.set(xlim=(-bound, bound), ylim=(-bound, bound), xticks=[-2, 0, 2], yticks=[-2, 0, 2])
            axis.set_aspect("equal")
            axis.axhline(0, color="#D1D5DB", lw=0.5, zorder=1)
            axis.axvline(0, color="#D1D5DB", lw=0.5, zorder=1)
            axis.tick_params(length=2, pad=2, labelsize=6)
            for spine in axis.spines.values():
                spine.set_color("#AAAAAA")
                spine.set_linewidth(0.5)
        axes[0].set_ylabel(("Selected score" if selected else "Evaluation score") + " (within-CP z)", fontsize=7)
        figure.suptitle(f"{label} · BFCL", fontsize=9, y=0.985)
        subtitle = "Argmax-selected candidates" if selected else "Gaussian approximation"
        figure.text(0.565, 0.895, subtitle + " · 50% / 90% Gaussian contours", ha="center", fontsize=6)
        figure.text(0.565, 0.21 if selected else 0.18, "Target performance (within-CP z)", ha="center", fontsize=7)
        gain = summary["aggregate"]["full_train"]["proposal_mean_gain"] * 100
        count = len(summary["checkpoints"])
        if selected:
            train = summary["aggregate"]["full_train"]
            val = summary["aggregate"]["full_validation"]
            train_gains = " / ".join(f"{train[s]['mean_selected_gain'] * 100:+.3f}" for s, _, _ in SELECTORS)
            val_lifts = " / ".join(f"{val[s]['mean_selection_gain'] * 100:+.3f}" for s, _, _ in SELECTORS)
            figure.text(0.5, 0.155, f"Train gain over parent: {train_gains} pp", ha="center", fontsize=6)
            figure.text(0.5, 0.10, f"Validation lift over proposal mean: {val_lifts} pp", ha="center", fontsize=5.5)
            figure.text(0.5, 0.045, f"All {count} CPs; Train 10 / Val 10. Proposal train gain: {gain:+.3f} pp",
                        ha="center", fontsize=5.5)
        else:
            figure.text(0.5, 0.095, f"Mean train proposal gain ({count} CPs): {gain:+.3f} pp", ha="center", fontsize=6)
            figure.text(0.5, 0.025, "Full targets include their respective 10-task score samples.",
                        ha="center", fontsize=5.5, color="#555555")
        stem = "selected_candidates_contours" if selected else (
            "alignment_contours_with_points" if scatter else "alignment_contours")
        for extension in ("pdf", "png"):
            figure.savefig(output / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(figure)
    caption = (
        r"\textbf{Proposal--evaluation alignment on BFCL (" + label + r").} "
        "Train 10 and Val 10 are compared against full training and validation performance. "
        "Each variable is standardized within a checkpoint before pooling five candidate pairs; "
        "both evaluators use the same paired-defined checkpoints. The resulting correlations equal "
        r"the mean within-checkpoint Pearson correlations. Curves enclose 50\% and 90\% of probability "
        "under a moment-matched bivariate Gaussian approximation, without assuming empirical normality. "
        "Full targets contain their respective 10-task score samples. For training, standardized "
        "performance equals standardized improvement over the shared parent. A validation parent "
        "baseline was not evaluated. Recorded execution errors retain their original zero scores. "
        "The proposal mean gain is reported across all checkpoints before standardization."
    )
    if selected:
        caption = (
            r"\textbf{Argmax-selected proposals on BFCL (" + label + r").} "
            "Selected candidates are reconstructed from Train 10 and Val 10 scores. Each tied maximum "
            "receives equal probability, with unit total weight per checkpoint. Variables are standardized "
            "using all five proposals before selecting maxima. Points and moment-matched Gaussian contours "
            "use the paired-defined checkpoints from the proposal-correlation analysis; crosses mark means. "
            r"Solid and dashed contours enclose 50\% and 90\% Gaussian probability mass. "
            "Selected-only distributions differ from the pre-selection distribution in Proposition 1. "
            "Gain annotations use all checkpoints, including constant-score groups. Training gain is relative "
            "to the parent; validation lift is relative to the proposal mean because no validation parent "
            "baseline was evaluated. Full targets include their score samples; recorded errors retain zero scores."
        )
    stem = "selected_candidates_contours" if selected else "alignment_contours"
    (output / f"{stem}.tex").write_text(
        "% Preamble: \\usepackage{graphicx}\n\\begin{figure}[t]\n    \\centering\n"
        "    \\includegraphics[width=\\columnwidth]{" + stem + ".pdf}\n"
        "    \\caption{" + caption + "}\n    \\label{fig:" + stem + "_" + label.lower()
        + "}\n\\end{figure}\n"
    )
    metadata = {
        "label": label, "source": str(source.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
        "models": models, "probability_mass_levels": probabilities.tolist(),
        "method": "Population z-score within CP, then a Gaussian with matching pooled mean and covariance",
        "normality_tested": False, "standardization_removes_proposal_mean_gain": True,
        "full_train_proposal_mean_gain_pp": summary["aggregate"]["full_train"]["proposal_mean_gain"] * 100,
        "proposal_gain_checkpoint_count": len(summary["checkpoints"]),
        "execution_errors": len(summary["candidate_execution_errors"]) + len(summary["parent_execution_errors"]),
        "error_policy": "Preserves existing zero-scored errors", "new_model_calls": 0,
        "scope": "Qwen parent contexts with model-specific proposals and evaluations",
        "selection": "Argmax reconstructed; uniform probability over ties" if selected else None,
        "adopted_candidate_log_available": False,
    }
    if selected:
        rows = selected_records(summary)
        with (output / "argmax_selected_candidates.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata["all_checkpoints_selection"] = {
            selector: {
                "checkpoints": len(summary["checkpoints"]),
                "selected_candidate_rows": sum(max(p["scores"][selector]) == value
                                               for p in summary["checkpoints"] for value in p["scores"][selector]),
                "tie_checkpoints": sum(p["scores"][selector].count(max(p["scores"][selector])) > 1
                                       for p in summary["checkpoints"]),
                "train_gain_over_parent_pp": summary["aggregate"]["full_train"][selector]["mean_selected_gain"] * 100,
                "validation_lift_over_proposal_mean_pp":
                    summary["aggregate"]["full_validation"][selector]["mean_selection_gain"] * 100,
            } for selector, _, _ in SELECTORS
        }
        metadata["method"] = "Argmax-weighted Gaussian moments after within-CP standardization of all five proposals"
    (output / "contour_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selected", action="store_true",
                        help="Use argmax-selected candidates and uniform tie weights")
    args = parser.parse_args()
    plot(args.summary, args.label, args.output, selected=args.selected)


if __name__ == "__main__":
    main()
