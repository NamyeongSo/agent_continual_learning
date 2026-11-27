# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Compare completed BFCL alignment summaries without additional model calls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

SELECTORS = ("train10", "val10")
COLORS = ("#138A72", "#DE8F05")


def load_summaries(inputs: list[tuple[str, Path]]) -> list[dict]:
    models = []
    for label, path in inputs:
        raw = path.read_bytes()
        data = json.loads(raw)
        if "partial_progress" in data:
            raise ValueError(f"Use a completed summary: {path}")
        if not data["checkpoints"] or not all(point["evaluation_complete"] for point in data["checkpoints"]):
            raise ValueError(f"Evaluation coverage is incomplete: {path}")
        errors = len(data["candidate_execution_errors"]) + len(data["parent_execution_errors"])
        total = sum(point["coverage"]["parent_full_train"] + sum(point["coverage"]["full_train"])
                    + sum(point["coverage"]["full_validation"]) for point in data["checkpoints"])
        models.append({"label": label, "source": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                       "summary": data, "errors": errors, "evaluations": total})
    if len({model["label"] for model in models}) != len(models):
        raise ValueError("Model labels must be distinct")
    if any({point["key"] for point in model["summary"]["checkpoints"]} != {
        point["key"] for point in models[0]["summary"]["checkpoints"]
    } for model in models[1:]):
        raise ValueError("Compare summaries with the same checkpoint groups")
    return models


def add_labels(axis, bars, values, *, decimals: int, signed: bool = False) -> None:
    for bar, value in zip(bars, values, strict=True):
        text = f"{value:+.{decimals}f}" if signed else f"{value:.{decimals}f}"
        axis.annotate(text, (bar.get_x() + bar.get_width() / 2, value),
                      xytext=(0, 5 if value >= 0 else -5), textcoords="offset points", ha="center",
                      va="bottom" if value >= 0 else "top", fontsize=11, fontweight="medium")


def save_figure(figure, output: Path, stem: str) -> None:
    for extension in ("pdf", "png"):
        figure.savefig(output / f"{stem}.{extension}", dpi=220, bbox_inches="tight")


def plot(models: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    error_note = "Recorded errors scored as zero: " + "; ".join(
        f"{m['label']} {m['errors']:,}/{m['evaluations']:,} ({m['errors'] / m['evaluations']:.1%})" for m in models)
    rows = []
    for target, title in (("full_train", "Correlation with full train experience"),
                          ("full_validation", "Correlation with full validation experience")):
        fig, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models), 3.4), sharey=True, squeeze=False)
        fig.subplots_adjust(top=0.74, bottom=0.29, left=0.09, right=0.98, wspace=0.18)
        for axis, model in zip(axes.flat, models, strict=True):
            summary = model["summary"]["aggregate"][target]
            values = [summary[selector]["mean_pearson"] for selector in SELECTORS]
            if any(value is None for value in values):
                raise ValueError("Undefined mean correlation")
            bars = axis.bar([0, 1], values, width=0.58, color=COLORS)
            add_labels(axis, bars, values, decimals=3)
            axis.axhline(0, color="#777777", linewidth=0.8)
            axis.set_xticks([0, 1], ["Train 10", "Val 10"])
            axis.set_ylim(-0.13, 0.76)
            axis.set_yticks([0, 0.2, 0.4, 0.6])
            axis.grid(axis="y", alpha=0.15)
            axis.set_axisbelow(True)
            axis.set_title(model["label"], fontsize=12, pad=18)
            axis.text(0.5, -0.30, f"Paired-defined CPs: {summary['paired_checkpoints']}/{summary['total_checkpoints']}",
                      transform=axis.transAxes, ha="center", fontsize=9, color="#444444")
            rows.append({"model": model["label"], "target": target,
                         "train10_pearson": values[0], "val10_pearson": values[1],
                         "paired_checkpoints": summary["paired_checkpoints"],
                         "proposal_mean_performance": summary["proposal_mean_performance"],
                         "parent_mean_performance": summary["mean_parent_performance"],
                         "proposal_mean_gain_pp": summary["proposal_mean_gain"] * 100
                         if summary["proposal_mean_gain"] is not None else None,
                         "train10_selected_gain_pp": summary["train10"]["mean_selected_gain"] * 100
                         if summary["train10"]["mean_selected_gain"] is not None else None,
                         "val10_selected_gain_pp": summary["val10"]["mean_selected_gain"] * 100
                         if summary["val10"]["mean_selected_gain"] is not None else None,
                         "train10_selection_gain_pp": summary["train10"]["mean_selection_gain"] * 100,
                         "val10_selection_gain_pp": summary["val10"]["mean_selection_gain"] * 100})
        axes[0, 0].set_ylabel("Mean within-CP Pearson r")
        fig.suptitle(title, fontsize=14, y=0.97)
        fig.text(0.5, 0.86, "50 checkpoints x 5 candidates · Qwen parent contexts", ha="center", color="#555555")
        fig.text(0.5, 0.085, "Full targets contain their respective 10-task score samples.", ha="center", fontsize=8)
        fig.text(0.5, 0.025, error_note, ha="center", fontsize=7.5, color="#555555")
        save_figure(fig, output, f"correlation_{target}")
        plt.close(fig)

    for target in ("full_train", "full_validation"):
        fig, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models), 3.6), sharey=True, squeeze=False)
        fig.subplots_adjust(top=0.73, bottom=0.31, left=0.09, right=0.98, wspace=0.18)
        all_values = []
        for axis, model in zip(axes.flat, models, strict=True):
            summary = model["summary"]["aggregate"][target]
            if target == "full_train":
                values = [summary["proposal_mean_gain"] * 100,
                          *(summary[selector]["mean_selected_gain"] * 100 for selector in SELECTORS)]
                labels, colors = ["Proposal\nmean", "Train 10\nselected", "Val 10\nselected"], ("#7A8793", *COLORS)
                caption = f"Parent mean reward: {summary['mean_parent_performance']:.2%}"
            else:
                values = [summary[selector]["mean_selection_gain"] * 100 for selector in SELECTORS]
                labels, colors = ["Train 10\nselected", "Val 10\nselected"], COLORS
                caption = f"Proposal mean reward: {summary['proposal_mean_performance']:.2%}"
            all_values.extend(values)
            bars = axis.bar(range(len(values)), values, width=0.58, color=colors)
            add_labels(axis, bars, values, decimals=2, signed=True)
            axis.set_xticks(range(len(values)), labels, fontsize=9)
            axis.axhline(0, color="#777777", linewidth=0.8)
            axis.grid(axis="y", alpha=0.15)
            axis.set_axisbelow(True)
            axis.set_title(model["label"], fontsize=12, pad=18)
            axis.text(0.5, -0.40, caption, transform=axis.transAxes, ha="center", fontsize=9, color="#444444")
        span = max(all_values) - min(all_values)
        margin = max(0.4, span * 0.25)
        axes[0, 0].set_ylim(min(0, *all_values) - margin, max(0, *all_values) + margin)
        title = "Gain on full train experience" if target == "full_train" else "Selection gain on full validation"
        ylabel = "Gain over parent (pp)" if target == "full_train" else "Gain over mean proposal (pp)"
        axes[0, 0].set_ylabel(ylabel)
        fig.suptitle(title, fontsize=14, y=0.97)
        fig.text(0.5, 0.85, "All 50 checkpoints · uniform tie-breaking expectation", ha="center", color="#555555")
        note = ("Proposal mean: average of all 5 candidates minus their parent."
                if target == "full_train" else "Validation parent baseline was not evaluated; zero = mean proposal.")
        fig.text(0.5, 0.085, note, ha="center", fontsize=8)
        fig.text(0.5, 0.025, error_note, ha="center", fontsize=7.5, color="#555555")
        save_figure(fig, output, f"gain_{target}")
        plt.close(fig)
    with (output / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"sources": [{k: v for k, v in model.items() if k != "summary"} for model in models],
                "correlation": "Arithmetic mean of paired-defined within-checkpoint correlations",
                "gain": "All fifty checkpoint groups; gains are percentage points",
                "errors": "Retains recorded error scores of zero; no new evaluations",
                "scope": "Completed proposal-evaluation alignment using Qwen parents; not the pending CP5 policy probe",
                "overlap": "Full targets include the score samples; remaining40 analyses are in source summaries",
                "scores": rows}
    (output / "comparison_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", nargs=2, action="append", required=True, metavar=("LABEL", "PATH"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(load_summaries([(label, Path(path)) for label, path in args.summary]), args.output.resolve())


if __name__ == "__main__":
    main()
