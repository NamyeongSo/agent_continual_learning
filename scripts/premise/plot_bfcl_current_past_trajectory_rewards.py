# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Plot current-plus-past reward selection with task and period weighting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.bfcl_trajectory_selection_rewards import REWARD_KEYS  # noqa: E402
from scripts.premise.plot_bfcl_trajectory_rewards import (  # noqa: E402
    LABELS,
    LINESTYLES,
    MARKERS,
    MODELS,
    OBJECTIVE_RECIPES,
    TARGET_STYLE,
    metric_value,
)

WEIGHTINGS = ("task_mean", "period_mean")
WEIGHTING_LABELS = {"task_mean": "Task mean", "period_mean": "Period mean"}
WEIGHTING_ALIASES = {"pooled": "task_mean", "period_balanced": "period_mean"}


def index_rows(summary: dict) -> dict[tuple[str, str, str, str, int], dict]:
    """Reject mismatched comparison cohorts before drawing comparable curves."""
    indexed = {}
    counts = {}
    for source_row in summary.get("aggregate", []):
        if source_row.get("candidate_count", 5) != 5:
            continue
        row = dict(source_row)
        weighting = WEIGHTING_ALIASES.get(row.get("weighting"), row.get("weighting"))
        model, cohort, recipe, n = (row.get(field) for field in ("model", "cohort", "recipe", "n"))
        if (model not in MODELS or weighting not in WEIGHTINGS or cohort not in ("objective", "primary")
                or recipe not in REWARD_KEYS or not isinstance(n, int) or not 1 <= n <= 10):
            raise ValueError("Unsupported model, weighting, cohort, recipe, or sample size")
        if cohort == "objective" and recipe not in OBJECTIVE_RECIPES:
            raise ValueError("Objective curves cannot include recipes requiring semantic judgments")
        key = (model, weighting, cohort, recipe, n)
        if key in indexed:
            raise ValueError(f"Duplicate aggregate row: {key}")
        count = row.get("checkpoints", 0)
        if not isinstance(count, int) or count < 0:
            raise ValueError("Checkpoint coverage must be a nonnegative integer")
        coverage_key = (model, cohort, n)
        if coverage_key in counts and counts[coverage_key] != count:
            raise ValueError(f"Weightings and rewards require matched checkpoint cohorts: {coverage_key}")
        counts[coverage_key] = count
        measured = metric_value(row, "validation_lift")
        if count == 0 and np.isfinite(measured):
            raise ValueError("Zero-checkpoint cohorts cannot have a measured validation gain")
        indexed[key] = row
    return indexed


def plot(summary: dict, output: Path) -> None:
    """Render four panels per cohort; missing judgments never become zero gains."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    indexed = index_rows(summary)
    output.mkdir(parents=True, exist_ok=True)
    sizes = np.arange(1, 11)
    color = TARGET_STYLE["full_validation"][0]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .65,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    diagnostics = {"analysis_created_at": summary.get("created_at"), "files": [], "panels": [],
                   "sample_definition": "One current task plus n-1 sampled past tasks; nested n=1..10",
                   "weighting_definitions": {
                       "task_mean": "Each of the n selected tasks receives weight 1/n",
                       "period_mean": "Current and past-period means each receive weight 1/2; current only at n=1",
                   }, "metric": "Selected full-validation score minus mean score of the five proposals"}

    def series(model: str, weighting: str, cohort: str, recipe: str) -> np.ndarray:
        return np.asarray([metric_value(indexed.get((model, weighting, cohort, recipe, int(n)), {}),
                                        "validation_lift") for n in sizes])

    for cohort, recipes, stem in (
        ("primary", REWARD_KEYS, "semantic_gain_full_validation_n1_10"),
        ("objective", OBJECTIVE_RECIPES, "objective_gain_full_validation_n1_10"),
    ):
        figure, axes = plt.subplots(2, 2, figsize=(7.2, 4.65), sharex=True, sharey=True, squeeze=False)
        figure.subplots_adjust(left=.10, right=.985, bottom=.115, top=.84 if cohort == "primary" else .88,
                               wspace=.14, hspace=.34)
        for row_index, model in enumerate(MODELS):
            for column, weighting in enumerate(WEIGHTINGS):
                axis = axes[row_index, column]
                plotted = []
                counts = []
                for recipe in recipes:
                    numbers = 100 * series(model, weighting, cohort, recipe)
                    if not np.isfinite(numbers).any():
                        continue
                    index = REWARD_KEYS.index(recipe)
                    axis.plot(sizes, numbers, color=color, marker=MARKERS[index], linestyle=LINESTYLES[index],
                              linewidth=1.35 if recipe == "success_count" else 1, markersize=3.1,
                              markerfacecolor=color if recipe == "success_count" else "white", markeredgewidth=.7)
                    plotted.append(recipe)
                for n in sizes:
                    item = indexed.get((model, weighting, cohort, "success_count", int(n)), {})
                    counts.append(int(item.get("checkpoints", 0)))
                count_text = str(min(counts)) if min(counts) == max(counts) else f"{min(counts)}-{max(counts)}"
                expected = 50
                partial = cohort == "primary" and any(count < expected for count in counts)
                status = "; partial" if partial else ""
                axis.set_title(f"{model} · {WEIGHTING_LABELS[weighting]} ({count_text} CPs{status})",
                               fontsize=8, pad=5)
                if not plotted:
                    message = "Semantic judgments pending" if cohort == "primary" else "No eligible trajectories"
                    axis.text(.5, .5, f"{message}\n0 eligible CPs", transform=axis.transAxes,
                              ha="center", va="center", fontsize=8, color="#666666")
                axis.axhline(0, color="#B9B9B9", linewidth=.5, zorder=0)
                axis.set(xticks=(1, 3, 5, 7, 10), xlim=(.7, 10.3))
                axis.tick_params(labelsize=7, length=2.5)
                axis.grid(axis="y", color="#E9E9E9", linewidth=.4, zorder=0)
                if column == 0:
                    axis.set_ylabel("Validation lift (pp)", fontsize=8)
                if row_index == 1:
                    axis.set_xlabel(r"Selection sample size $n$", fontsize=8)
                diagnostics["panels"].append({"cohort": cohort, "model": model, "weighting": weighting,
                                              "checkpoints_by_n": counts, "partial": partial,
                                              "plotted_recipes": plotted})
        handles = [Line2D([], [], color=color, linewidth=1, marker=MARKERS[REWARD_KEYS.index(recipe)],
                          linestyle=LINESTYLES[REWARD_KEYS.index(recipe)], markersize=3, label=LABELS[recipe])
                   for recipe in recipes]
        figure.legend(handles=handles, ncol=4 if cohort == "primary" else 3, loc="upper center",
                      frameon=False, fontsize=7, columnspacing=1.1, handlelength=2.4)
        for extension in ("pdf", "png"):
            figure.savefig(output / f"{stem}.{extension}", dpi=350, bbox_inches="tight")
        plt.close(figure)
        diagnostics["files"].append(stem)
        print(f"Saved {stem}.pdf", flush=True)

    (output / "figure_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
    (output / "README.md").write_text(
        "# Current + past trajectory reward figures\n\n"
        "Each figure has four panels: Qwen/Gemma rows and task-mean/period-mean columns. "
        "The selection pool contains the current task and n-1 past tasks, never future tasks. "
        "The same sampled task IDs are used for all five candidates, the parent, and both weightings.\n\n"
        "- Task mean: each task has weight 1/n.\n"
        "- Period mean: the current task has weight 1/2, and the past-task mean has weight 1/2. "
        "At n=1 only the current task is available and receives weight 1.\n"
        "- Validation lift is selected full-validation score minus the mean score of all five proposals; "
        "it is not parent-relative validation gain.\n"
        "- Semantic curves require complete valid judgments for every selected task and all six contexts. "
        "Within each model and n, all reward rules and both weightings use matched checkpoint cohorts. "
        "Counts may vary across n; panel titles disclose their range. Partial results are marked.\n"
        "- Objective curves use the separately labeled trace-only cohort. "
        "Do not attribute differences between these two figures solely to the reward rule.\n"
        "- Missing judgments are not plotted as zero reward. No confidence shading is shown.\n\n"
        + "\n".join(f"- [{stem}.pdf]({stem}.pdf) · [PNG]({stem}.png)" for stem in diagnostics["files"])
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot(json.loads(args.summary.read_text()), args.output or args.summary.parent / "figures")


if __name__ == "__main__":
    main()
