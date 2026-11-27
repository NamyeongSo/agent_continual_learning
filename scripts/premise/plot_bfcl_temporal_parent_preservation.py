# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Plot LLM-free parent preservation under seven chronological task samplers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODELS = ("Qwen", "Gemma")
WEIGHTS = ("task_mean", "period_mean")
WEIGHT_LABELS = {"task_mean": "Task mean", "period_mean": "Period mean"}
CASES = ("current", "past", "future", "current_future", "current_past", "past_future", "current_past_future")
CASE_LABELS = {
    "current": "Current (1 task)",
    "past": "Past",
    "future": "Future",
    "current_future": "Current + future",
    "current_past": "Current + past",
    "past_future": "Past + future",
    "current_past_future": "Current + past + future",
}
# Retain the existing chronological-analysis colors, adding the past+future case.
COLORS = {
    "current": "#6A6A6A", "past": "#287C85", "future": "#A34F65", "current_future": "#78618E",
    "current_past": "#15555D", "past_future": "#81734B", "current_past_future": "#AF8238",
}
FILLS = {
    "current": "#F2F2F2", "past": "#EAF4F5", "future": "#F9EEF1", "current_future": "#F3EFF7",
    "current_past": "#E5EFF0", "past_future": "#F2F0E9", "current_past_future": "#F8F2E7",
}
MARKERS = dict(zip(CASES, ("o", "s", "^", "v", "D", "P", "X"), strict=True))
STYLES = dict(zip(CASES, (":", "-", "-", "--", "--", "-.", "-"), strict=True))
METHODS = ("success_mean", "parent_preservation")
TARGET_METRICS = {
    "full_validation": "validation_lift",
    "validation_remaining40": "validation_remaining40_lift",
}


def metric_value(row: dict, metric: str, field: str = "mean") -> float:
    """Preserve unmeasured values as gaps, never as zero lift."""
    value = row.get(metric, {})
    value = value.get(field) if isinstance(value, dict) else value if field == "mean" else None
    return float(value) if value is not None else np.nan


def index_rows(summary: dict, *, paired: bool = False) -> dict[tuple, dict]:
    """Check measured comparisons share one fixed checkpoint coverage."""
    rows = summary.get("paired_parent_minus_success" if paired else "aggregate", [])
    indexed, counts = {}, {}
    for row in rows:
        if row.get("scope", "all") != "all":
            continue
        model, weight, case, n = (row.get(field) for field in ("model", "weight", "case", "n"))
        method = "paired_difference" if paired else row.get("method")
        if (model not in MODELS or weight not in WEIGHTS or case not in CASES
                or not isinstance(n, int) or not 1 <= n <= 10 or (not paired and method not in METHODS)):
            raise ValueError("Unsupported temporal parent-preservation aggregate identity")
        if case in ("current_future", "current_past", "past_future") and n < 2:
            raise ValueError("A two-period comparison cannot use fewer than two tasks")
        if case == "current_past_future" and n < 3:
            raise ValueError("A three-period comparison cannot use fewer than three tasks")
        count = row.get("checkpoints")
        if not isinstance(count, int) or count < 0:
            raise ValueError("Checkpoint coverage must be a nonnegative integer")
        if count == 0 and any(np.isfinite(metric_value(row, metric)) for metric in TARGET_METRICS.values()):
            raise ValueError("Zero-checkpoint groups cannot report measured validation lift")
        if model in counts and counts[model] != count:
            raise ValueError("Curves and weighting modes require a fixed checkpoint cohort within each model")
        counts[model] = count
        key = (model, weight, case, method, n)
        if key in indexed:
            raise ValueError(f"Repeated temporal aggregate row: {key}")
        indexed[key] = row
    return indexed


def reference_points(
    summary: dict, model: str, target: str, expected_checkpoints: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validation-sample references at their actual measured sample sizes."""
    rows = [row for row in summary.get("validation_reference", [])
            if row.get("model") == model and row.get("target") == target and row.get("scope", "all") == "all"]
    by_n = {}
    for row in rows:
        if expected_checkpoints is not None and row.get("checkpoints") != expected_checkpoints:
            raise ValueError("Validation-sample references must use the training-selector checkpoint cohort")
        n = row.get("n", 10)
        if not isinstance(n, int) or not 1 <= n <= 10 or n in by_n:
            raise ValueError("Validation references must have one measured row per model, target and sample size")
        value = row.get("selected_lift")
        value = value.get("mean") if isinstance(value, dict) else value
        if value is not None:
            by_n[n] = float(value)
    return np.asarray(sorted(by_n)), np.asarray([by_n[n] for n in sorted(by_n)])


def surface_grid(
    rows: list[dict], model: str, weight: str, case: str, metric: str = "validation_lift",
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve unmeasured cells and enforce exact uniform single-candidate behavior."""
    grid, counts = np.full((5, 10), np.nan), np.zeros((5, 10), dtype=int)
    observed = set()
    for row in rows:
        identity = (row.get("model"), row.get("weight"), row.get("case"), row.get("method"))
        if identity != (model, weight, case, "parent_preservation") or row.get("scope", "all") != "all":
            continue
        n, candidates = row.get("n"), row.get("candidate_count")
        if (not isinstance(n, int) or not isinstance(candidates, int)
                or not 1 <= n <= 10 or not 1 <= candidates <= 5):
            raise ValueError("Surface coordinates require n=1..10 and candidate_count=1..5")
        if ((case in ("current_past", "current_future", "past_future") and n < 2)
                or (case == "current_past_future" and n < 3)):
            raise ValueError("Surface cells cannot include infeasible temporal allocations")
        if (n, candidates) in observed:
            raise ValueError("Repeated temporal parent-preservation surface coordinate")
        observed.add((n, candidates))
        value, count = metric_value(row, metric), row.get("checkpoints", 0)
        if not isinstance(count, int) or count < 0 or (count == 0 and np.isfinite(value)):
            raise ValueError("Measured surface cells require positive checkpoint coverage")
        if candidates == 1 and np.isfinite(value) and abs(value) > 1e-10:
            raise ValueError("A uniformly sampled single candidate has zero lift over the proposal mean")
        grid[candidates - 1, n - 1], counts[candidates - 1, n - 1] = 100 * value, count
    measured_counts = counts[counts > 0]
    if len(measured_counts) and np.ptp(measured_counts):
        raise ValueError("Temporal candidate-count surfaces require a fixed checkpoint cohort")
    return grid, counts


def index_uniform_reference(summary: dict, reference: dict) -> dict[tuple[str, str, int], dict]:
    """Require the uniform-training control to share data and checkpoint identities."""
    if (set(summary.get("fixed_checkpoint_keys", [])) != set(reference.get("fixed_checkpoint_keys", []))
            or summary.get("source_sha256") != reference.get("source_sha256")
            or summary.get("llm_judgments_used") is not False
            or reference.get("llm_judgments_used") is not False):
        raise ValueError("Uniform-training comparisons require the same LLM-free data and checkpoint cohort")
    indexed = {}
    for row in reference.get("aggregate", []):
        if row.get("scope", "all") != "all":
            continue
        model, method, n = (row.get(field) for field in ("model", "method", "n"))
        if (model not in MODELS or method not in METHODS or row.get("weight") != "uniform"
                or not isinstance(n, int) or not 1 <= n <= 10):
            raise ValueError("Unsupported uniform-training control identity")
        key = (model, method, n)
        if key in indexed:
            raise ValueError("Repeated uniform-training control coordinate")
        expected_count = len(summary["fixed_checkpoint_keys"])
        if row.get("checkpoints") != expected_count:
            raise ValueError("Uniform-training control checkpoint count differs from the temporal comparison")
        indexed[key] = row
    return indexed


def plot_surfaces(summary: dict, output: Path) -> None:
    """Write a four-panel measured surface and heatmap for every temporal pool."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    output.mkdir(parents=True, exist_ok=True)
    rows = summary.get("aggregate", [])
    if any(row.get("model") not in MODELS or row.get("weight") not in WEIGHTS or row.get("case") not in CASES
           for row in rows):
        raise ValueError("Surface rows require known Qwen/Gemma temporal sampling identities")
    grids = {(model, weight, case): surface_grid(rows, model, weight, case)
             for case in CASES for model in MODELS for weight in WEIGHTS}
    finite = np.concatenate([grid[np.isfinite(grid)] for grid, _ in grids.values()])
    if not len(finite):
        raise ValueError("No measured temporal parent-preservation surfaces")
    bound = max(float(np.abs(finite).max()) * 1.08, .05)
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    cmap = LinearSegmentedColormap.from_list("validation_lift", ["#53434A", "#F9EEF1", "#A34F65"])
    cmap.set_bad("#EEEEEE")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                         "axes.linewidth": .65, "pdf.fonttype": 42, "ps.fonttype": 42})
    x, y = np.meshgrid(np.arange(1, 11), np.arange(1, 6))
    diagnostics = {
        "metric": "Full-validation lift over the mean score of all five proposals, in percentage points",
        "global_color_limit_pp": bound, "files": [], "panels": [],
        "candidate_subsets": "Exact expectation over every uniform subset of lambda out of five candidates",
        "current_only": "One actual task at every nominal sample size",
        "missing": "Infeasible allocations remain unmeasured and gray in heatmaps",
    }

    def save(figure, stem):
        for extension in ("pdf", "png"):
            figure.savefig(output / f"{stem}.{extension}", dpi=350, bbox_inches="tight")
        plt.close(figure)
        diagnostics["files"].append(stem)
        print(f"Saved {stem}.pdf", flush=True)

    for case in CASES:
        figure = plt.figure(figsize=(7.2, 6.15))
        figure.subplots_adjust(left=.025, right=.90, bottom=.05, top=.91, wspace=.09, hspace=.17)
        figure.suptitle(CASE_LABELS[case], fontsize=9, y=.985)
        for row_index, model in enumerate(MODELS):
            for column, weight in enumerate(WEIGHTS):
                values, counts = grids[model, weight, case]
                axis = figure.add_subplot(2, 2, row_index * 2 + column + 1, projection="3d")
                coverage = counts[counts > 0]
                count = int(coverage[0]) if len(coverage) else 0
                axis.set_title(f"{model} · {WEIGHT_LABELS[weight]} ({count} CPs)", fontsize=7.5, pad=2)
                axis.plot_surface(x, y, values, cmap=cmap, norm=norm, edgecolor="#C3A2AE", linewidth=.3,
                                  rstride=1, cstride=1, antialiased=True, shade=False, alpha=.97)
                defined = np.isfinite(values)
                axis.scatter(x[defined], y[defined], values[defined], color="#593140", s=3,
                             depthshade=False, alpha=.7)
                axis.plot(x[0], y[0], values[0], color="#777777", linewidth=.9)
                axis.set(xlim=(1, 10), ylim=(1, 5), zlim=(-bound, bound), xticks=(1, 5, 10), yticks=(1, 3, 5))
                axis.set_xlabel(r"Sample size $n$", fontsize=7, labelpad=1)
                axis.set_ylabel(r"Candidates $\lambda$", fontsize=7, labelpad=1)
                axis.set_zlabel("Validation lift (pp)", fontsize=7, labelpad=2)
                axis.tick_params(labelsize=6, pad=-1)
                axis.view_init(elev=27, azim=-128)
                axis.set_box_aspect((1.2, 1, .85))
                for coordinate in (axis.xaxis, axis.yaxis, axis.zaxis):
                    coordinate.pane.fill = False
                    coordinate._axinfo["grid"].update({"linewidth": .4, "color": (.84, .84, .84, .5)})
                diagnostics["panels"].append({"case": case, "model": model, "weight": weight,
                                              "checkpoints": counts.tolist(), "finite_cells": int(defined.sum())})
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = figure.colorbar(mappable, cax=figure.add_axes((.94, .19, .014, .61)))
        colorbar.set_label("Validation lift (pp)", fontsize=7)
        colorbar.ax.tick_params(labelsize=6.5)
        save(figure, f"parent_preservation_{case}_surface_full_validation")

        figure, axes = plt.subplots(2, 2, figsize=(7.2, 4.65), sharex=True, sharey=True, squeeze=False)
        figure.subplots_adjust(left=.08, right=.90, bottom=.12, top=.87, wspace=.13, hspace=.31)
        figure.suptitle(CASE_LABELS[case], fontsize=9, y=.985)
        for row_index, model in enumerate(MODELS):
            for column, weight in enumerate(WEIGHTS):
                axis = axes[row_index, column]
                values, counts = grids[model, weight, case]
                coverage = counts[counts > 0]
                count = int(coverage[0]) if len(coverage) else 0
                axis.imshow(values, origin="lower", aspect="auto", cmap=cmap, norm=norm,
                            extent=(.5, 10.5, .5, 5.5), interpolation="none")
                for candidate_index, n_index in np.argwhere(np.isfinite(values)):
                    value = values[candidate_index, n_index]
                    axis.text(n_index + 1, candidate_index + 1, f"{value:+.2f}", ha="center", va="center",
                              fontsize=5.2, color="white" if abs(value) > .65 * bound else "#3C3034")
                axis.set_title(f"{model} · {WEIGHT_LABELS[weight]} ({count} CPs)", fontsize=8, pad=4)
                axis.set(xticks=np.arange(1, 11), yticks=np.arange(1, 6))
                axis.tick_params(labelsize=6.5, length=2)
                if column == 0:
                    axis.set_ylabel(r"Candidates $\lambda$", fontsize=8)
                if row_index == 1:
                    axis.set_xlabel(r"Selection sample size $n$", fontsize=8)
        colorbar = figure.colorbar(mappable, cax=figure.add_axes((.94, .22, .014, .54)))
        colorbar.set_label("Validation lift (pp)", fontsize=7)
        colorbar.ax.tick_params(labelsize=6.5)
        save(figure, f"parent_preservation_{case}_heatmap_full_validation")

    (output / "figure_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
    (output / "README.md").write_text(
        "# Candidate-count supplement\n\n"
        "Every temporal pool has a four-panel surface and a corresponding heatmap: "
        "Qwen/Gemma rows and task-mean/period-mean columns. "
        "All figures share one color and height scale for comparisons across pools.\n\n"
        "The horizontal axes are sampled training-task count n and candidate count lambda. "
        "The height/color is selected full-validation score minus the mean validation score of all five "
        "proposals. Each cell averages every uniform subset of lambda candidates, with uniform ties. "
        "Lambda=1 therefore has exactly zero expected lift. "
        "Lambda=5 reproduces the main parent-preservation curves. "
        "Current-only uses one actual task at every nominal n. "
        "Infeasible mixed-period budgets remain blank; they are not zero measurements.\n\n"
        + "\n".join(f"- [{stem}.pdf]({stem}.pdf) · [PNG]({stem}.png)" for stem in diagnostics["files"])
        + "\n"
    )
    parent_index = output.parent / "README.md"
    if parent_index.exists():
        supplement = f"\n[Candidate-count surfaces and heatmaps]({output.name}/README.md)\n"
        content = parent_index.read_text()
        if supplement not in content:
            parent_index.write_text(content + supplement)


def plot(summary: dict, output: Path, *, uniform_reference: dict | None = None) -> None:
    """Draw four-panel comparisons; bands show sampling variability, not confidence intervals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    uniform_only = uniform_reference is not None
    indexed, paired = index_rows(summary), index_rows(summary, paired=True)
    uniform = index_uniform_reference(summary, uniform_reference) if uniform_reference is not None else {}
    if uniform_only and not uniform:
        raise ValueError("The uniform-only comparison requires a measured uniform-training reference")
    if not indexed:
        raise ValueError("No measured parent-preservation comparison rows")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .65,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    sizes = np.arange(1, 11)
    diagnostics = {
        "analysis_created_at": summary.get("created_at"), "files": [], "panels": [],
        "methods": {"parent_preservation": "Fewest weighted parent-success regressions; most weighted repairs; "
                                               "greatest weighted relative turn reduction on joint successes",
                    "success_mean": "Largest weighted success score on the same sampled tasks"},
        "weighting": {"task_mean": "Every sampled task has weight 1/n",
                      "period_mean": "Each represented temporal period has equal total weight"},
        "bands": "One standard deviation of checkpoint-aggregated Monte Carlo sampling draws, not a CI",
        "current_only": "The same single current task is shown across nominal n for comparison",
        "validation_target": "Lift over the unselected mean of the five candidates on the indicated target set",
        "missing": "Infeasible temporal allocations are omitted; missing data are not zero",
        "uniform_training_reference": bool(uniform),
    }

    def save(figure, stem):
        for extension in ("pdf", "png"):
            figure.savefig(output / f"{stem}.{extension}", dpi=350, bbox_inches="tight")
        plt.close(figure)
        diagnostics["files"].append(stem)
        print(f"Saved {stem}.pdf", flush=True)

    def curve_figure(method, metric, ylabel, stem, *, source=None, target=None, shade=True):
        source = indexed if source is None else source
        figure, axes = plt.subplots(2, 2, figsize=(7.2, 4.65), sharex=True, sharey=True, squeeze=False)
        figure.subplots_adjust(left=.105, right=.985, bottom=.11, top=.77 if uniform_only else .82,
                               wspace=.15, hspace=.32)
        any_reference = False
        for row_index, model in enumerate(MODELS):
            for column, weight in enumerate(WEIGHTS):
                axis = axes[row_index, column]
                coverage = set()
                plotted = []
                for case in CASES:
                    rows = [source.get((model, weight, case, method, int(n)), {}) for n in sizes]
                    values = np.asarray([metric_value(row, metric) for row in rows])
                    std = np.asarray([metric_value(row, metric, "std") for row in rows])
                    factor = 1 if metric.startswith("spearman") else 100
                    values, std = factor * values, factor * std
                    if not np.isfinite(values).any():
                        continue
                    for row in rows:
                        if "checkpoints" in row:
                            coverage.add(row["checkpoints"])
                    axis.plot(sizes, values, color=COLORS[case], marker=MARKERS[case], linestyle=STYLES[case],
                              linewidth=1.45 if case == "current_past" else 1.1, markersize=3,
                              markerfacecolor="white", markeredgewidth=.7, zorder=3)
                    if shade and np.isfinite(std).any():
                        axis.fill_between(sizes, values - std, values + std, color=FILLS[case], alpha=.65,
                                          linewidth=0, zorder=1)
                    plotted.append(case)
                if uniform:
                    reference_rows = [uniform.get((model, method, int(n)), {}) for n in sizes]
                    uniform_values = 100 * np.asarray([metric_value(row, metric) for row in reference_rows])
                    uniform_std = 100 * np.asarray([metric_value(row, metric, "std") for row in reference_rows])
                    axis.plot(sizes, uniform_values, color="#287C85", linestyle=(0, (5, 1, 1, 1)),
                              marker="h", markersize=4, markerfacecolor="white", markeredgewidth=.95,
                              linewidth=1.7, zorder=5)
                    axis.fill_between(sizes, uniform_values - uniform_std, uniform_values + uniform_std,
                                      color="#EAF4F5", alpha=.65, linewidth=0, zorder=1)
                if target is not None:
                    expected_count = next(iter(coverage)) if len(coverage) == 1 else None
                    ref_n, ref_values = reference_points(summary, model, target, expected_count)
                    if len(ref_n):
                        axis.plot(ref_n, 100 * ref_values, color="#333333", linestyle="--" if len(ref_n) > 1 else "",
                                  linewidth=1.25, marker="*", markersize=5, markerfacecolor="white", zorder=4)
                        any_reference = True
                count = ",".join(str(value) for value in sorted(coverage)) if coverage else "0"
                axis.set_title(f"{model} · {WEIGHT_LABELS[weight]} ({count} CPs)", fontsize=8.4, pad=5)
                axis.axhline(0, color="#B8B8B8", linewidth=.55, zorder=0)
                axis.set(xticks=(1, 3, 5, 7, 10), xlim=(.7, 10.3))
                axis.grid(axis="y", color="#EAEAEA", linewidth=.4, zorder=0)
                axis.tick_params(labelsize=7, length=2.5)
                if column == 0:
                    axis.set_ylabel(ylabel, fontsize=8)
                if row_index == 1:
                    axis.set_xlabel(r"Selection sample size $n$", fontsize=8)
                diagnostics["panels"].append({"figure": stem, "model": model, "weight": weight,
                                              "checkpoints": sorted(coverage), "cases": plotted})
        handles = [Line2D([], [], color=COLORS[case], marker=MARKERS[case], linestyle=STYLES[case],
                          linewidth=1.1, markersize=3, markerfacecolor="white", label=CASE_LABELS[case])
                   for case in CASES]
        if any_reference:
            handles.append(Line2D([], [], color="#333333", linestyle="--", marker="*", markersize=5,
                                  markerfacecolor="white", label="Val-sample success"))
        if uniform:
            handles.append(Line2D([], [], color="#287C85", linestyle=(0, (5, 1, 1, 1)), marker="h",
                                  linewidth=1.7, markersize=4, markerfacecolor="white",
                                  label="Uniform train (single pool)"))
        figure.legend(handles=handles, ncol=3 if uniform_only else 4, loc="upper center", frameon=False, fontsize=7.2,
                      columnspacing=1.2, handlelength=2.3)
        save(figure, stem)

    for target, metric in TARGET_METRICS.items():
        if uniform_only:
            curve_figure("parent_preservation", metric, "Validation lift (pp)",
                         f"parent_preservation_gain_{target}_with_uniform_train_n1_10", target=target)
            continue
        curve_figure("parent_preservation", metric, "Validation lift (pp)",
                     f"parent_preservation_gain_{target}_n1_10", target=target)
        curve_figure("success_mean", metric, "Validation lift (pp)",
                     f"success_mean_gain_{target}_n1_10", target=target)
        if paired:
            curve_figure("paired_difference", metric, "Lift over success selection (pp)",
                         f"parent_minus_success_gain_{target}_n1_10", source=paired)
    for target in (() if uniform_only else ("full_validation", "full_train")):
        curve_figure("parent_preservation", f"spearman_{target}", r"Spearman $\rho_{\mathrm{S}}$",
                     f"parent_preservation_correlation_{target}_n1_10", shade=False)

    diagnostic_name = "uniform_training_comparison_diagnostics.json" if uniform_only else "figure_diagnostics.json"
    (output / diagnostic_name).write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
    if uniform_only:
        supplement = (
            "\n## Matched uniform-training control\n\n"
            "These additional figures retain all seven temporal curves and add a uniform sample of n tasks "
            "from all fifty training tasks. This control uses the same thirty-six checkpoints and the same "
            "LLM-free parent-preservation rule. Its single-pool task weighting is identical in both columns, "
            "so the same measured control is repeated across the two weighting panels. "
            "The older semantic-judge surface uses a different five-checkpoint cohort and rule.\n\n"
            + "\n".join(f"- [{stem}.pdf]({stem}.pdf) · [PNG]({stem}.png)" for stem in diagnostics["files"])
            + "\n"
        )
        index_path = output / "README.md"
        content = index_path.read_text() if index_path.exists() else ""
        if supplement not in content:
            index_path.write_text(content + supplement)
        return
    (output / "README.md").write_text(
        "# Chronological sampling for LLM-free parent preservation\n\n"
        "Every figure has four panels: Qwen/Gemma rows and task-mean/period-mean columns. "
        "Curves compare seven sampling pools using the same checkpoint cohort and paired sampled task prefixes.\n\n"
        "- Parent preservation minimizes parent-success regressions, then maximizes repairs, then maximizes "
        "parent-relative turn reduction on joint successes. All three priorities use the indicated weights. "
        "It does not use an LLM progress or violation judge.\n"
        "- Task mean gives each sampled task weight 1/n; period mean gives each represented period equal weight.\n"
        "- Current appears at most once. Past and future receive equal sample counts when both are present; "
        "an odd spare task goes to the past. Current-only uses one task even where nominal n is larger.\n"
        "- Future-task conditions are retrospective analyses of archived evaluations.\n"
        "- Shading is one standard deviation across Monte Carlo draws of the checkpoint-averaged metric. "
        "It measures sample-selection variability, not confidence in a population effect.\n"
        "- Lift is selected validation score minus the average validation score of all five candidates; "
        "it is not parent-relative validation gain. Ties receive uniform selection probability.\n"
        "- Full-validation uses all 50 validation tasks; its validation-sample reference can overlap that target. "
        "The remaining-40 target excludes the frozen ten validation tasks used by its own reference, "
        "removing that reference's direct evaluation-set overlap.\n"
        "- Gray stars are validation-sample selection at the measured n values only. Validation samples "
        "are never used by the seven training-only selectors.\n"
        "- Mixed-period cases start at their minimum feasible n. Missing points are not zero.\n"
        "- Correlation uses the parent-preservation lexicographic ordering's ranks (Spearman), "
        "avoiding an arbitrary scalar encoding of its priorities. Undefined within-checkpoint correlations "
        "are omitted on a common valid cohort for both selection methods; counts are retained in summary.json.\n\n"
        + "\n".join(f"- [{stem}.pdf]({stem}.pdf) · [PNG]({stem}.png)" for stem in diagnostics["files"])
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--surface-summary", type=Path)
    parser.add_argument("--uniform-reference", type=Path)
    args = parser.parse_args()
    output = args.output or args.summary.parent / "figures"
    plot(json.loads(args.summary.read_text()), output,
         uniform_reference=json.loads(args.uniform_reference.read_text()) if args.uniform_reference else None)
    if args.surface_summary:
        plot_surfaces(json.loads(args.surface_summary.read_text()), output / "candidate_count")


if __name__ == "__main__":
    main()
