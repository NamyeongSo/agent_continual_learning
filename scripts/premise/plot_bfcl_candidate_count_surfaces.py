# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Plot measured sample-count/candidate-count validation selection surfaces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.bfcl_trajectory_selection_rewards import REWARD_KEYS  # noqa: E402
from scripts.premise.plot_bfcl_process_reward_ablations import LABELS as PROCESS_LABELS  # noqa: E402
from scripts.premise.plot_bfcl_trajectory_rewards import LABELS, MODELS, metric_value  # noqa: E402

COHORTS = ("primary", "objective", "process_ablation")
COHORT_LABELS = {"primary": "Semantic", "objective": "Trace-only", "process_ablation": "Process ablation"}


def surface_grid(rows: list[dict], model: str, cohort: str, recipe: str) -> tuple[np.ndarray, np.ndarray]:
    """Return measured 5-by-10 grids, preserving gaps and checking lambda-one gain."""
    gains = np.full((5, 10), np.nan)
    counts = np.zeros((5, 10), dtype=int)
    observed = set()
    for row in rows:
        if (row.get("model"), row.get("cohort"), row.get("recipe")) != (model, cohort, recipe):
            continue
        n, candidates = row.get("n"), row.get("candidate_count")
        if not isinstance(n, int) or not isinstance(candidates, int) or not 1 <= n <= 10 or not 1 <= candidates <= 5:
            raise ValueError("Surface axes require n=1..10 and candidate_count=1..5")
        key = (candidates, n)
        if key in observed:
            raise ValueError(f"Duplicate surface grid cell: {model}/{cohort}/{recipe}/{key}")
        observed.add(key)
        value, count = metric_value(row, "validation_lift"), row.get("checkpoints", 0)
        if not isinstance(count, int) or count < 0 or (count == 0 and np.isfinite(value)):
            raise ValueError("Measured surface cells require positive checkpoint coverage")
        if candidates == 1 and np.isfinite(value) and abs(value) > 1e-10:
            raise ValueError("One uniformly sampled candidate must have zero lift over the proposal mean")
        gains[candidates - 1, n - 1] = 100 * value
        counts[candidates - 1, n - 1] = count
    for column in counts.T:
        observed_counts = column[column > 0]
        if len(observed_counts) and not np.all(observed_counts == observed_counts[0]):
            raise ValueError("Candidate-count surfaces require matched checkpoint coverage at each n")
    return gains, counts


def plot(summary: dict, output: Path) -> None:
    """Write one two-model 3D surface and heatmap for each measured reward rule."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    rows = summary.get("aggregate", [])
    if any(row.get("model") not in MODELS or row.get("cohort") not in COHORTS for row in rows):
        raise ValueError("Only Qwen/Gemma measured trajectory-reward cohorts are supported")
    output.mkdir(parents=True, exist_ok=True)
    fixed_cohort = bool(summary.get("fixed_cohort") or summary.get("cohort_scope") == "fixed_across_n")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                         "axes.linewidth": .65, "pdf.fonttype": 42, "ps.fonttype": 42})
    cmap = LinearSegmentedColormap.from_list(
        "validation_signed", [(0, "#593140"), (.25, "#B68C9B"), (.5, "#F9EEF1"),
                              (.65, "#C88FA0"), (.8, "#B56780"), (1, "#A34F65")],
    )
    cmap.set_bad("#EEEEEE")
    x, y = np.meshgrid(np.arange(1, 11), np.arange(1, 6))
    diagnostics = {"analysis_created_at": summary.get("created_at"), "files": [], "surfaces": [],
                   "fixed_checkpoint_cohort_across_n": fixed_cohort,
                   "metric": "Selected validation score minus five-proposal mean, in percentage points",
                   "sample_pool": "Original frozen uniform random training-task prefix, not current + past",
                   "candidate_subsets": "Exact uniform average over all subsets of the five existing candidates"}

    def save(figure, stem):
        for extension in ("pdf", "png"):
            figure.savefig(output / f"{stem}.{extension}", dpi=350, bbox_inches="tight")
        plt.close(figure)
        diagnostics["files"].append(stem)
        print(f"Saved {stem}.pdf", flush=True)

    def model_title(model, cohort, counts):
        nonzero = counts[counts > 0]
        if not len(nonzero):
            return f"{model} (0 CPs)"
        minimum, maximum = int(nonzero.min()), int(nonzero.max())
        label = str(minimum) if minimum == maximum else f"{minimum}-{maximum}"
        suffix = "; partial" if cohort == "primary" and minimum < 50 else ""
        n10 = counts[-1, -1]
        scope = "; fixed across n" if fixed_cohort else f"; n=10: {n10}"
        return f"{model} ({label} CPs{suffix}{scope})"

    for cohort in COHORTS:
        recipes = list(dict.fromkeys(row["recipe"] for row in rows if row["cohort"] == cohort))
        for recipe in recipes:
            grids = {model: surface_grid(rows, model, cohort, recipe) for model in MODELS}
            finite = np.concatenate([grid[np.isfinite(grid)] for grid, _ in grids.values()])
            if not len(finite):
                continue
            bound = max(float(np.abs(finite).max()) * 1.08, .05)
            norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
            label = (LABELS | PROCESS_LABELS).get(recipe, recipe.replace("_", " ").title())
            prefix = f"{'semantic' if cohort == 'primary' else cohort}_{recipe}"
            figure = plt.figure(figsize=(8.4, 3.8))
            figure.subplots_adjust(left=.015, right=.91, bottom=.13, top=.86, wspace=.02)
            figure.suptitle(f"{label} · {COHORT_LABELS[cohort]}", fontsize=9, y=.98)
            for column, model in enumerate(MODELS):
                gains, counts = grids[model]
                axis = figure.add_subplot(1, 2, column + 1, projection="3d")
                axis.set_title(model_title(model, cohort, counts), fontsize=8, pad=4)
                if np.isfinite(gains).any():
                    axis.plot_surface(x, y, gains, cmap=cmap, norm=norm, linewidth=.3,
                                      edgecolor="#C3A2AE", antialiased=True, alpha=.97, shade=False,
                                      rstride=1, cstride=1)
                    axis.scatter(x[np.isfinite(gains)], y[np.isfinite(gains)], gains[np.isfinite(gains)],
                                 s=3.5, color="#593140", alpha=.7, depthshade=False)
                    axis.plot(x[0], y[0], np.zeros(10), color="#777777", linewidth=1.1)
                else:
                    axis.text2D(.5, .5, "No eligible CPs", transform=axis.transAxes, ha="center")
                axis.set(xlim=(1, 10), ylim=(1, 5), zlim=(-bound, bound), xticks=(1, 3, 5, 7, 10),
                         yticks=(1, 2, 3, 4, 5))
                axis.set_xlabel(r"Sample size $n$", fontsize=8, labelpad=2)
                axis.set_ylabel(r"Candidates $\lambda$", fontsize=8, labelpad=2)
                axis.set_zlabel("Validation lift (pp)", fontsize=8, labelpad=3)
                axis.tick_params(labelsize=6.5, pad=0)
                axis.view_init(elev=27, azim=-128)
                axis.set_box_aspect((1.2, 1, .85))
                for coordinate in (axis.xaxis, axis.yaxis, axis.zaxis):
                    coordinate.pane.fill = False
                    coordinate._axinfo["grid"].update({"linewidth": .4, "color": (.84, .84, .84, .5)})
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            colorbar = figure.colorbar(mappable, cax=figure.add_axes((.94, .22, .014, .52)))
            colorbar.set_label("Validation lift (pp)", fontsize=7)
            colorbar.ax.tick_params(labelsize=6.5)
            save(figure, f"{prefix}_surface_validation")

            figure, axes = plt.subplots(1, 2, figsize=(8.4, 2.95), sharex=True, sharey=True)
            figure.subplots_adjust(left=.07, right=.91, bottom=.19, top=.76, wspace=.12)
            figure.suptitle(f"{label} · {COHORT_LABELS[cohort]}", fontsize=9, y=.98)
            for axis, model in zip(axes, MODELS, strict=True):
                gains, counts = grids[model]
                axis.imshow(gains, origin="lower", aspect="auto", cmap=cmap, norm=norm,
                            extent=(.5, 10.5, .5, 5.5), interpolation="none")
                for candidate_index, n_index in np.argwhere(np.isfinite(gains)):
                    value = gains[candidate_index, n_index]
                    text_color = "white" if abs(value) > .65 * bound else "#3C3034"
                    axis.text(n_index + 1, candidate_index + 1, f"{value:+.2f}", ha="center", va="center",
                              fontsize=5.7, color=text_color)
                axis.set_title(model_title(model, cohort, counts), fontsize=8, pad=5)
                axis.set(xticks=np.arange(1, 11), yticks=np.arange(1, 6))
                axis.set_xlabel(r"Selection sample size $n$", fontsize=8)
                axis.tick_params(labelsize=7, length=2)
            axes[0].set_ylabel(r"Candidate count $\lambda$", fontsize=8)
            colorbar = figure.colorbar(mappable, cax=figure.add_axes((.94, .24, .014, .44)))
            colorbar.set_label("Validation lift (pp)", fontsize=7)
            colorbar.ax.tick_params(labelsize=6.5)
            save(figure, f"{prefix}_heatmap_validation")
            diagnostics["surfaces"].append({"cohort": cohort, "recipe": recipe, "color_limit_pp": bound,
                                             "models": [{"model": model, "checkpoints": counts.tolist(),
                                                         "finite_grid_cells": int(np.isfinite(gains).sum())}
                                                        for model, (gains, counts) in grids.items()]})

    # The seven-recipe overview shares one scale across all methods and models.
    # Individual recipe figures above retain a tighter scale for reading details.
    primary_recipes = [recipe for recipe in REWARD_KEYS
                       if any(row["cohort"] == "primary" and row["recipe"] == recipe for row in rows)]
    overview_grids = {(model, recipe): surface_grid(rows, model, "primary", recipe)
                      for recipe in primary_recipes for model in MODELS}
    finite = np.concatenate([grid[np.isfinite(grid)] for grid, _ in overview_grids.values()]) if overview_grids else []
    if len(finite):
        bound = max(float(np.abs(finite).max()) * 1.08, .05)
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        figure = plt.figure(figsize=(8.4, 2.65 * len(primary_recipes)))
        figure.subplots_adjust(left=.015, right=.91, bottom=.025, top=.975, wspace=.04, hspace=.18)
        for recipe_index, recipe in enumerate(primary_recipes):
            for column, model in enumerate(MODELS):
                gains, counts = overview_grids[model, recipe]
                axis = figure.add_subplot(len(primary_recipes), 2, recipe_index * 2 + column + 1, projection="3d")
                coverage = model_title(model, "primary", counts)
                axis.set_title(f"{LABELS[recipe]} · {coverage}", fontsize=7.3, pad=2)
                if np.isfinite(gains).any():
                    axis.plot_surface(x, y, gains, cmap=cmap, norm=norm, linewidth=.3, edgecolor="#C3A2AE",
                                      antialiased=True, alpha=.97, shade=False, rstride=1, cstride=1)
                    axis.scatter(x[np.isfinite(gains)], y[np.isfinite(gains)], gains[np.isfinite(gains)],
                                 s=3, color="#593140", alpha=.7, depthshade=False)
                    axis.plot(x[0], y[0], np.zeros(10), color="#777777", linewidth=.8)
                axis.set(xlim=(1, 10), ylim=(1, 5), zlim=(-bound, bound), xticks=(1, 5, 10),
                         yticks=(1, 3, 5))
                axis.set_xlabel(r"Sample size $n$", fontsize=7, labelpad=0)
                axis.set_ylabel(r"Candidates $\lambda$", fontsize=7, labelpad=0)
                axis.set_zlabel("Validation lift (pp)", fontsize=7, labelpad=2)
                axis.tick_params(labelsize=6, pad=-1)
                axis.view_init(elev=27, azim=-128)
                axis.set_box_aspect((1.2, 1, .85))
                for coordinate in (axis.xaxis, axis.yaxis, axis.zaxis):
                    coordinate.pane.fill = False
                    coordinate._axinfo["grid"].update({"linewidth": .35, "color": (.84, .84, .84, .5)})
        colorbar = figure.colorbar(mappable, cax=figure.add_axes((.94, .14, .014, .72)))
        colorbar.set_label("Validation lift (pp); shared across all methods/models", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
        save(figure, "semantic_candidate_count_surface_overview")

        figure, axes = plt.subplots(len(primary_recipes), 2, figsize=(8.4, 1.9 * len(primary_recipes)),
                                    sharex=True, sharey=True, squeeze=False)
        figure.subplots_adjust(left=.07, right=.91, bottom=.045, top=.975, wspace=.12, hspace=.38)
        for recipe_index, recipe in enumerate(primary_recipes):
            for column, model in enumerate(MODELS):
                axis = axes[recipe_index, column]
                gains, counts = overview_grids[model, recipe]
                axis.imshow(gains, origin="lower", aspect="auto", cmap=cmap, norm=norm,
                            extent=(.5, 10.5, .5, 5.5), interpolation="none")
                for candidate_index, n_index in np.argwhere(np.isfinite(gains)):
                    value = gains[candidate_index, n_index]
                    text_color = "white" if abs(value) > .65 * bound else "#3C3034"
                    axis.text(n_index + 1, candidate_index + 1, f"{value:+.2f}", ha="center", va="center",
                              fontsize=5.7, color=text_color)
                axis.set_title(f"{LABELS[recipe]} · {model_title(model, 'primary', counts)}", fontsize=7.3, pad=4)
                axis.set(xticks=np.arange(1, 11), yticks=np.arange(1, 6))
                axis.tick_params(labelsize=6.5, length=2)
                if column == 0:
                    axis.set_ylabel(r"Candidates $\lambda$", fontsize=7)
                if recipe_index == len(primary_recipes) - 1:
                    axis.set_xlabel(r"Selection sample size $n$", fontsize=8)
        colorbar = figure.colorbar(mappable, cax=figure.add_axes((.94, .14, .014, .72)))
        colorbar.set_label("Validation lift (pp); shared across all methods/models", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
        save(figure, "semantic_candidate_count_heatmap_overview")
        diagnostics["overview"] = {"cohort": "primary", "recipes": primary_recipes,
                                    "panels_per_figure": 2 * len(primary_recipes), "global_color_limit_pp": bound}

    (output / "figure_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
    (output / "README.md").write_text(
        "# Sample count and candidate count surfaces\n\n"
        "For each reward rule, a two-panel 3D surface and an annotated heatmap compare Qwen/Gemma. "
        "Axes are n=1..10 queried training tasks, lambda=1..5 candidate contexts, and full-validation "
        "lift over the mean of all five proposals, in percentage points. "
        "The color and vertical scales are shared between models within each reward rule.\n\n"
        "The semantic overview contains seven methods by two models, fourteen panels, with a single "
        "global color/vertical scale across every method and model. Its heatmap companion uses the same scale.\n\n"
        "These plots use the original frozen uniform-random training-task prefix, not the requested "
        "current-plus-past rerun. Candidate-count results enumerate all subsets of the five existing "
        "candidates and average selection uniformly over subsets and tied winners. "
        "No additional model calls are used. Lambda=1 is exactly zero by construction, and lambda=5 "
        "recovers the existing five-candidate result. The surface simply joins measured grid cells; "
        "it is not a fitted scaling law or additional generated candidate distribution.\n\n"
        + ("All n and lambda values use the same fixed checkpoint cohort within each model and source cohort. "
           if fixed_cohort else
           "Checkpoint composition can vary with n; lambda values share the same cohort within each n. "
           "Do not interpret variation across n as a sample-size effect alone. ")
        + "Semantic panels use the available complete-case cohort and are explicitly marked partial. "
        "Trace-only and process-ablation cohorts are separately labeled. Gray cells mean unavailable "
        "data, never zero gain. Semantic reward rules share the same matched complete-case cohort "
        "within each model and n; exact coverage is in figure_diagnostics.json. "
        "Surfaces are descriptive and do not indicate statistical significance.\n\n"
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
