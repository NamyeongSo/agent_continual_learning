# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Plot measured trajectory-grounded selection rewards for Qwen and Gemma."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.premise.bfcl_trajectory_selection_rewards import (  # noqa: E402
    LEXICOGRAPHIC_KEYS,
    REWARD_KEYS,
)
from scripts.premise.plot_bfcl_sample_size_contours import ellipse_points  # noqa: E402

MODELS = ("Qwen", "Gemma")
TARGET_STYLE = {"full_train": ("#287C85", "#EAF4F5"), "full_validation": ("#A34F65", "#F9EEF1")}
LABELS = {
    "success_count": "Success",
    "success_efficiency": "Efficient completion",
    "failure_progress": "Grounded progress",
    "certified_completion": "Certified completion",
    "parent_safe": "Parent preservation",
    "successful_recovery": "Recovery",
    "quality_gated": "Quality gates",
}
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
LINESTYLES = ("-", "--", "-.", ":", (0, (4, 1, 1, 1)), (0, (1, 1)), (0, (5, 2)))
PRESPECIFIED_BIVARIATE = ("success_count", "failure_progress", "certified_completion")
OBJECTIVE_RECIPES = ("success_count", "success_efficiency", "successful_recovery")


def metric_value(row: dict, metric: str, statistic: str = "mean") -> float:
    """Missing observations stay missing, never a plotted synthetic zero."""
    number = row.get(metric)
    if isinstance(number, dict):
        number = number.get(statistic)
    elif statistic != "mean":
        return np.nan
    if number is None:
        return np.nan
    number = float(number)
    return number if np.isfinite(number) else np.nan


def standardize_bivariate(
    rows: list[dict], target: str, recipes: tuple[str, ...] = PRESPECIFIED_BIVARIATE,
    *, cohort: str = "primary",
) -> tuple[dict[tuple[str, str], np.ndarray], list[dict]]:
    """Pair actual candidate clouds on jointly nonconstant within-CP scores.

    Each CP contributes exactly five standardized candidate pairs to each recipe.
    Constants are omitted jointly across the prespecified scalar recipes. The
    resulting pooled Pearson rho equals their mean within-CP Pearson rho.
    """
    if (target not in TARGET_STYLE or cohort not in ("primary", "objective", "process_ablation") or not recipes
            or any(key in LEXICOGRAPHIC_KEYS for key in recipes)):
        raise ValueError("Bivariate Pearson clouds require a known target and prespecified scalar recipes")
    selected = [row for row in rows if row.get("cohort") == cohort and row.get("n") == 10
                and row.get("recipe") in recipes]
    indexed = {}
    for row in selected:
        key = (row["model"], row["key"], row["recipe"])
        if key in indexed:
            raise ValueError(f"Duplicate checkpoint/recipe row: {key}")
        indexed[key] = row
    clouds, notes = {}, []
    for model in MODELS:
        keys = sorted({key for label, key, _ in indexed if label == model})
        point_lists = {recipe: [] for recipe in recipes}
        retained = []
        for key in keys:
            if not all((model, key, recipe) in indexed for recipe in recipes):
                continue
            candidates = [np.asarray(indexed[model, key, recipe]["candidate_scores"], float) for recipe in recipes]
            targets = [np.asarray(indexed[model, key, recipe][f"{target}_scores"], float) for recipe in recipes]
            if any(array.shape != (5,) or not np.isfinite(array).all() for array in [*candidates, *targets]):
                raise ValueError(f"Nonfinite/non-scalar candidate cloud: {model}/{key}")
            if not all(np.allclose(array, targets[0], rtol=0, atol=1e-12) for array in targets):
                raise ValueError(f"Target differs across recipe rows: {model}/{key}")
            if any(array.std() <= 1e-12 for array in [*candidates, targets[0]]):
                continue
            target_z = (targets[0] - targets[0].mean()) / targets[0].std()
            for recipe, candidate in zip(recipes, candidates, strict=True):
                candidate_z = (candidate - candidate.mean()) / candidate.std()
                point_lists[recipe].append(np.column_stack((candidate_z, target_z)))
            retained.append(key)
        for recipe in recipes:
            cloud = np.vstack(point_lists[recipe]) if retained else np.empty((0, 2))
            clouds[model, recipe] = cloud
            rho = float(np.corrcoef(cloud.T)[0, 1]) if len(cloud) else None
            notes.append({"model": model, "target": target, "recipe": recipe, "cohort": cohort,
                          "common_defined_checkpoints": len(retained), "candidate_pairs": len(cloud),
                          "rho": rho, "checkpoint_keys": retained})
    return clouds, notes


def plot(summary: dict, output: Path) -> None:
    """Write PDFs/PNGs only from measured summary records, allowing partial coverage."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = summary.get("aggregate", [])
    checkpoint_rows = summary.get("checkpoint_scores", [])
    semantic_is_partial = any(row["primary_checkpoints"] < row["expected_checkpoints"]
                              for row in summary.get("coverage", []))
    if any(row.get("model") not in MODELS for row in [*rows, *checkpoint_rows]):
        raise ValueError("Trajectory reward figures include Qwen and Gemma only")
    indexed = {}
    for row in rows:
        key = (row["model"], row["cohort"], row["recipe"], row["n"])
        if key in indexed:
            raise ValueError(f"Duplicate aggregate row: {key}")
        indexed[key] = row
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .65,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    sizes = np.arange(1, 11)
    diagnostics = {"files": [], "bivariate": [], "semantic_partial": semantic_is_partial,
                   "analysis_created_at": summary.get("created_at"),
                   "judged_groups_valid": summary.get("judged_groups_valid"),
                   "query_groups": summary.get("query_groups"), "aggregate_coverage": [
        {"model": row["model"], "cohort": row["cohort"], "recipe": row["recipe"], "n": row["n"],
         "checkpoints": row.get("checkpoints")} for row in rows]}

    def save(figure, stem):
        for extension in ("pdf", "png"):
            figure.savefig(output / f"{stem}.{extension}", dpi=350, bbox_inches="tight")
        plt.close(figure)
        diagnostics["files"].append(stem)
        print(f"Saved {stem}.pdf", flush=True)

    def series(model, cohort, recipe, metric):
        return np.asarray([metric_value(indexed.get((model, cohort, recipe, int(n)), {}), metric) for n in sizes])

    def handles(recipes, color="#444444"):
        return [Line2D([], [], color=color, linewidth=1, marker=MARKERS[REWARD_KEYS.index(recipe)],
                       linestyle=LINESTYLES[REWARD_KEYS.index(recipe)], markersize=3, label=LABELS[recipe])
                for recipe in recipes]

    def coverage_title(model, cohort):
        counts = [indexed[model, cohort, "success_count", int(n)]["checkpoints"] for n in sizes
                  if (model, cohort, "success_count", int(n)) in indexed]
        if not counts:
            return f"{model} (0 CPs)"
        minimum, maximum = min(counts), max(counts)
        label = str(minimum) if minimum == maximum else f"{minimum}-{maximum}"
        partial = "; partial" if cohort == "primary" and semantic_is_partial else ""
        return f"{model} ({label} CPs{partial})"

    def curve_figure(cohort, recipes, metric, factor, ylabel, target, stem):
        if not any(np.isfinite(series(model, cohort, recipe, metric)).any()
                   for model in MODELS for recipe in recipes):
            return
        color = TARGET_STYLE[target][0]
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), sharex=True, sharey=True, squeeze=False)
        figure.subplots_adjust(left=.10, right=.985, bottom=.19, top=.83 if len(recipes) <= 4 else .76,
                               wspace=.13)
        for column, model in enumerate(MODELS):
            axis = axes[0, column]
            for recipe in recipes:
                numbers = factor * series(model, cohort, recipe, metric)
                if not np.isfinite(numbers).any():
                    continue
                index = REWARD_KEYS.index(recipe)
                axis.plot(sizes, numbers, color=color, marker=MARKERS[index], linestyle=LINESTYLES[index],
                          linewidth=1.35 if recipe == "success_count" else 1, markersize=3.1,
                          markerfacecolor="white" if recipe != "success_count" else color, markeredgewidth=.7)
            axis.axhline(0, color="#B9B9B9", linewidth=.5, zorder=0)
            axis.set_title(coverage_title(model, cohort), fontsize=8.5, pad=4)
            axis.set_xticks((1, 3, 5, 7, 10))
            axis.set_xlim(.7, 10.3)
            axis.set_xlabel(r"Selection sample size $n$", fontsize=8)
            axis.tick_params(labelsize=7, length=2.5)
            axis.grid(axis="y", color="#E9E9E9", linewidth=.4, zorder=0)
            if column == 0:
                axis.set_ylabel(ylabel, fontsize=8)
        figure.legend(handles=handles(recipes), ncol=min(4, len(recipes)), loc="upper center",
                      frameon=False, fontsize=7, columnspacing=1.1, handlelength=2.4)
        save(figure, stem)

    primary = [recipe for recipe in REWARD_KEYS if any(row["cohort"] == "primary" and row["recipe"] == recipe
                                                       for row in rows)]
    scalar = [recipe for recipe in primary if recipe not in LEXICOGRAPHIC_KEYS]
    for target, gain_metric, gain_label in (
        ("full_validation", "validation_lift", "Validation lift (pp)"),
        ("full_train", "train_parent_gain", r"Training gain $\delta_t^{i^\star}$ (pp)"),
    ):
        curve_figure("primary", primary, gain_metric, 100, gain_label, target,
                     f"semantic_gain_{target}_n1_10")
        curve_figure("primary", scalar, f"rho_{target}", 1, r"Pearson $\rho$", target,
                     f"semantic_correlation_pearson_{target}_n1_10")
        curve_figure("primary", primary, f"spearman_{target}", 1, r"Spearman $\rho_{\mathrm{S}}$", target,
                     f"semantic_correlation_spearman_{target}_n1_10")
    objective = [recipe for recipe in OBJECTIVE_RECIPES
                 if any(row["cohort"] == "objective" and row["recipe"] == recipe for row in rows)]
    for target, gain_metric, gain_label in (
        ("full_validation", "validation_lift", "Validation lift (pp)"),
        ("full_train", "train_parent_gain", r"Training gain $\delta_t^{i^\star}$ (pp)"),
    ):
        curve_figure("objective", objective, gain_metric, 100, gain_label, target,
                     f"objective_gain_{target}_n1_10")
        curve_figure("objective", objective, f"rho_{target}", 1, r"Pearson $\rho$", target,
                     f"objective_correlation_pearson_{target}_n1_10")
        curve_figure("objective", objective, f"spearman_{target}", 1, r"Spearman $\rho_{\mathrm{S}}$", target,
                     f"objective_correlation_spearman_{target}_n1_10")

    # Scalar-only cloud comparison is prespecified; no recipe is chosen on validation gain.
    for cohort, cloud_recipes, target in (
        (cohort, recipes, target)
        for cohort, recipes in (("primary", PRESPECIFIED_BIVARIATE), ("objective", OBJECTIVE_RECIPES))
        for target in TARGET_STYLE
    ):
        color, fill = TARGET_STYLE[target]
        clouds, notes = standardize_bivariate(checkpoint_rows, target, cloud_recipes, cohort=cohort)
        diagnostics["bivariate"].extend(notes)
        if not any(len(cloud) for cloud in clouds.values()):
            continue
        note_index = {(row["model"], row["recipe"]): row for row in notes}
        figure, axes = plt.subplots(2, 3, figsize=(7.3, 4.7), sharex=True, sharey=True, squeeze=False)
        figure.subplots_adjust(left=.10, right=.99, bottom=.14, top=.91, hspace=.18, wspace=.13)
        for row_index, model in enumerate(MODELS):
            for column, recipe in enumerate(cloud_recipes):
                axis = axes[row_index, column]
                cloud = clouds[model, recipe]
                note = note_index[model, recipe]
                if len(cloud):
                    covariance = np.cov(cloud, rowvar=False, ddof=0)
                    axis.scatter(cloud[:, 0], cloud[:, 1], s=7, color=color, alpha=.23,
                                 edgecolors="none", rasterized=True, zorder=2)
                    if np.linalg.eigvalsh(covariance).min() > 1e-12:
                        inner, outer = (ellipse_points(covariance, mass, 181) for mass in (.5, .9))
                        axis.fill(inner[:, 0], inner[:, 1], color=fill, zorder=1)
                        axis.plot(inner[:, 0], inner[:, 1], color=color, linewidth=.9, zorder=3)
                        axis.plot(outer[:, 0], outer[:, 1], color=color, linewidth=.7, linestyle="--", zorder=3)
                    annotation = rf"$\rho={note['rho']:+.3f}$" + f"\n{note['common_defined_checkpoints']} CPs"
                else:
                    annotation = "N/A\n0 CPs"
                axis.text(.04, .96, annotation, transform=axis.transAxes, ha="left", va="top",
                          fontsize=7, color=color, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8})
                axis.axhline(0, color="#D6D6D6", linewidth=.4, zorder=0)
                axis.axvline(0, color="#D6D6D6", linewidth=.4, zorder=0)
                axis.set(xlim=(-3, 3), ylim=(-3, 3), xticks=(-2, 0, 2), yticks=(-2, 0, 2))
                axis.set_aspect("equal", adjustable="box")
                axis.tick_params(labelsize=7, length=2.5)
                if row_index == 0:
                    axis.set_title(LABELS[recipe], fontsize=8, pad=5)
                if column == 0:
                    partial = cohort == "primary" and semantic_is_partial
                    prefix = f"{model} (partial)" if partial else model
                    target_label = "Full training score" if target == "full_train" else "Full validation score"
                    axis.set_ylabel(f"{prefix}\n{target_label} ($z$)", fontsize=8)
                if row_index == 1:
                    axis.set_xlabel("Standardized selection score", fontsize=8)
        save(figure, f"{'semantic' if cohort == 'primary' else 'objective'}_bivariate_{target}_n10")

    # These are selected-sample diagnostics, not out-of-sample efficiency estimates.
    for cohort, recipes in (("primary", primary), ("objective", objective)):
        available = [indexed.get((model, cohort, recipe, 10), {}) for model in MODELS for recipe in recipes]
        if not (any(np.isfinite(metric_value(row, "selected_environment_steps")) for row in available)
                and any(np.isfinite(metric_value(row, "validation_lift")) for row in available)):
            continue
        color = TARGET_STYLE["full_validation"][0]
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), sharey=True, squeeze=False)
        figure.subplots_adjust(left=.10, right=.985, bottom=.18, top=.83 if len(recipes) <= 4 else .76,
                               wspace=.13)
        for column, model in enumerate(MODELS):
            axis = axes[0, column]
            for recipe in recipes:
                row = indexed.get((model, cohort, recipe, 10), {})
                x, y = metric_value(row, "selected_environment_steps"), 100 * metric_value(row, "validation_lift")
                if np.isfinite(x) and np.isfinite(y):
                    axis.scatter(x, y, marker=MARKERS[REWARD_KEYS.index(recipe)], s=31, color=color,
                                 facecolors=color if recipe == "success_count" else "none", linewidths=.9)
            n10 = indexed.get((model, cohort, "success_count", 10), {}).get("checkpoints", 0)
            partial = "; partial" if cohort == "primary" and semantic_is_partial else ""
            axis.set_title(f"{model} ({n10} CPs{partial})", fontsize=8.5)
            axis.set_xlabel("Environment steps / selected task", fontsize=8)
            axis.axhline(0, color="#B9B9B9", linewidth=.5)
            axis.tick_params(labelsize=7, length=2.5)
            if column == 0:
                axis.set_ylabel("Validation lift (pp)", fontsize=8)
        figure.legend(handles=handles(recipes), ncol=4, loc="upper center", frameon=False, fontsize=7)
        save(figure, f"{'semantic' if cohort == 'primary' else 'objective'}_execution_validation_tradeoff_n10")

    (output / "figure_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
    (output / "README.md").write_text(
        "# Trajectory reward figures\n\n"
        "All plotted points come from measured checkpoint summaries; missing values remain missing. "
        "Qwen and Gemma only. Recipe order is fixed independently of validation performance.\n\n"
        "- `semantic_gain_*`: semantic complete-case (`primary`) cohort. Validation lift is selected "
        "validation score minus the five-proposal mean, not parent-relative validation gain. Training "
        "gain uses the measured parent baseline.\n"
        "- `*_correlation_pearson_*`: scalar rewards only. `*_correlation_spearman_*`: all rewards, "
        "including lexicographic rewards, using average ascending candidate ranks.\n"
        "- `objective_*`: separately labeled larger trace-only cohort; do not compare its "
        "mean directly to semantic recipes evaluated on a different checkpoint cohort.\n"
        "- `*_bivariate_*`: n=10 actual five-candidate pairs, standardized separately within each CP; "
        "only CPs with nonconstant targets and all three prespecified scalar recipes are included. "
        "Solid/dashed curves are descriptive fitted-Gaussian 50%/90% ellipses, not simulated data "
        "or confidence regions. Each panel reports its matched CP count and mean within-CP Pearson rho. "
        "The matched bivariate cohort can differ from a curve's individual correlation coverage.\n"
        "- `*_execution_validation_tradeoff_n10`: execution cost on selection tasks versus full "
        "validation lift. The cost coordinate is a mechanism diagnostic on optimized samples.\n\n"
        "The frozen ten uniformly drawn training tasks provide nested first-n samples for n=1..10; "
        "these are not fresh independent full-pool task draws at every n. No confidence shading is "
        "drawn: summary SD describes between-checkpoint heterogeneity, not uncertainty in its mean. "
        "Coverage and exact bivariate checkpoint IDs are recorded in `figure_diagnostics.json`. "
        "Panel titles report available CP counts, which can vary across n. Partial semantic figures "
        "must not be interpreted as results for all 50 CPs per model. A completed judgment queue "
        "does not imply valid judgments for every task: failed requests remain excluded.\n\n"
        "## Checkpoint coverage at n=10\n\n"
        "| Model | Trace-only cohort | Semantic complete cases |\n"
        "| --- | ---: | ---: |\n"
        + "\n".join(
            f"| {model} | {indexed.get((model, 'objective', 'success_count', 10), {}).get('checkpoints', 0)} | "
            f"{indexed.get((model, 'primary', 'success_count', 10), {}).get('checkpoints', 0)} |"
            for model in MODELS
        )
        + "\n\n"
        "## Figure index\n\n"
        + "\n".join(f"- [{stem}.pdf]({stem}.pdf) · [PNG]({stem}.png)" for stem in diagnostics["files"])
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    plot(summary, args.output or args.summary.parent / "figures")


if __name__ == "__main__":
    main()
