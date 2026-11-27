# SPDX-License-Identifier: Apache-2.0
"""Apply all Gemma26 selected-case reruns to the existing corrected Gemma31 bundle."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from scripts.ace import plot_mean_abs_advantage_gap as distance_plot
from scripts.ace import update_gemma31_diagnostic_figures as prior

NOTE = (
    "Gemma26: 115 and Gemma31: 52 prior-format-error cases replaced by reruns; other outcomes retained. "
    "Selected-case diagnostic; changes include resampling variation. Gemma w/ GT unmeasured."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=prior.BUNDLE)
    parser.add_argument("--diagnostic", type=Path, default=prior.DIAGNOSTIC)
    args = parser.parse_args()
    bundle, diagnostic = args.bundle.resolve(), args.diagnostic.resolve()
    update = bundle / "updates/gemma26_format_rerun_20261127"
    if (update / "receipt.json").exists():
        print("Already updated: " + str(update / "receipt.json"), flush=True)
        return
    if prior.read(diagnostic / "status.json")["state"] != "completed":
        raise RuntimeError("Both Gemma diagnostic cohorts must have completed")
    backup, staging = update / "before", update / "staging"
    update.mkdir(parents=True, exist_ok=True)
    if not (update / "backup_manifest.json").exists():
        for path in bundle.iterdir():
            if path.name == "updates":
                continue
            destination = backup / path.name
            if path.is_dir():
                shutil.copytree(path, destination, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        prior.write(update / "backup_manifest.json", {
            str(p.relative_to(backup)): prior.digest(p) for p in backup.rglob("*") if p.is_file()
        })
    shutil.copytree(backup, staging, dirs_exist_ok=True)
    old_seeds = prior.csv_read(backup / "data/accumulated/metrics_by_seed.csv")
    new_seed, changes, comparisons, pins = prior.calculate(diagnostic, old_seeds, cohort="26b")
    by_seed = [dict(row) for row in old_seeds]
    for row in by_seed:
        if row["model"] == "gemma26" and row["arm"] == "without_gt":
            scores = new_seed[row["mode"], int(row["seed"]), row["strategy"]]
            for metric in prior.FIELDS:
                row[f"accumulated_{metric}"] = scores[metric]
    initial = prior.csv_read(backup / "data/accumulated/initial_metrics.csv")
    if any("pass_power_5" in row for row in initial):
        raise ValueError("The user removed pass^5 bars; preserve the current no-pass^5 layout")
    prior.BAR_NOTE = NOTE
    prior.SCATTER_NOTE = (
        "Gemma26/31 include 115/52 selected-case reruns; other outcomes retained. "
        "Selection bias applies; correlations are descriptive."
    )
    prior.render_bars(staging, by_seed, initial,
                      prior.csv_read(backup / "data/accumulated/metrics_mean_std.csv"))
    correlations = prior.render_scatters(staging, updated_models=("gemma26",))
    selected_stats = prior.read(diagnostic / "26b/summary.json")["selected_cases"]
    receipt = {
        "generated_at": datetime.now(UTC).isoformat(), "variant": "gemma26_selected115_rerun_after_gemma31_selected52",
        "rerun_counts": {"gemma26": 115, "gemma31": 52}, "gemma26_original_final_train_rollouts": 1500,
        "gemma26_original_outcomes_retained": 1385, "gemma31_previously_updated_outcomes_preserved": True,
        "selected_gemma26_cases": selected_stats, "backup": str(backup),
        "limitations": NOTE + " Gemma26 had zero actual Python-fence repairs in these new trajectories; "
        "its score changes cannot be attributed to a regex transformation.",
        "mean_std": "Mean and sample SD across seed42,43,44 Train50 means; paired sample SD for Best-Worst gaps",
        "candidate_violins_initial_metrics_and_x_coordinates": "Unchanged; no initial/candidate evaluations rerun",
        "correlations_descriptive": correlations, "source_result_sha256": pins,
    }
    # Stamp metadata before rendering the distance graph, so its cohort note is accurate.
    for relative in ("verification.json", "scatter_and_gain_verification.json", "data/accumulated/provenance.json"):
        record = prior.read(staging / relative)
        record["diagnostic_update"] = {k: v for k, v in receipt.items() if k != "source_result_sha256"}
        prior.write(staging / relative, record)
    points, group_rows, distance_pins = distance_plot.collect(staging)
    x, y = (np.array([p[field] for p in points]) for field in ("mean_abs_advantage", "gap_mean_pp"))
    slope, intercept = np.polyfit(x, y, 1)
    distance_stats = {
        "pearson_r": float(pearsonr(x, y).statistic), "spearman_rho": float(spearmanr(x, y).statistic),
        "ols_slope_gap_pp_per_mean_abs_advantage": float(slope), "ols_intercept_gap_pp": float(intercept),
    }
    distance_paths = distance_plot.draw(staging, points, distance_stats)
    distance_folder = staging / "data/mean_abs_advantage"
    distance_plot.write_csv(distance_folder / "scatter_points.csv", points)
    distance_plot.write_csv(distance_folder / "task_group_distances.csv", group_rows)
    prior.write(distance_folder / "provenance.json", {
        "generated_at": receipt["generated_at"], "statistics_descriptive": distance_stats,
        "formula": "sum(abs(A)) / 250; includes exact zero and all-tied task groups",
        "source_sha256": distance_pins, "diagnostic_update": receipt["variant"],
        "rerun_counts": receipt["rerun_counts"], "limitations": receipt["limitations"],
        "generated_figure_sha256": {p.name: prior.digest(p) for p in distance_paths}, "new_inference_calls": 0,
    })
    receipt["mean_abs_advantage_correlations_descriptive"] = distance_stats
    # Verify every other model's values and SDs, including the corrected Gemma31.
    for metric in prior.FIELDS:
        old = prior.csv_read(backup / "data/accumulated" / f"plotted_bars_{metric}.csv")
        new = prior.csv_read(staging / "data/accumulated" / f"plotted_bars_{metric}.csv")
        if len(new) != 100 or any(r["category"] == "pass_power_5" for r in new):
            raise ValueError("Unexpected bar layout")
        for a, b in zip(old, new, strict=True):
            if any(a[k] != b[k] for k in ("model", "mode", "arm", "category", "x")):
                raise ValueError("Bar identity or placement changed")
            if a["model"] != "gemma26" or a["category"] == "initial":
                for field in ("mean_percent", "std_percent"):
                    if a[field] != b[field]:
                        raise ValueError("An unrelated bar's value or SD changed")
        relative = Path("data/accumulated") / f"reference_lines_{metric}.csv"
        if prior.digest(backup / relative) != prior.digest(staging / relative):
            raise ValueError("Initial reference lines changed")
    for relative in (backup / "data/overlap_scatter").glob("*_points.csv"):
        a_rows = prior.csv_read(relative)
        b_rows = prior.csv_read(staging / relative.relative_to(backup))
        for a, b in zip(a_rows, b_rows, strict=True):
            for key in ("model", "mode", "arm", "overlap_metric", "overlap_percent"):
                if a[key] != b[key]:
                    raise ValueError("Overlap X or identity changed")
            if a["model"] != "gemma26":
                for key in ("gap_mean_pp", "gap_std_pp"):
                    if abs(float(a[key]) - float(b[key])) > 1e-12:
                        raise ValueError("An unrelated scatter Y or SD changed")
    old_distance = prior.csv_read(backup / "data/mean_abs_advantage/scatter_points.csv")
    for a, b in zip(old_distance, points, strict=True):
        if abs(float(a["mean_abs_advantage"]) - b["mean_abs_advantage"]) > 1e-12:
            raise ValueError("A raw candidate advantage distance changed")
    candidate_figures = [p for p in backup.iterdir() if p.suffix in {".png", ".pdf", ".svg"}
                         and "accumulated" not in p.name and "overlap_vs" not in p.name
                         and "mean_abs_grpo" not in p.name]
    if any(prior.digest(p) != prior.digest(staging / p.name) for p in candidate_figures):
        raise ValueError("A candidate violin figure changed")
    prior.write(update / "context_scores_before_after.json", changes)
    prior.write(update / "selected_cases_before_after.json", comparisons)
    for path in (Path(__file__), Path(prior.__file__), Path(distance_plot.__file__)):
        shutil.copy2(path, staging / "source_snapshot" / path.name)
    receipt["script_sha256"] = {p.name: prior.digest(p) for p in
                                 (Path(__file__), Path(prior.__file__), Path(distance_plot.__file__))}
    with (staging / "README.txt").open("a") as stream:
        stream.write("\nUPDATED with all Gemma26 selected115 reruns, keeping Gemma31 selected52 results. " +
                     receipt["limitations"] + "\nPCT/TGC bars, all overlap-gap scatter variants and mean-|A| "
                     "scatter refreshed; candidate violins and initial scores unchanged. pass^5 bars remain removed. "
                     "Backup/receipt: updates/gemma26_format_rerun_20261127/.\n")
    published = {}
    for path in staging.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(staging)
        if (backup / relative).exists() and prior.digest(backup / relative) == prior.digest(path):
            continue
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        published[str(relative)] = prior.digest(target)
    receipt["published_sha256"] = published
    receipt["unchanged_candidate_figure_files"] = len(candidate_figures)
    prior.write(update / "receipt.json", receipt)
    print(json.dumps({"bundle": str(bundle), "published_files": len(published), "gemma26_replaced_cases": 115,
                      "gemma31_replaced_cases_retained": 52, "distance_correlations": distance_stats,
                      "backup": str(backup), "verified": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
