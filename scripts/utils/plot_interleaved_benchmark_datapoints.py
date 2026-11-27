# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Replot a frozen interleaved export against each benchmark's task encounter count."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils.plot_interleaved_online_performance import (  # noqa: E402
    BENCHMARKS,
    aggregate_windows,
    fit_linear_trends,
    interleaved_cumulative,
    save_benchmark_panels,
    write_csv,
)
from scripts.utils.plot_sequential_online_performance import (  # noqa: E402
    COLORS,
    LABELS,
    PLOT_LABEL_GID,
    plot_sequential_axis,
    save_figure_pair,
)


def benchmark_datapoint_rows(tasks: list[dict]) -> list[dict]:
    """Align each seed before aggregation, retaining missing encounters in the x axis."""
    groups = defaultdict(list)
    for task in tasks:
        if task["benchmark"] not in BENCHMARKS:
            raise ValueError(f"Unsupported benchmark: {task['benchmark']}")
        score = task["score"]
        if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Task scores must be finite values between zero and one")
        groups[task["method"], task["seed"]].append(task)
    result = []
    for (method, seed), encounters in sorted(groups.items()):
        encounters.sort(key=lambda task: task["iteration"])
        if [task["iteration"] for task in encounters] != list(range(1, len(encounters) + 1)):
            raise ValueError("Each method/seed must have unique contiguous global iterations starting at one")
        order = [(task["benchmark"], task["task_id"]) for task in encounters]
        scores = {task["iteration"]: task["score"] for task in encounters if task["score"] is not None}
        for row in interleaved_cumulative(order, scores, len(order)):
            if row["benchmark"] != row["current_task_benchmark"]:
                continue
            k = row["benchmark_iteration"]
            result.append(
                {
                    "method": method,
                    "seed": seed,
                    **row,
                    "global_iteration": row["iteration"],
                    "iteration": k,
                    "window_start": 1,
                    "window_end": k,
                }
            )
    return result


def plot_benchmark_regions(ax, rows: list[dict], *, linear_regression: bool = False) -> list[tuple[str, int, int]]:
    """Arrange local datapoints in adjacent benchmark regions with one common y axis."""
    segments = []
    shifted = []
    offsets = {}
    offset = 0
    for benchmark in BENCHMARKS:
        selected = [row for row in rows if row["benchmark"] == benchmark]
        count = max(row["iteration"] for row in selected)
        segments.append((benchmark, offset + 1, offset + count))
        offsets[benchmark] = offset
        shifted.extend({**row, "iteration": row["iteration"] + offset} for row in selected)
        offset += count
    plot_sequential_axis(ax, shifted, "cumulative", segments)
    headings = {
        f"{LABELS[benchmark]}\nIterations {start}-{end}": f"{LABELS[benchmark]}\nDatapoints 1-{end - start + 1}"
        for benchmark, start, end in segments
    }
    for text in ax.texts:
        if text.get_text() in headings:
            text.set_text(headings[text.get_text()])
    ticks, labels = [], []
    for block, (_benchmark, start, end) in enumerate(segments):
        count = end - start + 1
        local_ticks = ([1] if block == 0 else []) + list(range(10, count + 1, 10))
        for k in local_ticks:
            ticks.append(start + k - 1)
            labels.append(str(k))
    ax.set_xticks(ticks, labels)
    ax.set_xlabel("Benchmark datapoint (counts restart in each region)", fontsize=11)
    ax.set_ylabel("Cumulative online score (%)", fontsize=11)
    endpoints = defaultdict(list)
    for annotation in ax.texts:
        if annotation.get_gid() == PLOT_LABEL_GID and hasattr(annotation, "xy"):
            endpoints[annotation.xy[0]].append(annotation)
    for group_labels in endpoints.values():
        group_labels.sort(key=lambda annotation: annotation.xy[1])
        positions = []
        for annotation in group_labels:
            score = annotation.xy[1]
            positions.append(max(score + 3, positions[-1] + 7) if positions else max(3, score - 5))
        if positions[-1] > 97:
            positions = [y - (positions[-1] - 97) for y in positions]
        for annotation, y in zip(group_labels, positions, strict=True):
            offset_y = (y - annotation.xy[1]) * ax.bbox.height / 100 * 72 / ax.figure.dpi
            annotation.set_position((-8, offset_y))
            connector = ax.annotate(
                "",
                xy=annotation.xy,
                xytext=annotation.xyann,
                textcoords="offset points",
                arrowprops={"arrowstyle": "-", "color": annotation.get_color(), "linewidth": 0.7, "alpha": 0.65},
            )
            connector.set_gid(PLOT_LABEL_GID)
    if linear_regression:
        for fit in fit_linear_trends(rows):
            offset = offsets[fit["benchmark"]]
            ax.plot(
                [offset + fit["start_iteration"], offset + fit["end_iteration"]],
                [fit["fitted_first_percent"], fit["fitted_last_percent"]],
                color=COLORS[fit["method"]],
                linewidth=1.2,
                alpha=0.25,
                label="_nolegend_",
                gid="performance-linear-regression",
            )
    return segments


def save_benchmark_regions(
    output: Path, rows: list[dict], caption: str, source_note: str, appworld_metric: str, *, linear_regression: bool
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": "#334155",
            "text.color": "#1E293B",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )
    fig, ax = plt.subplots(figsize=(14, 7.2))
    fig.subplots_adjust(top=0.70, bottom=0.25, left=0.075, right=0.985)
    plot_benchmark_regions(ax, rows, linear_regression=linear_regression)
    title = "Interleaved cumulative online performance  |  Benchmark datapoints"
    fig.suptitle(title, fontsize=18, fontweight="semibold", y=0.97)
    fig.text(0.5, 0.89, caption + "  |  mean and +/-1 sample SD", ha="center", fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.86), ncol=len(handles), frameon=False
    )
    legend.set_gid(PLOT_LABEL_GID)
    metric = "TGC" if appworld_metric == "tgc" else "partial task score"
    note = (
        "Each region counts its own benchmark's tasks; seed curves are aligned by task count before averaging.\n"
        f"AppWorld: {metric}; PREMiSE: best pre-update parent. Shading: +/-1 sample SD across seeds.\n"
        "Missing encounters retain their positions; missing scores are omitted from means.\n" + source_note
    )
    if linear_regression:
        note += "\nFaint straight lines: OLS fits against benchmark datapoints, equally weighting available points."
    fig.text(0.08, 0.015, note, fontsize=8.5, color="#64748B", linespacing=1.6, gid=PLOT_LABEL_GID)
    name = "interleaved_online_cumulative_benchmark_regions_datapoints"
    if linear_regression:
        name += "_regression"
    save_figure_pair(fig, output, name, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Frozen export with task_scores.csv and metadata.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("benchmark-panels", "benchmark-regions"), default="benchmark-panels")
    parser.add_argument("--linear-regression", action="store_true", help="Also export faint regression overlays")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        parser.error("Use a separate output directory to preserve the global-iteration figures")
    original = json.loads((source / "metadata.json").read_text())
    if original["setting"] != "interleaved" or original["windows"]:
        parser.error("Expected an interleaved cumulative export")
    with (source / "task_scores.csv").open(newline="") as stream:
        tasks = list(csv.DictReader(stream))
    for task in tasks:
        task["iteration"] = int(task["iteration"])
        task["seed"] = int(task["seed"])
        task["score"] = float(task["score"]) if task["score"] else None
    rows = benchmark_datapoint_rows(tasks)
    aggregate = aggregate_windows(rows)
    with (source / "aggregate_cumulative.csv").open(newline="") as stream:
        old_aggregate = list(csv.DictReader(stream))
    for method, benchmark in {(row["method"], row["benchmark"]) for row in aggregate}:
        latest = max(
            (row for row in aggregate if row["method"] == method and row["benchmark"] == benchmark),
            key=lambda row: row["iteration"],
        )
        old_latest = max(
            (row for row in old_aggregate if row["method"] == method and row["benchmark"] == benchmark),
            key=lambda row: int(row["iteration"]),
        )
        assert math.isclose(latest["mean"], float(old_latest["mean"]), abs_tol=1e-12), (method, benchmark)
    y_limits = {benchmark: tuple(limits) for benchmark, limits in original["panel_y_limits"]["cumulative"].items()}
    seeds = sorted({task["seed"] for task in tasks})
    caption = "Equal mean of available seeds: " + ", ".join(map(str, seeds)) if len(seeds) > 1 else f"Seed {seeds[0]}"
    premise_sources = "; ".join(
        f"seed {run['seed']}: "
        + ("completion-contract restart" if "completion_contract_restart" in run["run"] else "original training")
        for run in original["runs"]
        if run["method"] == "premise"
    )
    source_note = "PREMiSE source - " + premise_sources
    if original.get("skillopt_runs"):
        source_note += "\nSkillOpt: recorded training encounters in its own task order."
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "task_scores.csv", output / "task_scores.csv")
    write_csv(output / "seed_cumulative_datapoints.csv", rows)
    write_csv(output / "aggregate_cumulative_datapoints.csv", aggregate)
    fits = fit_linear_trends(aggregate) if args.linear_regression else []
    fits = [
        {
            "method": fit["method"],
            "benchmark": fit["benchmark"],
            "start_datapoint": fit["start_iteration"],
            "end_datapoint": fit["end_iteration"],
            "point_count": fit["point_count"],
            "slope_pp_per_datapoint": fit["slope_pp_per_iteration"],
            "intercept_percent": fit["intercept_percent"],
            "r_squared": fit["r_squared"],
            "fitted_first_percent": fit["fitted_first_percent"],
            "fitted_last_percent": fit["fitted_last_percent"],
        }
        for fit in fits
    ]
    if fits:
        write_csv(output / "linear_regression_datapoints.csv", fits)
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "setting": "interleaved",
        "grouping": original["grouping"],
        "selected_seeds": original["selected_seeds"],
        "x_axis": "benchmark_datapoint",
        "figure_layout": args.layout,
        "background": "white",
        "curve_connection": "linear" if args.layout == "benchmark-regions" else "steps-post",
        "curve_definition": (
            "At benchmark task k, average its first k recorded task positions separately for each seed, "
            "excluding missing scores from the denominator; then average available seed means equally. "
            "Missing encounters retain their x positions. No carry-forward points from other benchmarks."
        ),
        "benchmark_datapoint_counts": {
            benchmark: max(row["iteration"] for row in aggregate if row["benchmark"] == benchmark)
            for benchmark in BENCHMARKS
        },
        "panel_y_limits": dict.fromkeys(BENCHMARKS, (0, 100)) if args.layout == "benchmark-regions" else y_limits,
        "panel_y_definition": (
            "Common 0-100 score axis, matching the sequential style"
            if args.layout == "benchmark-regions"
            else "Preserved from source global-iteration figure; out-of-range values are clipped"
        ),
        "regression_definition": (
            "Separate OLS of plotted score (%) versus benchmark task k; one equal weight per point"
        ),
        "regression_fits": fits,
        "regression_style": {"linewidth": 1.2, "alpha": 0.25},
        "figure_variants": ["labeled", "no_labels"],
        "source_files": {
            name: {"path": str(source / name), "sha256": hashlib.sha256((source / name).read_bytes()).hexdigest()}
            for name in ("task_scores.csv", "metadata.json", "aggregate_cumulative.csv")
        },
        "source_metadata": original,
        "all_final_scores_verified_unchanged": True,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for regression in (False, True) if args.linear_regression else (False,):
        if args.layout == "benchmark-regions":
            save_benchmark_regions(
                output, aggregate, caption, source_note, original["appworld_metric"], linear_regression=regression
            )
            continue
        save_benchmark_panels(
            output,
            aggregate,
            caption,
            source_note,
            original["appworld_metric"],
            benchmark_datapoints=True,
            y_limits=y_limits,
            linear_regression=regression,
        )
    print(f"Figures and source data: {output}")


if __name__ == "__main__":
    main()
