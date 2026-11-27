# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Plot benchmark-specific cumulative or windowed online scores along interleaved iterations."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils.plot_sequential_online_performance import (  # noqa: E402
    BENCHMARKS,
    COLORS,
    DEFAULT_ACE_ROOT,
    LABELS,
    METHOD_LABELS,
    PLOT_LABEL_GID,
    aggregate_windows,
    load_runs,
    load_skillopt_runs,
    save_figure_pair,
    write_csv,
)

DEFAULT_PREMISE_ROOT = ROOT / "outputs/experiment_dashboard_8765/learning_manifests/premise"
LINESTYLES = {"bfcl": "-", "appworld": "--", "terminalbench2": ":"}


def interleaved_cumulative(order: list[tuple[str, str]], scores: dict[int, float], cursor: int) -> list[dict]:
    """Keep a separate cumulative history for each benchmark without resetting on switches."""
    if not 0 <= cursor <= len(order):
        raise ValueError("Cursor is outside the task order")
    planned = dict.fromkeys(BENCHMARKS, 0)
    counts = dict.fromkeys(BENCHMARKS, 0)
    totals = dict.fromkeys(BENCHMARKS, 0.0)
    first_seen = dict.fromkeys(BENCHMARKS)
    rows = []
    for iteration, (current, _task) in enumerate(order[:cursor], start=1):
        if current not in BENCHMARKS:
            raise ValueError(f"Unsupported benchmark: {current}")
        planned[current] += 1
        if first_seen[current] is None:
            first_seen[current] = iteration
        if iteration in scores:
            counts[current] += 1
            totals[current] += scores[iteration]
        for benchmark in BENCHMARKS:
            rows.append(
                {
                    "window": "cumulative",
                    "benchmark": benchmark,
                    "iteration": iteration,
                    "current_task_benchmark": current,
                    "benchmark_iteration": planned[benchmark],
                    "window_start": first_seen[benchmark],
                    "window_end": iteration,
                    "window_tasks": planned[benchmark],
                    "scored_tasks": counts[benchmark],
                    "missing_tasks": planned[benchmark] - counts[benchmark],
                    "score": totals[benchmark] / counts[benchmark] if counts[benchmark] else None,
                }
            )
    return rows


def interleaved_windows(
    order: list[tuple[str, str]], scores: dict[int, float], cursor: int, window: int, *, basis: str = "global"
) -> list[dict]:
    """Average benchmark scores within an exact global or benchmark-task window."""
    if window < 1:
        raise ValueError("Window size must be positive")
    if basis not in ("global", "benchmark"):
        raise ValueError(f"Unsupported window basis: {basis}")
    if not 0 <= cursor <= len(order):
        raise ValueError("Cursor is outside the task order")
    histories = {benchmark: deque() for benchmark in BENCHMARKS}
    seen = dict.fromkeys(BENCHMARKS, 0)
    rows = []
    for iteration, (current, _task) in enumerate(order[:cursor], start=1):
        if current not in BENCHMARKS:
            raise ValueError(f"Unsupported benchmark: {current}")
        histories[current].append((iteration, scores.get(iteration)))
        seen[current] += 1
        first = max(1, iteration - window + 1)
        for benchmark, history in histories.items():
            while history and (history[0][0] < first if basis == "global" else len(history) > window):
                history.popleft()
            values = [value for _index, value in history if value is not None]
            rows.append(
                {
                    "window": window,
                    "window_basis": basis,
                    "benchmark": benchmark,
                    "iteration": iteration,
                    "current_task_benchmark": current,
                    "benchmark_iteration": seen[benchmark],
                    "window_start": first if basis == "global" else history[0][0] if history else None,
                    "window_end": iteration,
                    "window_tasks": len(history),
                    "scored_tasks": len(values),
                    "missing_tasks": len(history) - len(values),
                    "score": statistics.mean(values) if values else None,
                }
            )
    return rows


def fit_linear_trends(rows: list[dict], *, start_iteration: int = 1) -> list[dict]:
    """Fit plotted scores to global iterations, equally weighting finite curve points."""
    if start_iteration < 1:
        raise ValueError("Regression starting iteration must be positive")
    groups = defaultdict(list)
    for row in rows:
        if row["iteration"] < start_iteration or row["mean"] is None:
            continue
        score = float(row["mean"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Regression requires finite plotted scores between zero and one")
        groups[row["method"], row["benchmark"], row.get("window", "cumulative")].append(row)
    fits = []
    for (method, benchmark, window), samples in sorted(groups.items()):
        samples = sorted(samples, key=lambda row: row["iteration"])
        x = [row["iteration"] for row in samples]
        if len(set(x)) != len(x):
            raise ValueError("Regression requires one plotted score per method, benchmark and iteration")
        if len(x) < 2:
            continue
        y = [100 * row["mean"] for row in samples]
        x_mean, y_mean = statistics.mean(x), statistics.mean(y)
        x_variation = sum((value - x_mean) ** 2 for value in x)
        slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True)) / x_variation
        intercept = y_mean - slope * x_mean
        residuals = sum((b - intercept - slope * a) ** 2 for a, b in zip(x, y, strict=True))
        y_variation = sum((value - y_mean) ** 2 for value in y)
        fits.append(
            {
                "method": method,
                "benchmark": benchmark,
                "window": window,
                "start_iteration": x[0],
                "end_iteration": x[-1],
                "point_count": len(x),
                "slope_pp_per_iteration": slope,
                "intercept_percent": intercept,
                "r_squared": 1 - residuals / y_variation if y_variation else None,
                "fitted_first_percent": intercept + slope * x[0],
                "fitted_last_percent": intercept + slope * x[-1],
            }
        )
    return fits


def plot_interleaved_axis(
    ax,
    rows: list[dict],
    *,
    window: int | str = "cumulative",
    compact: bool = False,
    linear_regression: bool = False,
    regression_start: int = 1,
) -> None:
    """Use one color per method and one line style per benchmark."""
    import numpy as np

    fits = (
        {(row["method"], row["benchmark"]): row for row in fit_linear_trends(rows, start_iteration=regression_start)}
        if linear_regression
        else {}
    )
    endpoints = []
    for method in (method for method in COLORS if any(row["method"] == method for row in rows)):
        for benchmark in BENCHMARKS:
            selected = sorted(
                (row for row in rows if row["method"] == method and row["benchmark"] == benchmark),
                key=lambda row: row["iteration"],
            )
            x = [row["iteration"] for row in selected]
            y = [100 * row["mean"] if row["mean"] is not None else np.nan for row in selected]
            fit = fits.get((method, benchmark))
            label = f"{METHOD_LABELS[method]} / {LABELS[benchmark]}"
            ax.plot(
                x,
                y,
                color=COLORS[method],
                linestyle=LINESTYLES[benchmark],
                drawstyle="steps-post",
                linewidth=2.1,
                alpha=1,
                label=label,
                gid="performance-online-curve",
            )
            if fit:
                ax.plot(
                    [fit["start_iteration"], fit["end_iteration"]],
                    [fit["fitted_first_percent"], fit["fitted_last_percent"]],
                    color=COLORS[method],
                    linestyle=LINESTYLES[benchmark],
                    linewidth=1.2,
                    alpha=0.25,
                    label="_nolegend_",
                    gid="performance-linear-regression",
                    zorder=2,
                )
            if selected and selected[-1]["mean"] is not None:
                endpoints.append((y[-1], x[-1], method, benchmark))
    endpoints.sort()
    label_positions = []
    for score, _iteration, _method, _benchmark in endpoints:
        label_positions.append(max(score, label_positions[-1] + 5.5 if label_positions else 3))
    if label_positions and label_positions[-1] > 97:
        label_positions = [y - (label_positions[-1] - 97) for y in label_positions]
    for (score, iteration, method, benchmark), label_y in zip(endpoints, label_positions, strict=True):
        ax.plot(iteration, score, "o", color=COLORS[method], markersize=3.5, label="_nolegend_")
        name = "Terminal" if benchmark == "terminalbench2" else LABELS[benchmark]
        annotation = ax.annotate(
            f"{METHOD_LABELS[method]} {name} {score:.2f}%",
            xy=(iteration, score),
            xytext=(iteration + 2, label_y),
            color=COLORS[method],
            fontsize=8 if compact else 9,
            fontweight="semibold",
            ha="left",
            va="center",
            annotation_clip=False,
            arrowprops={"arrowstyle": "-", "color": COLORS[method], "linewidth": 0.65, "alpha": 0.65},
        )
        annotation.set_gid(PLOT_LABEL_GID)
    last_iteration = max(row["iteration"] for row in rows)
    ax.set_xlim(0.5, last_iteration + 0.5)
    ax.set_ylim(0, 100)
    ax.set_xticks([1, *range(10, last_iteration + 1, 10)])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Interleaved iteration", fontsize=11)
    ax.set_ylabel("Cumulative online score (%)" if window == "cumulative" else "Online window score (%)", fontsize=11)
    ax.grid(axis="y", color="#D8DEE6", linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B5BFCD")


def figure_note(source_note: str, appworld_metric: str, window: int | str, basis: str) -> str:
    window_label = f"N={window}" if isinstance(window, int) else "N"
    if window == "cumulative":
        definition = "Each line averages only that benchmark's tasks observed up to the current global iteration."
    elif basis == "global":
        definition = f"Window {window_label}: benchmark-specific scores within the last N global iterations [M-N+1, M]."
    else:
        definition = (
            f"Window {window_label}: last N tasks from each benchmark, positioned on the global iteration axis."
        )
    metric = "TGC" if appworld_metric == "tgc" else "partial task score"
    return (
        definition + "\n"
        f"AppWorld: {metric}; PREMiSE: best pre-update parent. "
        "Missing scores are omitted; ACE recorded errors are retained.\n" + source_note
    )


def save_figure(
    output: Path,
    rows: list[dict],
    caption: str,
    source_note: str,
    appworld_metric: str,
    *,
    window: int | str = "cumulative",
    basis: str = "global",
    linear_regression: bool = False,
    regression_start: int = 1,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#1E293B", "axes.labelcolor": "#334155"})
    fig, ax = plt.subplots(figsize=(14, 7.4 if linear_regression else 6.8))
    fig.subplots_adjust(left=0.075, right=0.76, top=0.74, bottom=0.24 if linear_regression else 0.2)
    plot_interleaved_axis(
        ax, rows, window=window, linear_regression=linear_regression, regression_start=regression_start
    )
    title = (
        "Interleaved cumulative online performance"
        if window == "cumulative"
        else f"Interleaved online performance  |  N = {window}"
    )
    if linear_regression:
        title += "  |  Linear trends"
    fig.suptitle(title, fontsize=19, fontweight="semibold", y=0.97)
    caption_gid = PLOT_LABEL_GID if any(method in caption for method in METHOD_LABELS.values()) else None
    fig.text(0.5, 0.90, caption, fontsize=10, ha="center", gid=caption_gid)
    handles, labels = ax.get_legend_handles_labels()
    lookup = dict(zip(labels, handles, strict=True))
    labels = [
        label
        for benchmark in BENCHMARKS
        for method in COLORS
        if (label := f"{METHOD_LABELS[method]} / {LABELS[benchmark]}") in lookup
    ]
    handles = [lookup[label] for label in labels]
    legend = fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.87), ncol=3, frameon=False, fontsize=9
    )
    legend.set_gid(PLOT_LABEL_GID)
    note = figure_note(source_note, appworld_metric, window, basis)
    if linear_regression:
        note += (
            f"\nFaint straight lines: separate OLS fits from iteration {regression_start} onward; "
            "full-color step lines: observed scores."
            "\nFits weight each available global-iteration curve point equally; endpoint labels are observed scores."
        )
    fig.text(0.075, 0.025, note, fontsize=8.5, color="#64748B", linespacing=1.6, gid=PLOT_LABEL_GID)
    name = "interleaved_online_cumulative" if window == "cumulative" else f"interleaved_online_N{window:02d}"
    if linear_regression:
        name += "_regression"
    save_figure_pair(fig, output, name, dpi=200)
    plt.close(fig)


def benchmark_y_limits(rows: list[dict], benchmark: str, *, start_iteration: int = 20) -> tuple[float, float]:
    """Fit a zoomed score range to the later history, preserving the full x range."""
    selected = [row for row in rows if row["benchmark"] == benchmark and row["mean"] is not None]
    values = [100 * row["mean"] for row in selected if row["iteration"] >= start_iteration]
    if not values:
        values = [100 * row["mean"] for row in selected]
    if not values:
        return 0, 100
    low, high = min(values), max(values)
    padding = max(2.5, (high - low) * 0.15)
    return max(0, 5 * math.floor((low - padding) / 5)), min(100, 5 * math.ceil((high + padding) / 5))


def plot_benchmark_axis(
    ax,
    rows: list[dict],
    benchmark: str,
    *,
    window: int | str = "cumulative",
    focus_start: int = 20,
    benchmark_datapoints: bool = False,
    y_limits: tuple[float, float] | None = None,
    linear_regression: bool = False,
) -> None:
    """Plot one benchmark's method curves against global iterations or task counts."""
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    lower, upper = y_limits or benchmark_y_limits(rows, benchmark, start_iteration=focus_start)
    benchmark_rows = [row for row in rows if row["benchmark"] == benchmark]
    fits = {row["method"]: row for row in fit_linear_trends(benchmark_rows)} if linear_regression else {}
    endpoints = []
    for method in COLORS:
        selected = sorted(
            (row for row in rows if row["method"] == method and row["benchmark"] == benchmark),
            key=lambda row: row["iteration"],
        )
        if not selected:
            continue
        x = [row["iteration"] for row in selected]
        y = [100 * row["mean"] if row["mean"] is not None else np.nan for row in selected]
        ax.plot(
            x,
            y,
            color=COLORS[method],
            linestyle="-",
            drawstyle="steps-post",
            linewidth=2.1,
            label=METHOD_LABELS[method],
            gid="performance-online-curve",
        )
        if fit := fits.get(method):
            ax.plot(
                [fit["start_iteration"], fit["end_iteration"]],
                [fit["fitted_first_percent"], fit["fitted_last_percent"]],
                color=COLORS[method],
                linewidth=1.2,
                alpha=0.25,
                label="_nolegend_",
                gid="performance-linear-regression",
            )
        if selected[-1]["mean"] is not None:
            endpoints.append((y[-1], x[-1], method))
    endpoints.sort()
    label_positions = []
    for score, _iteration, _method in endpoints:
        label_positions.append(max(score, label_positions[-1] + 7 if label_positions else 3))
    label_upper = upper - max(2, (upper - lower) * 0.06) - 2
    if label_positions and label_positions[-1] > label_upper:
        label_positions = [y - (label_positions[-1] - label_upper) for y in label_positions]
    for (score, iteration, method), label_y in zip(endpoints, label_positions, strict=True):
        ax.plot(iteration, score, "o", color=COLORS[method], markersize=3.5, label="_nolegend_")
        annotation = ax.annotate(
            f"{METHOD_LABELS[method]} {score:.2f}%",
            xy=(iteration, score),
            xytext=(iteration - 3, label_y + 2),
            color=COLORS[method],
            fontsize=9,
            fontweight="semibold",
            ha="right",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
            arrowprops={"arrowstyle": "-", "color": COLORS[method], "linewidth": 0.65, "alpha": 0.65},
        )
        annotation.set_gid(PLOT_LABEL_GID)
    last_iteration = max(row["iteration"] for row in (benchmark_rows if benchmark_datapoints else rows))
    ax.set_xlim(0.5, last_iteration + 0.5)
    ax.set_ylim(lower, upper)
    tick_step = 10 if benchmark_datapoints else 20
    ax.set_xticks(sorted({1, *range(tick_step, last_iteration + 1, tick_step), last_iteration}))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    ax.set_xlabel("Benchmark datapoint" if benchmark_datapoints else "Interleaved iteration", fontsize=11)
    ax.set_ylabel("Cumulative online score (%)" if window == "cumulative" else "Online window score (%)", fontsize=11)
    ax.set_title(LABELS[benchmark], fontsize=13, fontweight="semibold", pad=15)
    ax.grid(axis="y", color="#D8DEE6", linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B5BFCD")
    ax.tick_params(labelsize=10)


def save_benchmark_panels(
    output: Path,
    rows: list[dict],
    caption: str,
    source_note: str,
    appworld_metric: str,
    *,
    window: int | str = "cumulative",
    basis: str = "global",
    focus_start: int = 20,
    benchmark_datapoints: bool = False,
    y_limits: dict[str, tuple[float, float]] | None = None,
    linear_regression: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#1E293B", "axes.labelcolor": "#334155"})
    fig, axes = plt.subplots(1, len(BENCHMARKS), figsize=(18, 7.4 if linear_regression else 6.8))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.73, bottom=0.27 if linear_regression else 0.235, wspace=0.3)
    for ax, benchmark in zip(axes, BENCHMARKS, strict=True):
        plot_benchmark_axis(
            ax,
            rows,
            benchmark,
            window=window,
            focus_start=focus_start,
            benchmark_datapoints=benchmark_datapoints,
            y_limits=y_limits[benchmark] if y_limits else None,
            linear_regression=linear_regression,
        )
    title = (
        "Interleaved cumulative online performance"
        if window == "cumulative"
        else f"Interleaved online performance  |  N = {window}"
    )
    if benchmark_datapoints:
        title += "  |  Benchmark datapoints"
    if linear_regression:
        title += "  |  Linear trends"
    fig.suptitle(title, fontsize=19 if benchmark_datapoints else 21, fontweight="semibold", y=0.97)
    caption_gid = PLOT_LABEL_GID if any(method in caption for method in METHOD_LABELS.values()) else None
    fig.text(0.5, 0.89, caption, fontsize=11, ha="center", gid=caption_gid)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.85),
        ncol=len(handles),
        frameon=False,
        fontsize=11,
    )
    legend.set_gid(PLOT_LABEL_GID)
    note = figure_note(source_note, appworld_metric, window, basis)
    if benchmark_datapoints:
        note = note.replace(
            "Each line averages only that benchmark's tasks observed up to the current global iteration.",
            "At datapoint k, each seed averages that benchmark's first k tasks; seed means are then averaged equally.",
        )
        note += "\nX axes count each benchmark's task encounters, including tasks with missing scores."
    if y_limits:
        note += "\nY ranges match the original global-iteration figure; values outside these ranges are clipped."
    else:
        note += (
            f"\nIndependent y axes fit the score range from {'datapoint' if benchmark_datapoints else 'iteration'} "
            f"{focus_start} onward; earlier values outside the displayed ranges are clipped."
        )
    if linear_regression:
        note += "\nFaint straight lines: separate OLS fits, equally weighting available plotted datapoints."
    fig.text(0.055, 0.025, note, fontsize=8.5, color="#64748B", linespacing=1.6, gid=PLOT_LABEL_GID)
    name = "interleaved_online_cumulative" if window == "cumulative" else f"interleaved_online_N{window:02d}"
    name += "_by_benchmark"
    if benchmark_datapoints:
        name += "_datapoints"
    if linear_regression:
        name += "_regression"
    save_figure_pair(fig, output, name, dpi=200)
    plt.close(fig)


def save_overview(
    output: Path, rows: list[dict], windows: list[int], caption: str, source_note: str, appworld_metric: str, basis: str
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(windows), 1, figsize=(14, 4.2 * len(windows) + 2), squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.76, top=0.91, bottom=0.055, hspace=0.4)
    for ax, window in zip(axes[:, 0], windows, strict=True):
        plot_interleaved_axis(ax, [row for row in rows if row["window"] == window], window=window, compact=True)
        ax.set_title(f"Window N = {window}", loc="left", fontsize=12, fontweight="semibold")
    fig.suptitle("Interleaved online performance windows", y=0.986, fontsize=21, fontweight="semibold")
    caption_gid = PLOT_LABEL_GID if any(method in caption for method in METHOD_LABELS.values()) else None
    fig.text(0.5, 0.971, caption, ha="center", fontsize=11, gid=caption_gid)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    lookup = dict(zip(labels, handles, strict=True))
    labels = [
        label
        for benchmark in BENCHMARKS
        for method in COLORS
        if (label := f"{METHOD_LABELS[method]} / {LABELS[benchmark]}") in lookup
    ]
    legend = fig.legend(
        [lookup[label] for label in labels],
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    legend.set_gid(PLOT_LABEL_GID)
    note = (
        figure_note(source_note, appworld_metric, "N", basis)
        if basis == "benchmark"
        else (
            "Each window uses [max(1, M-N+1), M] on the global iteration axis, "
            "averaging tasks separately by benchmark.\n"
            "Missing scores are omitted; a benchmark with no scored task in the window has no point.\n" + source_note
        )
    )
    fig.text(0.075, 0.012, note, fontsize=9, color="#64748B", linespacing=1.6, gid=PLOT_LABEL_GID)
    save_figure_pair(fig, output, "interleaved_online_windows_overview", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ace-root", type=Path, default=DEFAULT_ACE_ROOT)
    parser.add_argument("--premise-root", type=Path, default=DEFAULT_PREMISE_ROOT)
    parser.add_argument("--skillopt-csv", type=Path, help="Add SkillOpt's percentage training-encounter CSV")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--grouping", choices=("by-seed", "mean", "best"), default="by-seed")
    parser.add_argument(
        "--linear-regression", action="store_true", help="Add separate OLS trend lines to overlay curves"
    )
    parser.add_argument("--regression-start", type=int, default=1, help="First global iteration included in OLS fits")
    parser.add_argument(
        "--layout",
        choices=("overlay", "benchmark-panels"),
        default="overlay",
        help="Overlay all benchmarks, or use three separate axes with independent y ranges and full iteration ranges",
    )
    parser.add_argument(
        "--panel-y-start", type=int, default=20, help="Fit panel y ranges using scores from this iteration"
    )
    parser.add_argument(
        "--windows", type=int, nargs="+", help="Trailing windows; omit for benchmark-wise cumulative means"
    )
    parser.add_argument("--window-basis", choices=("global", "benchmark"), default="global")
    parser.add_argument("--appworld-metric", choices=("tgc", "partial"), default="tgc")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/interleaved_online_by_seed_20261009")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    if args.regression_start < 1:
        parser.error("Regression starting iteration must be positive")
    if args.linear_regression and args.layout != "overlay":
        parser.error("Linear regression currently uses the overlay layout")
    if args.panel_y_start < 1:
        parser.error("Panel y-range starting iteration must be positive")
    if args.windows and (any(window < 1 for window in args.windows) or len(set(args.windows)) != len(args.windows)):
        parser.error("Window sizes must be positive and unique")
    runs = load_runs(
        args.ace_root.resolve(),
        args.premise_root.resolve(),
        args.seeds,
        "best-parent",
        args.appworld_metric,
        mode="interleaved",
    )
    skillopt_runs = (
        load_skillopt_runs(args.skillopt_csv, runs, args.appworld_metric, mode="interleaved")
        if args.skillopt_csv
        else []
    )
    runs.extend(skillopt_runs)
    methods = [method for method in COLORS if any(run["method"] == method for run in runs)]
    if any(run["cursor"] != len(run["order"]) for run in runs):
        parser.error("Expected completed interleaved runs")
    cumulative_rows = [
        {"method": run["method"], "seed": run["seed"], **row}
        for run in runs
        for row in interleaved_cumulative(run["order"], run["scores"], run["cursor"])
    ]
    windows = args.windows or ["cumulative"]
    seed_rows = (
        cumulative_rows
        if not args.windows
        else [
            {"method": run["method"], "seed": run["seed"], **row}
            for run in runs
            for window in windows
            for row in interleaved_windows(run["order"], run["scores"], run["cursor"], window, basis=args.window_basis)
        ]
    )
    rankings = []
    for run in runs:
        final = [
            row["score"]
            for row in cumulative_rows
            if row["method"] == run["method"] and row["seed"] == run["seed"] and row["iteration"] == run["cursor"]
        ]
        rankings.append({"method": run["method"], "seed": run["seed"], "macro_mean": statistics.mean(final)})
    if args.grouping == "by-seed":
        groups = [(f"seed{seed}", {method: [seed] for method in methods}, f"Seed {seed}") for seed in args.seeds]
    elif args.grouping == "mean":
        groups = [
            (
                "mean",
                dict.fromkeys(methods, args.seeds),
                "Equal mean of available seeds: " + ", ".join(map(str, args.seeds)),
            )
        ]
    else:
        best = {
            method: max((row for row in rankings if row["method"] == method), key=lambda r: r["macro_mean"])["seed"]
            for method in methods
        }
        groups = [
            (
                "best_seeds",
                {method: [seed] for method, seed in best.items()},
                "Highest final three-benchmark online mean: "
                + " / ".join(f"{METHOD_LABELS[method]} seed {seed}" for method, seed in best.items()),
            )
        ]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "seed_rankings.csv", rankings)
    for name, chosen, caption in groups:
        directory = output / name
        directory.mkdir(exist_ok=True)
        selected_runs = [run for run in runs if run["seed"] in chosen[run["method"]]]
        rows = [row for row in seed_rows if row["seed"] in chosen[row["method"]]]
        aggregate = aggregate_windows(rows)
        regression_fits = (
            fit_linear_trends(aggregate, start_iteration=args.regression_start) if args.linear_regression else []
        )
        tasks = [
            {
                "method": run["method"],
                "seed": run["seed"],
                "iteration": i,
                "benchmark": benchmark,
                "task_id": task,
                "score": run["scores"].get(i),
                "source_run": run["run"],
                "encounter_source": run.get("encounter_sources", {}).get(i),
            }
            for run in selected_runs
            for i, (benchmark, task) in enumerate(run["order"], start=1)
        ]
        write_csv(directory / "task_scores.csv", tasks)
        write_csv(directory / ("seed_windows.csv" if args.windows else "seed_cumulative.csv"), rows)
        write_csv(directory / ("aggregate_windows.csv" if args.windows else "aggregate_cumulative.csv"), aggregate)
        if regression_fits:
            write_csv(directory / "linear_regression.csv", regression_fits)
        metadata = {
            "created_at": datetime.now(UTC).isoformat(),
            "setting": "interleaved",
            "grouping": args.grouping,
            "figure_variants": ["labeled", "no_labels"],
            "figure_layout": args.layout,
            "linear_regression": args.linear_regression,
            "regression_start": args.regression_start if args.linear_regression else None,
            "regression_definition": (
                "Separate ordinary least squares fits of plotted scores (%) versus global iteration; "
                "equal weight per available curve point, including carried-forward values"
            )
            if args.linear_regression
            else None,
            "regression_fits": regression_fits,
            "benchmark_panel_iterations": [1, len(selected_runs[0]["order"])]
            if args.layout == "benchmark-panels"
            else None,
            "panel_y_start": args.panel_y_start if args.layout == "benchmark-panels" else None,
            "panel_y_limits": {
                str(window): {
                    benchmark: benchmark_y_limits(
                        [row for row in aggregate if row["window"] == window],
                        benchmark,
                        start_iteration=args.panel_y_start,
                    )
                    for benchmark in BENCHMARKS
                }
                for window in windows
            }
            if args.layout == "benchmark-panels"
            else None,
            "selected_seeds": chosen,
            "appworld_metric": args.appworld_metric,
            "premise_score": "best-parent",
            "windows": args.windows,
            "window_basis": args.window_basis if args.windows else None,
            "curve_definition": (
                "Benchmark-specific mean within [max(1, M-N+1), M] global iterations; missing scores omitted"
                if args.windows and args.window_basis == "global"
                else "Last N task positions of each benchmark; missing scores omitted without extending the window"
                if args.windows
                else "Mean over this benchmark's scored tasks at iterations <= M; no reset on switches"
            ),
            "colors": {method: COLORS[method] for method in methods},
            "line_styles": dict.fromkeys(BENCHMARKS, "-") if args.layout == "benchmark-panels" else LINESTYLES,
            "rankings": rankings,
            "skillopt_runs": [
                {
                    key: value
                    for key, value in run.items()
                    if key not in ("order", "scores", "raw_scores", "encounter_sources")
                }
                for run in selected_runs
                if run["method"] == "skillopt"
            ],
            "runs": [
                {key: run[key] for key in ("method", "seed", "run", "cursor", "unscored_iterations")}
                for run in selected_runs
            ],
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        premise_sources = "; ".join(
            f"seed {run['seed']}: "
            + ("completion-contract restart" if "completion_contract_restart" in run["run"] else "original training")
            for run in selected_runs
            if run["method"] == "premise"
        )
        source_note = "PREMiSE source - " + premise_sources
        if skillopt_runs:
            source_note += "\nSkillOpt: CSV training encounters in its own task order."
            if any(not run["task_order_matches_ace"] for run in skillopt_runs):
                source_note += " Task schedule differs from ACE/PREMiSE."
        for window in windows:
            save_plot = save_benchmark_panels if args.layout == "benchmark-panels" else save_figure
            options = {"focus_start": args.panel_y_start} if args.layout == "benchmark-panels" else {}
            if args.linear_regression:
                options.update(linear_regression=True, regression_start=args.regression_start)
            save_plot(
                directory,
                [row for row in aggregate if row["window"] == window],
                caption,
                source_note,
                args.appworld_metric,
                window=window,
                basis=args.window_basis,
                **options,
            )
        if args.windows and len(args.windows) > 1 and args.layout == "overlay" and not args.linear_regression:
            save_overview(
                directory, aggregate, args.windows, caption, source_note, args.appworld_metric, args.window_basis
            )
        print(f"Figure and source data: {directory}")


if __name__ == "__main__":
    main()
