# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Replot frozen online score curves with independent benchmark y ranges."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils.plot_sequential_online_performance import (  # noqa: E402
    BENCHMARKS,
    COLORS,
    LABELS,
    METHOD_LABELS,
    PLOT_LABEL_GID,
    save_figure_pair,
    write_csv,
)

SEGMENTS = {"bfcl": (1, 50), "appworld": (51, 100), "terminalbench2": (101, 130)}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def panel_y_limits(rows: list[dict], reference: float | None, *, focus_start: int = 10) -> tuple[float, float]:
    """Fit later mean values with padding; include any displayed initial-context reference."""
    values = [
        row["mean_percent"] for row in rows if row["local_step"] >= focus_start and row["mean_percent"] is not None
    ]
    if reference is not None:
        values.append(reference)
    lower, upper = min(values), max(values)
    padding = max(2.5, (upper - lower) * 0.15)
    return max(0, math.floor((lower - padding) / 5) * 5), min(100, math.ceil((upper + padding) / 5) * 5)


def make_figure(
    rows: list[dict],
    setting: str,
    references: list[dict],
    statuses: list[dict],
    *,
    focus_start: int,
    interleaved_curve: str = "rolling10",
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#1E293B", "axes.labelcolor": "#334155"})
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.2), gridspec_kw={"width_ratios": [5, 5, 3]})
    fig.subplots_adjust(left=0.055, right=0.985, top=0.70, bottom=0.24, wspace=0.32)
    limits = {}
    for ax, benchmark in zip(axes, BENCHMARKS, strict=True):
        selected = [row for row in rows if row["benchmark"] == benchmark]
        reference = next((100 * row["mean"] for row in references if row["benchmark"] == benchmark), None)
        lower, upper = panel_y_limits(selected, reference, focus_start=focus_start)
        limits[benchmark] = (lower, upper)
        endpoints = []
        for index, (method, color) in enumerate(COLORS.items()):
            method_rows = sorted((row for row in selected if row["method"] == method), key=lambda row: row["x"])
            x = [row["x"] for row in method_rows]
            y = np.array([row["mean_percent"] for row in method_rows], dtype=float)
            sd = np.array([row["std_percent"] for row in method_rows], dtype=float)
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.1, linewidth=0)
            ax.plot(x, y, color=color, linewidth=2.1, label=METHOD_LABELS[method], gid="performance-online-curve")
            if setting == "interleaved" and interleaved_curve == "rolling10":
                result = next(row for row in statuses if row["method"] == method and row["benchmark"] == benchmark)
                step = result["stable_onset_step"] or result["first_plateau_onset_step"]
                if step:
                    stable = result["status"] == "stable_tail"
                    step = int(step)
                    ax.plot(
                        step,
                        method_rows[step - 1]["mean_percent"],
                        marker="*" if stable else "o",
                        markersize=9,
                        markerfacecolor=color if stable else "white",
                        markeredgecolor=color,
                        linestyle="none",
                    )
                    label = f"{'stable' if stable else 'temporary'} k={step}"
                else:
                    label = "insufficient data" if result["status"] == "insufficient_data" else "not stabilized"
                ax.text(
                    0.98,
                    0.96 - 0.055 * index,
                    f"{METHOD_LABELS[method]}: {label}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8.5,
                    color=color,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
                    gid=PLOT_LABEL_GID,
                )
            else:
                endpoints.append((y[-1], x[-1], method))
        endpoints.sort()
        positions = []
        gap = (upper - lower) * 0.085
        for score, _x, _method in endpoints:
            positions.append(max(score, positions[-1] + gap) if positions else score)
        if positions and positions[-1] > upper - gap:
            positions = [y - (positions[-1] - upper + gap) for y in positions]
        for (score, x, method), y in zip(endpoints, positions, strict=True):
            ax.annotate(
                f"{METHOD_LABELS[method]} {score:.2f}%",
                xy=(x, score),
                xytext=(x - 1, y),
                ha="right",
                va="bottom",
                color=COLORS[method],
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
                arrowprops={"arrowstyle": "-", "color": COLORS[method], "linewidth": 0.65, "alpha": 0.65},
                gid=PLOT_LABEL_GID,
            )
        start, end = (
            SEGMENTS[benchmark] if setting == "sequential" else (1, SEGMENTS[benchmark][1] - SEGMENTS[benchmark][0] + 1)
        )
        if reference is not None:
            ax.plot([start - 0.5, end + 0.5], [reference, reference], color="black", linestyle="--", linewidth=1.5)
            ax.text(
                start + 1,
                reference + (upper - lower) * 0.02,
                f"Initial {reference:.2f}%",
                fontsize=9,
                gid=PLOT_LABEL_GID,
            )
        ax.set_xlim(start - 0.5, end + 0.5)
        ax.set_ylim(lower, upper)
        ax.set_xticks([start, *range(10 * math.ceil((start + 1) / 10), end + 1, 10)])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.set_xlabel("Sequential iteration" if setting == "sequential" else "Benchmark datapoint", fontsize=11)
        score_label = "Cumulative online score (%)"
        if setting == "interleaved" and interleaved_curve == "rolling10":
            score_label = "Recent-10 online score (%)"
        ax.set_ylabel(score_label, fontsize=11)
        span = f"Iterations {start}-{end}" if setting == "sequential" else f"Datapoints {start}-{end}"
        ax.set_title(f"{LABELS[benchmark]}\n{span}", fontsize=12, fontweight="semibold", pad=12)
        ax.grid(axis="y", color="#D8DEE6", linewidth=0.65, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#B5BFCD")
    title = "Sequential online performance  |  N = 50"
    if setting == "interleaved":
        title = (
            "Interleaved online performance  |  Recent 10 tasks"
            if interleaved_curve == "rolling10"
            else "Interleaved cumulative online performance  |  Benchmark datapoints"
        )
    fig.suptitle(title, fontsize=21, fontweight="semibold", y=0.97)
    fig.text(
        0.5,
        0.89,
        "Seeds 42, 43, 44  |  mean and +/-1 sample SD  |  independent y ranges",
        ha="center",
        fontsize=11,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.85), ncol=3, frameon=False)
    legend.set_gid(PLOT_LABEL_GID)
    note = (
        "Frozen mean curves and sample standard deviations are preserved; the background is white.\n"
        f"Each y range fits mean scores from local task {focus_start} onward, with 15% padding (at least 2.5 pp).\n"
        "Earlier mean values and parts of standard-deviation bands outside these zoomed ranges are clipped.\n"
    )
    if setting == "sequential":
        note += "Black dashed lines: shared initial context; checkpoint test lines are omitted."
    elif interleaved_curve == "rolling10":
        note += (
            "Stability decisions are unchanged; incomplete assessment data are distinguished from no detected plateau."
        )
    else:
        note += "Benchmark-local cumulative means are aligned by task count before averaging seeds."
    fig.text(0.055, 0.02, note, fontsize=9, color="#64748B", linespacing=1.6, gid=PLOT_LABEL_GID)
    return fig, axes, limits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focus-start", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "acr_oct/independent_y_ranges")
    parser.add_argument("--setting", choices=("both", "sequential", "interleaved"), default="both")
    parser.add_argument("--interleaved-curve", choices=("rolling10", "cumulative"), default="rolling10")
    args = parser.parse_args()
    if args.focus_start < 1:
        parser.error("Focus start must be positive")
    sources = {
        "sequential": ROOT / "acr_oct/sequential/mean/no_checkpoint_white/aggregate_windows.csv",
        "interleaved": ROOT / "acr_oct/stabilization/smoothed_curves_and_metrics.csv",
    }
    if args.interleaved_curve == "cumulative":
        sources["interleaved"] = (
            ROOT / "acr_oct/interleaved/mean/benchmark_datapoints_seq_style/aggregate_cumulative_datapoints.csv"
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / ".gitignore").write_text("*\n")
    for setting, source in sources.items():
        if args.setting != "both" and args.setting != setting:
            continue
        records = read_csv(source)
        references, statuses = [], []
        rolling = setting == "interleaved" and args.interleaved_curve == "rolling10"
        if rolling:
            records = [
                row
                for row in records
                if row["setting"] == setting and row["smoother"] == "rolling10" and row["scope"] == "mean"
            ]
            summary_source = ROOT / "acr_oct/stabilization/stabilization_summary.csv"
            statuses = [
                row
                for row in read_csv(summary_source)
                if row["setting"] == setting and row["smoother"] == "rolling10" and row["scope"] == "mean"
            ]
            basename = "interleaved_rolling10_stabilization_by_benchmark"
        elif setting == "interleaved":
            basename = "interleaved_benchmark_datapoints_seq_style_independent_y"
        else:
            reference_source = ROOT / "acr_oct/sequential/mean/no_checkpoint_white/metadata.json"
            references = json.loads(reference_source.read_text())["test_references"]
            basename = "seq_latest_no_checkpoint_by_benchmark"
        rows = []
        for row in records:
            start = SEGMENTS[row["benchmark"]][0]
            x = int(row["step"]) if rolling else int(row["iteration"])
            mean = row["smoothed_score_percent"] if rolling else row["mean"]
            sd = row["std_percent"] if rolling else row["std"]
            rows.append(
                {
                    "method": row["method"],
                    "benchmark": row["benchmark"],
                    "x": x,
                    "local_step": x if setting == "interleaved" else x - start + 1,
                    "mean_percent": float(mean) * (1 if rolling else 100) if mean else None,
                    "std_percent": float(sd) * (1 if rolling else 100) if sd else None,
                }
            )
        family = "interleaved_cumulative" if setting == "interleaved" and not rolling else setting
        directory = output / family
        directory.mkdir(exist_ok=True)
        fig, _axes, limits = make_figure(
            rows,
            setting,
            references,
            statuses,
            focus_start=args.focus_start,
            interleaved_curve=args.interleaved_curve,
        )
        save_figure_pair(fig, directory, basename, dpi=200)
        import matplotlib.pyplot as plt

        plt.close(fig)
        write_csv(directory / "curve_points.csv", rows)
        metadata = {
            "created_at": datetime.now(UTC).isoformat(),
            "setting": setting,
            "curve": "cumulative" if not rolling else "rolling10",
            "focus_start_local_task": args.focus_start,
            "y_ranges": limits,
            "range_definition": (
                "Later mean scores and initial references, 15% padding (minimum 2.5 pp), rounded to 5 pp"
            ),
            "clipping": "Early means and standard-deviation band portions outside the displayed y limits are clipped",
            "source_csv": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "references": references,
            "stability_decisions": statuses,
        }
        if setting == "interleaved" and not rolling:
            original_pdf = ROOT / "acr_oct/interleaved_benchmark_datapoints_seq_style_no_labels.pdf"
            metadata["source_figure"] = str(original_pdf)
            metadata["source_figure_sha256"] = hashlib.sha256(original_pdf.read_bytes()).hexdigest()
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        for suffix in ("", "_no_labels"):
            for extension in ("png", "pdf"):
                shutil.copy2(
                    directory / f"{basename}{suffix}.{extension}",
                    ROOT / "acr_oct" / f"{basename}{suffix}.{extension}",
                )
        print(setting, limits)


if __name__ == "__main__":
    main()
