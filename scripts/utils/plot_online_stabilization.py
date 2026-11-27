# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Plot EWMA/recent-task online scores with sustained stability assessments from frozen encounters."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils.plot_sequential_online_performance import (  # noqa: E402
    COLORS,
    LABELS,
    METHOD_LABELS,
    PLOT_LABEL_GID,
    save_figure_pair,
    write_csv,
)


def smooth_scores(scores: list[float | None], kind: str, *, window: int = 10, alpha: float = 0.2) -> list[dict]:
    """Smooth raw percentage scores; preserve missing task positions and expose coverage."""
    if kind not in ("ewma", "rolling10") or window < 1 or not 0 < alpha <= 1:
        raise ValueError("Invalid smoother or smoothing parameters")
    result = []
    state = None
    seen = 0
    for step, score in enumerate(scores, 1):
        if score is not None:
            if not math.isfinite(score) or not 0 <= score <= 100:
                raise ValueError("Expected finite percentage scores")
            seen += 1
            state = score if state is None else alpha * score + (1 - alpha) * state
        recent = [value for value in scores[max(0, step - window) : step] if value is not None]
        value = (
            (state if score is not None else None) if kind == "ewma" else statistics.mean(recent) if recent else None
        )
        eligible = step >= window and score is not None
        eligible = eligible and (seen >= window if kind == "ewma" else len(recent) == window)
        result.append(
            {
                "step": step,
                "raw_score_percent": score,
                "smoothed_score_percent": value,
                "eligible": eligible,
                "window_scored_tasks": len(recent),
            }
        )
    return result


def assess_stability(
    samples: list[dict],
    *,
    assessment_window: int = 10,
    slope_limit: float = 0.2,
    range_limit: float = 5,
    patience: int = 5,
) -> tuple[list[dict], dict]:
    """Assess complete eligible windows; distinguish temporary plateaus from final stable tails."""
    if assessment_window < 2 or patience < 1 or slope_limit < 0 or range_limit < 0:
        raise ValueError("Invalid stability parameters")
    metrics = []
    passing_run = 0
    episodes = []
    for index, sample in enumerate(samples):
        recent = samples[max(0, index - assessment_window + 1) : index + 1]
        valid = len(recent) == assessment_window and all(row["eligible"] for row in recent)
        slope = spread = None
        if valid:
            x = [row["step"] for row in recent]
            y = [row["smoothed_score_percent"] for row in recent]
            center_x, center_y = statistics.mean(x), statistics.mean(y)
            slope = sum((a - center_x) * (b - center_y) for a, b in zip(x, y, strict=True)) / sum(
                (a - center_x) ** 2 for a in x
            )
            spread = max(y) - min(y)
        passed = valid and abs(slope) <= slope_limit + 1e-10 and spread <= range_limit + 1e-10
        passing_run = passing_run + 1 if passed else 0
        if passing_run == patience:
            episodes.append(
                {"onset_step": sample["step"] - patience + 1, "confirmation_step": sample["step"], "last_step": None}
            )
        if not passed and episodes and episodes[-1]["last_step"] is None:
            episodes[-1]["last_step"] = samples[index - 1]["step"]
        metrics.append(
            {
                **sample,
                "assessment_eligible": valid,
                "slope_pp_per_step": slope,
                "range_pp": spread,
                "criterion_passed": passed,
                "consecutive_passes": passing_run,
            }
        )
    stable_tail = passing_run >= patience
    if episodes and episodes[-1]["last_step"] is None:
        episodes[-1]["last_step"] = samples[-1]["step"]
    eligible_points = sum(row["assessment_eligible"] for row in metrics)
    summary = {
        "status": (
            "stable_tail"
            if stable_tail
            else "temporary_only"
            if episodes
            else "not_stabilized"
            if eligible_points
            else "insufficient_data"
        ),
        "stable_onset_step": episodes[-1]["onset_step"] if stable_tail else None,
        "stable_confirmation_step": episodes[-1]["confirmation_step"] if stable_tail else None,
        "first_plateau_onset_step": episodes[0]["onset_step"] if episodes else None,
        "first_plateau_confirmation_step": episodes[0]["confirmation_step"] if episodes else None,
        "episode_count": len(episodes),
        "episodes": episodes,
        "eligible_assessment_points": eligible_points,
        "observed_steps": len(samples),
    }
    return metrics, summary


def read_tasks(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        score = row["online_score"] if "online_score" in row else row["score"]
        row["score_percent"] = 100 * float(score) if score else None
        row["iteration"] = int(row["iteration"])
        row["seed"] = int(row["seed"])
    return rows


def build_curves(tasks: list[dict], setting: str, kind: str, args) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for task in tasks:
        groups[task["method"], task["benchmark"], task["seed"]].append(task)
    curves = []
    summaries = []
    means = defaultdict(list)
    for (method, benchmark, seed), encounters in sorted(groups.items()):
        encounters.sort(key=lambda row: row["iteration"])
        samples = smooth_scores(
            [row["score_percent"] for row in encounters], kind, window=args.window, alpha=args.alpha
        )
        metrics, summary = assess_stability(
            samples,
            assessment_window=args.assessment_window,
            slope_limit=args.slope_limit,
            range_limit=args.range_limit,
            patience=args.patience,
        )
        identity = {"setting": setting, "smoother": kind, "method": method, "benchmark": benchmark, "scope": str(seed)}
        curves.extend({**identity, **row} for row in metrics)
        summaries.append({**identity, **summary})
        for sample in samples:
            means[method, benchmark, sample["step"]].append(sample)
    for method, benchmark in sorted({(method, benchmark) for method, benchmark, _step in means}):
        samples = []
        for key in sorted(key for key in means if key[:2] == (method, benchmark)):
            members = means[key]
            scores = [row["smoothed_score_percent"] for row in members if row["smoothed_score_percent"] is not None]
            samples.append(
                {
                    "step": key[2],
                    "raw_score_percent": None,
                    "smoothed_score_percent": statistics.mean(scores) if scores else None,
                    "eligible": all(row["eligible"] for row in members) and len(members) == 3,
                    "window_scored_tasks": sum(row["window_scored_tasks"] for row in members),
                    "std_percent": statistics.stdev(scores) if len(scores) > 1 else None,
                    "seed_count": len(scores),
                }
            )
        metrics, summary = assess_stability(
            samples,
            assessment_window=args.assessment_window,
            slope_limit=args.slope_limit,
            range_limit=args.range_limit,
            patience=args.patience,
        )
        identity = {"setting": setting, "smoother": kind, "method": method, "benchmark": benchmark, "scope": "mean"}
        curves.extend({**identity, **row} for row in metrics)
        summaries.append({**identity, **summary})
    return curves, summaries


def save_plot(output: Path, curves: list[dict], summaries: list[dict], setting: str, kind: str, args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#1E293B", "axes.labelcolor": "#334155"})
    fig, ax = plt.subplots(figsize=(14, 7.2))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.7, bottom=0.25)
    segments = [("bfcl", 1, 50), ("appworld", 51, 100), ("terminalbench2", 101, 130)]
    for block, (benchmark, start, end) in enumerate(segments):
        for index, (method, color) in enumerate(COLORS.items()):
            selected = sorted(
                (
                    row
                    for row in curves
                    if row["scope"] == "mean" and row["method"] == method and row["benchmark"] == benchmark
                ),
                key=lambda row: row["step"],
            )
            x = [start + row["step"] - 1 for row in selected]
            y = np.array([row["smoothed_score_percent"] for row in selected], dtype=float)
            sd = np.array([row["std_percent"] for row in selected], dtype=float)
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.1, linewidth=0)
            ax.plot(x, y, color=color, linewidth=2.1, label=METHOD_LABELS[method] if block == 0 else "_nolegend_")
            result = next(
                row
                for row in summaries
                if row["scope"] == "mean" and row["method"] == method and row["benchmark"] == benchmark
            )
            step = result["stable_onset_step"] or result["first_plateau_onset_step"]
            if step:
                value = selected[step - 1]["smoothed_score_percent"]
                stable = result["status"] == "stable_tail"
                ax.plot(
                    start + step - 1,
                    value,
                    marker="*" if stable else "o",
                    markersize=10 if stable else 7,
                    markerfacecolor=color if stable else "white",
                    markeredgecolor=color,
                    markeredgewidth=1.6,
                    linestyle="none",
                    zorder=5,
                )
                status = f"stable k={step}" if stable else f"temporary k={step}"
            else:
                status = "insufficient data" if result["status"] == "insufficient_data" else "not stabilized"
            ax.text(
                end - 1,
                96 - index * 5,
                f"{METHOD_LABELS[method]}: {status}",
                ha="right",
                va="center",
                color=color,
                fontsize=9,
                fontweight="semibold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2},
                gid=PLOT_LABEL_GID,
            )
        if block:
            ax.axvline(start - 0.5, color="#475569", linewidth=1.4, linestyle="--")
        span = f"Iterations {start}-{end}" if setting == "sequential" else f"Datapoints 1-{end - start + 1}"
        ax.text(
            (start + end) / 2,
            1.025,
            f"{LABELS[benchmark]}\n{span}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            linespacing=1.45,
        )
    if setting == "sequential":
        reference_path = ROOT / "acr_oct/sequential/mean/no_checkpoint_white/metadata.json"
        reference_metadata = json.loads(reference_path.read_text())
        for reference in reference_metadata["test_references"]:
            benchmark = reference["benchmark"]
            _, start, end = next(segment for segment in segments if segment[0] == benchmark)
            value = 100 * reference["mean"]
            ax.plot([start - 0.5, end + 0.5], [value, value], color="black", linewidth=1.5, linestyle="--")
            ax.text(start + 1, value + 2, f"Initial {value:.2f}%", color="black", fontsize=9, gid=PLOT_LABEL_GID)
    ax.set_xlim(0.5, 130.5)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ticks = [1, *range(10, 131, 10)]
    labels = (
        [str(k) for k in ticks]
        if setting == "sequential"
        else [str(k if k <= 50 else k - 50 if k <= 100 else k - 100) for k in ticks]
    )
    ax.set_xticks(ticks, labels)
    ax.set_xlabel(
        "Sequential iteration" if setting == "sequential" else "Benchmark datapoint (counts restart in each region)"
    )
    ax.set_ylabel("Smoothed online score (%)")
    ax.grid(axis="y", color="#D8DEE6", linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B5BFCD")
    description = f"EWMA alpha = {args.alpha:g}" if kind == "ewma" else f"Recent {args.window} tasks"
    fig.suptitle(f"{setting.title()} online performance  |  {description}", fontsize=19, fontweight="semibold", y=0.97)
    fig.text(0.5, 0.89, "Seeds 42, 43, 44  |  mean and +/-1 sample SD", ha="center", fontsize=10)
    legend = fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.86), ncol=3, frameon=False)
    legend.set_gid(PLOT_LABEL_GID)
    note = (
        "Raw task scores are smoothed separately for each seed and benchmark, then averaged equally.\n"
        f"Assessment: {args.assessment_window} consecutive eligible points; |slope| <= {args.slope_limit:g} pp/step; "
        f"max-min <= {args.range_limit:g} pp; {args.patience} consecutive passes.\n"
        "Star: stable onset with criteria maintained through the observed end; hollow circle: temporary plateau only.\n"
        "Warm-up and incomplete score coverage are excluded from stability assessments. "
        "Missing scores retain task positions.\n"
        "Status labels refer to the plotted mean curve; individual-seed decisions are exported separately."
    )
    fig.text(0.08, 0.015, note, fontsize=8.5, color="#64748B", linespacing=1.6, gid=PLOT_LABEL_GID)
    save_figure_pair(fig, output, f"{setting}_{kind}_stabilization", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--assessment-window", type=int, default=10)
    parser.add_argument("--slope-limit", type=float, default=0.2)
    parser.add_argument("--range-limit", type=float, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "acr_oct/stabilization")
    args = parser.parse_args()
    sources = {
        "sequential": ROOT / "outputs/online_skillopt_csv_updated_20261009/sequential_mean/task_scores.csv",
        "interleaved": ROOT / "acr_oct/interleaved/mean/benchmark_datapoints_seq_style/task_scores.csv",
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_curves, all_summaries = [], []
    for setting, source in sources.items():
        tasks = read_tasks(source)
        for kind in ("ewma", "rolling10"):
            curves, summaries = build_curves(tasks, setting, kind, args)
            directory = output / setting / kind
            directory.mkdir(parents=True, exist_ok=True)
            save_plot(directory, curves, summaries, setting, kind, args)
            all_curves.extend(curves)
            all_summaries.extend(summaries)
    fields = list(dict.fromkeys(key for row in all_curves for key in row))
    write_csv(output / "smoothed_curves_and_metrics.csv", [{key: row.get(key) for key in fields} for row in all_curves])
    write_csv(
        output / "stabilization_summary.csv",
        [{key: value for key, value in row.items() if key != "episodes"} for row in all_summaries],
    )
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "parameters": {key: value for key, value in vars(args).items() if key != "output"},
        "curve_definition": "Smooth raw scores per seed/benchmark before averaging; datapoint-local histories",
        "ewma": (
            "Recursive alpha*x+(1-alpha)*state, initialized at first recorded score; state pauses across missing scores"
        ),
        "rolling": "Last 10 task positions, excluding missing scores from the denominator; partial windows plotted",
        "eligibility": (
            "Full warm-up; finite observed points; rolling windows require complete score coverage; "
            "mean assessments require all three eligible seeds"
        ),
        "onset_definition": (
            "First assessment endpoint of an episode with at least patience consecutive passing windows"
        ),
        "stable_tail_definition": (
            "Final passing episode lasts at least patience steps and continues through the observed end"
        ),
        "temporary_definition": "A confirmed episode occurred, but criteria are not maintained at the observed end",
        "insufficient_data_definition": "No complete eligible assessment window is available",
        "results": all_summaries,
        "sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources.values()},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / ".gitignore").write_text("*\n")
    means = [row for row in all_summaries if row["scope"] == "mean"]
    for row in means:
        print(row["setting"], row["smoother"], row["benchmark"], row["method"], row["status"], row["stable_onset_step"])
    print(f"Figures and metrics: {output}")


if __name__ == "__main__":
    main()
