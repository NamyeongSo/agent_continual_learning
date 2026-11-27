# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Plot complete seed-44 interleaved validation scores across training checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/premise_ace_interleaved_s44_val10_then_best_test_queue100_20261009/all_validation_scores.json"
OUTPUT = ROOT / "acr_oct/interleaved/seed44/validation"
METHODS = ("ace", "premise")
LABELS = {"ace": "ACE", "premise": "PREMiSE"}
COLORS = {"ace": "#2474B5", "premise": "#E06B35"}
MARKERS = {"ace": "o", "premise": "s"}
POINTS = tuple(range(10, 131, 10))
COUNTS = {"bfcl": 50, "appworld": 50, "terminalbench2": 30}


def load_scores(source: Path) -> list[dict]:
    rows = json.loads(source.read_text())["scores"]
    expected = {(method, point) for method in METHODS for point in POINTS}
    if len(rows) != len(expected) or {(r["method"], r["checkpoint"]) for r in rows} != expected:
        raise ValueError("Expected 26 unique validation checkpoints")
    for row in rows:
        if (row["seed"], row["mode"], row["split"], row["complete"], row["total"]) != (
            44, "interleaved", "val", True, 130,
        ):
            raise ValueError("Expected complete seed-44 interleaved validation scores")
        for benchmark, count in COUNTS.items():
            if row["by_benchmark"][benchmark]["total"] != count:
                raise ValueError("Validation benchmark task count differs")
        for metric in ("hard", "soft"):
            weighted = sum(row["by_benchmark"][b][metric] * n for b, n in COUNTS.items()) / 130
            if not 0 <= row[metric] <= 1 or not math.isclose(row[metric], weighted, abs_tol=1e-9):
                raise ValueError("Overall score differs from the task-weighted benchmark scores")
    return sorted(rows, key=lambda r: (r["method"], r["checkpoint"]))


def style_axis(ax, title: str, ylim: tuple[float, float], step: int = 5) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=13)
    ax.set_xlim(5, 135)
    ax.set_xticks(POINTS)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Training iteration (checkpoint)", labelpad=10)
    ax.set_ylabel("Validation score", labelpad=9)
    ax.grid(axis="y", color="#DDE3EB", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#BBC4D0")
    ax.tick_params(labelsize=9, color="#BBC4D0")


def draw_lines(ax, rows: list[dict], metric: str, benchmark: str | None = None, annotate: bool = False) -> None:
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        x = [r["checkpoint"] for r in selected]
        y = [100 * (r["by_benchmark"][benchmark][metric] if benchmark else r[metric]) for r in selected]
        ax.plot(x, y, color=COLORS[method], linestyle="-" if method == "ace" else "--", linewidth=2.2,
                marker=MARKERS[method], markersize=5.8, markeredgewidth=1.1, label=LABELS[method], zorder=3)
        for index, point in enumerate(x):
            if point in (50, 100, 130):
                ax.plot(point, y[index], marker=MARKERS[method], markersize=5.8,
                        markerfacecolor="white", markeredgecolor=COLORS[method], markeredgewidth=1.3, zorder=4)
        if annotate:
            best = max(range(len(y)), key=lambda i: y[i])
            ax.scatter([x[best]], [y[best]], marker="*", s=155, color=COLORS[method],
                       edgecolor="white", linewidth=0.8, zorder=5)
            ax.annotate(f"CP{x[best]}  {y[best]:.2f}%", xy=(x[best], y[best]),
                        xytext=(0, 14 if method == "ace" else 21), textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold", color=COLORS[method],
                        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.9})


def add_legend(fig, anchor: tuple[float, float]) -> None:
    handles = [Line2D([], [], color=COLORS[m], linestyle="-" if m == "ace" else "--",
                      marker=MARKERS[m], linewidth=2.2, markersize=5.8, label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=anchor, frameon=False, ncol=2, fontsize=11)


def save_figure(fig, output: Path, basename: str) -> list[Path]:
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"{basename}.{suffix}"
        fig.savefig(path, dpi=240, facecolor="white", bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot(source: Path, output: Path) -> list[Path]:
    rows = load_scores(source)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.size": 10, "text.color": "#1E293B", "axes.labelcolor": "#334155",
                         "xtick.color": "#475569", "ytick.color": "#475569"})
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.20, top=0.73, wspace=0.20)
    fig.text(0.065, 0.94, "Validation performance across checkpoints", fontsize=19, fontweight="bold")
    fig.text(0.065, 0.878, "Interleaved setting · seed 44 · 130 validation tasks per checkpoint",
             fontsize=11, color="#64748B")
    add_legend(fig, (0.985, 0.945))
    for ax, metric, title, limits in zip(axes, ("hard", "soft"),
                                        ("Hard · full-task success", "Soft · partial-credit score"),
                                        ((30, 57), (45, 64)), strict=True):
        style_axis(ax, title, limits)
        draw_lines(ax, rows, metric, annotate=True)
    fig.text(0.065, 0.065, "BFCL 50 + AppWorld 50 + TerminalBench 30; overall scores are weighted by task count.",
             fontsize=9, color="#64748B")
    fig.text(0.065, 0.022, "Errors score zero. Hollow markers: reused CP50 / CP100 / CP130. "
             "Stars: highest score in each panel.",
             fontsize=9, color="#64748B")
    artifacts = save_figure(fig, output, "validation_curves_seed44")

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.5))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.12, top=0.81, hspace=0.45, wspace=0.20)
    fig.text(0.065, 0.945, "Validation performance by benchmark", fontsize=19, fontweight="bold")
    fig.text(0.065, 0.905, "Interleaved setting · seed 44 · ACE vs PREMiSE", fontsize=11, color="#64748B")
    add_legend(fig, (0.985, 0.95))
    panels = [("bfcl", "hard", "BFCL · Hard = Soft · n=50", (50, 85)),
              ("terminalbench2", "hard", "TerminalBench · Hard = Soft · n=30", (5, 40)),
              ("appworld", "hard", "AppWorld · Hard (TGC) · n=50", (0, 50)),
              ("appworld", "soft", "AppWorld · Soft (SGC) · n=50", (35, 70))]
    for ax, (benchmark, metric, title, limits) in zip(axes.flat, panels, strict=True):
        style_axis(ax, title, limits)
        draw_lines(ax, rows, metric, benchmark)
    fig.text(0.065, 0.038, "Errors score zero. Hollow markers: reused CP50 / CP100 / CP130. "
             "All other checkpoints were newly evaluated.",
             fontsize=9, color="#64748B")
    artifacts += save_figure(fig, output, "validation_by_benchmark_seed44")
    with (output / "validation_curve_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "iteration", "seed", "overall_hard_pct", "overall_soft_pct", "errors",
                         *[f"{b}_{m}_pct" for b in COUNTS for m in ("hard", "soft")]])
        for row in rows:
            writer.writerow([row["method"], row["checkpoint"], 44, row["hard"] * 100, row["soft"] * 100,
                             row["execution_errors"],
                             *[row["by_benchmark"][b][m] * 100 for b in COUNTS for m in ("hard", "soft")]])
    manifest = {"created_at": datetime.now(UTC).isoformat(), "source": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "seed": 44,
                "mode": "interleaved", "checkpoints": list(POINTS), "methods": list(METHODS),
                "tasks_per_checkpoint": 130, "benchmark_counts": COUNTS, "std": None,
                "std_reason": "One seed; no uncertainty band inferred from checkpoint variation",
                "reused_checkpoints": [50, 100, 130], "error_scoring": "zero", "smoothing": False,
                "artifacts": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                              for p in artifacts]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Seed 44 interleaved checkpoint validation curves\n\n"
        "validation_curves_seed44.png/pdf: overall task-weighted Hard and Soft across checkpoint ordinals 10-130.\n\n"
        "validation_by_benchmark_seed44.png/pdf: BFCL, TerminalBench, AppWorld Hard, and AppWorld Soft.\n\n"
        "All 26 checkpoints have complete validation scores. CP50/100/130 reuse their previously completed results. "
        "Hollow markers identify those reused points. Every other checkpoint was newly evaluated. "
        "Errors count as zero. Points are connected directly without smoothing. Only seed 44 is shown.\n\n"
        "Stars in the overall figure show the maximum score for each method in each panel. Test checkpoint selection "
        "uses overall Hard, then Soft for ties. A Soft-panel maximum is not necessarily "
        "the selected test checkpoint.\n\n"
        "validation_curve_scores.csv contains the exact percent scores. "
        "manifest.json records source and image hashes.\n"
    )
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    for path in plot(args.source.resolve(), args.output.resolve()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
