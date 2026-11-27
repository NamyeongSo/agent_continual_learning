# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Plot Qwen same-history policy shift: BFCL on the left, AppWorld on the right."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "openai/Qwen/Qwen3.6-35B-A3B"
POINTS = tuple(range(0, 51, 5))
SEEDS = (42, 43, 44)
METRICS = (("strict_accuracy", r"$1-\mathrm{ACC}$", "#2467A9", "o", "-"),
           ("strict_f1", r"$1-\mathrm{F1}$", "#D07825", "s", "--"))


def aggregate(analysis: Path, benchmark: str, model_name: str = MODEL) -> tuple[list[dict], dict]:
    summary = json.loads((analysis / "summary.json").read_text())
    # The original BFCL-only analyzer predates the explicit benchmark field.
    recorded_benchmark = summary.get("benchmark", "BFCL")
    if summary["model"]["model"] != model_name or recorded_benchmark != benchmark:
        raise ValueError(f"Expected the {model_name} {benchmark} study")
    path = analysis / "pairwise_metrics.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(seed, left, right) for seed in SEEDS for i, left in enumerate(POINTS) for right in POINTS[i + 1:]}
    keys = [(int(row["seed"]), int(row["left"]), int(row["right"])) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Expected all 55 distinct checkpoint pairs for each of three seeds")
    for row in rows:
        if int(row["distance"]) != int(row["right"]) - int(row["left"]):
            raise ValueError("Incorrect checkpoint distance")
        for metric, *_ in METRICS:
            value = float(row[metric])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {metric}")
    data = []
    for distance in POINTS[1:]:
        group = [row for row in rows if int(row["distance"]) == distance]
        for metric, *_ in METRICS:
            values = [1 - float(row[metric]) for row in group]
            data.append({"benchmark": benchmark, "checkpoint_distance": distance,
                         "metric": metric, "mean_shift": statistics.mean(values),
                         "pair_sample_sd": statistics.stdev(values), "checkpoint_pairs": len(values),
                         "seeds": len({row["seed"] for row in group})})
    return data, summary


def run(bfcl: Path, appworld: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = (("BFCL", "PREMiSE", bfcl), ("AppWorld", "ACE", appworld))
    datasets, provenance, all_rows = [], [], []
    for benchmark, method, analysis in sources:
        data, summary = aggregate(analysis, benchmark)
        datasets.append(data)
        all_rows.extend(data)
        provenance.append({"benchmark": benchmark, "method": method, "analysis": str(analysis),
                           "coverage": summary["coverage"], "source_hashes": {
                               name: hashlib.sha256((analysis / name).read_bytes()).hexdigest()
                               for name in ("pairwise_metrics.csv", "summary.json")}})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "qwen_policy_shift.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
                         "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7.5, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.65})
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.95), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.23, top=0.88, wspace=0.17)
    maximum = max(row["mean_shift"] + row["pair_sample_sd"] for row in all_rows)
    upper = min(1.0, max(0.5, math.ceil((maximum + 0.03) * 10) / 10))
    correlations = []
    for axis, data, (benchmark, _method, _) in zip(axes, datasets, sources, strict=True):
        for metric, label, color, marker, line in METRICS:
            group = [row for row in data if row["metric"] == metric]
            x = [row["checkpoint_distance"] for row in group]
            y = [row["mean_shift"] for row in group]
            sd = [row["pair_sample_sd"] for row in group]
            pearson = statistics.correlation(x, y) if len(set(y)) > 1 else None
            correlations.append({"benchmark": benchmark, "metric": metric, "points": len(x), "pearson_r": pearson})
            value = f"{pearson:.4f}" if pearson is not None else r"\mathrm{undefined}"
            legend_label = label[:-1] + rf",\ r={value}$"
            axis.fill_between(x, [max(0, mean - spread) for mean, spread in zip(y, sd, strict=True)],
                              [min(1, mean + spread) for mean, spread in zip(y, sd, strict=True)],
                              color=color, alpha=0.12, linewidth=0)
            axis.plot(x, y, color=color, marker=marker, markersize=2.8, markerfacecolor="white",
                      markeredgewidth=0.7, linewidth=1.1, linestyle=line, label=legend_label)
        axis.set_title(benchmark.upper(), pad=5)
        axis.set_xlim(1, 52)
        axis.set_ylim(0, upper)
        axis.set_xticks([10, 20, 30, 40, 50])
        axis.set_yticks([tick / 10 for tick in range(0, round(upper * 10) + 1, 2)])
        axis.grid(axis="y", color="#D5D5D5", linewidth=0.45, alpha=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2.5, width=0.6, pad=2)
        axis.legend(loc="lower right", frameon=False, fontsize=6.5, handlelength=1.3,
                    handletextpad=0.4, borderaxespad=0.45, labelspacing=0.4)
    axes[0].set_ylabel("Policy shift", labelpad=4)
    fig.supxlabel("Checkpoint distance (training tasks)", y=0.025, fontsize=8)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(output / f"qwen_policy_shift.{extension}", dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    with (output / "qwen_policy_shift_correlations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(correlations[0]))
        writer.writeheader()
        writer.writerows(correlations)
    (output / "summary.json").write_text(json.dumps({
        "model": MODEL, "panel_order": ["BFCL", "AppWorld"], "figure_width_inches": 3.35,
        "metrics": "Strict: function names and parsed JSON arguments; 1 - ACC and 1 - F1",
        "aggregation": "Mean across within-seed checkpoint pairs at each distance; source scores are task-balanced",
        "bands": "One sample SD across checkpoint pairs and seeds; descriptive, not confidence intervals",
        "histories": "All-checkpoint successful-history intersection, separately within each benchmark",
        "correlation_definition": "Pearson r between checkpoint distance and mean policy shift across ten distances",
        "correlations": correlations,
        "sources": provenance, "created_at": datetime.now(UTC).isoformat(),
    }, ensure_ascii=False, indent=2) + "\n")
    caption = (
        "Qwen policy shift on BFCL (PREMiSE; left) and AppWorld (ACE; right). "
        "We compare checkpoint predictions under identical saved histories using strict action agreement "
        "(function names and arguments). Curves show $1-\\mathrm{ACC}$ and $1-\\mathrm{F1}$ versus "
        "checkpoint distance; shading is one sample standard deviation across within-seed checkpoint pairs "
        "from three seeds. Each benchmark uses histories with valid predictions at all checkpoints. "
        "Corner legends report Pearson $r$ between checkpoint distance and mean shift across ten distances."
    )
    (output / "qwen_policy_shift.tex").write_text(
        "\\begin{figure}[t]\n"
        "  \\centering\n"
        "  \\includegraphics[width=\\columnwidth]{figures/qwen_policy_shift.pdf}\n"
        f"  \\caption{{{caption}}}\n"
        "  \\label{fig:qwen-policy-shift}\n"
        "\\end{figure}\n"
    )
    print(f"Saved {output / 'qwen_policy_shift.pdf'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfcl-analysis", type=Path, default=ROOT / "outputs"
                        / "bfcl_same_history_policy_shift_cp5_steps20_queue150_20261011/qwen/analysis")
    parser.add_argument("--appworld-analysis", type=Path, default=ROOT / "outputs"
                        / "appworld_ace_qwen_policy_shift_cp5_restored_history_queue150_20261013/analysis")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/qwen_bfcl_appworld_policy_shift_20261013")
    args = parser.parse_args()
    run(args.bfcl_analysis.resolve(), args.appworld_analysis.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
