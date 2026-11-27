# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Render matched-sample and fixed-valid-prefix action agreement diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCHMARKS = (("bfcl", "BFCL"), ("appworld", "AppWorld"), ("terminalbench2", "Terminal-Bench"))
COLORS = {0: "#235789", 1: "#c55a11"}
LABELS = {"precision": "Precision", "recall": "Recall", "f1": "F1", "bleu1": "BLEU-1",
          "bleu4": "BLEU-4 (effective order)", "meteor": "METEOR", "content_f1": "Normalized content F1"}


def save(fig, output, name):
    files = []
    for suffix in ("png", "pdf"):
        path = output / f"{name}.{suffix}"
        fig.savefig(path, dpi=200)
        files.append(path.name)
    plt.close(fig)
    return files


def rows_for(summary, benchmark, cohort, mode, temp, view="cp0_bank"):
    return sorted((r for r in summary["curves"] if r["benchmark"] == benchmark and r["cohort"] == cohort and
                   r["mode"] == mode and r["temperature"] == temp and r["view"] == view),
                  key=lambda r: r["distance"])


def decorate(ax, benchmark, title, ylabel):
    ax.set(title=title, xlabel="Update distance", ylabel=ylabel, ylim=(-.03, 1.03))
    ax.set_xticks(range(0, 31 if benchmark == "terminalbench2" else 51, 10))
    ax.grid(alpha=.2)


def metric_panels(summary, output, cohort, mode, metrics, group):
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 9.5))
    for i, (benchmark, title) in enumerate(BENCHMARKS):
        sample = rows_for(summary, benchmark, cohort, mode, 0)
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            for temp in (0, 1):
                rows = rows_for(summary, benchmark, cohort, mode, temp)
                ax.plot([r["distance"] for r in rows], [r[metric] for r in rows], "o-", color=COLORS[temp],
                        markersize=4, label=f"T={temp}")
                if rows:
                    ax.axhline(rows[0][metric], color=COLORS[temp], linestyle=":", alpha=.5)
            decorate(ax, benchmark, f"{title} | {sample[0]['tasks']} tasks, {sample[0]['prefixes']} prefixes",
                     LABELS[metric])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=2, frameon=False)
    fig.suptitle(f"Aligned-prefix {mode} {group} | {cohort} | CP0 bank reference", fontsize=14)
    fig.text(.5, .02, "Score each identical prefix separately, then average within task and equally across tasks. "
             "Dotted: independent CP0 self control. No cross-prefix matching.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .9))
    return save(fig, output, f"{cohort}_{mode}_{group}")


def normalization_panels(summary, output, cohort, all_pairs=False):
    fig, axes = plt.subplots(3, 2, figsize=(11, 9))
    view = "all_pairs" if all_pairs else "cp0_bank"
    for i, (benchmark, title) in enumerate(BENCHMARKS):
        for temp in (0, 1):
            rows = rows_for(summary, benchmark, cohort, "strict", temp, view)
            for metric, style, suffix in (("f1", "-", "raw"), ("content_f1", "--", "normalized")):
                axes[i, 0].plot([r["distance"] for r in rows], [r[metric] for r in rows], "o" + style,
                                color=COLORS[temp], markersize=3, label=f"T={temp} {suffix}")
                axes[i, 1].plot([r["distance"] for r in rows], [r[f"gap_{metric}"] for r in rows], "o" + style,
                                color=COLORS[temp], markersize=3, label=f"T={temp} {suffix}")
        for ax in axes[i]:
            ax.set_xticks(range(0, 31 if benchmark == "terminalbench2" else 51, 10))
            ax.grid(alpha=.2)
        axes[i, 0].set(title=title, xlabel="Update distance", ylabel="Prefix-wise F1", ylim=(-.03, 1.03))
        axes[i, 1].axhline(0, color="#555555", linewidth=1)
        axes[i, 1].set(title=title, xlabel="Update distance", ylabel="Self control minus cross-context F1")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=4, frameon=False, fontsize=8)
    fig.suptitle(f"Raw / normalized prefix agreement and descriptive gaps | {cohort} | {view}", fontsize=12)
    fig.text(.5, .02, "AppWorld normalization removes Python formatting; Terminal removes duration; BFCL unchanged. "
             "Neither proves semantic equivalence. Gaps can be negative.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .9))
    return save(fig, output, f"{cohort}_{view}_normalization_gaps")


def unit_comparison(summary, paired, output):
    fig, axes = plt.subplots(3, 2, figsize=(11, 9))
    comparisons = []
    for i, (benchmark, title) in enumerate(BENCHMARKS):
        for j, mode in enumerate(("strict", "loose")):
            ax = axes[i, j]
            for temp in (0, 1):
                rows = rows_for(summary, benchmark, "matched_sequences", mode, temp)
                old = sorted((r for r in paired["summaries"][temp]["cp0_complete_sequences"] if
                              r["benchmark"] == benchmark and r["mode"] == mode), key=lambda r: r["distance"])
                if [r["distance"] for r in rows] != [r["distance"] for r in old]:
                    raise ValueError("Scoring-unit comparison has different checkpoint grids")
                ax.plot([r["distance"] for r in rows], [r["f1"] for r in rows], "o-", color=COLORS[temp],
                        markersize=3, label=f"T={temp} prefix")
                ax.plot([r["distance"] for r in old], [r["f1"] for r in old], "x:", color=COLORS[temp],
                        label=f"T={temp} concat")
                comparisons.extend({"benchmark": benchmark, "mode": mode, "temperature": temp,
                                    "distance": a["distance"], "prefix_f1": a["f1"], "concat_f1": b["f1"],
                                    "tasks": a["tasks"], "prefixes": a["prefixes"]}
                                   for a, b in zip(rows, old, strict=True))
            decorate(ax, benchmark, f"{title} | {mode}", "F1")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Only the scoring unit changes: identical tasks, prefixes and generated draw pairs", fontsize=13)
    fig.text(.5, .02, "Solid: compare at each prefix, then average. "
             "Dotted: concatenate full regenerated actions first. Both use identical saved samples and task weights.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .9))
    (output / "concat_vs_prefix_values.json").write_text(json.dumps(comparisons, indent=2))
    return save(fig, output, "concat_vs_aligned_prefix_f1")


def prefix_distribution(prefixes, output):
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    for i, (benchmark, title) in enumerate(BENCHMARKS):
        final = 30 if benchmark == "terminalbench2" else 50
        for temp in (0, 1):
            ax = axes[i, temp]
            for distance, label, color in [(0, "CP0 self", "#235789"), (final, f"CP{final}", "#c55a11")]:
                rows = [r for r in prefixes if r["benchmark"] == benchmark and r["cohort"] == "valid_prefixes"
                        and r["view"] == "cp0_bank" and r["temperature"] == temp and r["mode"] == "strict"
                        and r["distance"] == distance]
                counts = Counter(r["task"] for r in rows)
                ordered = sorted(rows, key=lambda r: r["f1"])
                total, y = 0, []
                for row in ordered:
                    total += 1 / (len(counts) * counts[row["task"]])
                    y.append(total)
                ax.step([0, *[r["f1"] for r in ordered], 1], [0, *y, 1], where="post", label=label, color=color)
            ax.set(title=f"{title} | T={temp}", xlabel="Per-prefix repeat-pair mean F1",
                   ylabel="Task-balanced cumulative weight", xlim=(-.02, 1.02), ylim=(-.02, 1.02))
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Strict prefix F1 distributions | fixed valid-prefix cohort", fontsize=14)
    fig.text(.5, .02, "Each prefix first averages independent draw pairs. Each task has equal total weight. "
             "Dependent descriptive samples; no confidence intervals.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .95))
    return save(fig, output, "valid_prefixes_strict_f1_distribution")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--paired-summary", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    paired = json.loads(args.paired_summary.read_text())
    prefixes = [json.loads(line) for line in (args.run / "prefix_means.jsonl").read_text().splitlines()]
    output = args.run / "figures"
    output.mkdir(exist_ok=True)
    files = []
    for cohort in ("matched_sequences", "valid_prefixes"):
        for mode in ("strict", "loose"):
            files += metric_panels(summary, output, cohort, mode, ("precision", "recall", "f1"), "prf")
            files += metric_panels(summary, output, cohort, mode, ("bleu1", "bleu4", "meteor"), "bleu_meteor")
        files += normalization_panels(summary, output, cohort)
    files += normalization_panels(summary, output, "valid_prefixes", all_pairs=True)
    files += unit_comparison(summary, paired, output)
    files += prefix_distribution(prefixes, output)
    (output / "manifest.json").write_text(json.dumps({"figures": files}, indent=2))
    print(json.dumps({"figures": len(files), "output": str(output)}))


if __name__ == "__main__":
    main()
