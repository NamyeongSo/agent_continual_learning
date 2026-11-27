# SPDX-License-Identifier: Apache-2.0
"""Reproduce the descriptive three-seed ADD-only summary without inference."""

import contextlib
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.ace.analyze_reflection_accumulation import analyze
from scripts.ace.compare_reflection_accumulation import compare
from scripts.ace.schedule_nonthinking_replication import predecessor_ready


def main():
    run = Path(__file__).resolve().parents[1]
    assert predecessor_ready(run)
    output = run / "analysis"
    summaries, expectations = {}, {}
    with (output / "three_seeds_analysis_stdout.log").open("w") as log, contextlib.redirect_stdout(log):
        for seed in (42, 43, 44):
            summaries[seed] = analyze(run, seed)
            expectations[seed] = compare(run, seed)
            assert set(summaries[seed]["completed_phases"]) == {"train_final", "bwt_checkpoint", "val"}
    metrics = {"train_pct": ("train_final", "mean_pct"), "train_tgc": ("train_final", "tgc_rate"),
               "heldout_pct": ("val", "mean_pct"), "heldout_tgc": ("val", "tgc_rate"),
               "bwt_pct_all50": ("bwt", "all50"), "bwt_pct_prior49": ("bwt", "prior49"),
               "bwt_tgc_all50": ("bwt", "tgc_all50"), "bwt_tgc_prior49": ("bwt", "tgc_prior49")}
    result = {"seeds": [42, 43, 44], "unique_train_tasks": 50, "unique_val_tasks": 50,
              "actor_evaluations": 2700, "logical_curator_steps": 900, "actual_curator_calls": 901,
              "aggregation": "Equal mean and sample standard deviation across 3 seed-level task means",
              "limitations": ["The same 50 tasks recur across seeds; not 150 unique tasks",
                              "Seed variation includes selection, Curator and rollout randomness, not only order",
                              "Expected scores reuse the same selection bank and are not new independent samples",
                              "No independent-reflection/empty-context held-out baseline was measured"],
              "conditions": {}, "source_sha256": {}}
    rows = []
    for arm in ("without_gt", "with_gt"):
        for mode in ("random", "best", "worst"):
            key = f"{arm}_{mode}"
            values = {}
            for metric, (phase, field) in metrics.items():
                v = [summaries[s]["conditions"][f"s{s}_{key}"][phase][field] for s in (42, 43, 44)]
                values[metric] = {"values": v, "mean": statistics.mean(v), "sample_std": statistics.stdev(v)}
            for metric in ("expected_independent_pct", "expected_independent_tgc", "delta_pct", "delta_tgc"):
                v = [expectations[s]["conditions"][f"s{s}_{key}"][metric] for s in (42, 43, 44)]
                values[metric] = {"values": v, "mean": statistics.mean(v), "sample_std": statistics.stdev(v)}
            result["conditions"][key] = values
            for seed in (42, 43, 44):
                c = summaries[seed]["conditions"][f"s{seed}_{key}"]
                assert all(c[p]["tasks"] == c[p]["tgc_recorded_outcomes"] == 50
                           for p in ("train_final", "bwt_checkpoint", "val"))
                rows.append({"seed": seed, "condition": key,
                             **{m: c[p][f] for m, (p, f) in metrics.items()},
                             **{m: expectations[seed]["conditions"][f"s{seed}_{key}"][m]
                                for m in ("expected_independent_pct", "expected_independent_tgc",
                                          "delta_pct", "delta_tgc")}})
    for seed in (42, 43, 44):
        for label in ("complete", "independent_vs_accumulated"):
            path = output / f"seed{seed}_{label}.json"
            result["source_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["source_sha256"][str(run / "protocol.json")] = hashlib.sha256((run / "protocol.json").read_bytes()).hexdigest()
    result["analysis_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / "three_seeds_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (output / "three_seeds_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(result, output)
    print(json.dumps(result["conditions"], ensure_ascii=False))


def plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    keys = list(result["conditions"])
    labels = [k.replace("without_gt_", "No GT\n").replace("with_gt_", "GT\n") for k in keys]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    panels = [("train_pct", "Train PCT (%)"), ("heldout_pct", "Held-out PCT (%)"),
              ("bwt_pct_prior49", "PCT BWT prior49 (pp)"), ("train_tgc", "Train TGC (%)"),
              ("heldout_tgc", "Held-out TGC (%)"), ("bwt_tgc_prior49", "TGC BWT prior49 (pp)")]
    for ax, (metric, title) in zip(axes.flat, panels, strict=True):
        for i, key in enumerate(keys):
            v = result["conditions"][key][metric]
            ax.errorbar(i, v["mean"] * 100, yerr=v["sample_std"] * 100,
                        fmt="o", capsize=4, color="#175c88" if key.startswith("without") else "#c46721")
            ax.scatter(i + np.array([-.11, 0, .11]), np.array(v["values"]) * 100,
                       s=20, color="#777777", alpha=.7)
        if "bwt" in metric:
            ax.axhline(0, color="black", lw=.8)
        ax.set_title(title)
        ax.set_xticks(range(6), labels, fontsize=8)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Thinking ADD-only: seed means +/- sample SD (n=3); gray dots are individual seeds")
    fig.tight_layout()
    fig.savefig(output / "three_seeds_performance.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ("pct", "tgc"), strict=True):
        for i, key in enumerate(keys):
            expected = result["conditions"][key][f"expected_independent_{metric}"]["mean"] * 100
            actual = result["conditions"][key][f"train_{metric}"]["mean"] * 100
            ax.plot([i-.12, i+.12], [expected, actual], color="#aaaaaa")
            ax.scatter(i-.12, expected, marker="s", color="#777777", label="Selected independent expectation" if i == 0 else None)
            ax.scatter(i+.12, actual, color="#175c88", label="Accumulated C50 actual" if i == 0 else None)
        ax.set_xticks(range(6), labels)
        ax.set_title(f"Train {metric.upper()} (%) - mean across 3 seeds")
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Matched selected reflections; same-bank selection bias and unequal repeats apply")
    fig.tight_layout()
    fig.savefig(output / "three_seeds_expectation_vs_actual.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
