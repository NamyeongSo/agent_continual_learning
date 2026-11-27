# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Compare final-CP train/val/test and validation-selected test using one frozen evaluation queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils.plot_checkpoint_learning_curves import BENCHMARKS, LABELS, read_scores, write_csv  # noqa: E402

METRICS = {"train": "Offline Train", "val": "Offline Val", "test": "Offline Test",
           "test_early": "Offline Test (early)"}
COLORS = {"train": "#8060A8", "val": "#2C9279", "test": "#347ABD", "test_early": "#D28A24"}


def select_samples(scores, config, training_selections=None):
    lookup = {(r["variant"], r["benchmark"], r["seed"], r["checkpoint"], r["split"]): r for r in scores}
    if len(lookup) != len(scores):
        raise ValueError("Duplicate checkpoint evaluation rows")
    result = []
    for variant in LABELS:
        for benchmark in BENCHMARKS:
            points = config["offline_early"]["points"][benchmark]
            final = max(points)
            reuse_final = variant in config["offline_early"].get("reuse_training_val_best_variants", [])
            for seed in config["seeds"]:
                vals = [lookup[variant, benchmark, seed, point, "val"] for point in points]
                if not reuse_final and any(r["score"] is None or r["measured"] != r["total"] or r["errors"]
                                           for r in vals):
                    raise ValueError("All candidate validation checkpoints must be fully evaluated")
                if reuse_final:
                    audit = (training_selections or {}).get((variant, benchmark, seed))
                    if not audit or audit["final_cp"] != final or not audit["frozen_policy_matches_final_cp"]:
                        raise ValueError("Training Val-best final policy requires a verified frozen-policy audit")
                    chosen = lookup[variant, benchmark, seed, final, "val"]
                    selection_val = audit["training_val_best_score"]
                    selection_rule = "training_val_best_final_policy"
                else:
                    chosen = min(vals, key=lambda r: (-r["score"], r["checkpoint"]))
                    selection_val = chosen["score"]
                    selection_rule = "reevaluated_cp_val_best"
                for metric in METRICS:
                    point = chosen["checkpoint"] if metric == "test_early" else final
                    split = "test" if metric == "test_early" else metric
                    row = lookup[variant, benchmark, seed, point, split]
                    complete = row["score"] is not None and row["measured"] == row["total"] and not row["errors"]
                    result.append({"variant": variant, "benchmark": benchmark, "seed": seed, "metric": metric,
                                   "checkpoint": point, "checkpoint_id": row["checkpoint_id"],
                                   "canonical": row["canonical"], "score": row["score"] if complete else None,
                                   "measured_score": row["score"], "measured": row["measured"],
                                   "total": row["total"], "errors": row["errors"],
                                   "status": "complete" if complete else "missing_task_outcomes",
                                   "selection_val": selection_val if metric == "test_early" else None,
                                   "selection_rule": selection_rule if metric == "test_early" else "final_cp",
                                   "test_result_reused": reuse_final and metric == "test_early"})
    return result


def aggregate(samples, expected_seeds):
    result = []
    for variant in LABELS:
        for benchmark in BENCHMARKS:
            for metric in METRICS:
                rows = [r for r in samples if (r["variant"], r["benchmark"], r["metric"]) ==
                        (variant, benchmark, metric)]
                if {r["seed"] for r in rows} != set(expected_seeds) or len(rows) != len(expected_seeds):
                    raise ValueError("Missing or duplicate seeds")
                values = [100 * r["score"] for r in rows if r["score"] is not None]
                mean = statistics.mean(values) if values else None
                sd = statistics.stdev(values) if len(values) > 1 else None
                display = "—" if mean is None else f"{mean:.2f}" + (f" ({sd:.2f})" if sd is not None else "")
                result.append({"variant": variant, "benchmark": benchmark, "metric": metric,
                               "mean_percent": mean, "sample_sd_percent": sd, "seeds": len(values),
                               "expected_seeds": len(expected_seeds),
                               "included_seeds": json.dumps([r["seed"] for r in rows if r["score"] is not None]),
                               "display": display + ("*" if values and len(values) < len(expected_seeds) else "")})
    return result


def plot(output, summary, seeds):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.4))
    lookup = {(r["variant"], r["benchmark"], r["metric"]): r for r in summary}
    labels = ["ACE\nAUD", "GEPA\nVal", "GEPA\nTrain-as-val", "SkillOpt\nVal", "SkillOpt\nTrain-as-val",
              "PEACE\nAUD\nw/o alignment"]
    x = np.arange(len(LABELS))
    width = .19
    handles, ranges = [], {}
    for ax, (benchmark, title) in zip(axes, BENCHMARKS.items(), strict=True):
        upper = 0
        for i, (metric, label) in enumerate(METRICS.items()):
            rows = [lookup[variant, benchmark, metric] for variant in LABELS]
            means = [r["mean_percent"] if r["mean_percent"] is not None else np.nan for r in rows]
            sd = [r["sample_sd_percent"] or 0 for r in rows]
            positions = x + (i - 1.5) * width
            bars = ax.bar(positions, means, width=width * .93, color=COLORS[metric], label=label,
                          yerr=sd, capsize=2, edgecolor="white", linewidth=.5,
                          hatch="//" if metric == "test_early" else None,
                          error_kw={"linewidth": .85, "capthick": .85, "ecolor": "#394C5C"})
            if benchmark == "bfcl":
                handles.append(bars)
            for pos, mean, spread in zip(positions, means, sd, strict=True):
                if not math.isfinite(mean):
                    ax.text(pos, 1, "NA", ha="center", fontsize=7, rotation=90)
                    continue
                text = f"{mean:.1f}"
                ax.annotate(text, (pos, mean + spread), xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", rotation=90, fontsize=7.5)
                upper = max(upper, mean + spread)
        ceiling = min(100, math.ceil((upper + 9) / 10) * 10)
        ranges[benchmark] = [0, ceiling]
        final = 30 if benchmark == "terminalbench2" else 50
        ax.set(title=f"{title} · final CP{final}", ylabel="Score (%)", ylim=(0, ceiling),
               xticks=x, xticklabels=labels)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Offline evaluation · Train / Val / Test / Test (early)", fontsize=18, y=.98)
    fig.text(.5, .92, "Qwen3.6 · Original AppWorld policy · PEACE AUD w/o alignment · "
             f"seed mean ± sample SD ({', '.join(map(str, seeds))})", ha="center", fontsize=12, color="#526271")
    fig.legend(handles, list(METRICS.values()), loc="lower center", bbox_to_anchor=(.5, .17),
               frameon=False, ncol=4, fontsize=11)
    notes = [
        "Train, Val and Test use the final frozen checkpoint's evaluation from the SAME CP queue. "
        "GEPA Val / SkillOpt Val: Test (early) reuses Test of the training Val-best final policy.",
        "Other settings: Test (early) uses the best re-evaluated Val CP per seed, including CP0; "
        "exact ties choose the earliest CP. Selection never uses Test scores.",
        "Only fully measured seed/split evaluations are averaged; seed counts are specified below. "
        "One seed has no SD. Incomplete seed/split results are omitted, never filled with zero.",
        "Terminal-Bench: Train has n=2 for ACE, SkillOpt Val, SkillOpt Train-as-val and PEACE; "
        "ACE Test (early) has n=1 (seed43). All remaining bars have n=3.",
        "AppWorld uses official TGC. Test is a final-CP re-evaluation and can differ from the main table's "
        "separately executed final Test. CP0/context aliases reuse outcomes; Train is not Online.",
    ]
    for i, note in enumerate(notes):
        fig.text(.015, .17 - i * .033, note, fontsize=9, color="#526271")
    fig.tight_layout(rect=(0, .31, 1, .88), w_pad=2)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"offline_train_val_test_early.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return ranges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--table-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.table_manifest.read_text())
    if args.queue.name != config["offline_early"]["checkpoint_queue_id"]:
        raise ValueError("Wrong checkpoint evaluation cohort")
    scores, fingerprint = read_scores(args.queue, config)
    training_selections = {}
    if config["offline_early"].get("reuse_training_val_best_variants"):
        audit = json.loads(Path(config["offline_early"]["training_val_best_audit"]).read_text())
        for path, sha in audit["source_hashes"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
                raise ValueError("A verified training Val-best source changed")
        training_selections = {(r["variant"], r["benchmark"], r["seed"]): r for r in audit["rows"]}
    samples = select_samples(scores, config, training_selections)
    summary = aggregate(samples, config["seeds"])
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "offline_split_seed_scores.csv", samples)
    write_csv(args.output / "offline_split_summary.csv", summary)
    ranges = plot(args.output, summary, config["seeds"])
    manifest = {"created_at": datetime.now(UTC).isoformat(), "queue": str(args.queue.resolve()),
                "table_manifest": str(args.table_manifest.resolve()), "journal_id": config["journal_id"],
                "seeds": config["seeds"], "source_score_fingerprint_sha256": fingerprint,
                "method_labels": LABELS, "metrics": METRICS,
                "final_cp": {b: max(p) for b, p in config["offline_early"]["points"].items()},
                "test_early": "GEPA Val / SkillOpt Val reuse Test of the verified training Val-best final policy. "
                              "Other settings: best re-evaluated Val CP per seed, CP0 included; earliest exact tie; "
                              "selected Test availability never affects checkpoint selection",
                "reuse_training_val_best_variants": config["offline_early"].get(
                    "reuse_training_val_best_variants", []),
                "training_val_best_audit": config["offline_early"].get("training_val_best_audit"),
                "aggregation": "Only complete seed/split evaluations; equal seed mean and sample SD; "
                               "no missing-value zero imputation",
                "figure_markers": "Asterisks hidden; incomplete seed counts disclosed in figure footnote",
                "test_source": "Final-CP queue reevaluation, independent of the main-table final test run",
                "partial_groups": [r for r in summary if r["seeds"] < r["expected_seeds"]],
                "panel_y_limits_percent": ranges,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "new_inference": False,
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()
                          if p.suffix in {".png", ".pdf", ".csv"}}}
    (args.output / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "seed_rows": len(samples), "groups": len(summary),
                      "partial_groups": manifest["partial_groups"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
