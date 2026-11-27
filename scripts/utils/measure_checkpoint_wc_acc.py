# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Compute observed WC-ACC on ten-task training blocks using the frozen CP evaluation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils.measure_checkpoint_stability_gap import task_order  # noqa: E402
from scripts.utils.plot_checkpoint_learning_curves import (  # noqa: E402
    BENCHMARKS,
    COLORS,
    LABELS,
    panel_limits,
    read_scores,
    write_csv,
)


def stage_score(order, outcomes, stage, block_size=10):
    """Minimize block accuracies after learning, with a fixed observed cohort for each block."""
    current_cp = stage * block_size
    blocks = []
    for learned in range(1, stage + 1):
        tasks = order[(learned - 1) * block_size:learned * block_size]
        if len(tasks) != block_size:
            raise ValueError("Incomplete training block")
        observed = [current_cp] if learned == stage else [j * block_size for j in range(learned + 1, stage + 1)]
        common = [task for task in tasks if all(outcomes[cp].get(task) is not None for cp in observed)]
        scores = {cp: statistics.mean(outcomes[cp][task] for task in common) for cp in observed} if common else {}
        worst_cp = min(scores, key=lambda cp: (scores[cp], cp)) if scores else None
        blocks.append({"learned_block": learned, "learned_at_cp": learned * block_size,
                       "original_tasks": len(tasks), "common_tasks": len(common),
                       "excluded_task_ids": json.dumps([task for task in tasks if task not in common]),
                       "common_task_ids": json.dumps(common), "observed_checkpoints": json.dumps(observed),
                       "scores_by_cp": json.dumps(scores), "minimum_cp": worst_cp,
                       "minimum_score": scores.get(worst_cp), "current_score": scores.get(current_cp)})
    complete = all(row["common_tasks"] > 0 for row in blocks)
    wc_acc = statistics.mean(row["minimum_score"] for row in blocks) if complete else None
    current = statistics.mean(row["current_score"] for row in blocks) if complete else None
    current_values = [outcomes[current_cp].get(task) for task in order[:current_cp]]
    measured = [value for value in current_values if value is not None]
    if complete and wc_acc > current + 1e-12:
        raise ValueError("WC-ACC must not exceed the matched current accuracy")
    return {"stage": stage, "checkpoint": current_cp, "wc_acc": wc_acc,
            "matched_current_accuracy": current,
            "full_seen_train_accuracy": statistics.mean(measured) if measured else None,
            "current_block_accuracy": blocks[-1]["current_score"],
            "past_min_accuracy": statistics.mean(row["minimum_score"] for row in blocks[:-1])
            if complete and stage > 1 else None,
            "common_tasks": sum(row["common_tasks"] for row in blocks), "expected_tasks": current_cp,
            "strict_full_task_wc_acc": wc_acc if all(row["common_tasks"] == block_size for row in blocks) else None,
            "status": "measured" if complete else "missing_block"}, blocks


def collect(queue, config):
    _, fingerprint = read_scores(queue, config)
    variants = {row["variant"] for row in config["rows"]}
    sources = {source["batch"]: source for source in config["sources"]}
    stages, blocks, hashes, cache = [], [], {}, {}
    with sqlite3.connect((queue / "queue.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        for row in db.execute("SELECT spec FROM streams ORDER BY id"):
            spec = json.loads(row[0])
            source = sources.get(spec["variant"])
            if (not source or spec["run_variant"] not in variants or spec["seed"] not in config["seeds"]
                    or spec["benchmark"] not in source.get("benchmarks", BENCHMARKS)):
                continue
            order = task_order(spec, hashes)
            if len(order) % 10:
                raise ValueError("Expected complete ten-task training blocks")
            checkpoints = db.execute("SELECT point,canonical FROM checkpoints WHERE stream=? ORDER BY point",
                                     (spec["id"],)).fetchall()
            outcomes = {}
            for cp in checkpoints:
                if cp["canonical"] not in cache:
                    cache[cp["canonical"]] = {r["task_id"]: r["score"] if r["status"] == "completed" else None
                                              for r in db.execute("""SELECT task_id,status,
                        json_extract(result,'$.score') AS score FROM tasks WHERE cp=? AND split='train'""",
                                                                  (cp["canonical"],))}
                outcomes[cp["point"]] = cache[cp["canonical"]]
            identity = {"variant": spec["run_variant"], "benchmark": spec["benchmark"], "seed": spec["seed"],
                        "stream": spec["id"]}
            for stage in range(1, len(order) // 10 + 1):
                result, block_rows = stage_score(order, outcomes, stage)
                stages.append({**identity, **result, "final": stage == len(order) // 10})
                blocks.extend({**identity, "checkpoint": result["checkpoint"], **r} for r in block_rows)
    return stages, blocks, hashes, fingerprint


def aggregate(stages):
    groups = defaultdict(list)
    for row in stages:
        groups[row["variant"], row["benchmark"], row["checkpoint"]].append(row)
    curves = []
    for (variant, benchmark, checkpoint), rows in sorted(groups.items()):
        values = [r["wc_acc"] for r in rows if r["wc_acc"] is not None]
        curves.append({"variant": variant, "benchmark": benchmark, "checkpoint": checkpoint,
                       "mean": statistics.mean(values) if values else None,
                       "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                       "seeds": len(values), "partial": any(r["common_tasks"] < r["expected_tasks"] for r in rows)})
    finals = [dict(row) for row in stages if row["final"]]
    for variant in LABELS:
        for seed in sorted({row["seed"] for row in stages}):
            rows = [r for r in finals if (r["variant"], r["seed"]) == (variant, seed) and r["benchmark"] in BENCHMARKS]
            if len(rows) == len(BENCHMARKS) and all(r["wc_acc"] is not None for r in rows):
                finals.append({"variant": variant, "benchmark": "avg", "seed": seed, "stream": "",
                               "stage": None, "checkpoint": None,
                               **{field: statistics.mean(r[field] for r in rows) for field in
                                  ("wc_acc", "matched_current_accuracy", "full_seen_train_accuracy",
                                   "current_block_accuracy", "past_min_accuracy")},
                               "common_tasks": sum(r["common_tasks"] for r in rows),
                               "expected_tasks": sum(r["expected_tasks"] for r in rows),
                               "strict_full_task_wc_acc": statistics.mean(r["strict_full_task_wc_acc"] for r in rows)
                               if all(r["strict_full_task_wc_acc"] is not None for r in rows) else None,
                               "status": "measured", "final": True})
    summaries = []
    for variant in LABELS:
        for benchmark in (*BENCHMARKS, "avg"):
            rows = [r for r in finals if (r["variant"], r["benchmark"]) == (variant, benchmark)
                    and r["wc_acc"] is not None]
            values = [r["wc_acc"] for r in rows]
            mean = statistics.mean(values) if values else None
            sd = statistics.stdev(values) if len(values) > 1 else None
            summaries.append({"variant": variant, "benchmark": benchmark, "mean_wc_acc": mean,
                              "sample_sd": sd, "seeds": len(values),
                              "display_percent": "—" if mean is None else f"{100 * mean:.2f}" +
                              (f" ({100 * sd:.2f})" if sd is not None else ""),
                              "partial_seed_cohorts": sum(r["common_tasks"] < r["expected_tasks"] for r in rows),
                              "common_tasks": sum(r["common_tasks"] for r in rows),
                              "expected_tasks": sum(r["expected_tasks"] for r in rows)})
    return curves, finals, summaries


def plot(output, curves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.7))
    handles = []
    for ax, (benchmark, title) in zip(axes, BENCHMARKS.items(), strict=True):
        for (variant, label), color in zip(LABELS.items(), COLORS, strict=True):
            rows = sorted((r for r in curves if (r["variant"], r["benchmark"]) == (variant, benchmark)),
                          key=lambda r: r["checkpoint"])
            x = np.array([r["checkpoint"] for r in rows])
            y = np.array([100 * r["mean"] if r["mean"] is not None else np.nan for r in rows])
            sd = np.array([100 * (r["sample_sd"] or 0) for r in rows])
            line, = ax.plot(x, y, color=color, label=label, marker="o", markersize=4,
                            linestyle="--" if "train_as_val" in variant else "-", linewidth=1.8)
            ax.fill_between(x, np.clip(y - sd, 0, 100), np.clip(y + sd, 0, 100), color=color, alpha=.09)
            partial = np.array([r["partial"] for r in rows])
            ax.scatter(x[partial], y[partial], facecolors="none", edgecolors=color, s=90, linewidths=1.1)
            if benchmark == "bfcl":
                handles.append(line)
        ax.set(title=title, ylabel="Observed train-block WC-ACC (%) ↑",
               xlabel="Checkpoint (training task IDs consumed)",
               ylim=panel_limits([r for r in curves if r["benchmark"] == benchmark]),
               xticks=sorted({r["checkpoint"] for r in curves if r["benchmark"] == benchmark}))
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Observed WC-ACC · ten-task training blocks · seed mean ± sample SD (42, 43, 44)", fontsize=16)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .14), ncol=3, frameon=False)
    fig.text(.02, .105, "At CP 10t: mean of current block accuracy and each past block's minimum accuracy "
             "over observed CPs strictly after that block was learned.", fontsize=9, color="#526271")
    fig.text(.02, .073, "Ringed points use the common officially measured task subset per block across "
             "its compared CPs. Zero scores are included; missing outcomes are never zero.",
             fontsize=9, color="#526271")
    fig.text(.02, .041, "Training-task retention measure, separate from held-out test accuracy. "
             "CP-only observation can miss temporary dips; context aliases reuse evaluation outcomes.",
             fontsize=9, color="#526271")
    fig.tight_layout(rect=(0, .28, 1, .94))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"observed_wc_acc_cp10.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--table-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.table_manifest.read_text())
    if args.queue.name != config["offline_early"]["checkpoint_queue_id"]:
        raise ValueError("Expected the configured CP10 main-table cohort")
    stages, blocks, hashes, fingerprint = collect(args.queue, config)
    curves, finals, summaries = aggregate(stages)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("stages", stages), ("blocks", blocks), ("seed_means", curves),
                       ("final_seeds", finals), ("summary", summaries)):
        write_csv(args.output / f"wc_acc_{name}.csv", rows)
    plot(args.output, curves)
    lines = ["# Observed WC-ACC on ten-task training blocks", "",
             "Final WC-ACC (%): mean (sample SD) across seeds 42/43/44. Higher is better.", "",
             "| Method | BFCL CP50 | AppWorld CP50 | Terminal-Bench CP30 | Avg. |", "|---|---:|---:|---:|---:|"]
    for variant, label in LABELS.items():
        lines.append("| " + label + " | " + " | ".join(next(r["display_percent"] for r in summaries if
                     (r["variant"], r["benchmark"]) == (variant, benchmark)) for benchmark in (*BENCHMARKS, "avg"))
                     + " |")
    lines += ["", "Operational task i is the i-th block of ten training task IDs in the actual per-seed training "
              "order. R(k,i) is the mean official outcome on that block at CP k. At stage t, the current block uses "
              "R(10t,t); each past block i<t uses min R(k,i) over observed k=10(i+1),…,10t. The block's own "
              "end-of-learning checkpoint 10i is excluded, as the paper's minimum is strictly after learning i.",
              "", "WC-ACC_t = [R(10t,t) + sum_{i<t} min_k R(k,i)] / t. Unlike SG, all zero task outcomes are included. "
              "Minimize the block mean, not each individual task's outcome before averaging.",
              "", "For each block and stage, compare the same task IDs across the required checkpoints: use their "
              "intersection of officially measured outcomes, without filtering by success. Missing tasks are not zero. "
              "A wholly unmeasured block makes that WC-ACC undefined. CSV retains excluded IDs, every block score, "
              "the CP attaining the minimum, matched current accuracy and strict full-task coverage alternatives.",
              "", "Final report uses the last stage (CP50 for BFCL/AppWorld, CP30 for Terminal), rather than averaging "
              "over stages. Avg. is the within-seed macro-average across benchmarks, followed by mean/sample SD across "
              "seeds. WC-ACC is checked against the same-cohort current block-macro accuracy at every stage.",
              "", "This is observed TRAIN-block WC-ACC, separate from held-out test accuracy. It is an adaptation "
              "using arbitrary ten-instance learning blocks, not a claim that these blocks are stationary task "
              "distributions. Ten-task checkpoint sampling cannot reveal intra-block transient dips. Temperature0.7 "
              "rollout variation and exact-context evaluation reuse remain part of the recorded protocol. "
              "No new inference.",
              "", "Paper: https://arxiv.org/html/2205.13452v2#S3.SS3 ; min-ACC definition: §3.1, Eq.2; WC-ACC: Eq.6."]
    (args.output / "wc_acc_report.md").write_text("\n".join(lines) + "\n")
    manifest = {"created_at": datetime.now(UTC).isoformat(), "queue": str(args.queue.resolve()),
                "protocol": "observed_train_block_wc_acc_cp10", "block_size": 10,
                "minimum_range": "Strictly after learning block i through current CP; current block uses current CP",
                "missing_policy": "Per-block common measured task intersection across compared CPs; no zero imputation",
                "zero_scores": "Included", "primary_summary": "Final stage per seed, then mean/sample SD",
                "stage_rows": len(stages), "block_rows": len(blocks), "mean_curve_points": len(curves),
                "training_order_hashes": hashes, "source_fingerprint_sha256": fingerprint,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "reference": "https://arxiv.org/html/2205.13452v2#S3.SS3", "new_inference": False,
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()
                          if p.suffix in {".csv", ".png", ".pdf", ".md"}}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    for row in summaries:
        print(row["variant"], row["benchmark"], row["display_percent"], f"seeds={row['seeds']}",
              f"matched_tasks={row['common_tasks']}/{row['expected_tasks']}")


if __name__ == "__main__":
    main()
