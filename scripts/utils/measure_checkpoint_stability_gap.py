# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Estimate the user-defined stability gap on past training tasks at observed CP10 boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils.plot_checkpoint_learning_curves import (  # noqa: E402
    BENCHMARKS,
    COLORS,
    LABELS,
    read_scores,
    write_csv,
)


def gap(reference, observed):
    if reference is None:
        return None, "missing_reference"
    if reference == 0:
        return None, "zero_reference"
    if not observed or any(value is None for value in observed):
        return None, "missing_window"
    return (reference - min(observed)) / reference, "measured"


def task_order(spec, hashes):
    folder = Path(spec["train_dir"])
    method = spec["method"]
    path = (folder / "experiment_config.json" if method == "ace" else folder / "state.json"
            if method == "premise" else Path(spec["output"]) / "config.json")
    raw = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    if method in {"ace", "premise"}:
        sequence = data["task_order"] if method == "ace" else data["config"]["order"]
        if any(benchmark != spec["benchmark"] for benchmark, _ in sequence):
            raise ValueError("Expected isolated training order")
        order = [str(task) for _, task in sequence]
    else:
        order = [str(item["task_id"]) for item in data["train"]]
    if len(set(order)) != len(order) or set(order) != set(spec["splits"]["train"]):
        raise ValueError("Training order differs from frozen evaluation split")
    return order


def collect(queue, config):
    _, fingerprint = read_scores(queue, config)
    variants = {row["variant"] for row in config["rows"]}
    sources = {source["batch"]: source for source in config["sources"]}
    pairs, windows, hashes, cache = [], [], {}, {}
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
            checkpoints = db.execute("SELECT * FROM checkpoints WHERE stream=? ORDER BY point",
                                     (spec["id"],)).fetchall()
            identity = {"variant": spec["run_variant"], "benchmark": spec["benchmark"], "seed": spec["seed"]}

            def outcomes(checkpoint):
                canonical = checkpoint["canonical"]
                if canonical not in cache:
                    cache[canonical] = {r["task_id"]: dict(r) for r in db.execute("""SELECT task_id,status,
                        json_extract(result,'$.score') AS score FROM tasks WHERE cp=? AND split='train'""",
                        (canonical,))}
                return cache[canonical]

            for before, after in pairwise(checkpoints):
                if after["point"] - before["point"] != 10:
                    raise ValueError("This analysis requires consecutive ten-task blocks")
                start, stop = before["point"], after["point"]
                left, right = outcomes(before), outcomes(after)
                items = []
                for position, task in enumerate(order[:start], 1):
                    a, b = left[task], right[task]
                    ref = a["score"] if a["status"] == "completed" else None
                    score = b["score"] if b["status"] == "completed" else None
                    value, reason = gap(ref, [score])
                    item = {**identity, "start_cp": start, "end_cp": stop, "task_position": position,
                            "task_id": task, "reference": ref, "observed_min": score,
                            "sg": value, "reason": reason, "before_status": a["status"],
                            "after_status": b["status"], "before_canonical": before["canonical"],
                            "after_canonical": after["canonical"],
                            "same_policy": before["sha"] == after["sha"]}
                    pairs.append(item)
                    items.append(item)
                counts = Counter(item["reason"] for item in items)
                values = [item["sg"] for item in items if item["sg"] is not None]
                windows.append({**identity, "start_cp": start, "end_cp": stop, "past_tasks": len(items),
                                "valid_pairs": len(values), "zero_reference": counts["zero_reference"],
                                "missing_reference": counts["missing_reference"],
                                "missing_window": counts["missing_window"],
                                "sg": statistics.mean(values) if values else None,
                                "reason": "measured" if values else "no_past_tasks" if not items else
                                          "no_defined_task_gaps",
                                "same_policy": before["sha"] == after["sha"]})
    return pairs, windows, hashes, fingerprint


def aggregate(windows):
    groups = defaultdict(list)
    for row in windows:
        groups[row["variant"], row["benchmark"], row["seed"]].append(row)
    seeds = []
    for (variant, benchmark, seed), rows in sorted(groups.items()):
        values = [r["sg"] for r in rows if r["sg"] is not None]
        seeds.append({"variant": variant, "benchmark": benchmark, "seed": seed,
                      "sg": statistics.mean(values) if values else None,
                      "maximum_window_sg": max(values) if values else None, "defined_windows": len(values),
                      "past_task_windows": sum(r["past_tasks"] > 0 for r in rows),
                      "valid_pairs": sum(r["valid_pairs"] for r in rows),
                      "zero_reference": sum(r["zero_reference"] for r in rows),
                      "missing_reference": sum(r["missing_reference"] for r in rows),
                      "missing_window": sum(r["missing_window"] for r in rows)})
    for variant in LABELS:
        for seed in sorted({r["seed"] for r in seeds}):
            rows = [r for r in seeds if r["variant"] == variant and r["seed"] == seed
                    and r["benchmark"] in BENCHMARKS]
            if len(rows) == len(BENCHMARKS) and all(r["sg"] is not None for r in rows):
                seeds.append({"variant": variant, "benchmark": "avg", "seed": seed,
                              "sg": statistics.mean(r["sg"] for r in rows),
                              "maximum_window_sg": max(r["maximum_window_sg"] for r in rows),
                              **{field: sum(r[field] for r in rows) for field in
                                 ("defined_windows", "past_task_windows", "valid_pairs", "zero_reference",
                                  "missing_reference", "missing_window")}})
    summaries = []
    for variant in LABELS:
        for benchmark in (*BENCHMARKS, "avg"):
            rows = [r for r in seeds if (r["variant"], r["benchmark"]) == (variant, benchmark)
                    and r["sg"] is not None]
            values = [r["sg"] for r in rows]
            mean = statistics.mean(values) if values else None
            sd = statistics.stdev(values) if len(values) > 1 else None
            summaries.append({"variant": variant, "benchmark": benchmark, "mean_sg": mean,
                              "sample_sd": sd, "seeds": len(values),
                              "mean_maximum_window_sg": statistics.mean(r["maximum_window_sg"] for r in rows)
                              if rows else None,
                              "display_percent": "—" if mean is None else f"{100 * mean:.2f}" +
                              (f" ({100 * sd:.2f})" if sd is not None else ""),
                              **{field: sum(r[field] for r in rows) for field in
                                 ("defined_windows", "past_task_windows", "valid_pairs", "zero_reference",
                                  "missing_reference", "missing_window")}})
    return seeds, summaries


def plot(output, seeds, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.7))
    labels = ["ACE\nAUD", "GEPA\nVal", "GEPA\nTrain-as-val", "SkillOpt\nVal", "SkillOpt\nTrain-as-val",
              "PEACE\nAUD\nw/o alignment"]
    for ax, (benchmark, title) in zip(axes, BENCHMARKS.items(), strict=True):
        rows = [next(r for r in summaries if (r["variant"], r["benchmark"]) == (variant, benchmark))
                for variant in LABELS]
        means = [100 * r["mean_sg"] if r["mean_sg"] is not None else np.nan for r in rows]
        sd = [100 * (r["sample_sd"] or 0) for r in rows]
        bars = ax.bar(range(6), means, yerr=sd, color=COLORS, alpha=.85, width=.66,
                      capsize=4, error_kw={"linewidth": 1})
        for index, (bar, row) in enumerate(zip(bars, rows, strict=True)):
            if "train_as_val" in row["variant"]:
                bar.set_hatch("//")
            points = [100 * r["sg"] for r in seeds if
                      (r["variant"], r["benchmark"]) == (row["variant"], benchmark) and r["sg"] is not None]
            ax.scatter(np.linspace(index - .13, index + .13, len(points)), points,
                       s=21, color="#25364a", edgecolors="white", linewidths=.5, zorder=4)
            if row["mean_sg"] is not None:
                ax.annotate(f"{means[index]:.1f}", (index, max([means[index] + sd[index], *points])),
                            xytext=(0, 6), textcoords="offset points", ha="center", fontsize=10)
        seed_values = [100 * r["sg"] for r in seeds if r["benchmark"] == benchmark and r["sg"] is not None]
        upper = [m + s for m, s in zip(means, sd, strict=True) if math.isfinite(m)] + seed_values
        lower = [m - s for m, s in zip(means, sd, strict=True) if math.isfinite(m)] + seed_values
        maximum, minimum = max(upper, default=0), min(lower, default=0)
        ax.set(title=title, ylabel="Observed SG (%) ↓", xticks=range(6), xticklabels=labels,
               ylim=(min(-1, 5 * math.floor(minimum / 5)), max(10, math.ceil((maximum + 6) / 5) * 5)))
        ax.axhline(0, color="#73808d", linewidth=.6)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)
    fig.suptitle("Observed stability gap · 10-task blocks · seed mean ± sample SD (42, 43, 44)", fontsize=16, y=.98)
    notes = [
        "Past train tasks only. For each CP a→b, use tasks learned before a; "
        "observed K contains b only. CP0→10 has no past tasks.",
        "Apply (R_before - R_end) / R_before per task; undefined zero-reference and missing outcomes "
        "are excluded. No clipping of negative gaps.",
        "Average tasks within block, then blocks equally within seed, then seeds equally. "
        "Dots = seeds. Smaller values indicate less observed loss.",
        "This endpoint-only proxy misses within-block temporary dips. CP aliases reuse evaluations; "
        "rollout noise can also affect measured losses.",
    ]
    for index, text in enumerate(notes):
        fig.text(.02, .13 - .032 * index, text, fontsize=9, color="#526271")
    fig.tight_layout(rect=(0, .19, 1, .92))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"stability_gap_cp10.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--table-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.table_manifest.read_text())
    if args.queue.name != config["offline_early"]["checkpoint_queue_id"]:
        raise ValueError("Expected the configured main-table CP cohort")
    pairs, windows, hashes, fingerprint = collect(args.queue, config)
    seeds, summaries = aggregate(windows)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("task_pairs", pairs), ("windows", windows), ("seeds", seeds), ("summary", summaries)):
        write_csv(args.output / f"stability_gap_{name}.csv", rows)
    plot(args.output, seeds, summaries)
    lines = ["# Observed stability gap at ten-task boundaries", "",
             "Values (%): equal-block mean within seed, then mean (sample SD) across seeds 42/43/44. Lower is better.",
             "", "| Method | BFCL | AppWorld | Terminal-Bench | Avg. |", "|---|---:|---:|---:|---:|"]
    for variant, label in LABELS.items():
        lines.append("| " + label + " | " + " | ".join(next(r["display_percent"] for r in summaries if
                     (r["variant"], r["benchmark"]) == (variant, benchmark)) for benchmark in (*BENCHMARKS, "avg"))
                     + " |")
    lines += ["", "Each ten-task block is a training stage t. Old tasks i are individual training task IDs "
              "already seen at its start. The observed checkpoint set contains only the block endpoint; "
              "an unobserved intra-block "
              "minimum cannot be reconstructed. This is an endpoint-only adaptation of the user formula, not the "
              "paper's per-iteration evaluation or its named min-ACC/WC-ACC metrics.",
              "", "The current official task outcomes are binary: eligible tasks were successful at the reference CP; "
              "SG is 1 when they fail at the next CP, otherwise 0. Previously unsuccessful tasks have zero "
              "denominators and are undefined, rather than zero-gap observations. Reference or endpoint "
              "infrastructure failures are missing, not measured failures. "
              "Training-order files are hashed and checked against split membership.",
              "", "Avg. is a within-seed macro-average over the three benchmarks. Blocks are weighted equally, "
              "not by the growing number of past tasks. Blocks without defined task gaps are excluded with coverage "
              "reported in CSV. The first block has no old tasks. Raw pair gaps and window maxima are retained.",
              "", "Evaluation temperature is 0.7; no new inference was run. Context aliases reuse the original "
              "outcomes. Observed drops can include rollout variability and endpoint-only sampling "
              "can miss transient drops.",
              "", "Reference: https://arxiv.org/abs/2205.13452 (De Lange et al., ICLR 2023)."]
    (args.output / "stability_gap_report.md").write_text("\n".join(lines) + "\n")
    manifest = {"created_at": datetime.now(UTC).isoformat(), "queue": str(args.queue.resolve()),
                "protocol": "cp10_endpoint_observed_stability_gap_on_past_train_task_ids",
                "formula": "(R_start_i - min(R_end_i)) / R_start_i", "window": "10 training task IDs; endpoint only",
                "first_block": "Excluded: no previously trained tasks", "reference_zero": "undefined; excluded",
                "negative_gaps": "Retained without clipping", "missing": "Excluded; never imputed with zero",
                "aggregation": "Equal tasks within block; equal defined blocks within seed; equal seeds; sample SD",
                "source_fingerprint_sha256": fingerprint, "training_order_hashes": hashes,
                "pair_reasons": dict(Counter(r["reason"] for r in pairs)),
                "window_reasons": dict(Counter(r["reason"] for r in windows)),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "environment_execution": False, "new_inference": False,
                "reference": "https://arxiv.org/abs/2205.13452",
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()
                          if p.suffix in {".png", ".pdf", ".csv", ".md"}}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("pair_reasons", "window_reasons")}, ensure_ascii=False))
    for row in summaries:
        print(row["variant"], row["benchmark"], row["display_percent"], f"seeds={row['seeds']}",
              f"windows={row['defined_windows']}/{row['past_task_windows']}")


if __name__ == "__main__":
    main()
