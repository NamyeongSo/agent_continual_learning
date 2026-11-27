# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Plot seed-averaged frozen-checkpoint scores and descriptive checkpoint correlations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

BENCHMARKS = {"bfcl": "BFCL", "appworld": "AppWorld", "terminalbench2": "Terminal-Bench"}
SPLITS = {"train": "Train", "val": "Val", "test": "Test"}
LABELS = {"ace_aud": "ACE AUD", "gepa_val": "GEPA Val", "gepa_train_as_val": "GEPA Train-as-val",
          "skillopt_val": "SkillOpt Val", "skillopt_train_as_val": "SkillOpt Train-as-val",
          "peace_no_alignment_aud": "PEACE AUD w/o alignment"}
COLORS = ["#7753A6", "#D58A21", "#D58A21", "#2B9472", "#2B9472", "#2877B6"]


def read_scores(queue, config):
    sources = {source["batch"]: source for source in config["sources"]}
    variants = {row["variant"]: row["method"] for row in config["rows"]}
    seeds = config["seeds"]
    points = config["offline_early"]["points"]
    samples, seen, cache = [], set(), {}
    fingerprint = hashlib.sha256()
    with sqlite3.connect((queue / "queue.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        invalid = {row[0] for row in db.execute("SELECT stream FROM stream_invalidations")}
        for row in db.execute("SELECT spec FROM streams ORDER BY id"):
            spec = json.loads(row[0])
            variant, benchmark, seed = spec["run_variant"], spec["benchmark"], spec["seed"]
            source = sources.get(spec["variant"])
            if (not source or variant not in variants or benchmark not in source.get("benchmarks", BENCHMARKS)
                    or seed not in seeds):
                continue
            key = (variant, benchmark, seed)
            if key in seen or spec["id"] in invalid:
                raise ValueError("Ambiguous or invalidated checkpoint stream")
            seen.add(key)
            if (spec["method"] != variants[variant] or spec["split_sha256"] != config["split_manifest_sha256"]
                    or spec["model"].removeprefix("openai/") != config["model"].removeprefix("openai/")
                    or spec["temperature"] != config["temperature"]):
                raise ValueError("Checkpoint cohort differs from the main table")
            if benchmark == "appworld" and spec["appworld_policy_sha256"] != config["appworld_policy_sha256"]:
                raise ValueError("AppWorld environment policies differ")
            for name in ("input_tokens", "output_tokens", "max_turns"):
                if spec["actor_limits"][name] != config["actor_limits"][benchmark][name]:
                    raise ValueError("Actor budgets differ")
            checkpoints = db.execute("SELECT * FROM checkpoints WHERE stream=? ORDER BY point",
                                     (spec["id"],)).fetchall()
            if [cp["point"] for cp in checkpoints] != points[benchmark]:
                raise ValueError("Missing or unexpected checkpoint points")
            fingerprint.update(json.dumps(spec, sort_keys=True).encode())
            for cp in checkpoints:
                fingerprint.update(json.dumps(dict(cp), sort_keys=True).encode())
                for split in SPLITS:
                    cache_key = (cp["canonical"], split)
                    if cache_key not in cache:
                        tasks = db.execute("""SELECT task_id,status,json_extract(result,'$.score') AS score,
                            json_extract(result,'$.score_metric') AS metric FROM tasks
                            WHERE cp=? AND split=? ORDER BY task_id""", cache_key).fetchall()
                        cache[cache_key] = [dict(task) for task in tasks]
                        fingerprint.update(json.dumps(cache[cache_key], sort_keys=True).encode())
                    tasks = cache[cache_key]
                    if {task["task_id"] for task in tasks} != set(spec["splits"][split]):
                        raise ValueError("Canonical checkpoint task membership differs")
                    measured = [task for task in tasks if task["status"] == "completed"]
                    values = [task["score"] for task in measured]
                    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                        raise ValueError("Invalid official task score")
                    if benchmark == "appworld" and any(task["metric"] != "appworld_tgc_v1" for task in measured):
                        raise ValueError("AppWorld scores must be official TGC")
                    samples.append({"variant": variant, "benchmark": benchmark, "seed": seed,
                                    "checkpoint": cp["point"], "split": split,
                                    "score": statistics.mean(values) if values else None,
                                    "measured": len(values), "total": len(tasks),
                                    "errors": sum(task["status"] == "error" for task in tasks),
                                    "canonical": cp["canonical"], "checkpoint_id": cp["id"]})
    expected = {(variant, benchmark, seed) for variant in variants for benchmark in BENCHMARKS for seed in seeds}
    if seen != expected:
        raise ValueError("Expected all method/benchmark/seed streams from the main table")
    return samples, fingerprint.hexdigest()


def aggregate(samples, seeds):
    groups = defaultdict(list)
    for row in samples:
        groups[row["variant"], row["benchmark"], row["checkpoint"], row["split"]].append(row)
    means = []
    for (variant, benchmark, checkpoint, split), rows in sorted(groups.items()):
        if {row["seed"] for row in rows} != set(seeds) or len(rows) != len(seeds):
            raise ValueError("Missing or duplicate seed rows")
        values = [row["score"] for row in rows if row["score"] is not None]
        means.append({"variant": variant, "benchmark": benchmark, "checkpoint": checkpoint, "split": split,
                      "mean": statistics.mean(values) if values else None,
                      "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                      "seeds": len(values), "expected_seeds": len(seeds),
                      "measured": sum(row["measured"] for row in rows), "total": sum(row["total"] for row in rows),
                      "partial": any(row["measured"] != row["total"] for row in rows)})
    return means


def ranks(values):
    ordered = sorted(values)
    return [statistics.mean(i + 1 for i, v in enumerate(ordered) if v == value) for value in values]


def correlations(means):
    groups = defaultdict(list)
    for row in means:
        groups[row["variant"], row["benchmark"], row["split"]].append(row)
    result = []
    for (variant, benchmark, split), rows in sorted(groups.items()):
        rows = sorted((row for row in rows if row["mean"] is not None), key=lambda r: r["checkpoint"])
        x, y = [row["checkpoint"] for row in rows], [row["mean"] for row in rows]
        reason = "too_few_checkpoints" if len(rows) < 3 else "constant_score" if max(y) - min(y) < 1e-12 else "ok"
        result.append({"variant": variant, "benchmark": benchmark, "split": split, "checkpoints": len(rows),
                       "pearson_r": statistics.correlation(x, y) if reason == "ok" else None,
                       "spearman_rho": statistics.correlation(ranks(x), ranks(y)) if reason == "ok" else None,
                       "reason": reason, "minimum_seeds": min((row["seeds"] for row in rows), default=0),
                       "partial_points": sum(row["partial"] for row in rows)})
    return result


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def panel_limits(rows):
    rows = [row for row in rows if row["mean"] is not None]
    if not rows:
        return 0, 100
    lower = min(max(0, 100 * (row["mean"] - (row["sample_sd"] or 0))) for row in rows)
    upper = max(min(100, 100 * (row["mean"] + (row["sample_sd"] or 0))) for row in rows)
    padding = max(1, .08 * max(upper - lower, 4))
    return max(0, 2 * math.floor((lower - padding) / 2)), min(100, 2 * math.ceil((upper + padding) / 2))


def plot(output, means, correlations_, seeds, *, include_correlations=True):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 3, figsize=(17, 12), sharex="col")
    legend = []
    limits = {}
    for column, (benchmark, title) in enumerate(BENCHMARKS.items()):
        for row_index, (split, split_title) in enumerate(SPLITS.items()):
            ax = axes[row_index, column]
            limits[f"{benchmark}.{split}"] = panel_limits(
                [r for r in means if (r["benchmark"], r["split"]) == (benchmark, split)])
            for (variant, label), color in zip(LABELS.items(), COLORS, strict=True):
                rows = sorted((r for r in means if (r["variant"], r["benchmark"], r["split"]) ==
                               (variant, benchmark, split)), key=lambda r: r["checkpoint"])
                x = np.array([r["checkpoint"] for r in rows])
                y = np.array([100 * r["mean"] if r["mean"] is not None else np.nan for r in rows])
                sd = np.array([100 * r["sample_sd"] if r["sample_sd"] is not None else np.nan for r in rows])
                line, = ax.plot(x, y, label=label, color=color, marker="o", markersize=4, linewidth=1.9,
                                linestyle="--" if "train_as_val" in variant else "-")
                ax.fill_between(x, np.clip(y - sd, 0, 100), np.clip(y + sd, 0, 100), color=color, alpha=.085)
                partial = np.array([r["partial"] for r in rows])
                ax.scatter(x[partial], y[partial], s=95, facecolors="none", edgecolors=color, linewidths=1.1, zorder=5)
                if column == row_index == 0:
                    legend.append(line)
            ax.set(title=f"{title} · {split_title}", ylim=limits[f"{benchmark}.{split}"], ylabel="Score (%)",
                   xticks=sorted({r["checkpoint"] for r in means if r["benchmark"] == benchmark}))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
            ax.grid(alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
            if row_index == 2:
                ax.set_xlabel("Checkpoint (training task IDs consumed)")
    fig.suptitle(f"Checkpoint learning curves · seed mean ± sample SD ({', '.join(map(str, seeds))})",
                 fontsize=17, y=.98)
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .075), ncol=3, frameon=False, fontsize=11)
    fig.text(.02, .061, "Y-axis ranges are fitted separately for each panel and include mean ± SD; "
             "numeric scales are shown on every panel.", fontsize=9, color="#566573")
    fig.text(.02, .042, "Train = frozen-checkpoint evaluation on the train split. "
             "AppWorld = Original policy + code generation, official TGC.", fontsize=9, color="#566573")
    fig.text(.02, .023, "Ringed points contain missing task outcomes: each seed uses measured tasks only, "
             "then seeds receive equal weight. Missing outcomes are never zero.", fontsize=9, color="#566573")
    fig.text(.02, .004, "CP0 is shared within each baseline across seeds/variations; unchanged contexts reuse "
             "evaluation results. Shading is descriptive across-seed SD.", fontsize=9, color="#566573")
    fig.tight_layout(rect=(0, .145, 1, .95))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"checkpoint_learning_curves_seed_mean.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    if not include_correlations:
        return limits

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    lookup = {(r["variant"], r["benchmark"], r["split"]): r for r in correlations_}
    columns = [(benchmark, split) for benchmark in BENCHMARKS for split in SPLITS]
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#e9edf1")
    for ax, (metric, title) in zip(axes, (("pearson_r", "Pearson r"), ("spearman_rho", "Spearman rho")), strict=True):
        values = np.array([[lookup[variant, benchmark, split][metric] for benchmark, split in columns]
                           for variant in LABELS], dtype=float)
        image = ax.imshow(values, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(columns)), [BENCHMARKS[b] + "\n" + SPLITS[s] for b, s in columns], fontsize=9)
        ax.set_yticks(range(len(LABELS)), LABELS.values(), fontsize=10)
        ax.set_title(title + " · checkpoint vs. seed-mean split score", fontsize=14, pad=13)
        for i, variant in enumerate(LABELS):
            for j, (benchmark, split) in enumerate(columns):
                entry = lookup[variant, benchmark, split]
                value = entry[metric]
                text = "—" if value is None else f"{value:+.2f}" + ("*" if entry["partial_points"] else "")
                ax.text(j, i, text, ha="center", va="center", fontsize=10,
                        color="white" if value is not None and abs(value) > .65 else "#213547")
        for boundary in (2.5, 5.5):
            ax.axvline(boundary, color="white", linewidth=2.5)
        ax.spines[:].set_visible(False)
    fig.subplots_adjust(left=.20, right=.92, top=.93, bottom=.12, hspace=.46)
    bar = fig.colorbar(image, ax=axes, fraction=.023, pad=.02)
    bar.set_label("Correlation coefficient")
    fig.text(.02, .065, "Each coefficient uses the plotted CP-level seed means: 6 checkpoints for BFCL/AppWorld, "
             "4 for Terminal-Bench, including CP0.", fontsize=10)
    fig.text(.02, .04, "* At least one CP has missing task outcomes. — Constant score or fewer than 3 points; "
             "correlation is undefined. Exact rank ties use average ranks.", fontsize=10)
    fig.text(.02, .015, "Descriptive association only: checkpoints are sequentially related "
             "and CP0 / unchanged policies can share evaluation data.", fontsize=10)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"checkpoint_split_correlations.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--table-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.table_manifest.read_text())
    if args.queue.name != config["offline_early"]["checkpoint_queue_id"]:
        raise ValueError("Checkpoint queue differs from configured main table")
    samples, fingerprint = read_scores(args.queue, config)
    means = aggregate(samples, config["seeds"])
    coefficients = correlations(means)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "checkpoint_seed_scores.csv", samples)
    write_csv(args.output / "checkpoint_seed_mean_scores.csv", means)
    write_csv(args.output / "checkpoint_split_correlations.csv", coefficients)
    limits = plot(args.output, means, coefficients, config["seeds"])
    manifest = {"created_at": datetime.now(UTC).isoformat(), "queue": str(args.queue.resolve()),
                "table_manifest": str(args.table_manifest.resolve()), "seeds": config["seeds"],
                "streams": len({r["checkpoint_id"].split("_cp")[0] for r in samples}),
                "checkpoint_seed_split_rows": len(samples), "seed_mean_points": len(means),
                "panel_y_limits_percent": limits,
                "correlations": len(coefficients),
                "undefined_correlations": [r for r in coefficients if r["reason"] != "ok"],
                "source_score_fingerprint_sha256": fingerprint,
                "table_manifest_sha256": hashlib.sha256(args.table_manifest.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "aggregation": "Mean official scores within seed (missing tasks excluded); "
                               "equal mean across seeds; sample SD.",
                "correlation": "Pearson r and Spearman rho on CP-level seed means; "
                               "CP0 included; not pooled individual tasks.",
                "main_table_difference": "Curves retain measured task outcomes in incomplete splits, "
                                         "marked with rings; Offline (early) requires complete selected-test seeds.",
                "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()
                          if p.suffix in {".png", ".pdf", ".csv"}}}
    (args.output / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "streams", "seed_mean_points", "correlations", "undefined_correlations")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
