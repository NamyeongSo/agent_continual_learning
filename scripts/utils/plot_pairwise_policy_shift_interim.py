# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Create a read-only interim ΔCP policy-shift report from completed responses."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils.policy_shift_metrics import score_actions, write_json  # noqa: E402

METHODS = ("ace", "ace_add", "gepa", "skillopt", "premise", "premise_add")
DISPLAY = {
    "ace": "ACE",
    "ace_add": "ACE ADD",
    "gepa": "GEPA",
    "skillopt": "SkillOpt",
    "premise": "PEACE",
    "premise_add": "PEACE ADD",
}
METRICS = ("loose_accuracy", "strict_accuracy", "loose_f1", "strict_f1", "accuracy")
COLORS = {
    "ace": "#7753A6",
    "ace_add": "#7753A6",
    "gepa": "#D58A21",
    "skillopt": "#2B9472",
    "premise": "#2877B6",
    "premise_add": "#2877B6",
}


def now():
    return datetime.now(UTC).astimezone(timezone(timedelta(hours=9))).isoformat()


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_partial(db_path: Path, *, require_all_checkpoints=False):
    db = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("BEGIN")  # Hold one consistent SQLite WAL snapshot while the live runner continues.
    job_counts = dict(db.execute("SELECT status,count(*) FROM jobs GROUP BY status").fetchall())
    errors = [
        {"error": row[0], "count": row[1]}
        for row in db.execute(
            "SELECT error,count(*) FROM jobs WHERE status='error' GROUP BY error ORDER BY count(*) DESC"
        )
    ]
    streams = db.execute(
        "SELECT DISTINCT stream,benchmark,method,seed FROM policies ORDER BY benchmark,method,seed"
    ).fetchall()
    pair_totals = defaultdict(lambda: [0, defaultdict(float)])
    coverage = []

    for stream in streams:
        points = [
            row[0]
            for row in db.execute("SELECT point FROM policies WHERE stream=? ORDER BY point", (stream["stream"],))
        ]
        rows = db.execute(
            """SELECT b.anchor,a.task,b.point,
                   CASE WHEN b.point=0 THEN r.status ELSE j.status END AS status,
                   CASE WHEN b.point=0 THEN r.action ELSE j.action END AS action
               FROM bindings b JOIN anchors a ON a.id=b.anchor
               JOIN jobs j ON j.id=b.job JOIN jobs r ON r.id=b.reference
               WHERE b.stream=? ORDER BY b.anchor,b.point""",
            (stream["stream"],),
        )
        history_count = 0
        histories_with_two = 0
        observed_pairs = 0
        completed_points = 0
        task_ids = set()
        common_histories = 0
        compared_tasks = set()
        for _, anchor_rows in itertools.groupby(rows, key=lambda row: row["anchor"]):
            anchor_rows = list(anchor_rows)
            history_count += 1
            task = anchor_rows[0]["task"]
            task_ids.add(task)
            actions = {
                row["point"]: json.loads(row["action"])
                for row in anchor_rows
                if row["status"] == "completed" and row["action"] is not None
            }
            completed_points += len(actions)
            if set(actions) == set(points):
                common_histories += 1
            elif require_all_checkpoints:
                continue
            if len(actions) >= 2:
                histories_with_two += 1
                compared_tasks.add(task)
            for left, right in itertools.combinations(sorted(actions), 2):
                metrics = score_actions(actions[left], actions[right], stream["benchmark"])
                count, totals = pair_totals[
                    (stream["stream"], stream["benchmark"], stream["method"], stream["seed"], left, right, task)
                ]
                count += 1
                for metric, value in metrics.items():
                    totals[metric] += value
                pair_totals[
                    (stream["stream"], stream["benchmark"], stream["method"], stream["seed"], left, right, task)
                ] = [count, totals]
                observed_pairs += 1
        coverage.append(
            {
                "stream": stream["stream"],
                "benchmark": stream["benchmark"],
                "method": stream["method"],
                "seed": stream["seed"],
                "checkpoints": points,
                "histories": history_count,
                "tasks": len(task_ids),
                "completed_policy_actions": completed_points,
                "histories_with_two_or_more_actions": histories_with_two,
                "observed_history_pairs": observed_pairs,
                "all_checkpoint_common_histories": common_histories,
                "compared_tasks": len(compared_tasks),
            }
        )

    task_rows = []
    per_pair = defaultdict(lambda: defaultdict(list))
    for key, (count, totals) in sorted(pair_totals.items()):
        stream, benchmark, method, seed, left, right, task = key
        for metric, total in totals.items():
            agreement = total / count
            task_rows.append(
                {
                    "stream": stream,
                    "benchmark": benchmark,
                    "method": method,
                    "seed": seed,
                    "left": left,
                    "right": right,
                    "checkpoint_distance": right - left,
                    "task_id": task,
                    "metric": metric,
                    "agreement": agreement,
                    "histories": count,
                }
            )
            per_pair[(stream, benchmark, method, seed, left, right)][metric].append(agreement)

    pair_rows = []
    distance_groups = defaultdict(list)
    for key, metrics in sorted(per_pair.items()):
        stream, benchmark, method, seed, left, right = key
        pair_row = {
            "stream": stream,
            "benchmark": benchmark,
            "method": method,
            "seed": seed,
            "left": left,
            "right": right,
            "checkpoint_distance": right - left,
        }
        for metric, values in metrics.items():
            pair_row[metric] = statistics.mean(values)
            distance_groups[(benchmark, method, right - left, metric)].append((seed, 1 - pair_row[metric]))
        pair_row["tasks"] = max((len(values) for values in metrics.values()), default=0)
        pair_rows.append(pair_row)

    distance_rows = []
    for (benchmark, method, distance, metric), values in sorted(distance_groups.items()):
        shifts = [item[1] for item in values]
        distance_rows.append(
            {
                "benchmark": benchmark,
                "method": method,
                "checkpoint_distance": distance,
                "metric": metric,
                "mean_shift": statistics.mean(shifts),
                "pair_sample_sd": statistics.stdev(shifts) if len(shifts) > 1 else 0,
                "checkpoint_pairs": len(values),
                "seeds": len({item[0] for item in values}),
            }
        )

    total_effective = sum(count for status, count in job_counts.items() if status != "deduplicated")
    completed = job_counts.get("completed", 0)
    summary = {
        "generated_at": now(),
        "phase": "interim",
        "completed_requests": completed,
        "effective_requests": total_effective,
        "completion_fraction": completed / total_effective if total_effective else 0,
        "job_counts": job_counts,
        "measurement_errors_excluded": errors,
        "partial_pair_policy": (
            "Require completed primary actions at every checkpoint within each stream; "
            "use the same common histories for every CP pair. Missing/error actions exclude that history."
            if require_all_checkpoints else
            "Compare only completed actions on the same fixed history; "
            "missing/error actions are omitted, not scored as disagreement."),
        "common_history_scope": "all_checkpoints_within_stream" if require_all_checkpoints else "available_cp_pair",
        "aggregation": "Mean histories within task, equal task weight within a checkpoint pair; "
        "aggregate available checkpoint-pair/seed rows by ΔCP.",
        "pairwise_points": len(pair_rows),
        "task_pair_points": len(task_rows),
        "scores": distance_rows,
        "coverage": coverage,
    }
    db.close()
    return summary, pair_rows, task_rows


def curve_correlations(rows):
    """Descriptive Pearson r across the equally weighted points shown in each curve."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["benchmark"], row["method"], row["metric"])].append(row)
    result = []
    for (benchmark, method, metric), group in sorted(grouped.items()):
        group.sort(key=lambda row: row["checkpoint_distance"])
        x = [float(row["checkpoint_distance"]) for row in group]
        y = [float(row["mean_shift"]) for row in group]
        reason = ""
        if len(x) < 2:
            value, reason = None, "fewer_than_two_points"
        elif len(set(x)) < 2 or len(set(y)) < 2:
            value, reason = None, "constant_curve_or_distance"
        else:
            value = statistics.correlation(x, y)
            if not math.isfinite(value):
                value, reason = None, "nonfinite_correlation"
        result.append(
            {
                "benchmark": benchmark,
                "method": method,
                "metric": metric,
                "pearson_r": value,
                "distance_points": len(x),
                "min_distance": min(x),
                "max_distance": max(x),
                "undefined_reason": reason,
            }
        )
    return result


def plot(summary: dict, out_dir: Path, file_suffix: str = "delta_cp_interim"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    correlations = curve_correlations(summary["scores"])
    correlation_lookup = {(row["benchmark"], row["method"], row["metric"]): row["pearson_r"] for row in correlations}
    write_csv(out_dir / "correlations.csv", correlations)
    write_json(
        out_dir / "correlation_definition.json",
        {
            "coefficient": "Pearson r",
            "x": "checkpoint distance (ΔCP)",
            "y": "plotted mean policy shift",
            "weighting": "One equally weighted plotted point per available ΔCP; not individual history/pair rows",
            "interpretation": "Descriptive correlation; constant curves are undefined (N/A); no significance test",
            "source_generated_at": summary.get("generated_at"),
            "rendered_at": now(),
        },
    )
    has_add = any(row["method"].endswith("_add") for row in summary["scores"])
    for benchmark in ("bfcl", "appworld", "terminalbench2"):
        metrics = (
            ["accuracy"]
            if benchmark == "terminalbench2"
            else ["loose_accuracy", "strict_accuracy", "loose_f1", "strict_f1"]
        )
        width = 7 if len(metrics) == 1 else 5.1 * len(metrics)
        fig, axes = plt.subplots(1, len(metrics), figsize=(width, 4.8), squeeze=False)
        for ax, metric in zip(axes[0], metrics, strict=True):
            for method in METHODS:
                group = sorted(
                    (
                        row
                        for row in summary["scores"]
                        if (row["benchmark"], row["method"], row["metric"]) == (benchmark, method, metric)
                    ),
                    key=lambda row: row["checkpoint_distance"],
                )
                if not group:
                    continue
                x = [row["checkpoint_distance"] for row in group]
                y = [row["mean_shift"] for row in group]
                sd = [row["pair_sample_sd"] for row in group]
                r = correlation_lookup[(benchmark, method, metric)]
                coefficient = "N/A" if r is None else f"{r:.3f}"
                name = DISPLAY[method]
                if has_add and method in ("ace", "premise"):
                    name += " AUD"
                ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=3,
                    color=COLORS[method],
                    linestyle="--" if method.endswith("_add") else "-",
                    label=f"{name} (r={coefficient})",
                )
                ax.fill_between(
                    x,
                    [max(0, a - b) for a, b in zip(y, sd, strict=True)],
                    [min(1, a + b) for a, b in zip(y, sd, strict=True)],
                    color=COLORS[method],
                    alpha=0.12,
                )
            ax.set(
                xlabel="Checkpoint distance (ΔCP)",
                ylabel="Policy shift",
                ylim=(-0.02, 1.02),
                title="1 - " + metric.replace("_", " "),
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        local_time = summary.get("generated_at", now())[:16].replace("T", " ") + " KST"
        scope = (
            "Completed same-history pairs; coverage varies by ΔCP"
            if summary.get("phase") == "interim"
            else "All-checkpoint common histories"
        )
        fig.suptitle(
            f"{benchmark} · temperature 0 · {local_time}\n{scope} · r = Pearson(ΔCP, mean shift)",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        for ext in ("png", "pdf", "svg"):
            fig.savefig(out_dir / f"{benchmark}_{file_suffix}.{ext}", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Active ΔCP experiment directory")
    parser.add_argument("--out-dir", type=Path, help="Destination for this interim snapshot")
    parser.add_argument("--replot-summary", type=Path, help="Render a saved summary without querying inference data")
    parser.add_argument("--file-suffix", default="delta_cp_interim")
    args = parser.parse_args()
    output = args.output.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir
        else output
        / "analysis/interim"
        / datetime.now(UTC).astimezone(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=bool(args.replot_summary))
    if args.replot_summary:
        summary = json.loads(args.replot_summary.read_text())
        plot(summary, out_dir, args.file_suffix)
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "source": str(args.replot_summary),
                    "correlations": curve_correlations(summary["scores"]),
                },
                ensure_ascii=False,
            )
        )
        return
    summary, pair_rows, task_rows = analyze_partial(output / "policy_shift.sqlite3")
    write_json(out_dir / "summary.json", summary)
    write_csv(out_dir / "pairwise_metrics.csv", pair_rows)
    write_csv(out_dir / "pairwise_task_metrics.csv", task_rows)
    write_json(out_dir / "coverage.json", summary["coverage"])
    plot(summary, out_dir)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "generated_at": summary["generated_at"],
                "completed_requests": summary["completed_requests"],
                "effective_requests": summary["effective_requests"],
                "completion_fraction": summary["completion_fraction"],
                "pairwise_points": summary["pairwise_points"],
                "task_pair_points": summary["task_pair_points"],
                "errors": summary["measurement_errors_excluded"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
