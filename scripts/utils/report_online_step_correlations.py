# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Measure Pearson correlations between benchmark task steps and frozen online mean curves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ("bfcl", "appworld", "terminalbench2")
METHODS = ("ace", "premise", "skillopt")


def curve_correlations(source: Path, setting: str, starts: dict[str, int]) -> tuple[list[dict], list[dict]]:
    """Use recorded task positions, excluding missing means without renumbering later points."""
    with source.open(newline="") as stream:
        curves = list(csv.DictReader(stream))
    grouped = defaultdict(list)
    points = []
    for row in curves:
        if not row["mean"]:
            continue
        x = int(row["iteration"]) - starts[row["benchmark"]] + 1
        y = 100 * float(row["mean"])
        if x < 1 or not math.isfinite(y):
            raise ValueError("Expected positive task positions and finite means")
        grouped[row["benchmark"], row["method"]].append((x, y))
        points.append(
            {
                "setting": setting,
                "benchmark": row["benchmark"],
                "baseline": row["method"],
                "step": x,
                "score_percent": y,
            }
        )
    correlations = []
    for benchmark in BENCHMARKS:
        for method in METHODS:
            samples = sorted(grouped[benchmark, method])
            x, y = zip(*samples, strict=True)
            if len(set(x)) != len(x):
                raise ValueError("Expected one mean curve point per baseline, benchmark and step")
            r = statistics.correlation(x, y) if len(samples) > 1 and len(set(y)) > 1 else None
            correlations.append(
                {
                    "setting": setting,
                    "benchmark": benchmark,
                    "baseline": method,
                    "pearson_r": r,
                    "point_count": len(samples),
                    "first_step": x[0],
                    "last_step": x[-1],
                    "first_score_percent": y[0],
                    "last_score_percent": y[-1],
                    "first_to_last_change_pp": y[-1] - y[0],
                    "source_csv": str(source),
                }
            )
    return correlations, points


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=ROOT / "acr_oct")
    parser.add_argument("--output", type=Path, default=ROOT / "acr_oct/analysis/update_step_correlation")
    args = parser.parse_args()
    folder = args.figures.resolve()
    output = args.output.resolve()
    seq_metadata = folder / "sequential/mean/no_checkpoint_white/metadata.json"
    starts = {benchmark: start for benchmark, start, _end in json.loads(seq_metadata.read_text())["benchmark_segments"]}
    sources = {
        "sequential": folder / "sequential/mean/no_checkpoint_white/aggregate_windows.csv",
        "interleaved": folder / "interleaved/mean/benchmark_datapoints_seq_style/aggregate_cumulative_datapoints.csv",
    }
    figure_files = [
        folder / "seq_latest_no_checkpoint_no_labels.png",
        folder / "interleaved_benchmark_datapoints_seq_style_no_labels.png",
    ]
    provenance = json.loads((folder / "white_background_manifest.json").read_text())
    for figure in figure_files:
        expected = next(row for row in provenance["files"] if row["file"] == figure.name)["sha256"]
        if hashlib.sha256(figure.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Figure hash differs from its source manifest: {figure}")
    results, points = [], []
    for setting, source in sources.items():
        local_starts = starts if setting == "sequential" else dict.fromkeys(BENCHMARKS, 1)
        correlations, curve_points = curve_correlations(source, setting, local_starts)
        results.extend(correlations)
        points.extend(curve_points)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "pearson_correlations.csv", results)
    write_csv(output / "curve_points.csv", points)
    report = [
        "Pearson r: task step versus seed-mean cumulative online score (%)",
        "",
        "| Setting | Benchmark | n | ACE | PREMiSE | SkillOpt |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for setting in sources:
        for benchmark in BENCHMARKS:
            rows = [row for row in results if row["setting"] == setting and row["benchmark"] == benchmark]
            r_values = " | ".join(
                f"{row['pearson_r']:+.4f}" if row["pearson_r"] is not None else "undefined" for row in rows
            )
            report.append(f"| {setting} | {benchmark} | {rows[0]['point_count']} | {r_values} |")
    report.extend(
        [
            "",
            "x is the benchmark-local task position, including missing encounters; "
            "it is not a count of verified successful optimizer updates.",
            "y is the plotted equal-weight seed mean of cumulative online scores. "
            "All available points in each region are included (1-50 or 1-30).",
            "This correlates task position with the performance level, rather than with the per-step change in score.",
            "Sequential N=50 covers each entire benchmark region "
            "and therefore equals benchmark-wise cumulative averaging.",
            "The reported r summarizes the complete curve, rather than only the first-to-last difference. "
            "No p-values or causal effects are inferred from serially dependent cumulative curves.",
        ]
    )
    (output / "README.md").write_text("\n".join(report) + "\n")
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "statistic": "Pearson correlation coefficient",
        "x": "Benchmark-local task encounter position (proxy for update step)",
        "y": "Plotted equal-weight seed mean cumulative online score (%)",
        "seeds": [42, 43, 44],
        "fitting_interval": "All available points in each benchmark region",
        "missing_means": "Omitted without renumbering step positions",
        "aggregation": "Correlation of mean curves, not mean correlation over seeds",
        "no_p_values": "Cumulative curve points are serially dependent; only descriptive r is reported",
        "results": results,
        "source_files": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*sources.values(), seq_metadata, *figure_files]
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("\n".join(report[:10]))
    print(f"Results and source points: {output}")


if __name__ == "__main__":
    main()
