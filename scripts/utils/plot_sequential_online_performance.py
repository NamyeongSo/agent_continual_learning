# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Plot trailing online scores, restarting each window at a benchmark transition."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ace.checkpoint_recovery import completed_prefix  # noqa: E402

BENCHMARKS = ("bfcl", "appworld", "terminalbench2")
LABELS = {"bfcl": "BFCL", "appworld": "AppWorld", "terminalbench2": "Terminal-Bench 2.0"}
COLORS = {"ace": "#2474B5", "premise": "#E06B35", "skillopt": "#27945C"}
METHOD_LABELS = {"ace": "ACE", "premise": "PREMiSE", "skillopt": "SkillOpt"}
DEFAULT_ACE_ROOT = ROOT / "scripts/ace/outputs/ace_appworld_scorefix_resume_bfcl50_20260928"
DEFAULT_PREMISE_ROOT = (
    ROOT / "outputs/premise_bfcl_appworld_terminalbench2_seed42_44_scorefix_20260928"  # pragma: allowlist secret
)
DEFAULT_TEST_MANIFEST = ROOT / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json"
PLOT_LABEL_GID = "performance-plot-label"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_segments(order: list[tuple[str, str]]) -> list[tuple[str, int, int]]:
    segments = []
    for iteration, (benchmark, _task) in enumerate(order, start=1):
        if not segments or segments[-1][0] != benchmark:
            segments.append((benchmark, iteration, iteration))
        else:
            name, first, _last = segments[-1]
            segments[-1] = (name, first, iteration)
    return segments


def load_run(
    root: Path, job: dict, method: str, premise_score: str, appworld_metric: str, *, mode: str = "sequential"
) -> dict:
    if mode not in ("sequential", "interleaved"):
        raise ValueError(f"Unsupported task setting: {mode}")
    seed = int(job["seed"])
    run = Path(job.get("output_dir") or root / job.get("name", f"{method}_{mode}_s{seed}"))
    if not run.is_absolute():
        run = ROOT / run
    order = [(str(benchmark), str(task)) for benchmark, task in job["task_order"]]
    if mode == "sequential" and [segment[0] for segment in benchmark_segments(order)] != list(BENCHMARKS):
        raise ValueError(f"Expected sequential BFCL / AppWorld / Terminal-Bench blocks: {run}")
    if mode == "interleaved" and {benchmark for benchmark, _task in order} != set(BENCHMARKS):
        raise ValueError(f"Expected all three benchmarks in the interleaved order: {run}")
    scores, errors = {}, []
    if method == "ace":
        prefix = completed_prefix(run, order)
        cursor = len(prefix)
        for iteration, item in enumerate(prefix, start=1):
            record = item.get("metrics") or item["result"]
            if record.get("score") is not None:
                scores[iteration] = float(record["score"])
            if record.get("status") in ("error", "cancelled"):
                errors.append(iteration)
    else:
        state = read_json(run / "state.json")
        config = state["config"]
        saved_order = [(str(benchmark), str(task)) for benchmark, task in config["order"]]
        if config["mode"] != mode or int(config["seed"]) != seed or saved_order != order:
            raise ValueError(f"Saved PREMiSE configuration differs from its manifest: {run}")
        cursor = int(state["cursor"])
        if not 0 <= cursor <= len(order):
            raise ValueError(f"Invalid training cursor {cursor}: {run}")
        for iteration in range(1, cursor + 1):
            artifact = read_json(run / "iterations" / f"iter_{iteration:04d}.json")
            if (
                artifact["iteration"] != iteration
                or (artifact["benchmark"], str(artifact["task_id"])) != order[iteration - 1]
            ):
                raise ValueError(f"Iteration {iteration} differs from the manifest: {run}")
            parents = artifact.get("parents") or []
            if premise_score == "rank1":
                parents = parents[:1]
            values = [float(parent["score"]) for parent in parents if parent and parent.get("score") is not None]
            if values:
                scores[iteration] = max(values)
            if artifact.get("errors"):
                errors.append(iteration)
    for iteration, score in scores.items():
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid score at iteration {iteration}: {run}")
    raw_scores = dict(scores)
    if appworld_metric == "tgc":
        scores = {
            i: float(value >= 1.0 - 1e-9) if order[i - 1][0] == "appworld" else value for i, value in scores.items()
        }
    return {
        "method": method,
        "seed": seed,
        "run": str(run.resolve()),
        "order": order,
        "cursor": cursor,
        "scores": scores,
        "raw_scores": raw_scores,
        "benchmark_sources": dict.fromkeys(BENCHMARKS, str(run.resolve())),
        "error_iterations": errors,
        "unscored_iterations": [i for i in range(1, cursor + 1) if i not in scores],
    }


def load_runs(
    ace_root: Path,
    premise_root: Path,
    seeds: list[int],
    premise_score: str,
    appworld_metric: str,
    *,
    mode: str = "sequential",
) -> list[dict]:
    runs = []
    for method, root in (("ace", ace_root), ("premise", premise_root)):
        manifest = read_json(root / "launch_manifest.json")
        for seed in seeds:
            jobs = [
                job
                for job in manifest["jobs"]
                if job.get("mode") == mode and int(job["seed"]) == seed and job.get("method", method) == method
            ]
            if len(jobs) != 1:
                raise ValueError(f"Expected one {method} {mode} job for seed {seed}: {root}")
            runs.append(load_run(root, jobs[0], method, premise_score, appworld_metric, mode=mode))
    reference = runs[0]["order"]
    for run in runs:
        same_seed = next(item for item in runs if item["method"] == "ace" and item["seed"] == run["seed"])
        if run["order"] != same_seed["order"]:
            raise ValueError(f"Methods have different task orders for seed {run['seed']}")
        if mode == "sequential" and benchmark_segments(run["order"]) != benchmark_segments(reference):
            raise ValueError("Seed runs have different benchmark boundaries")
    return runs


def replace_premise_appworld(runs: list[dict], root: Path, premise_score: str, appworld_metric: str) -> dict:
    """Replace only AppWorld observations from matching sequential restart runs."""
    read_json(root / "launch_manifest.json")
    replaced, retained = [], []
    for run in runs:
        if run["method"] != "premise":
            continue
        source = root / f"s{run['seed']}_sequential_t3"
        if not (source / "state.json").exists():
            retained.append(run["seed"])
            continue
        replacement = load_run(
            root,
            {"seed": run["seed"], "output_dir": str(source), "task_order": run["order"]},
            "premise",
            premise_score,
            appworld_metric,
        )
        indices = {i for i, (benchmark, _task) in enumerate(run["order"], start=1) if benchmark == "appworld"}
        if replacement["cursor"] < max(indices) or run["cursor"] < max(indices):
            raise ValueError(f"AppWorld replacement requires a completed AppWorld training block: {source}")
        for field in ("scores", "raw_scores"):
            run[field] = {i: score for i, score in run[field].items() if i not in indices}
            run[field].update({i: score for i, score in replacement[field].items() if i in indices})
        run["error_iterations"] = sorted(
            [i for i in run["error_iterations"] if i not in indices]
            + [i for i in replacement["error_iterations"] if i in indices]
        )
        run["unscored_iterations"] = [i for i in range(1, run["cursor"] + 1) if i not in run["scores"]]
        run["benchmark_sources"]["appworld"] = replacement["run"]
        replaced.append(run["seed"])
    if not replaced:
        raise ValueError(f"No matching sequential PREMiSE restart runs: {root}")
    return {
        "benchmark": "appworld",
        "method": "premise",
        "root": str(root),
        "replaced_seeds": replaced,
        "retained_original_seeds": retained,
    }


def load_skillopt_runs(
    path: Path, reference_runs: list[dict], appworld_metric: str, *, mode: str
) -> list[dict]:
    """Read percentage encounter scores in their native task order, without reordering results."""
    if mode not in ("sequential", "interleaved"):
        raise ValueError(f"Unsupported task setting: {mode}")
    path = path.resolve()
    groups = defaultdict(list)
    references = {run["seed"]: run for run in reference_runs if run["method"] == "ace"}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"mode", "seed", "iteration", "benchmark", "task_id", "score", "source"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"SkillOpt CSV is missing required columns: {path}")
        for row in reader:
            if row["mode"] == mode and int(row["seed"]) in references:
                groups[int(row["seed"])].append(row)
    runs = []
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for seed, reference in references.items():
        rows = sorted(groups[seed], key=lambda row: int(row["iteration"]))
        iterations = [int(row["iteration"]) for row in rows]
        if not rows or iterations != list(range(1, len(rows) + 1)):
            raise ValueError(f"SkillOpt iterations must be contiguous and unique for {mode} seed {seed}: {path}")
        order = [(row["benchmark"], row["task_id"]) for row in rows]
        counts = {benchmark: sum(b == benchmark for b, _task in order) for benchmark in BENCHMARKS}
        reference_counts = {
            benchmark: sum(b == benchmark for b, _task in reference["order"]) for benchmark in BENCHMARKS
        }
        if len(order) != len(reference["order"]) or counts != reference_counts:
            raise ValueError(f"SkillOpt task count or benchmark coverage differs for seed {seed}: {path}")
        if mode == "sequential" and benchmark_segments(order) != benchmark_segments(reference["order"]):
            raise ValueError(f"SkillOpt sequential benchmark boundaries differ for seed {seed}: {path}")
        raw_scores = {}
        for iteration, row in enumerate(rows, 1):
            if not row["score"].strip():
                continue
            score = float(row["score"])
            if not math.isfinite(score) or not 0 <= score <= 100:
                raise ValueError(f"Invalid SkillOpt percentage score at iteration {iteration}: {path}")
            raw_scores[iteration] = score / 100
        scores = {
            i: float(value >= 1.0 - 1e-9) if appworld_metric == "tgc" and order[i - 1][0] == "appworld" else value
            for i, value in raw_scores.items()
        }
        differences = sum(task != original for task, original in zip(order, reference["order"], strict=True))
        runs.append(
            {
                "method": "skillopt",
                "seed": seed,
                "run": str(path),
                "order": order,
                "cursor": len(order),
                "scores": scores,
                "raw_scores": raw_scores,
                "benchmark_sources": dict.fromkeys(BENCHMARKS, str(path)),
                "error_iterations": [],
                "unscored_iterations": [i for i in iterations if i not in scores],
                "encounter_sources": {i: row["source"] for i, row in enumerate(rows, 1)},
                "csv_sha256": digest,
                "csv_score_scale": "0..100; divided by 100 before aggregation",
                "error_status_available": False,
                "task_order_matches_ace": differences == 0,
                "task_position_mismatches": differences,
                "benchmark_position_mismatches": sum(
                    task[0] != original[0] for task, original in zip(order, reference["order"], strict=True)
                ),
                "task_set_matches_ace": set(order) == set(reference["order"]),
            }
        )
    return runs


def read_test_scores(
    path: Path, benchmark: str, expected_ids: list[str], manifest_hash: str, appworld_metric: str
) -> tuple[dict, dict]:
    """Validate frozen test coverage and count recorded execution errors as zero."""
    summary = read_json(path)
    identity = summary.get("identity", summary)
    if identity.get("benchmark") != benchmark or identity.get("manifest_sha256") != manifest_hash:
        raise ValueError(f"Test benchmark or split manifest differs: {path}")
    if identity.get("learning", False) or not summary.get("evaluation_mode", True):
        raise ValueError(f"Expected an evaluation without learning: {path}")
    if identity.get("beam_rank", 1) != 1:
        raise ValueError(f"Expected the frozen rank-1 PREMiSE checkpoint: {path}")
    records = summary.get("results", summary.get("sessions", []))
    observed = [str(record["task_id"]) for record in records if record is not None]
    if len(observed) != len(expected_ids) or len(set(observed)) != len(observed) or set(observed) != set(expected_ids):
        raise ValueError(f"Test records do not cover the complete frozen test split: {path}")
    if set(identity.get("task_ids", [])) != set(expected_ids):
        raise ValueError(f"Test identity differs from the frozen task split: {path}")
    scores, errors = [], 0
    for record in records:
        failed = record.get("status") in ("error", "cancelled")
        errors += int(failed)
        score = 0.0 if failed else record.get("score")
        if score is None or not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
            raise ValueError(f"Invalid test score: {path}")
        score = float(score)
        if benchmark == "appworld" and appworld_metric == "tgc":
            score = float(score >= 1.0 - 1e-9)
        scores.append(score)
    return summary, {
        "score": statistics.mean(scores),
        "test_tasks": len(scores),
        "execution_errors": errors,
        "source_file": str(path.resolve()),
        "source_official_score": summary.get("benchmark_score", summary.get("mean_official_score")),
    }


def load_test_references(
    runs: list[dict],
    ace_root: Path,
    manifest: Path,
    appworld_metric: str,
    *,
    checkpoints: bool = False,
    initial_root: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """Load only each benchmark's finishing checkpoint and optional shared initial context."""
    split = read_json(manifest)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    segments = benchmark_segments(runs[0]["order"])
    seed_rows = []
    checkpoint_runs = [run for run in runs if run["method"] in ("ace", "premise")] if checkpoints else []
    for run in checkpoint_runs:
        for benchmark, _start, checkpoint in segments:
            source = Path(run["benchmark_sources"][benchmark])
            if run["method"] == "ace":
                directory = Path(run["run"]).name
                relative = Path(directory) / f"checkpoint_{checkpoint:03d}" / benchmark / "summary.json"
                path = ace_root / "test_evaluation" / relative
                retry = ace_root / "test_evaluation_terminalbench2_retry" / relative
                if benchmark == "terminalbench2" and retry.exists():
                    path = retry
                checkpoint_path = (
                    ace_root / "test_evaluation/snapshots" / directory / f"checkpoint_{checkpoint:03d}.json"
                )
            else:
                path = source / "test_evaluation" / f"cp{checkpoint:03d}_{benchmark}.json"
                checkpoint_path = source / "checkpoints" / f"checkpoint_{checkpoint:03d}.json"
            summary, values = read_test_scores(
                path, benchmark, split[f"{benchmark}_{run['seed']}"]["test"], manifest_hash, appworld_metric
            )
            identity = summary.get("identity", summary)
            expected_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            if identity.get("checkpoint_sha256") != expected_hash:
                raise ValueError(f"Test evaluation differs from the selected checkpoint: {path}")
            if int(identity.get("checkpoint", identity.get("stage", -1))) != checkpoint:
                raise ValueError(f"Test evaluation has the wrong checkpoint iteration: {path}")
            if run["method"] == "ace" and Path(summary["train_dir"]).resolve() != source.resolve():
                raise ValueError(f"Test evaluation comes from a different training run: {path}")
            seed_rows.append(
                {
                    "series": run["method"],
                    "benchmark": benchmark,
                    "checkpoint": checkpoint,
                    "seed": run["seed"],
                    **values,
                    "checkpoint_sha256": expected_hash,
                }
            )
    if initial_root:
        from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK

        initial_hash = hashlib.sha256(DEFAULT_PLAYBOOK.encode()).hexdigest()
        seeds = sorted({run["seed"] for run in runs})
        for benchmark, _start, _end in segments:
            for seed in seeds:
                path = initial_root / f"{benchmark}_{seed}" / "summary.json"
                summary, values = read_test_scores(
                    path, benchmark, split[f"{benchmark}_{seed}"]["test"], manifest_hash, appworld_metric
                )
                if summary.get("initial_playbook_sha256") != initial_hash or summary.get("training_time", True):
                    raise ValueError(f"Expected the unchanged shared initial playbook without training: {path}")
                seed_rows.append(
                    {
                        "series": "initial",
                        "benchmark": benchmark,
                        "checkpoint": 0,
                        "seed": seed,
                        **values,
                        "checkpoint_sha256": initial_hash,
                    }
                )
    groups = defaultdict(list)
    for row in seed_rows:
        groups[row["series"], row["benchmark"], row["checkpoint"]].append(row)
    references = []
    for (series, benchmark, checkpoint), samples in sorted(groups.items()):
        values = [sample["score"] for sample in samples]
        references.append(
            {
                "series": series,
                "benchmark": benchmark,
                "checkpoint": checkpoint,
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else None,
                "seed_count": len(values),
                "test_tasks": sum(sample["test_tasks"] for sample in samples),
                "execution_errors": sum(sample["execution_errors"] for sample in samples),
            }
        )
    return seed_rows, references


def rolling_windows(order: list[tuple[str, str]], scores: dict[int, float], cursor: int, window: int) -> list[dict]:
    """Keep iteration positions when scores are absent; never extend the window."""
    if window < 1:
        raise ValueError("Window size must be positive")
    if not 0 <= cursor <= len(order):
        raise ValueError("Cursor is outside the task order")
    rows = []
    segment_start = 1
    previous_benchmark = None
    for iteration, (benchmark, _task) in enumerate(order[:cursor], start=1):
        if benchmark != previous_benchmark:
            segment_start = iteration
            previous_benchmark = benchmark
        first = max(segment_start, iteration - window + 1)
        values = [scores[i] for i in range(first, iteration + 1) if i in scores]
        count = iteration - first + 1
        rows.append(
            {
                "window": window,
                "benchmark": benchmark,
                "iteration": iteration,
                "benchmark_iteration": iteration - segment_start + 1,
                "segment_start": segment_start,
                "window_start": first,
                "window_end": iteration,
                "window_tasks": count,
                "scored_tasks": len(values),
                "missing_tasks": count - len(values),
                "score": statistics.mean(values) if values else None,
            }
        )
    return rows


def cumulative_windows(order: list[tuple[str, str]], scores: dict[int, float], cursor: int) -> list[dict]:
    """Average all observed scores since the current benchmark block began."""
    rows = rolling_windows(order, scores, cursor, max(1, len(order)))
    return [{**row, "window": "cumulative"} for row in rows]


def aggregate_windows(rows: list[dict]) -> list[dict]:
    """Average seed-level window means equally; report sample SD and coverage."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"], row["window"], row["benchmark"], row["iteration"]].append(row)
    aggregated = []
    for (method, window, benchmark, iteration), samples in sorted(groups.items()):
        values = [sample["score"] for sample in samples if sample["score"] is not None]
        first = samples[0]
        aggregated.append(
            {
                "method": method,
                "window": window,
                "benchmark": benchmark,
                "iteration": iteration,
                "window_start": first["window_start"],
                "window_end": first["window_end"],
                "window_tasks_per_seed": first["window_tasks"],
                "seed_count": len(values),
                "seed_ids": ",".join(str(sample["seed"]) for sample in samples if sample["score"] is not None),
                "scored_tasks": sum(sample["scored_tasks"] for sample in samples),
                "missing_tasks": sum(sample["missing_tasks"] for sample in samples),
                "mean": statistics.mean(values) if values else None,
                "std": statistics.stdev(values) if len(values) > 1 else None,
            }
        )
    return aggregated


def seed_figure_data(rows: list[dict], test_rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """Use only the requested seed for both online curves and test references."""
    online = aggregate_windows([row for row in rows if row["seed"] == seed])
    references = [
        {
            "series": row["series"],
            "benchmark": row["benchmark"],
            "checkpoint": row["checkpoint"],
            "mean": row["score"],
            "std": None,
            "seed_count": 1,
            "test_tasks": row["test_tasks"],
            "execution_errors": row["execution_errors"],
        }
        for row in test_rows
        if row["seed"] == seed
    ]
    return online, references


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure_pair(fig, output: Path, basename: str, *, dpi: int) -> None:
    """Export the labeled figure and a version without legends, annotations, or notes."""
    labels = fig.findobj(lambda artist: artist.get_gid() == PLOT_LABEL_GID)
    visibility = [artist.get_visible() for artist in labels]
    try:
        for suffix in ("", "_no_labels"):
            if suffix:
                for artist in labels:
                    artist.set_visible(False)
            for extension in ("png", "pdf"):
                fig.savefig(output / f"{basename}{suffix}.{extension}", dpi=dpi, facecolor="white")
    finally:
        for artist, visible in zip(labels, visibility, strict=True):
            artist.set_visible(visible)


def plot_sequential_axis(
    ax,
    rows: list[dict],
    window: int | str,
    segments: list[tuple[str, int, int]],
    *,
    compact: bool = False,
    references: list[dict] | None = None,
    shade_window_filling: bool = True,
) -> None:
    """Draw all benchmark blocks on one axis, breaking curves at each shift."""
    import numpy as np

    online_annotations = []
    for block, (benchmark, start, end) in enumerate(segments):
        if window != "cumulative" and shade_window_filling:
            ax.axvspan(start - 0.5, min(end + 0.5, start + window - 1.5), color="#E8ECF1", alpha=0.7, zorder=0)
        for method in (method for method in COLORS if any(row["method"] == method for row in rows)):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["window"] == window and row["benchmark"] == benchmark and row["method"] == method
                ),
                key=lambda row: row["iteration"],
            )
            x = [row["iteration"] for row in selected]
            means = np.array([100 * row["mean"] if row["mean"] is not None else np.nan for row in selected])
            std = np.array([100 * row["std"] if row["std"] is not None else np.nan for row in selected])
            if np.isfinite(std).any():
                ax.fill_between(x, means - std, means + std, color=COLORS[method], alpha=0.13, linewidth=0)
            ax.plot(
                x,
                means,
                color=COLORS[method],
                linewidth=1.9 if compact else 2.3,
                label=METHOD_LABELS[method] if block == 0 else "_nolegend_",
            )
            if selected and selected[-1]["mean"] is not None:
                offset = {"ace": -18, "premise": 13, "skillopt": 28}[method]
                if references and benchmark == "terminalbench2":
                    offset = {"ace": -28, "premise": 42, "skillopt": 15}[method]
                annotation = ax.annotate(
                    f"{'Online ' if references else METHOD_LABELS[method] + ' '}{means[-1]:.2f}%",
                    xy=(x[-1], means[-1]),
                    xytext=(-8, offset),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    color=COLORS[method],
                    fontsize=8.5 if compact else 10,
                    fontweight="semibold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
                )
                annotation.set_gid(PLOT_LABEL_GID)
                online_annotations.append(annotation)
        block_references = sorted(
            (row for row in references or [] if row["benchmark"] == benchmark), key=lambda row: row["mean"]
        )
        label_y = []
        for row in block_references:
            label_y.append(max(100 * row["mean"] + 2, label_y[-1] + 7 if label_y else 2))
        if label_y and label_y[-1] > 97:
            label_y = [y - (label_y[-1] - 97) for y in label_y]
        for row, y in zip(block_references, label_y, strict=True):
            series = row["series"]
            color = "black" if series == "initial" else COLORS[series]
            label = (
                "Shared initial context (test)" if series == "initial" else f"{METHOD_LABELS[series]} checkpoint test"
            )
            ax.plot(
                [start - 0.5, end + 0.5],
                [100 * row["mean"]] * 2,
                linestyle=({"ace": 0, "premise": 4, "initial": 2}[series], (5, 3)),
                linewidth=1.7,
                color=color,
                label=label if block == 0 else "_nolegend_",
                zorder=3,
            )
            name = "Initial" if series == "initial" else f"{METHOD_LABELS[series]} CP{row['checkpoint']} test"
            marker = "*" if row["execution_errors"] else ""
            annotation = ax.annotate(
                f"{name} {100 * row['mean']:.2f}%{marker}",
                xy=(start + 1, 100 * row["mean"]),
                xytext=(start + 1, y),
                ha="left",
                va="bottom",
                fontsize=8 if compact else 9,
                color=color,
                fontweight="semibold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
                arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.65}
                if y > 100 * row["mean"] + 4
                else None,
                zorder=5,
            )
            annotation.set_gid(PLOT_LABEL_GID)
        ax.text(
            (start + end) / 2,
            1.025,
            f"{LABELS[benchmark]}\nIterations {start}-{end}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9 if compact else 11,
            fontweight="semibold",
            linespacing=1.45,
        )
        if block:
            ax.axvline(start - 0.5, color="#475569", linewidth=1.4, linestyle="--")
        if not any(row["window"] == window and row["benchmark"] == benchmark for row in rows):
            ax.text((start + end) / 2, 50, "No completed tasks", ha="center", color="#64748B")
    ax.set_xlim(segments[0][1] - 0.5, segments[-1][2] + 0.5)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks([segments[0][1], *range(10, segments[-1][2] + 1, 10)])
    ax.set_xlabel("Sequential iteration", fontsize=9 if compact else 11)
    ax.set_ylabel(
        "Cumulative score (%)" if window == "cumulative" else "Online score (%)", fontsize=9 if compact else 11
    )
    ax.grid(axis="y", color="#D8DEE6", linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B5BFCD")
    ax.tick_params(labelsize=9 if compact else 10, color="#B5BFCD")
    if references:
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        occupied = [
            text.get_bbox_patch().get_window_extent(renderer)
            for text in ax.texts
            if text.get_bbox_patch() is not None and text not in online_annotations
        ]
        for annotation in online_annotations:
            x_offset, original_offset = annotation.xyann
            direction = 1 if original_offset >= 0 else -1
            for shift in range(0, 151, 10):
                annotation.set_position((x_offset, original_offset + direction * shift))
                bounds = annotation.get_window_extent(renderer).expanded(1.02, 1.1)
                if bounds.ymin >= ax.bbox.ymin and bounds.ymax <= ax.bbox.ymax:
                    if not any(bounds.overlaps(other) for other in occupied):
                        occupied.append(bounds)
                        break
            else:
                annotation.set_position((x_offset, original_offset))
            connector = ax.annotate(
                "",
                xy=annotation.xy,
                xytext=annotation.xyann,
                textcoords="offset points",
                arrowprops={"arrowstyle": "-", "color": annotation.get_color(), "linewidth": 0.7, "alpha": 0.65},
                zorder=2.5,
            )
            connector.set_gid(PLOT_LABEL_GID)


def save_figures(
    output: Path,
    rows: list[dict],
    windows: list[int | str],
    segments: list[tuple[str, int, int]],
    seeds: list[int],
    appworld_metric: str,
    premise_score: str,
    source_note: str | None = None,
    references: list[dict] | None = None,
    *,
    shade_window_filling: bool = True,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": "#334155",
            "text.color": "#1E293B",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )
    seed_note = ", ".join(str(seed) for seed in seeds)
    seed_caption = (
        f"Seed {seeds[0]}  |  single-seed online scores"
        if len(seeds) == 1
        else f"Seeds {seed_note}  |  mean and +/-1 sample SD"
    )
    metric_note = "AppWorld: TGC" if appworld_metric == "tgc" else "AppWorld: partial task score"
    parent_note = "best pre-update parent" if premise_score == "best-parent" else "rank-1 pre-update parent"
    cumulative = windows == ["cumulative"]
    mode_note = (
        "Cumulative means restart at each benchmark shift."
        if cumulative
        else "Gray: window filling after reset."
        if shade_window_filling
        else "Window means restart at each benchmark shift."
    )
    note = (
        f"{metric_note}; PREMiSE: {parent_note}. {mode_note}\n"
        "Missing scores stay at their iteration and are omitted from the mean; recorded ACE error scores are retained."
    )
    if source_note:
        note += "\n" + source_note
    if references:
        if any(row["series"] != "initial" for row in references):
            note += (
                "\nDashed: full-test scores at each benchmark's finishing checkpoint; black: shared initial context."
            )
        else:
            note += "\nBlack dashed lines: shared initial context (test)."
        if any(row["execution_errors"] for row in references):
            note += (
                "\n* Recorded test mean with execution-error tasks counted as zero; these evaluations retain errors."
            )
    for window in windows:
        fig, ax = plt.subplots(figsize=(14, 7.2 if references else 6.2))
        fig.subplots_adjust(
            top=0.7 if references else 0.735,
            bottom=0.25 if references else 0.205 if source_note else 0.17,
            left=0.075,
            right=0.985,
        )
        plot_sequential_axis(
            ax, rows, window, segments, references=references, shade_window_filling=shade_window_filling
        )
        title = (
            "Sequential cumulative online performance"
            if cumulative
            else f"Sequential online performance  |  N = {window}"
        )
        fig.suptitle(title, fontsize=19, fontweight="semibold", y=0.97)
        fig.text(0.5, 0.89, seed_caption, ha="center", fontsize=10)
        handles, labels = ax.get_legend_handles_labels()
        legend_order = [
            *METHOD_LABELS.values(),
            *(f"{label} checkpoint test" for label in METHOD_LABELS.values()),
            "Shared initial context (test)",
        ]
        ordered = sorted(zip(handles, labels, strict=True), key=lambda item: legend_order.index(item[1]))
        handles, labels = zip(*ordered, strict=True)
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.86),
            ncol=3 if references else len(handles),
            frameon=False,
        )
        legend.set_gid(PLOT_LABEL_GID)
        fig.text(0.08, 0.015, note, fontsize=8.5, color="#64748B", linespacing=1.6, gid=PLOT_LABEL_GID)
        basename = "sequential_online_cumulative" if cumulative else f"sequential_online_N{window:02d}"
        save_figure_pair(fig, output, basename, dpi=200)
        plt.close(fig)

    if cumulative:
        return

    fig, axes = plt.subplots(len(windows), 1, figsize=(14, 4.2 * len(windows) + 1.5), squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.12, hspace=0.9)
    for ax, window in zip(axes[:, 0], windows, strict=True):
        plot_sequential_axis(
            ax, rows, window, segments, compact=True, references=references, shade_window_filling=shade_window_filling
        )
        ax.set_title(f"Window N = {window}", loc="left", y=1.19, fontsize=13, fontweight="semibold")
    fig.suptitle("Sequential online performance", fontsize=22, fontweight="semibold", y=0.985)
    fig.text(
        0.5,
        0.943,
        f"{seed_caption}  |  windows reset at every benchmark shift",
        ha="center",
        fontsize=11,
    )
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.928),
        ncol=3 if references else len(handles),
        frameon=False,
    )
    legend.set_gid(PLOT_LABEL_GID)
    fig.text(0.06, 0.02, note, fontsize=9, color="#64748B", linespacing=1.7, gid=PLOT_LABEL_GID)
    save_figure_pair(fig, output, "sequential_online_overview", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ace-root", type=Path, default=DEFAULT_ACE_ROOT)
    parser.add_argument("--premise-root", type=Path, default=DEFAULT_PREMISE_ROOT)
    parser.add_argument("--skillopt-csv", type=Path, help="Add SkillOpt's percentage training-encounter CSV")
    parser.add_argument(
        "--premise-appworld-root",
        type=Path,
        help="Replace only AppWorld from matching s<seed>_sequential_t3 restart directories",
    )
    parser.add_argument("--windows", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument(
        "--cumulative",
        action="store_true",
        help="Plot all tasks since each benchmark began, resetting at benchmark shifts",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--by-seed", action="store_true", help="Write a separate figure and source data for each seed")
    parser.add_argument("--appworld-metric", choices=("tgc", "partial"), default="tgc")
    parser.add_argument("--premise-score", choices=("best-parent", "rank1"), default="best-parent")
    parser.add_argument("--test-checkpoints", action="store_true", help="Add benchmark-finishing frozen test means")
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--initial-context-root", type=Path, help="Shared unchanged-initial-playbook test evaluations")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sequential_online_windows_20261009")
    args = parser.parse_args()
    if any(window < 1 for window in args.windows) or len(set(args.windows)) != len(args.windows):
        parser.error("Window sizes must be positive and unique")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    runs = load_runs(
        args.ace_root.resolve(), args.premise_root.resolve(), args.seeds, args.premise_score, args.appworld_metric
    )
    override = None
    source_note = None
    if args.premise_appworld_root:
        override = replace_premise_appworld(
            runs, args.premise_appworld_root.resolve(), args.premise_score, args.appworld_metric
        )
        replaced = ", ".join(str(seed) for seed in override["replaced_seeds"])
        source_note = f"PREMiSE AppWorld: restarted training for seeds {replaced}."
        if override["retained_original_seeds"]:
            retained = ", ".join(str(seed) for seed in override["retained_original_seeds"])
            source_note += f" Original training retained for seeds {retained}."
    skillopt_runs = (
        load_skillopt_runs(args.skillopt_csv, runs, args.appworld_metric, mode="sequential")
        if args.skillopt_csv
        else []
    )
    runs.extend(skillopt_runs)
    skillopt_note = "SkillOpt: CSV training encounters, native task order; checkpoint test scores not supplied."
    if skillopt_runs:
        source_note = (source_note + "\n" if source_note else "") + skillopt_note
    if not any(run["cursor"] for run in runs):
        parser.error("No completed sequential training tasks")
    windows = ["cumulative"] if args.cumulative else args.windows
    rows = [
        {"method": run["method"], "seed": run["seed"], **row}
        for run in runs
        for window in windows
        for row in (
            cumulative_windows(run["order"], run["scores"], run["cursor"])
            if args.cumulative
            else rolling_windows(run["order"], run["scores"], run["cursor"], window)
        )
    ]
    aggregate = aggregate_windows(rows)
    test_rows, references = [], []
    if args.test_checkpoints or args.initial_context_root:
        test_rows, references = load_test_references(
            runs,
            args.ace_root.resolve(),
            args.test_manifest.resolve(),
            args.appworld_metric,
            checkpoints=args.test_checkpoints,
            initial_root=args.initial_context_root.resolve() if args.initial_context_root else None,
        )
    tasks = [
        {
            "method": run["method"],
            "seed": run["seed"],
            "iteration": i,
            "benchmark": benchmark,
            "task_id": task,
            "raw_score": run["raw_scores"].get(i),
            "online_score": run["scores"].get(i),
            "recorded_errors": None if run["method"] == "skillopt" else i in run["error_iterations"],
            "source_run": run["benchmark_sources"][benchmark],
            "encounter_source": run.get("encounter_sources", {}).get(i),
        }
        for run in runs
        for i, (benchmark, task) in enumerate(run["order"][: run["cursor"]], start=1)
    ]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "task_scores.csv", tasks)
    write_csv(output / ("seed_cumulative.csv" if args.cumulative else "seed_windows.csv"), rows)
    write_csv(output / ("aggregate_cumulative.csv" if args.cumulative else "aggregate_windows.csv"), aggregate)
    if references:
        write_csv(output / "seed_test_references.csv", test_rows)
        write_csv(output / "aggregate_test_references.csv", references)
    segments = benchmark_segments(runs[0]["order"])
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "windows": None if args.cumulative else args.windows,
        "aggregation_mode": "cumulative" if args.cumulative else "rolling",
        "seeds": args.seeds,
        "figure_grouping": "per_seed" if args.by_seed else "seed_mean",
        "figure_variants": ["labeled", "no_labels"],
        "benchmark_segments": segments,
        "graph_layout": "One axis for iterations 1-130, divided at 50.5 and 100.5",
        "appworld_metric": args.appworld_metric,
        "premise_score": args.premise_score,
        "aggregation": "Equal mean of seed-level means; sample SD",
        "window": "[benchmark_start, M]; reset at benchmark shifts"
        if args.cumulative
        else "[max(benchmark_start, M-N+1), M]; partial windows allowed after reset",
        "missing_scores": "Excluded from denominator without extending the iteration window",
        "ace_errors": "Recorded online error scores are retained, matching the existing report",
        "skillopt_runs": [
            {
                key: value
                for key, value in run.items()
                if key not in ("order", "scores", "raw_scores", "encounter_sources")
            }
            for run in skillopt_runs
        ],
        "benchmark_override": override,
        "test_references": references,
        "test_reference_definition": (
            "Equal seed means over the entire frozen test split; execution errors count as zero"
        ),
        "initial_context_root": str(args.initial_context_root.resolve()) if args.initial_context_root else None,
        "initial_context_definition": (
            "Shared DEFAULT_PLAYBOOK and generator prompt, evaluated using ACE without learning"
        ),
        "runs": [
            {
                key: run[key]
                for key in (
                    "method",
                    "seed",
                    "run",
                    "cursor",
                    "unscored_iterations",
                    "error_iterations",
                    "benchmark_sources",
                )
            }
            for run in runs
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.by_seed:
        for seed in args.seeds:
            seed_output = output / f"seed{seed}"
            seed_output.mkdir(parents=True, exist_ok=True)
            seed_online, seed_references = seed_figure_data(rows, test_rows, seed)
            seed_note = None
            if override:
                origin = "restarted" if seed in override["replaced_seeds"] else "original"
                seed_note = f"PREMiSE AppWorld: {origin} training for seed {seed}."
            if skillopt_runs:
                seed_note = (seed_note + "\n" if seed_note else "") + skillopt_note
            write_csv(seed_output / "task_scores.csv", [row for row in tasks if row["seed"] == seed])
            write_csv(
                seed_output / ("seed_cumulative.csv" if args.cumulative else "seed_windows.csv"),
                [row for row in rows if row["seed"] == seed],
            )
            write_csv(
                seed_output / ("aggregate_cumulative.csv" if args.cumulative else "aggregate_windows.csv"), seed_online
            )
            if seed_references:
                write_csv(seed_output / "seed_test_references.csv", [row for row in test_rows if row["seed"] == seed])
                write_csv(seed_output / "aggregate_test_references.csv", seed_references)
            seed_metadata = {
                **metadata,
                "seeds": [seed],
                "aggregation": "Single-seed scores; no across-seed mean or SD",
                "test_reference_definition": "Single-seed full-test scores; execution errors count as zero",
                "test_references": seed_references,
                "runs": [run for run in metadata["runs"] if run["seed"] == seed],
                "skillopt_runs": [run for run in metadata["skillopt_runs"] if run["seed"] == seed],
            }
            (seed_output / "metadata.json").write_text(json.dumps(seed_metadata, indent=2) + "\n", encoding="utf-8")
            save_figures(
                seed_output,
                seed_online,
                windows,
                segments,
                [seed],
                args.appworld_metric,
                args.premise_score,
                source_note=seed_note,
                references=seed_references,
            )
    else:
        save_figures(
            output,
            aggregate,
            windows,
            segments,
            args.seeds,
            args.appworld_metric,
            args.premise_score,
            source_note=source_note,
            references=references,
        )
    print(f"Figures and source data: {output}")
    for run in runs:
        print(
            f"{run['method']} seed={run['seed']}: {run['cursor']}/{len(run['order'])} tasks, "
            f"unscored={run['unscored_iterations']}"
        )


if __name__ == "__main__":
    main()
