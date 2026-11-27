# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Show live training progress and checkpoint scores for ACE and PREMiSE baselines.

``initial`` is the benchmark-specific isolated setting. Online scores use
cumulative training rollouts for interleaved runs and checkpoint intervals otherwise; offline scores are
frozen-checkpoint test evaluations. Incomplete and error-bearing results stay
visible but are excluded from the final three-seed aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from exgentic.agents.premise.pipeline import manifest_order  # noqa: E402
from scripts.ace.checkpoint_recovery import completed_prefix  # noqa: E402
from scripts.utils import monitor_experiments as monitor  # noqa: E402

BENCHMARKS = ("bfcl", "appworld", "terminalbench2")
METHODS = ("ace", "premise", "ps_st_pareto")
SEEDS = (42, 43, 44)
POINTS = (50, 100, 130)
KST = timezone(timedelta(hours=9))
DEFAULT_MONITOR_CONFIG = ROOT / "outputs/experiment_monitor_20260929/config.json"
DEFAULT_INITIAL_ROOT = ROOT / "outputs/isolated_baselines_20260930"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def score_summary(values: list[float]) -> dict:
    """Keep sample standard deviation undefined for a single seed."""
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else None,
    }


def live_pid(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rpartition(")")[2].split()
        return fields[0] not in ("Z", "X")
    except (OSError, IndexError):
        return False


def ace_scores(run: Path, order: list[tuple[str, str]]) -> dict[int, float]:
    """Use completed training sessions when online_metrics has not been flushed yet."""
    prefix = completed_prefix(run, order)
    scores = {}
    for index, item in enumerate(prefix, start=1):
        result = item.get("metrics") or item.get("result") or {}
        if result.get("score") is not None:
            scores[index] = float(result["score"])
    return scores


def premise_scores(run: Path, cursor: int, order: list[tuple[str, str]]) -> dict[int, float]:
    """Use the highest observed parent rollout score for each training task."""
    scores = {}
    for index in range(1, min(cursor, len(order)) + 1):
        item = read_json(run / "iterations" / f"iter_{index:04d}.json", {})
        if item and (item.get("benchmark"), str(item.get("task_id"))) != order[index - 1]:
            raise ValueError(f"Iteration {index} differs from planned task order: {run}")
        parents = item.get("parents") or []
        parent_scores = [float(parent["score"]) for parent in parents if parent and parent.get("score") is not None]
        if parent_scores:
            scores[index] = max(parent_scores)
    return scores


def online_windows(
    order: list[tuple[str, str]], scores: dict[int, float], points: tuple[int, ...], cursor: int,
    *, cumulative: bool = False,
) -> list[dict]:
    """Aggregate rollouts, using full task success (TGC) for AppWorld."""
    rows = []
    start = 0
    for point in points:
        if not start < point <= len(order):
            raise ValueError(f"Invalid checkpoint {point} for {len(order)} tasks")
        first = 1 if cumulative else start + 1
        for benchmark in BENCHMARKS:
            indices = [i for i in range(first, point + 1) if order[i - 1][0] == benchmark]
            if not indices:
                continue
            values = [scores[i] for i in indices if i in scores]
            if benchmark == "appworld":
                # AppWorld rollout scores are partial test pass rates. TGC counts
                # only tasks that passed every official check.
                values = [float(value >= 1.0 - 1e-9) for value in values]
            observed = sum(i in scores for i in indices)
            rows.append(
                {
                    "checkpoint": point,
                    "benchmark": benchmark,
                    "score": statistics.mean(values) if values else None,
                    "scored": observed,
                    "planned": len(indices),
                    "complete": cursor >= point,
                }
            )
        start = point
    return rows


def offline_row(group: dict, method: str, metric: str) -> dict:
    summary = read_json(Path(group["result_path"]), {})
    task_mean = group.get("score")
    if method == "ace":
        official = summary.get("benchmark_score")
    elif group["benchmark"] == "appworld" and group["complete"]:
        # PREMiSE stores per-task official pass percentages, but no aggregate success rate.
        records = summary.get("results", [])
        official = (
            sum(float(item.get("score") or 0) >= 1.0 - 1e-9 for item in records) / len(records) if records else None
        )
    else:
        official = task_mean
    return {
        "checkpoint": group["checkpoint"],
        "benchmark": group["benchmark"],
        "score": official if metric == "official" else task_mean,
        "official_score": official,
        "task_mean_score": task_mean,
        "processed": group["processed"],
        "total": group["total"],
        "errors": group["errors"],
        "complete": group["complete"],
        "clean_complete": group["clean_complete"],
        "protocol_errors": group.get("protocol_errors", []),
        "owner_pids": group.get("owner_pids", []),
    }


def evaluation_progress(groups: list[dict]) -> dict:
    return {
        "clean_groups": sum(group["clean_complete"] for group in groups),
        "groups": len(groups),
        "processed_tasks": sum(group["processed"] for group in groups),
        "total_tasks": sum(group["total"] for group in groups),
        "errors": sum(group["errors"] for group in groups),
    }


def multibenchmark_runs(config_path: Path, offline_metric: str) -> list[dict]:
    config = read_json(config_path)
    if not isinstance(config, dict) or not config.get("learning_roots"):
        raise ValueError(f"Monitor configuration is missing learning_roots: {config_path}")
    # Collect fresh state without writing files, making model requests, or scanning unrelated milestone runs.
    config = {**config, "model_health_url": None, "milestone_evaluation_roots": []}
    snapshot = monitor.collect(config)
    learning = {(row["method"], row["mode"], row["seed"]): row for row in snapshot["learning"]}
    evaluations: dict[tuple[str, str, int], list[dict]] = {}
    for group in snapshot["checkpoint_evaluations"]:
        evaluations.setdefault((group["method"], group["mode"], group["seed"]), []).append(
            offline_row(group, group["method"], offline_metric)
        )
    runs = []
    for root_name in config["learning_roots"]:
        root = monitor.resolve(root_name)
        plan = read_json(root / "launch_manifest.json", {})
        if not plan:
            raise FileNotFoundError(f"Learning launch manifest is missing: {root / 'launch_manifest.json'}")
        method = str(plan.get("method", plan.get("agent", ""))).lower()
        if method not in METHODS:
            continue
        for job in plan.get("jobs", []):
            mode, seed = job["mode"], int(job["seed"])
            if mode not in ("sequential", "interleaved") or seed not in SEEDS:
                continue
            row = learning[(method, mode, seed)]
            run = monitor.resolve(job.get("output_dir", str(root / job.get("name", f"{mode}_s{seed}"))))
            order = [(str(slug), str(task)) for slug, task in job["task_order"]]
            scores = ace_scores(run, order) if method == "ace" else premise_scores(run, row["cursor"], order)
            all_groups = evaluations.get((method, mode, seed), [])
            evaluation_workers = sorted({pid for group in all_groups for pid in group["owner_pids"]})
            status = (
                "evaluating"
                if row["status"] == "trained"
                and evaluation_workers
                and any(not group["complete"] for group in all_groups)
                else row["status"]
            )
            runs.append(
                {
                    "method": method,
                    "setting": mode,
                    "seed": seed,
                    "train_benchmark": None,
                    "status": status,
                    "cursor": row["cursor"],
                    "total": row["total"],
                    "counts": row["counts"],
                    "skipped": (row.get("skipped_iterations") or {}).get("count", 0),
                    "online": online_windows(order, scores, POINTS, row["cursor"], cumulative=mode == "interleaved"),
                    "offline": all_groups,
                    "evaluation_progress": evaluation_progress(all_groups),
                    "evaluation_workers": evaluation_workers,
                    "own_benchmark_at_checkpoint": {point: order[point - 1][0] for point in POINTS},
                    "path": str(run),
                }
            )
    return runs


def initial_runs(root: Path, offline_metric: str) -> list[dict]:
    plan = read_json(root / "launch_manifest.json", {})
    if not plan:
        raise FileNotFoundError(f"Initial launch manifest is missing: {root / 'launch_manifest.json'}")
    split_path = Path(plan["source_manifest"])
    split = read_json(split_path, {})
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if split_hash != plan.get("source_manifest_sha256"):
        raise ValueError(f"Initial split manifest changed: {split_path}")
    status = read_json(root / "matrix_status.json", {}).get("jobs", {})
    runs = []
    for job in plan["jobs"]:
        method, benchmark, seed = job["method"], job["benchmark"], int(job["seed"])
        if method not in METHODS or benchmark not in BENCHMARKS or seed not in SEEDS:
            continue
        output = Path(job["output"])
        count = int(job["train_count"])
        ids = split[f"{benchmark}_{seed}"]["test"]
        if len(ids) != job["test_count"]:
            raise ValueError(f"Initial test split size changed: {method}/{benchmark}/{seed}")
        progress = status.get(job["name"], {})
        if method == "ace":
            train = output / "train"
            settings = read_json(train / "experiment_config.json", {})
            order = [(str(slug), str(task)) for slug, task in settings.get("task_order", [])]
            if order and (len(order) != count or any(slug != benchmark for slug, _ in order)):
                raise ValueError(f"Initial ACE task order differs: {train}")
            scores = ace_scores(train, order) if order else {}
            cursor = len(completed_prefix(train, order)) if order else 0
            checkpoint = (
                output / "evaluation/snapshots" / f"isolated_{benchmark}_s{seed}" / f"checkpoint_{count:03d}.json"
            )
            path = (
                output
                / "evaluation"
                / f"isolated_{benchmark}_s{seed}"
                / f"checkpoint_{count:03d}"
                / benchmark
                / "summary.json"
            )
            identity = {
                "stage": count,
                "benchmark": benchmark,
                "seed": seed,
                "mode": "isolated",
                "task_ids": ids,
                "evaluation_mode": True,
                "training_time": False,
                "manifest_sha256": split_hash,
            }
        else:
            train = output / "run"
            state = read_json(train / "state.json", {})
            cursor = int(state.get("cursor", 0))
            order, _configs = manifest_order(split_path, [benchmark], seed, None, "isolated")
            scores = premise_scores(train, cursor, order)
            checkpoint = train / "checkpoints" / f"checkpoint_{count:03d}.json"
            path = train / "test_evaluation" / f"cp{count:03d}_{benchmark}.json"
            identity = {
                "checkpoint": count,
                "benchmark": benchmark,
                "task_ids": ids,
                "beam_rank": 1,
                "learning": False,
                "manifest_sha256": split_hash,
            }
        checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.is_file() else None
        group = monitor.group_progress([path], ids, identity=identity, checkpoint_hash=checksum)
        group.update({"checkpoint": count, "benchmark": benchmark})
        state = progress.get("status", "pending")
        if cursor >= count:
            if group["clean_complete"]:
                state = "done"
            elif group["complete"] and (group["errors"] or group["protocol_errors"]):
                state = "evaluation_errors"
            elif state == "failed":
                state = "failed"
            elif state == "running" and live_pid(progress.get("pid")):
                state = "evaluating"
            else:
                state = "trained"
        elif state == "running" and not live_pid(progress.get("pid")):
            state = "stopped"
        runs.append(
            {
                "method": method,
                "setting": "initial",
                "seed": seed,
                "train_benchmark": benchmark,
                "status": state,
                "cursor": cursor,
                "total": count,
                "counts": {benchmark: cursor},
                "skipped": max(0, cursor - len(scores)) if method != "ace" else 0,
                "online": online_windows(order, scores, (count,), cursor) if order else [],
                "offline": [offline_row(group, method, offline_metric)],
                "evaluation_progress": evaluation_progress([group]),
                "own_benchmark_at_checkpoint": {count: benchmark},
                "path": str(train),
            }
        )
    return runs


def aggregate(runs: list[dict], metric: str) -> list[dict]:
    """Keep clean scores separate from interim task means with explicit coverage."""
    grouped: dict[tuple[str, str, str | None, int, str], list[tuple[int, dict]]] = {}
    for run in runs:
        for row in run[metric]:
            key = (run["method"], run["setting"], run["train_benchmark"], row["checkpoint"], row["benchmark"])
            grouped.setdefault(key, []).append((run["seed"], row))
    results = []
    for (method, setting, train_benchmark, point, benchmark), samples in grouped.items():
        selected = [
            (seed, row["score"])
            for seed, row in samples
            if row["score"] is not None and (row["complete"] if metric == "online" else row["clean_complete"])
        ]
        summary = score_summary([value for _, value in selected])
        interim = (
            [(seed, float(row["task_mean_score"]), row) for seed, row in samples
             if row.get("task_mean_score") is not None and row.get("processed", 0) > 0
             and not row.get("protocol_errors")]
            if metric == "offline" else []
        )
        interim_summary = score_summary([value for _, value, _row in interim])
        results.append(
            {
                "method": method,
                "setting": setting,
                "train_benchmark": train_benchmark,
                "checkpoint": point,
                "benchmark": benchmark,
                "seeds": [seed for seed, _ in selected],
                "expected_seeds": list(SEEDS),
                **summary,
                "interim_task_mean": interim_summary["mean"],
                "interim_task_std": interim_summary["std"],
                "interim_n": interim_summary["n"],
                "interim_seeds": [seed for seed, _value, _row in interim],
                "interim_processed": sum(row["processed"] for _seed, _value, row in interim),
                "interim_total": sum(row["total"] for _seed, _value, row in interim),
                "interim_errors": sum(row["errors"] for _seed, _value, row in interim),
            }
        )
    return sorted(
        results,
        key=lambda row: (
            METHODS.index(row["method"]),
            ("initial", "sequential", "interleaved").index(row["setting"]),
            BENCHMARKS.index(row["train_benchmark"]) if row["train_benchmark"] else -1,
            row["checkpoint"],
            BENCHMARKS.index(row["benchmark"]),
        ),
    )


def build_report(
    monitor_config: Path = DEFAULT_MONITOR_CONFIG,
    initial_root: Path = DEFAULT_INITIAL_ROOT,
    offline_metric: str = "official",
) -> dict:
    if offline_metric not in ("official", "task-mean"):
        raise ValueError(f"Unknown offline metric: {offline_metric}")
    runs = multibenchmark_runs(monitor_config, offline_metric) + initial_runs(initial_root, offline_metric)
    runs.sort(
        key=lambda row: (
            METHODS.index(row["method"]),
            ("initial", "sequential", "interleaved").index(row["setting"]),
            BENCHMARKS.index(row["train_benchmark"]) if row["train_benchmark"] else -1,
            row["seed"],
        )
    )
    return {
        "generated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "online_definition": (
            "training rollout scores by benchmark: interleaved uses [1, checkpoint], "
            "other settings use (previous_checkpoint, checkpoint]; "
            "PREMiSE and ps_st_pareto use the higher observed parent score per task; "
            "AppWorld uses TGC (full-success tasks / tasks with recorded scores), not partial test pass rates"
        ),
        "offline_metric": offline_metric,
        "offline_definition": (
            "benchmark success rate of frozen checkpoint"
            if offline_metric == "official"
            else "mean test-task pass percentage of frozen checkpoint"
        ),
        "runs": runs,
        "online_aggregates": aggregate(runs, "online"),
        "offline_aggregates": aggregate(runs, "offline"),
    }


def percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def seed_cell(row: dict | None, metric: str) -> str:
    if row is None:
        return "—"
    if metric == "online":
        if row["score"] is None:
            return f"— [{row['scored']}/{row['planned']}]"
        suffix = (
            " partial"
            if not row["complete"]
            else f" skipped={row['planned'] - row['scored']}"
            if row["scored"] < row["planned"]
            else ""
        )
        return f"{percent(row['score'])} [{row['scored']}/{row['planned']}]{suffix}"
    if row["score"] is None:
        suffix = f" errors={row['errors']}" if row["errors"] else ""
        if row.get("task_mean_score") is not None and row["processed"] > 0:
            suffix += f" interim_task={percent(row['task_mean_score'])}"
        return f"— [{row['processed']}/{row['total']}]{suffix}"
    suffix = f" errors={row['errors']}" if row["errors"] else "" if row["clean_complete"] else " partial"
    return f"{percent(row['score'])} [{row['processed']}/{row['total']}]{suffix}"


def aggregate_cell(row: dict | None, metric: str) -> str:
    if row is None:
        return "—"
    if row["n"]:
        deviation = f" ± {100 * row['std']:.2f}" if row["std"] is not None else ""
        final = f"{100 * row['mean']:.2f}{deviation}% (n={row['n']}/3)"
    else:
        final = "—"
    if metric != "offline" or not row["interim_n"] or (
        row["interim_n"] == row["n"] and row["interim_errors"] == 0
        and row["interim_processed"] == row["interim_total"]
    ):
        return final
    deviation = f" ± {100 * row['interim_task_std']:.2f}" if row["interim_task_std"] is not None else ""
    interim = (f"중간 task 평균 {100 * row['interim_task_mean']:.2f}{deviation}% "
               f"(n={row['interim_n']}/3, task={row['interim_processed']}/{row['interim_total']}, "
               f"errors={row['interim_errors']})")
    return f"{final}; {interim}"


def render_table(report: dict, metric: str, include_pending: bool = False) -> list[str]:
    lookup = {(r["method"], r["setting"], r["train_benchmark"], r["seed"]): r for r in report["runs"]}
    totals = {
        (r["method"], r["setting"], r["train_benchmark"], r["checkpoint"], r["benchmark"]): r
        for r in report[f"{metric}_aggregates"]
    }
    keys = {
        (r["method"], r["setting"], r["train_benchmark"], x["checkpoint"], x["benchmark"])
        for r in report["runs"]
        for x in r[metric]
    }
    rows = [
        "| Method | Setting | Checkpoint | Test/rollout benchmark | Seed 42 | Seed 43 | Seed 44 | Mean ± SD |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for method, setting, target, point, benchmark in sorted(
        keys,
        key=lambda x: (
            METHODS.index(x[0]),
            ("initial", "sequential", "interleaved").index(x[1]),
            BENCHMARKS.index(x[2]) if x[2] else -1,
            x[3],
            BENCHMARKS.index(x[4]),
        ),
    ):
        cells = []
        items = []
        for seed in SEEDS:
            run = lookup.get((method, setting, target, seed))
            item = (
                next((v for v in run[metric] if v["checkpoint"] == point and v["benchmark"] == benchmark), None)
                if run
                else None
            )
            items.append(item)
            cells.append(seed_cell(item, metric))
        if not include_pending and all(
            item is None or (item["score"] is None and
                             (item["scored"] if metric == "online" else item["processed"]) == 0)
            for item in items
        ):
            continue
        label = f"initial/{target}" if target else setting
        aggregate_row = totals.get((method, setting, target, point, benchmark))
        rows.append(
            f"| {method} | {label} | {point} | {benchmark} | "
            + " | ".join(cells)
            + f" | {aggregate_cell(aggregate_row, metric)} |"
        )
    return rows


def render_markdown(report: dict, include_pending: bool = False) -> str:
    lines = [
        f"# Baseline experiment results — {report['generated_at']}",
        "",
        "Initial = benchmark-specific isolated training. Scores are percentages.",
        "",
        "## Training progress",
        "",
        "| Method | Setting | Seed | Status | Tasks | BFCL/AppWorld/Terminal | Skipped | "
        "Clean eval groups | Test tasks | Eval errors |",
        "|---|---|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        label = f"initial/{run['train_benchmark']}" if run["train_benchmark"] else run["setting"]
        counts = "/".join(str(run["counts"].get(benchmark, 0)) for benchmark in BENCHMARKS)
        evaluated = run["evaluation_progress"]
        lines.append(
            f"| {run['method']} | {label} | {run['seed']} | {run['status']} | "
            f"{run['cursor']}/{run['total']} | {counts} | {run['skipped']} | "
            f"{evaluated['clean_groups']}/{evaluated['groups']} | "
            f"{evaluated['processed_tasks']}/{evaluated['total_tasks']} | {evaluated['errors']} |"
        )
    lines.extend(
        [
            "",
            "## Online: training rollouts by checkpoint",
            "",
            "[scored/planned] counts benchmark rollouts: interleaved uses tasks 1 through the checkpoint; "
            "other settings use the interval since the previous checkpoint.",
            "PREMiSE and ps_st_pareto use the higher observed score of the two parents for each task.",
            "AppWorld online scores use TGC: full-success tasks divided by tasks with recorded scores.",
            "",
        ]
    )
    lines.extend(render_table(report, "online", include_pending))
    lines.extend(
        [
            "",
            f"## Offline: frozen checkpoint test ({report['offline_metric']})",
            "",
            "[processed/total] counts test tasks. Error-bearing and incomplete evaluations are excluded "
            "from the official Mean ± SD. Interim task means use only recorded tasks; "
            "execution errors have placeholder score zero.",
            "Every benchmark's frozen test evaluation is shown at each checkpoint, including interleaved runs. "
            "A dash with processed tasks and errors means no valid official score is available.",
            "",
        ]
    )
    lines.extend(render_table(report, "offline", include_pending))
    lines.extend(
        [
            "",
            "Mean ± SD uses sample standard deviation across completed seeds; n/3 gives coverage.",
            "Online aggregates use reached checkpoints and average scored rollouts in each setting's range; "
            "skipped rollouts are shown separately.",
            "An online cell marked partial has not reached its checkpoint yet.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor-config", type=Path, default=DEFAULT_MONITOR_CONFIG)
    parser.add_argument("--initial-root", type=Path, default=DEFAULT_INITIAL_ROOT)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument(
        "--offline-metric",
        choices=("official", "task-mean"),
        default="official",
        help="Official benchmark success rate, or mean test-task pass percentage",
    )
    parser.add_argument("--include-pending", action="store_true", help="Show score rows with no observations yet")
    parser.add_argument("--output", type=Path, help="Save report instead of printing to stdout")
    args = parser.parse_args()
    report = build_report(args.monitor_config.resolve(), args.initial_root.resolve(), args.offline_metric)
    value = (
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.format == "json"
        else render_markdown(report, args.include_pending)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(value, encoding="utf-8")
        print(args.output.resolve())
    else:
        print(value, end="")


if __name__ == "__main__":
    main()
