# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Audit learning and evaluation independently; wait with a real sleep process between reports."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BENCHMARKS = ("bfcl", "appworld", "terminalbench2")
POINTS = (50, 100, 130)
KST = timezone(timedelta(hours=9))
KNOWN_FLAGS = (
    "--output-dir", "--output-root", "--output", "--launch-manifest", "--worker", "--worker-run", "--worker-job",
    "--execution-tag",
)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def processes() -> list[dict]:
    """Keep only known experiment flags; never record environments, keys, or full commands."""
    rows = []
    for directory in Path("/proc").glob("[0-9]*"):
        try:
            fields = (directory / "stat").read_text().rpartition(")")[2].split()
            if fields[0] in ("Z", "X"):
                continue
            argv = (directory / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
            scripts = [arg for arg in argv if arg.endswith(".py") and "/exgentic/scripts/" in arg]
            if not scripts:
                continue
            flags = {flag: argv[argv.index(flag) + 1] for flag in KNOWN_FLAGS if flag in argv}
            if "--retry-infrastructure-errors" in argv:
                flags["--retry-infrastructure-errors"] = True
            rows.append({"pid": int(directory.name), "script": Path(scripts[0]).name,
                         "flags": flags, "start_ticks": fields[19]})
        except (OSError, IndexError, UnicodeDecodeError):
            continue
    return rows


def find_workers(rows: list[dict], script: str, flag: str, target: str) -> list[dict]:
    return [row for row in rows if row["script"] == script and row["flags"].get(flag) == target]


def active_shared_controller(root: Path, run: Path, checkpoint: int,
                             *, proc_root: Path = Path("/proc")) -> dict | None:
    """Recognize a fresh PREMiSE controller launched from either scripts or outputs."""
    status_path = root / "restart_status.json"
    if time.time() - file_time(status_path) > 30:
        return None
    status = read_json(status_path, {})
    queue = status.get("queue", {})
    if not queue.get("running") and not queue.get("queued"):
        return None
    if not any(row.get("output") == str(run) and row.get("evaluations", {}).get(str(checkpoint)) == "running"
               for row in status.get("experiments", [])):
        return None
    pid = status.get("controller_pid")
    if not isinstance(pid, int) or pid < 1:
        return None
    try:
        directory = proc_root / str(pid)
        state = (directory / "stat").read_text().rsplit(")", 1)[1].split()[0]
        argv = (directory / "cmdline").read_bytes().decode().split("\0")
    except (OSError, IndexError, UnicodeDecodeError):
        return None
    scripts = {Path(arg).name for arg in argv if arg.endswith(".py")}
    if state in ("Z", "X") or not scripts.intersection({"restart_sequential.py", "resume_controller.py",
                                                           "watch_premise.py"}):
        return None
    if "watch_premise.py" in scripts and "--controller" not in argv:
        return None
    return {"pid": pid}


def error_kind(record: dict) -> str | None:
    if record.get("status") not in ("error", "cancelled"):
        return None
    metadata = (record.get("details") or {}).get("session_metadata") or {}
    message = str(record.get("error") or metadata.get("error") or "")
    if record.get("context_error") or "context_length_exceeded" in message or "maximum context length" in message:
        return "context"
    if "all predefined address pools" in message:
        return "docker_network_pool"
    if "Pool overlaps with other one" in message:
        return "docker_subnet_collision"
    if any(marker in message for marker in (
        "litellm.Timeout", "APITimeoutError", "is not accessible: TimeoutError(",
    )):
        return "provider_timeout"
    if (metadata.get("error_source") == "benchmark" and message == "timed out") or (
        "'error_source': 'benchmark'" in message and "'error': 'timed out'" in message
    ):
        return "runner_rpc_timeout"
    if "TaskTimeLimitError" in message or metadata.get("error_type") == "TimeoutError" or (
        "'error_source': 'execution'" in message
        and "'error_type': 'TimeoutError'" in message
        and "'error': ''" in message
    ):
        return "task_deadline"
    if "Venv service" in message:
        return "runner_service"
    return metadata.get("error_type") or "other_execution_error"


def file_time(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def log_errors(path: Path) -> list[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 65536))
            tail = handle.read().decode(errors="replace")
    except OSError:
        return []
    selected = [line[:400] for line in tail.splitlines()
                if re.match(r"^(?:[\w.]+(?:Error|Exception|Timeout):|Traceback|FAILED )", line)]
    return [re.sub(r"(?i)(password|api[_-]?key|secret)\s*[:=]\s*\S+", r"\1=<redacted>", line)
            for line in selected[-3:]]


def skipped_iterations(run: Path) -> dict:
    skipped = []
    for path in (run / "iterations").glob("iter_*.json"):
        try:
            with path.open(encoding="utf-8") as handle:
                prefix = handle.read(8192)
        except OSError:
            continue
        if re.search(r'^  "status": "skipped"', prefix, re.MULTILINE):
            task = re.search(r'^  "task_id": "([^"]+)"', prefix, re.MULTILINE)
            skipped.append({"iteration": int(path.stem.split("_")[-1]),
                            "task_id": task[1] if task else "unknown"})
    return {"count": len(skipped), "latest": sorted(skipped, key=lambda row: row["iteration"])[-5:]}


def iteration_activity(run: Path, cursor: int, previous: dict | None = None,
                       *, execution_tag: str | None = None) -> dict:
    """Measure progress inside a task; a live PID alone is not execution evidence."""
    iteration = cursor + 1
    selected = {}
    pattern = f"*_i{iteration:04d}_*"
    if execution_tag is not None:
        namespace_source = str(run) + (f"\n{execution_tag}" if execution_tag else "")
        namespace = hashlib.sha256(namespace_source.encode()).hexdigest()[:10]
        pattern = f"premise_{namespace}_i{iteration:04d}_*"
    for folder in (run / "sessions").glob(pattern):
        label = folder.name.split(f"_i{iteration:04d}_", 1)[-1]
        attempt = re.search(r"_try(\d+)_", label)
        job_key = re.sub(r"_try\d+(?=_)", "", label)
        for session in folder.glob("sessions/*"):
            priority = (int(attempt[1]) if attempt else 1, file_time(session / "config.json"))
            if job_key in selected and priority <= selected[job_key][0]:
                continue
            result = read_json(session / "results.json", {})
            manifest = read_json(session / "session.json", {})
            actor_dir = session / "benchmark/harbor/agent"
            episodes = read_json(actor_dir / "premise_episodes.json", {})
            trajectory = read_json(actor_dir / "trajectory.json", {})
            event_path = session / "trajectory.jsonl"
            try:
                with event_path.open("rb") as handle:
                    event_count = sum(1 for _line in handle)
            except OSError:
                event_count = 0
            call_logs = list(session.glob("*/litellm/trace.jsonl"))
            activity_paths = [event_path, session / "results.json", *call_logs,
                              actor_dir / "premise_episodes.json", actor_dir / "trajectory.json"]
            selected[job_key] = (priority, {
                "label": label, "kind": "parent" if label.startswith("parent") else "replay",
                "task_id": str(result.get("task_id", manifest.get("task_id", "unknown"))),
                "benchmark": next((slug for slug in BENCHMARKS if label.endswith("_" + slug)), "unknown"),
                "status": result.get("status", "running"), "finished": bool(result),
                "events": event_count, "llm_log_bytes": sum(path.stat().st_size for path in call_logs if path.exists()),
                "actor_turns": sum(step.get("source") == "agent" for step in trajectory.get("steps", [])),
                "episodes": len(episodes.get("episodes", [])), "prefix_length": episodes.get("prefix_length"),
                "updated_at_epoch": max((file_time(path) for path in activity_paths), default=0),
            })
    jobs = {key: row[1] for key, row in selected.items()}
    parents = [row for row in jobs.values() if row["kind"] == "parent"]
    replays = [row for row in jobs.values() if row["kind"] == "replay"]
    phase = ("replay_execution" if any(not row["finished"] for row in replays)
             else "alignment_or_selection" if replays
             else "parent_execution" if any(not row["finished"] for row in parents)
             else "reflection_or_mutation" if parents else "starting")
    old_jobs = previous.get("jobs", {}) if previous and previous.get("iteration") == iteration else {}
    changed = [key for key, row in jobs.items() if old_jobs and (
        key not in old_jobs or any(row[field] != old_jobs[key].get(field)
                                  for field in ("status", "events", "llm_log_bytes", "actor_turns", "episodes"))
    )]
    return {"iteration": iteration, "phase": phase, "parents_finished": sum(row["finished"] for row in parents),
            "parents_total": len(parents), "replays_finished": sum(row["finished"] for row in replays),
            "replays_total": len(replays), "changed_jobs": changed, "jobs": jobs,
            "updated_at_epoch": max((row["updated_at_epoch"] for row in jobs.values()), default=0)}


def group_progress(paths: list[Path], ids: list[str], *, identity: dict, checkpoint_hash: str | None) -> dict:
    """Deduplicate published copies and count live ACE session results before a summary exists."""
    candidates = [(file_time(path), path, read_json(path)) for path in paths if path.is_file()]
    candidates = [item for item in candidates if isinstance(item[2], dict)]
    summary_path, summary = (
        max(candidates, key=lambda item: (item[0], paths.index(item[1])))[1:]
        if candidates else (paths[0], {})
    )
    protocol_errors = []
    records = summary.get("results", summary.get("sessions", []))
    complete = bool(summary.get("complete", "sessions" in summary))
    if summary:
        actual = summary.get("identity", summary)
        for key, expected in identity.items():
            if actual.get(key) != expected:
                protocol_errors.append(f"{key} differs from the frozen test protocol")
        if checkpoint_hash and actual.get("checkpoint_sha256") != checkpoint_hash:
            protocol_errors.append("checkpoint hash differs")
        if len(records) != len(ids):
            protocol_errors.append("result array size differs from test size")
    else:
        # ACE writes one small results.json per completed test task.
        records = []
        for path in summary_path.parent.glob("*/sessions/*/results.json"):
            row = read_json(path, {})
            if row.get("task_id") is not None:
                records.append(row)
    by_id = {str(row["task_id"]): row for row in records if row and str(row.get("task_id")) in ids}
    if len(by_id) != sum(row is not None for row in records):
        protocol_errors.append("duplicate or unexpected test task IDs")
    if complete and set(by_id) != set(ids):
        protocol_errors.append("complete result is missing test tasks")
    # Older ACE summaries retain status and score but omit exception metadata.
    # Recover the error classification from matching session artifacts without changing recorded outcomes.
    search_paths = [summary_path, *[path for path in paths if path != summary_path]]
    for task, row in list(by_id.items()):
        if error_kind(row) != "other_execution_error" or not row.get("session_id"):
            continue
        for path in search_paths:
            matches = path.parent.glob(f"*/sessions/{row['session_id']}/results.json")
            details = next((raw for file in matches if (raw := read_json(file, {})).get("task_id") == task
                            and raw.get("status") == row["status"] and raw.get("score") == row.get("score")), None)
            if details:
                by_id[task] = {**row, "details": details.get("details", {})}
                break
    kinds = Counter(kind for row in by_id.values() if (kind := error_kind(row)))
    errors = sum(kinds.values())
    observed_score = summary.get("mean_official_score")
    if observed_score is None:
        observed_score = summary.get("mean_task_score")
    if observed_score is None and by_id:
        # Before a summary exists, report the mean of recorded tasks only.
        # Execution errors have placeholder score zero; pending tasks are not in the denominator.
        observed_score = sum(float(row.get("score") or 0) for row in by_id.values()) / len(by_id)
    return {"complete": complete and not protocol_errors, "processed": len(by_id), "total": len(ids),
            "valid_results": len(by_id) - errors, "errors": errors, "error_kinds": dict(kinds),
            "error_task_ids": [task for task, row in by_id.items() if error_kind(row)],
            "score": observed_score,
            "clean_complete": complete and not protocol_errors and errors == 0,
            "protocol_errors": protocol_errors, "result_path": str(summary_path),
            "updated_at_epoch": max((item[0] for item in candidates), default=0)}


def collect(config: dict, previous: dict | None = None) -> dict:
    now = time.time()
    live = processes()
    training, evaluations, controllers, alerts = [], [], [], []
    early = {}
    for value in config.get("early_evaluation_roots", []):
        root = resolve(value)
        plan = read_json(root / "evaluation_plan.json", {})
        status = read_json(root / "status.json", {})
        control = find_workers(live, "evaluate_bfcl_checkpoints.py", "--output", str(root))
        controllers.append({"kind": "early_checkpoint_evaluation", "root": str(root),
                            "phase": status.get("phase"), "pids": [p["pid"] for p in control]})
        for job in plan.get("jobs", []):
            early[str(resolve(job["source_run"]))] = job
    for value in config["learning_roots"]:
        root = resolve(value)
        plan = read_json(root / "launch_manifest.json", {})
        if not plan:
            alerts.append({"severity": "error", "kind": "missing_manifest", "root": str(root)})
            continue
        method = plan.get("method", plan.get("agent", "unknown")).lower()
        split = read_json(resolve(plan["source_manifest"]), {})
        split_hash = hashlib.sha256(resolve(plan["source_manifest"]).read_bytes()).hexdigest()
        if split_hash != plan.get("source_manifest_sha256"):
            alerts.append({"severity": "error", "kind": "split_changed", "root": str(root)})
        status = read_json(root / "matrix_status.json", {})
        statuses = {row["name"]: row for row in status.get("experiments", [])}
        matrix = find_workers(live, "run_matrix.py", "--output-root", str(root))
        controllers.append({"kind": "learning_matrix", "root": str(root), "pids": [p["pid"] for p in matrix]})
        ace_evaluators = find_workers(live, "evaluate_test_checkpoints.py", "--launch-manifest",
                                     str(root / "launch_manifest.json"))
        for job in plan["jobs"]:
            name = job.get("name", f"{job['mode']}_s{job['seed']}")
            run = resolve(job.get("output_dir", str(root / name)))
            state = read_json(run / "state.json", {})
            order = job["task_order"]
            done_flag = read_json(run / "training_complete.json", {})
            cursor = len(order) if done_flag else int(state.get("cursor", 0))
            workers = find_workers(live, "run_experiment.py", "--output-dir", str(run))
            row_status = statuses.get(name, {})
            label = "trained" if cursor == len(order) else "running" if workers else row_status.get("status", "stopped")
            if label == "running" and not workers:
                label = "stopped"
            if cursor < len(order) and label in ("stopped", "failed", "cancelled"):
                alerts.append({"severity": "error", "kind": "learning_stopped", "method": method, "run": name})
            if len(workers) > 1:
                alerts.append({"severity": "error", "kind": "duplicate_learning_workers", "run": name})
            if cursor > len(order):
                alerts.append({"severity": "error", "kind": "cursor_out_of_range", "run": name})
            log = resolve(job.get("log", str(root / "logs" / f"{name}.log")))
            key = f"{method}:{name}"
            old = next((row for row in (previous or {}).get("learning", []) if row["key"] == key), {})
            execution_tag = workers[0]["flags"].get("--execution-tag") if workers else None
            internal = iteration_activity(run, cursor, old.get("activity"),
                                          execution_tag=execution_tag) if method != "ace" else None
            activity = max(file_time(run / "state.json"), file_time(log), (internal or {}).get("updated_at_epoch", 0))
            idle_minutes = (now - activity) / 60 if activity else None
            if workers and idle_minutes is not None and idle_minutes > config.get("inactivity_minutes", 90):
                alerts.append({"severity": "warning", "kind": "no_recent_activity", "run": name,
                               "minutes": round(idle_minutes, 1)})
            instances = [{"pid": worker["pid"], "start_ticks": worker["start_ticks"]} for worker in workers]
            restarted = bool(old.get("pids") and workers and (
                instances != old["worker_instances"] if old.get("worker_instances")
                else [worker["pid"] for worker in workers] != old["pids"]
            ))
            if restarted:
                alerts.append({"severity": "warning", "kind": "learning_worker_restarted", "run": name,
                               "previous_pids": old["pids"], "current_pids": [worker["pid"] for worker in workers],
                               "cursor_before": old.get("cursor"), "cursor_now": cursor})
            train_row = {"key": key, "method": method, "name": name, "mode": job["mode"], "seed": job["seed"],
                         "status": label, "cursor": cursor, "total": len(order),
                         "counts": dict(Counter(slug for slug, _task in order[:cursor])),
                         "next_task": order[cursor] if cursor < len(order) else None,
                         "pids": [worker["pid"] for worker in workers],
                         "worker_instances": instances, "restarted_since_previous": restarted,
                         "delta_tasks": cursor - old.get("cursor", cursor),
                         "attempts": row_status.get("attempts"), "inactive_minutes": idle_minutes,
                         "skipped_iterations": skipped_iterations(run) if method != "ace" else None,
                         "log": str(log), "recent_exception_lines": log_errors(log)}
            train_row["activity"] = internal
            training.append(train_row)
            for point in job.get("evaluation_checkpoints", POINTS):
                if method == "ace":
                    checkpoint = root / "test_evaluation/snapshots" / name / f"checkpoint_{point:03d}.json"
                else:
                    checkpoint = run / "checkpoints" / f"checkpoint_{point:03d}.json"
                checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint.is_file() else None
                for slug in BENCHMARKS:
                    ids = split[f"{slug}_{job['seed']}"]["test"]
                    if method == "ace":
                        paths = [root / "test_evaluation" / name / f"checkpoint_{point:03d}" / slug / "summary.json"]
                        if slug == "terminalbench2":
                            paths.append(root / "test_evaluation_terminalbench2_retry" / name /
                                         f"checkpoint_{point:03d}" / slug / "summary.json")
                        identity = {"stage": point, "benchmark": slug, "seed": job["seed"], "mode": job["mode"],
                                    "task_ids": ids, "evaluation_mode": True, "training_time": False,
                                    "manifest_sha256": split_hash}
                    else:
                        paths = [run / "test_evaluation" / f"cp{point:03d}_{slug}.json"]
                        if point == 50 and str(run) in early:
                            paths.append(resolve(early[str(run)]["evaluation_output"]) / "test_evaluation" /
                                         f"cp{point:03d}_{slug}.json")
                        identity = {"checkpoint": point, "benchmark": slug, "task_ids": ids, "beam_rank": 1,
                                    "learning": False, "manifest_sha256": split_hash}
                    group = group_progress(paths, ids, identity=identity, checkpoint_hash=checkpoint_hash)
                    group.update({"key": f"{key}:cp{point}:{slug}", "method": method, "run": name,
                                  "seed": job["seed"], "mode": job["mode"], "checkpoint": point,
                                  "benchmark": slug, "checkpoint_ready": checkpoint_hash is not None})
                    if method == "ace":
                        owners = [p for p in ace_evaluators if p["flags"].get("--worker-run") == name]
                        if slug == "terminalbench2":
                            owners += [p for p in live if p["script"] == "retry_terminalbench2_test.py"
                                       and p["flags"].get("--worker-run") == name
                                       and p["flags"].get("--launch-manifest") == str(root / "launch_manifest.json")]
                    else:
                        owners = workers if cursor == len(order) else []
                        if job.get("evaluation_controller_root") and checkpoint_hash and not group["complete"]:
                            controller_root = resolve(job["evaluation_controller_root"])
                            owners += find_workers(live, "restart_sequential.py", "--output-root",
                                                   str(controller_root))
                            shared_controller = active_shared_controller(controller_root, run, point)
                            if shared_controller:
                                owners.append(shared_controller)
                    if point == 50 and str(run) in early:
                        owners += [p for p in live if p["script"] == "evaluate_bfcl_checkpoints.py"
                                   and p["flags"].get("--worker") == early[str(run)]["name"]]
                    group["owner_pids"] = sorted({p["pid"] for p in owners})
                    old_group = next((g for g in (previous or {}).get("checkpoint_evaluations", [])
                                      if g["key"] == group["key"]), {})
                    group["delta_processed"] = group["processed"] - old_group.get("processed", group["processed"])
                    evaluations.append(group)
                    if group["protocol_errors"]:
                        alerts.append({"severity": "error", "kind": "evaluation_protocol", "group": group["key"],
                                       "errors": group["protocol_errors"]})
                    if group["errors"]:
                        alerts.append({"severity": "warning", "kind": "evaluation_execution_errors",
                                       "group": group["key"], "count": group["errors"], "types": group["error_kinds"]})
                    if cursor == len(order) and not group["complete"] and not owners:
                        alerts.append({"severity": "error", "kind": "evaluation_unowned", "group": group["key"]})
    milestones = []
    from scripts.ace import run_swemilestone_isolated as milestone

    for value in config.get("milestone_evaluation_roots", []):
        root = resolve(value)
        status = read_json(root / "status.json", {})
        milestone_workers = find_workers(live, "run_swemilestone_isolated.py", "--output-root", str(root))
        milestone_pids = {worker["pid"] for worker in milestone_workers}
        controllers.append({"kind": "milestone_evaluation", "root": str(root), "phase": status.get("status"),
                            "pids": sorted(milestone_pids)})
        plan = read_json(root / "launch_manifest.json", {})
        source_jobs = {job["name"]: job for job in plan.get("jobs", [])}
        recovery = read_json(root / "recovery_status.json", {})
        recovery_index = read_json(root / "recovery_index.json", {}).get("jobs", {})
        for row in status.get("jobs", []):
            try:
                result = milestone.read_job_result(plan, root, source_jobs.get(row["name"], row)) or {}
            except (OSError, ValueError, KeyError, StopIteration) as exc:
                alerts.append({"severity": "error", "kind": "milestone_recovery_protocol",
                               "job": row["name"], "error": str(exc)[:400]})
                result = {}
            effective_root = root
            effective_row = row
            if row["name"] in recovery_index:
                effective_root = resolve(recovery_index[row["name"]]["result_path"]).parents[1]
            if recovery.get("phase") == "running" and row["name"] in recovery.get("jobs", []):
                effective_root = resolve(recovery["output_root"])
                effective_row = next((job for job in read_json(effective_root / "status.json", {}).get("jobs", [])
                                      if job["name"] == row["name"]), row)
            workers = find_workers(live, "run_swemilestone_isolated.py", "--output-root", str(effective_root))
            progress = read_json(effective_root / row["name"] / "progress.json", effective_row.get("progress", {}))
            phase = "done" if result.get("valid_for_statistics") is True else effective_row["status"]
            if result.get("valid_for_statistics") is False and phase != "running":
                phase = "failed"
            milestones.append({"name": row["name"], "status": phase, "progress": progress,
                               "original_status": row["status"], "recovery_state": recovery.get("phase"),
                               "result_root": str(effective_root), "pid": effective_row.get("pid"),
                               "worker_alive": effective_row.get("pid") in {p["pid"] for p in workers},
                               "valid_for_statistics": result.get("valid_for_statistics"), "score": result.get("score"),
                               "infrastructure_errors": result.get("infrastructure_errors", [])})
            if phase == "running" and effective_row.get("pid") not in {p["pid"] for p in workers}:
                alerts.append({"severity": "error", "kind": "milestone_worker_stopped", "job": row["name"]})
            if phase == "failed" or result.get("valid_for_statistics") is False:
                alerts.append({"severity": "error", "kind": "milestone_evaluation_failed", "job": row["name"]})
    model_url = config.get("model_health_url")
    service = {"url": model_url, "healthy": None}
    if model_url:
        try:
            with urllib.request.urlopen(model_url, timeout=5) as response:
                service["healthy"] = response.status == 200
        except (OSError, urllib.error.URLError):
            service["healthy"] = False
            alerts.append({"severity": "error", "kind": "model_unreachable"})
    totals = {"learning_runs": len(training), "trained": sum(row["status"] == "trained" for row in training),
              "learning_active": sum(row["status"] == "running" for row in training),
              "evaluation_groups": len(evaluations), "processed_groups": sum(g["complete"] for g in evaluations),
              "clean_groups": sum(g["clean_complete"] for g in evaluations),
              "processed_test_tasks": sum(g["processed"] for g in evaluations),
              "valid_test_results": sum(g["valid_results"] for g in evaluations),
              "planned_test_tasks": sum(g["total"] for g in evaluations),
              "execution_errors": sum(g["errors"] for g in evaluations)}
    return {"checked_at": datetime.now(KST).isoformat(timespec="seconds"), "checked_at_epoch": now,
            "totals": totals, "learning": training, "checkpoint_evaluations": evaluations,
            "milestone_evaluations": milestones, "controllers": controllers,
            "model_service": service, "alerts": alerts}


def render(snapshot: dict) -> str:
    totals = snapshot["totals"]
    lines = [f"# Experiment audit — {snapshot['checked_at']}", "", "## Learning", "",
             f"Active {totals['learning_active']}; trained {totals['trained']}/{totals['learning_runs']}.", "",
             "| Method | Setting | Seed | State | Tasks | BFCL/AppWorld/Terminal | Change | Next task |",
             "|---|---|---:|---|---:|---|---:|---|"]
    for row in snapshot["learning"]:
        counts = "/".join(str(row["counts"].get(slug, 0)) for slug in BENCHMARKS)
        next_task = "/".join(row["next_task"]) if row["next_task"] else "—"
        lines.append(f"| {row['method']} | {row['mode']} | {row['seed']} | {row['status']} | "
                     f"{row['cursor']}/{row['total']} | {counts} | {row['delta_tasks']:+d} | {next_task} |")
    restarted = [row["key"] for row in snapshot["learning"] if row.get("restarted_since_previous")]
    if restarted:
        lines += ["", "Workers restarted since the previous audit: " + ", ".join(restarted) + "."]
    lines += ["", "## Progress inside the current training task", "",
              "Session completion includes officially unsuccessful results; retries count as one replay job.", "",
              "| Method | Setting | Seed | Iteration | Phase | Parents finished | Replays finished | Jobs changed |",
              "|---|---|---:|---:|---|---:|---:|---:|"]
    for row in snapshot["learning"]:
        activity = row.get("activity")
        if not activity or row["status"] != "running":
            continue
        lines.append(f"| {row['method']} | {row['mode']} | {row['seed']} | {activity['iteration']} | "
                     f"{activity['phase']} | {activity['parents_finished']}/{activity['parents_total']} | "
                     f"{activity['replays_finished']}/{activity['replays_total']} | "
                     f"{len(activity['changed_jobs'])} |")
    lines += ["", "## Frozen checkpoint evaluation", "",
              f"Processed groups {totals['processed_groups']}/{totals['evaluation_groups']}; "
              f"error-free groups {totals['clean_groups']}. "
              f"Tasks {totals['processed_test_tasks']}/{totals['planned_test_tasks']}; "
              f"valid results {totals['valid_test_results']}; "
              f"execution errors {totals['execution_errors']}.", "",
              "Published copies and early evaluations count once. "
              "Completed official zero scores are valid results.", "",
              "| Method | Setting | Seed | Checkpoint | BFCL | AppWorld | Terminal | Clean groups |",
              "|---|---|---:|---:|---|---|---|---:|"]
    for row in snapshot["learning"]:
        for point in POINTS:
            groups = [g for g in snapshot["checkpoint_evaluations"]
                      if g["key"].startswith(row["key"] + ":") and g["checkpoint"] == point]
            cells = [f"{g['processed']}/{g['total']} (errors {g['errors']})" for g in groups]
            lines.append(f"| {row['method']} | {row['mode']} | {row['seed']} | {point} | "
                         + " | ".join(cells) + f" | {sum(g['clean_complete'] for g in groups)}/3 |")
    if snapshot["milestone_evaluations"]:
        lines += ["", "## SWE-Milestone baseline evaluation", ""]
        for row in snapshot["milestone_evaluations"]:
            progress = row["progress"]
            lines.append(f"- {row['name']}: {row['status']}, "
                         f"{progress.get('completed_sessions', 0)}/{progress.get('planned_sessions', '?')}, "
                         f"valid_for_statistics={row['valid_for_statistics']}")
    lines += ["", "## Alerts", ""]
    lines += [f"- {json.dumps(alert, ensure_ascii=False)}" for alert in snapshot["alerts"]] or ["- None."]
    return "\n".join(lines) + "\n"


def launch_known_error_retries(config: dict, output: Path) -> list[dict]:
    """Reuse frozen evaluation workers, with bounded retries and disjoint service ports."""
    live = processes()
    registry_path = output / "repair_workers.json"
    registry = read_json(registry_path, {})
    started = []
    environment = os.environ.copy()
    environment_path = config.get("worker_environment_file")
    if environment_path:
        for key, value in read_json(resolve(environment_path), {}).items():
            environment[key] = value if isinstance(value, str) else json.dumps(value)
    environment["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + environment.get("PATH", "")
    environment["PYTHONPATH"] = str(ROOT / "src")
    for value in config.get("early_evaluation_roots", []):
        root = resolve(value)
        plan = read_json(root / "evaluation_plan.json", {})
        for index, job in enumerate(j for j in plan.get("jobs", []) if j["method"] != "ace"):
            key = f"{root}:{job['name']}"
            if any(p["script"] == "evaluate_bfcl_checkpoints.py"
                   and p["flags"].get("--output") == str(root)
                   and p["flags"].get("--worker") == job["name"] for p in live):
                continue
            affected = []
            for slug in BENCHMARKS:
                path = resolve(job["evaluation_output"]) / "test_evaluation" / f"cp050_{slug}.json"
                summary = read_json(path, {})
                affected += [{"benchmark": slug, "task_id": record["task_id"]}
                             for record in summary.get("results", [])
                             if record and error_kind(record) in (
                                 "docker_network_pool", "docker_subnet_collision", "provider_timeout",
                                 "runner_rpc_timeout",
                             )]
            if not affected or registry.get(key, {}).get("attempts", 0) >= config.get("max_repair_attempts", 3):
                continue
            port = config.get("repair_port_base", 35000) + 1000 * index
            command = [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "scripts/utils/evaluate_bfcl_checkpoints.py"),
                       "--output", str(root), "--worker", job["name"], "--retry-infrastructure-errors"]
            log = output / "repairs" / f"{job['name']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as handle:
                process = subprocess.Popen(command, cwd=ROOT,
                                           env=environment | {"EXGENTIC_PORT_BLOCK_START": str(port)},
                                           stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            entry = {"pid": process.pid, "attempts": registry.get(key, {}).get("attempts", 0) + 1,
                     "affected_tasks": affected, "port_block_start": port, "log": str(log),
                     "started_at": datetime.now(KST).isoformat(timespec="seconds")}
            registry[key] = entry
            started.append({"job": job["name"], **entry})
    for value in config["learning_roots"]:
        root = resolve(value)
        plan = read_json(root / "launch_manifest.json", {})
        if plan.get("agent") != "ace":
            continue
        for index, job in enumerate(plan["jobs"]):
            name = f"{job['mode']}_s{job['seed']}"
            key = f"ace_recovery:{root}:{name}"
            if any(p["script"] == "retry_terminalbench2_test.py" and p["flags"].get("--worker-run") == name
                   and p["flags"].get("--launch-manifest") == str(root / "launch_manifest.json") for p in live):
                continue
            recovery = root / "test_evaluation_terminalbench2_retry"
            if (recovery / name / "status.json").exists():
                status = read_json(recovery / name / "status.json", {})
                if status.get("phase") == "complete":
                    continue
            if registry.get(key, {}).get("attempts", 0) >= config.get("max_repair_attempts", 3):
                continue
            ports = config.get("ace_recovery_ports", [28000, 29000, 41000, 58000, 59000, 60000])
            command = [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "scripts/ace/retry_terminalbench2_test.py"),
                       "--launch-manifest", str(root / "launch_manifest.json"), "--worker-run", name,
                       "--wait-for-summary"]
            log = output / "repairs" / f"ace_{name}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as handle:
                process = subprocess.Popen(command, cwd=ROOT,
                                           env=environment | {"EXGENTIC_PORT_BLOCK_START": str(ports[index])},
                                           stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            entry = {"pid": process.pid, "attempts": registry.get(key, {}).get("attempts", 0) + 1,
                     "port_block_start": ports[index], "log": str(log),
                     "started_at": datetime.now(KST).isoformat(timespec="seconds")}
            registry[key] = entry
            started.append({"job": f"ace_{name}", **entry})
    from scripts.ace import run_swemilestone_isolated as milestone

    for value in config.get("milestone_evaluation_roots", []):
        root = resolve(value)
        if any(p["script"] == "run_swemilestone_isolated.py" and p["flags"].get("--output-root") == str(root)
               and p["flags"].get("--retry-infrastructure-errors") for p in live):
            continue
        plan = read_json(root / "launch_manifest.json", {})
        affected = [job["name"] for job in plan.get("jobs", [])
                    if milestone.retryable_repository_result(milestone.read_job_result(plan, root, job))]
        key = f"milestone_recovery:{root}"
        if not affected or registry.get(key, {}).get("attempts", 0) >= config.get("max_repair_attempts", 3):
            continue
        command = [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "scripts/ace/run_swemilestone_isolated.py"),
                   "--output-root", str(root), "--retry-infrastructure-errors"]
        log = output / "repairs" / "swemilestone_recovery.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        entry = {"pid": process.pid, "attempts": registry.get(key, {}).get("attempts", 0) + 1,
                 "affected_jobs": affected, "log": str(log),
                 "started_at": datetime.now(KST).isoformat(timespec="seconds")}
        registry[key] = entry
        started.append({"job": "swemilestone_recovery", **entry})
    save_json(registry_path, registry)
    return started


def write_report(config: dict, output: Path) -> dict:
    snapshot = collect(config, read_json(output / "latest.json", {}))
    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    save_json(output / "snapshots" / f"{stamp}.json", snapshot)
    save_json(output / "latest.json", snapshot)
    markdown = render(snapshot)
    (output / "latest.md").write_text(markdown, encoding="utf-8")
    (output / "snapshots" / f"{stamp}.md").write_text(markdown, encoding="utf-8")
    with (output / "history.jsonl").open("a", encoding="utf-8") as history:
        history.write(json.dumps({"checked_at": snapshot["checked_at"], **snapshot["totals"],
                                  "alerts": snapshot["alerts"]}, ensure_ascii=False) + "\n")
    print(json.dumps({"checked_at": snapshot["checked_at"], **snapshot["totals"],
                      "alerts": len(snapshot["alerts"]), "report": str(output / "latest.md")}), flush=True)
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=1200)
    parser.add_argument("--repair-known-infrastructure", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("interval must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        timer = None
        stopped = False

        def stop(_signum: int, _frame: Any) -> None:
            nonlocal stopped
            stopped = True
            if timer is not None and timer.poll() is None:
                timer.terminate()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while not stopped:
            config = read_json(args.config)
            if not isinstance(config, dict):
                raise ValueError("Monitor config must be a JSON object")
            snapshot = write_report(config, output)
            if args.repair_known_infrastructure and snapshot["model_service"]["healthy"]:
                started = launch_known_error_retries(config, output)
                if started:
                    print(json.dumps({"started_infrastructure_retries": [
                        {key: value for key, value in row.items() if key != "affected_tasks"}
                        for row in started
                    ]}, ensure_ascii=False), flush=True)
            if not args.watch:
                break
            save_json(output / "monitor.json", {"pid": os.getpid(), "phase": "waiting",
                                               "interval_seconds": args.interval,
                                               "next_check_at_epoch": time.time() + args.interval})
            # During this wait, the monitor performs no experiment inspection,
            # file polling, model requests, or repairs.
            timer = subprocess.Popen(["sleep", str(args.interval)])
            print(f"Waiting: sleep {args.interval} pid={timer.pid}", flush=True)
            timer.wait()
            timer = None


if __name__ == "__main__":
    main()
