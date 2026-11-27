# SPDX-License-Identifier: Apache-2.0
"""Read registered standalone evaluation queues without changing their artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path


def native_workers():
    """Project only task IDs and output roots from the native worker processes."""
    result = {}
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / "cmdline").read_bytes().decode().split("\0")
            if "--task" not in args or "--output" not in args:
                continue
            state = (process / "stat").read_text().rpartition(")")[2].split()[0]
            if state in {"Z", "X"}:
                continue
            output = args[args.index("--output") + 1]
            task = args[args.index("--task") + 1]
            result.setdefault(output, []).append(task)
        except (OSError, ValueError, IndexError, UnicodeError):
            continue
    return result


def evaluations(base_dir, files, allowed, live_pid):
    result = []
    workers = native_workers()
    for marker in sorted((base_dir / "outputs").glob("*/evaluation_monitor.json")):
        root = marker.parent
        if not allowed(root):
            continue
        plan = files.read(marker, default={})
        config = files.read(root / "config.json", (
            "tasks", "workers", "model", "dataset_version", "agent", "retry_timeout_source", "retry_source",
        ), {})
        state = files.read(root / "status.json", default={})
        owner = files.read(root / plan.get("process_file", "controller_process.json"), ("pid",), {})
        phase = state.get("phase", "planned")
        alive = live_pid(owner.get("pid"), root)
        if phase not in {"completed", "infrastructure_errors", "planned"} and not alive:
            phase = "stopped"
        tasks = config.get("tasks", [])
        records = {}
        for task in tasks:
            benchmark = task.get("benchmark", "terminalbench2")
            path = root / "records"
            if plan.get("nested_records"):
                path /= benchmark
            path /= f"{task['task_id']}.json"
            records[benchmark, task["task_id"]] = files.read(
                path, ("status", "success", "attempt", "ended_at"), {},
            )
        errors = Counter()
        for path in (root / "errors").glob("**/*.json"):
            if not allowed(path):
                continue
            error = files.read(path, ("benchmark",), {})
            errors[error.get("benchmark", "terminalbench2")] += 1
        rows = []
        for benchmark in dict.fromkeys(task.get("benchmark", "terminalbench2") for task in tasks):
            splits = {}
            for split in ("test", "val", "train", "all"):
                selected = [task for task in tasks if task.get("benchmark", "terminalbench2") == benchmark
                            and (split == "all" or task["split"] == split)]
                outcomes = [records[benchmark, task["task_id"]] for task in selected]
                measured = [r for r in outcomes if r.get("status") == "completed"]
                failures = sum(r.get("status") == "infrastructure_error" for r in outcomes)
                successes = sum(bool(r.get("success")) for r in measured)
                splits[split] = {
                    "total": len(selected), "measured": len(measured), "successes": successes,
                    "failed": len(measured) - successes, "errors": failures,
                    "processed": len(measured) + failures,
                    "remaining": len(selected) - len(measured) - failures,
                    "score_percent": 100 * successes / len(measured) if measured else None,
                    "complete": bool(selected) and len(measured) == len(selected),
                }
            rows.append({"benchmark": benchmark, "splits": splits, "failed_attempts": errors[benchmark],
                         "retries": sum(max(0, int(records[benchmark, t['task_id']].get('attempt', 1)) - 1)
                                        for t in tasks if t.get("benchmark", "terminalbench2") == benchmark)})
        updated_at = max([state.get("at") or "", *[r.get("ended_at") or "" for r in records.values()]]) or None
        source_ids = set(plan.get("source_journal_ids", []))
        for key in ("retry_timeout_source", "retry_source"):
            if config.get(key):
                source = Path(config[key])
                if not source.is_absolute():
                    source = base_dir / source
                if allowed(source):
                    journal = files.read(source / "journal.json", ("id",), {})
                    if journal.get("id"):
                        source_ids.add(journal["id"])
        active_tasks = sorted(workers.get(str(root.resolve()), [])) if config.get("agent") == "terminus2" else None
        result.append({"id": root.name, "label": plan.get("label", root.name), "phase": phase,
                       "order": plan.get("order", 0), "workers": plan.get("workers", config.get("workers")),
                       "model": config.get("model"), "protocol": plan.get("protocol", ""),
                       "after_label": plan.get("after_label"), "journal_ids": plan.get("journal_ids", []),
                       "source_journal_ids": sorted(source_ids), "active_tasks": active_tasks,
                       "updated_at": updated_at, "controller_alive": alive, "rows": rows})
    return sorted(result, key=lambda row: row["order"])
