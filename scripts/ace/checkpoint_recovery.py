# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Recover the completed training prefix from immutable per-session checkpoints."""

from __future__ import annotations

import json
from pathlib import Path


def session_records(output: Path) -> dict[tuple[str, str], dict]:
    records = {}
    for config_path in sorted(output.glob("*/sessions/*/config.json"), key=lambda p: p.stat().st_mtime_ns):
        session = config_path.parent
        try:
            config = json.loads(config_path.read_text())
            result = json.loads((session / "results.json").read_text())
            checkpoint_path = session / "agent/playbook_checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else None
        except (OSError, ValueError):
            continue
        key = (config["benchmark"], str(config["task_id"]))
        if checkpoint is None and result.get("status") not in ("error", "cancelled"):
            continue  # Results are written before the reflector and curator finish.
        records[key] = {
            "result": result, "checkpoint": checkpoint,
            "checkpoint_path": str(checkpoint_path) if checkpoint else None,
            "session_dir": str(session),
        }
    return records


def completed_prefix(output: Path, task_order: list, *, require_valid: bool = False) -> list[dict]:
    records = session_records(output)
    metrics_path = output / "online_metrics.jsonl"
    metrics = []
    if metrics_path.exists():
        lines = metrics_path.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except ValueError:
                if index == len(lines) - 1:
                    break  # A writer may still be appending its last record.
                raise
            if (index >= len(task_order)
                    or (row["benchmark_slug"], str(row["task_id"])) != tuple(task_order[index])
                    or row["session_index"] != index):
                raise ValueError(f"Completed metrics differ from the planned task prefix: {metrics_path}")
            metrics.append(row)
    prefix = []
    for index, (slug, task) in enumerate(task_order):
        record = records.get((slug, str(task)))
        if record is None and index < len(metrics):
            # Resumed tasks retain their metrics even when older run folders
            # are unavailable. These records prove completion, not a playbook state.
            record = {"result": metrics[index], "metrics": metrics[index],
                      "checkpoint": None, "checkpoint_path": None, "session_dir": None}
        if record is None:
            break
        if require_valid and record["result"].get("status") in ("error", "cancelled"):
            break
        prefix.append({"benchmark": slug, "task_id": str(task), **record})
    return prefix


def recover_checkpoint(output: Path, task_order: list, mode: str) -> list[dict]:
    prefix = completed_prefix(output, task_order, require_valid=True)
    if mode == "isolated":
        if len({slug for slug, _task in task_order}) != 1:
            raise ValueError("Isolated recovery requires one benchmark per run")
        from exgentic.agents.ace.playbook_store import DEFAULT_PLAYBOOK

        slug = task_order[0][0]
        store_id = f"ace_isolated_{slug}"
        latest = next((item["checkpoint"] for item in reversed(prefix) if item["checkpoint"]), None)
        if latest is None:
            latest = {"store_id": store_id, "playbook": DEFAULT_PLAYBOOK,
                      "next_global_id": 1, "session_count": 0, "benchmark_counts": {}}
        if latest["store_id"] != store_id:
            raise ValueError("Recovered checkpoint belongs to a different isolated playbook store")
        path = output / f"playbook_{store_id}.json"
        path.write_text(json.dumps({**latest, "completed_task_count": len(prefix)}, indent=2,
                                   ensure_ascii=False) + "\n")
        return prefix
    latest = next((item["checkpoint"] for item in reversed(prefix) if item["checkpoint"]), None)
    path = output / f"playbook_ace_{mode}_global.json"
    if latest is None and path.exists():
        existing = json.loads(path.read_text())
        if existing.get("completed_task_count") == len(prefix):
            latest = existing
    if latest is None:
        return []
    if latest["store_id"] != f"ace_{mode}_global":
        raise ValueError("Recovered checkpoint belongs to a different playbook store")
    payload = {**latest, "completed_task_count": len(prefix)}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return prefix
