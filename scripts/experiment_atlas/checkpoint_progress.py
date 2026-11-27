# SPDX-License-Identifier: Apache-2.0
"""Read the current checkpoint queue without scanning historical model traces."""

from __future__ import annotations

import json
from pathlib import Path

QUEUE = Path(__file__).resolve().parents[2] / "outputs/qwen_shared_reflection_cp10_train_val_test_20261112"


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def snapshot(folder=None):
    root = Path(folder) if folder else QUEUE
    plan, data = read(root / "checkpoint_queue_manifest.json"), read(root / "dashboard.json")
    if not plan or not data:
        return {"available": False}
    rows = []
    for item in data.get("rows", []):
        rows.append({**{key: item.get(key) for key in (
            "method", "benchmark", "seed", "point", "alias", "state", "policy_hash")},
            "splits": {name: {key: value.get(key) for key in ("done", "total", "running", "errors", "score")}
                       for name, value in item.get("splits", {}).items() if name in {"train", "val", "test"}}})
    splits = {}
    for name in ("train", "val", "test"):
        groups = [row["splits"].get(name, {}) for row in rows if not row.get("alias")]
        splits[name] = {key: sum(group.get(key) or 0 for group in groups)
                        for key in ("done", "total", "running", "errors")}
    return {"available": True, "updated_at": data.get("updated_at"), "phase": data.get("phase"),
            "counts": data.get("task_counts", {}), "checkpoints": data.get("checkpoints"),
            "expected_checkpoints": plan.get("expected_checkpoints"), "aliases": data.get("skipped", 0),
            "reused_outcomes": plan.get("initial_reused_outcomes", 0), "splits": splits, "rows": rows,
            "gpu_indices": plan.get("serving_gpu_indices", []), "query_limits": plan.get("query_limits", {}),
            "gpu_queries": {key: value for key, value in data.get("gpu_queries", {}).items()
                            if key in {"active", "waiting", "per_endpoint_query_limits", "maximum", "errors"}},
            "journal": "http://166.104.110.49:8878/#experiment/" + plan.get("journal_id", ""),
            "live_monitor": "http://166.104.110.49:8878/#live"}
