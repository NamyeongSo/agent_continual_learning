# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Recover first-parent reporting evidence without replaying or editing a run."""

from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.utils.benchmark_metrics import reported_score


def read_inside(path, root):
    if not path.resolve().is_relative_to(root):
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def valid_outcome(record):
    score = record.get("score")
    return (type(score) in {int, float} and math.isfinite(score) and 0 <= score <= 1
            and type(record.get("success")) is bool)


def first_parent_outcome(folder, iteration, task_id, benchmark):
    """Require a unique exact parent1 session with a recorded official outcome.

    A later replay error can replace the iteration summary after parent1 finished.
    The only recoverable error session is the known empty-action model protocol
    failure, with the same separately recorded official proof used by the actor.
    Infrastructure errors and missing outcomes remain unmeasured.
    """
    if benchmark not in {"bfcl", "appworld", "terminalbench2"} or task_id is None:
        return None
    root = Path(folder).resolve()
    matches = []
    pattern = f"sessions/premise_*_i{iteration:04d}_parent1_{benchmark}/sessions/*/results.json"
    for path in root.glob(pattern):
        raw = read_inside(path, root)
        config = read_inside(path.parent / "config.json", root)
        if not (raw.get("task_id") == task_id == config.get("task_id")
                and config.get("benchmark") == benchmark
                and config.get("run_id") == path.parents[2].name
                and raw.get("session_id") == path.parent.name and valid_outcome(raw)):
            continue
        source = path
        if raw.get("status") in {"success", "unsuccessful", "completed"}:
            score = reported_score(benchmark, raw)
        else:
            metadata = (raw.get("details") or {}).get("session_metadata") or {}
            if not (benchmark in {"bfcl", "appworld"} and raw.get("status") == "error"
                    and raw["score"] == 0 and raw["success"] is False
                    and metadata.get("error_source") == "agent"
                    and metadata.get("error") == "LiteLLM completion missing assistant content."):
                continue
            name = "bfcl_score.json" if benchmark == "bfcl" else "results.json"
            source = path.parent / "benchmark" / name
            official = read_inside(source, root)
            if not valid_outcome(official):
                continue
            if benchmark == "bfcl" and (official["score"] != 0 or official["success"] is not False):
                continue
            if benchmark == "appworld" and not (
                official.get("session_id") == raw["session_id"] and official.get("task_id") == task_id
            ):
                continue
            score = reported_score(benchmark, official)
        if score is not None:
            matches.append({"score": score, "score_source": str(source.relative_to(root))})
    return matches[0] if len(matches) == 1 else None
