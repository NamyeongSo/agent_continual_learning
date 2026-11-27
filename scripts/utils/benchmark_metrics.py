# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Project recorded task outcomes into the metrics used in experiment reports."""

from __future__ import annotations

import math

APPWORLD_METRIC = "appworld_tgc_v1"


def reported_score(benchmark: str, record: dict) -> float | None:
    """Use AppWorld's official success flag; missing evidence is not a failure.

    Raw AppWorld scores can be rounded partial test pass rates, including 1.0
    for an unsuccessful task. Never infer TGC by thresholding those scores.
    """
    if benchmark != "appworld":
        value = record.get("score")
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None
    if record.get("status") in {"error", "cancelled"}:
        return None
    details = record.get("details") or {}
    metadata = details.get("session_metadata") or record.get("session_metadata") or {}
    if (record.get("budget_failure") or record.get("experiment_budget_failure")
            or metadata.get("experiment_budget_exceeded") or metadata.get("experiment_budget_failure")):
        return 0.0
    evidence = (record, details, details.get("session_metrics") or {}, record.get("session_metrics") or {},
                metadata.get("test_tracker") or {}, record.get("test_tracker") or {})
    for source in evidence:
        success = source.get("success")
        if isinstance(success, bool):
            return float(success)
    if record.get("tgc") in (0, 1):
        return float(record["tgc"])
    # Legacy ACE metrics and frozen evaluations preserve the official status.
    for key in ("session_status", "status"):
        if record.get(key) in {"success", "unsuccessful"}:
            return float(record[key] == "success")
    if record.get("score_metric") == APPWORLD_METRIC and record.get("score") in (0, 1):
        return float(record["score"])
    return None
