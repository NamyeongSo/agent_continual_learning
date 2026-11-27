# SPDX-License-Identifier: Apache-2.0
"""Explicit dense candidate rewards; official reporting scores remain independent."""

from __future__ import annotations

import json
import math
from pathlib import Path

METRICS = {"bfcl": "bfcl_turn_pass_rate_v1", "terminalbench2": "terminalbench_test_pass_rate_v1"}


def candidate_metric(trace, benchmark):
    if benchmark not in METRICS:
        raise ValueError("Auxiliary candidate metric supports BFCL and Terminal-Bench only")
    metadata = (trace.get("details") or {}).get("session_metadata") or {}
    if metadata.get("experiment_budget_exceeded") or metadata.get("experiment_budget_failure"):
        return {"metric": METRICS[benchmark], "score": 0.0, "source": "actor_budget_failure"}
    if benchmark == "bfcl":
        result = metadata.get("auxiliary_turn_metric")
        if not isinstance(result, dict) or result.get("metric") != METRICS[benchmark]:
            raise ValueError("Missing BFCL official turn-predicate evidence")
        value = result.get("score")
        if type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid BFCL turn pass rate")
        return result
    path = Path(trace["session_dir"]) / "benchmark/harbor/verifier/ctrf.json"
    data = json.loads(path.read_text())["results"]
    summary, tests = data["summary"], data["tests"]
    total = summary.get("tests")
    if (type(total) is not int or total < 1 or total != len(tests)
            or any(summary.get(key, 0) for key in ("pending", "other"))
            or any(row.get("status") not in {"passed", "failed", "skipped"} for row in tests)):
        raise ValueError("CTRF is missing, incomplete, or not a completed discrete test report")
    passed = sum(row["status"] == "passed" for row in tests)
    if passed != summary.get("passed"):
        raise ValueError("CTRF summary and individual outcomes disagree")
    return {"metric": METRICS[benchmark], "score": passed / total, "passed": passed, "total": total,
            "source": str(path), "definition": "Passed / all collected tests, including skipped tests in denominator"}
