# SPDX-License-Identifier: Apache-2.0
"""Read-only status for the separately pinned shared-reflector experiment cohort."""

from __future__ import annotations

import hashlib
import json
import runpy
import threading
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

COHORT = Path(__file__).resolve().parents[2] / "outputs/qwen_shared_reflection_medium_fresh1_four_baselines_20261111"
LOCK = threading.Lock()
CACHE = None
CACHE_AT = 0


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def lines(path):
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return []


def snapshot():
    global CACHE, CACHE_AT
    with LOCK:
        if CACHE is not None and time.monotonic() - CACHE_AT < 15:
            return CACHE
        plan = read(COHORT / "launch_manifest.json")
        if plan is None:
            return {"available": False}
        progress = read(COHORT / "progress.json", {})
        jobs = []
        for job in plan["jobs"]:
            state = progress.get("jobs", {}).get(job["id"], {"status": "pending"})
            folder = Path(job["train_dir"])
            detail = read(Path(job["output"]) / "job_status.json", {})
            if job["method"] == "ace":
                trained = len(lines(folder / "online_metrics.jsonl"))
            elif job["method"] == "premise":
                trained = read(folder / "state.json", {}).get("cursor", 0)
            else:
                trained = len(list((folder / "online").glob("*.json")))
            initial_val = None
            if job["method"] in {"gepa", "skillopt"}:
                label = "val_evaluation_0001" if job["method"] == "gepa" else "validation"
                initial = []
                for request in (folder / "rollouts").glob("*/request.json"):
                    value = read(request, {})
                    if value.get("label") == label and value.get("learned_policy") == "":
                        initial.append(request.parent)
                initial_val = {
                    "requested": len(initial),
                    "completed": sum((path / "result.json").exists() for path in initial),
                    "total": len(job["val_ids"]),
                }
            anchor = (
                progress.get("optimizer_started_at", {}).get(f"{job['benchmark']}:{job['method']}")
                if job["method"] in {"gepa", "skillopt"}
                else progress.get("benchmark_started_at", {}).get(job["benchmark"])
            )
            jobs.append(
                {
                    **job,
                    "runtime": state,
                    "detail": detail,
                    "trained": trained,
                    "initial_validation": initial_val,
                    "release_at": state.get(
                        "release_at", anchor + job["release_delay_seconds"] if anchor is not None else None
                    ),
                }
            )
        requests = lines(COHORT / "serving/requests.jsonl")
        excluded = set()
        for path in (COHORT / "monitor/checkpoint_usage_exclusions").glob("*.json"):
            excluded.update(read(path, {}).get("response_ids", []))
        requests = [row for row in requests if row.get("id") not in excluded]
        usage = {}
        for role in ("actor", "analysis"):
            rows = [r for r in requests if r.get("role") == role]
            choices = [c for r in rows for c in r["choices"]]
            usage[role] = {
                "requests": len(rows),
                "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in rows),
                "completion_tokens": sum(r["usage"]["completion_tokens"] for r in rows),
                "reasoning_tokens": sum(c["reasoning_tokens"] for c in choices),
                "visible_tokens": sum(c["visible_tokens"] for c in choices),
                "temperature": sorted({r["temperature"] for r in rows}),
                "reasoning_effort": sorted({r["reasoning_effort"] for r in rows}),
                "reasoning_limit": sorted({r["reasoning_limit"] for r in rows}),
                "repetitions": sum(c["repetition_detected"] for c in choices),
                "length_finishes": sum(c["finish_reason"] == "length" for c in choices),
                "profile_matches": all(r["profile_sha256"] == plan["inference_profile_sha256"] for r in rows),
            }
        template = Path(plan["snapshot"]) / "src/exgentic/reflection.py"
        reflection = runpy.run_path(str(template))
        result = {
            "available": True,
            "checked_at": datetime.now(UTC).isoformat(),
            "journal": "http://166.104.110.49:8878/#experiment/exp-20261111-8497707d",
            "status": read(COHORT / "status.json", {}),
            "plan": {**plan, "jobs": jobs},
            "counts": dict(Counter(j["runtime"]["status"] for j in jobs)),
            "validation": read(COHORT / "prompt_validation_summary.json", {}),
            "calibration": read(COHORT / "calibration_summary.json", {}),
            "usage": usage,
            "usage_attribution": {"excluded_checkpoint_eval_requests": len(excluded),
                                  "basis": "Exact response IDs recorded by checkpoint SDK traces"},
            "prompt": {
                "system": reflection["SYSTEM"],
                "rules": reflection["RULES"]
                + "\n\n"
                + reflection["INSTRUCTION_AWARE"]
                + "\n\n"
                + reflection["TIMELINE_CAUSALITY"],
                "source_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
            },
        }
        CACHE, CACHE_AT = result, time.monotonic()
        return result
