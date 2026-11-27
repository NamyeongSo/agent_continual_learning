# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Read learning diagnostics without treating a skipped analysis as an actor failure."""

import json
import re
from collections import Counter
from pathlib import Path


def jsonl_records(path):
    """An active writer may not have finished its last line yet."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return records


def context_overflow(message):
    text = str(message).lower()
    return any(marker in text for marker in (
        "maximum context length", "context_length_exceeded", "context length exceeded",
        "contextwindowexceedederror", "contextlengthexceedederror",
    ))


def audit_training(job):
    """Count recorded events; SkillOpt analyst failures do not prove a whole update was skipped."""
    folder = Path(job["train_dir"])
    result = {
        "job_id": job["id"], "method": job["method"],
        "analysis_failed_attempts": 0, "analysis_context_failed_attempts": 0,
        "analysis_other_failed_attempts": 0, "gepa_skipped_batches": 0,
        "gepa_context_skipped_batches": 0, "premise_skipped_iterations": 0,
        "ace_context_skipped_tasks": 0, "sources": [],
    }
    if job["method"] in {"gepa", "skillopt"}:
        errors = jsonl_records(folder / "analysis_errors.jsonl")
        result["analysis_failed_attempts"] = len(errors)
        result["analysis_context_failed_attempts"] = sum(context_overflow(row.get("error")) for row in errors)
        result["analysis_other_failed_attempts"] = len(errors) - result["analysis_context_failed_attempts"]
        result["failed_attempts_by_stage"] = dict(Counter(row.get("stage", "unknown") for row in errors))
        if errors:
            result["sources"].append(str(folder / "analysis_errors.jsonl"))
        if job["method"] == "gepa":
            skipped = jsonl_records(folder / "skipped_updates.jsonl")
            result["gepa_skipped_batches"] = len({row["batch"] for row in skipped})
            result["gepa_context_skipped_batches"] = len({
                row["batch"] for row in skipped if context_overflow(row.get("error"))
            })
            if skipped:
                result["sources"].append(str(folder / "skipped_updates.jsonl"))
    elif job["method"] == "premise":
        for path in sorted((folder / "iterations").glob("*.json")):
            with path.open() as stream:
                prefix = stream.read(1024)
            if re.search(r'^\s*"status":\s*"skipped"', prefix, re.MULTILINE):
                result["premise_skipped_iterations"] += 1
                result["sources"].append(str(path))
    elif job["method"] == "ace":
        for path in sorted(folder.rglob("task_skip.json")):
            row = json.loads(path.read_text())
            if context_overflow(row.get("reason")):
                result["ace_context_skipped_tasks"] += 1
                result["sources"].append(str(path))
    return result


def has_diagnostics(row):
    return any(row[key] for key in (
        "analysis_failed_attempts", "gepa_skipped_batches", "premise_skipped_iterations", "ace_context_skipped_tasks",
    ))
