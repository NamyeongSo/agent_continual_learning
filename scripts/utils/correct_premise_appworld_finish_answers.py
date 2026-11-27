# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Audit the AppWorld score effect of omitting an unnecessary finish answer.

This writes a separate counterfactual artifact. Recorded evaluation summaries
and official scores are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ANSWER_REQUIREMENT = "assert answers match."
NULL_ANSWER = re.compile(r"==\s*\n'null'\s*$")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def unnecessary_finish_answer(tracker: dict[str, Any], events: list[dict[str, Any]]) -> bool:
    """Require both an official null target and an executed, nonempty finish answer."""
    answer_failures = [
        failure for failure in tracker.get("failures", [])
        if failure.get("requirement") == ANSWER_REQUIREMENT
    ]
    if len(answer_failures) != 1 or not NULL_ANSWER.search(str(answer_failures[0].get("trace", ""))):
        return False

    finishes = [
        event["action"] for event in events
        if event.get("event") == "action" and (event.get("action") or {}).get("name") == "finish"
    ]
    if not finishes:
        return False
    finish = finishes[-1]
    if not finish.get("validation", {}).get("valid"):
        return False
    if (finish.get("arguments") or {}).get("answer") in (None, ""):
        return False
    return any(
        event.get("event") == "observation"
        and any(
            action.get("id") == finish.get("id")
            for action in (event.get("observation") or {}).get("invoking_actions") or []
        )
        and (event.get("observation") or {}).get("result") == {"message": "Marked the active task complete."}
        for event in events
    )


def correct_group(run: Path, summary_path: Path) -> dict[str, Any]:
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    point = int(summary_path.stem[2:5])
    records = summary["results"]
    if not summary.get("complete") or any(record is None for record in records):
        return {"run": run.name, "checkpoint": point, "status": "incomplete_evaluation"}

    raw_full = sum(float(record["score"]) >= 1.0 - 1e-9 for record in records)
    result: dict[str, Any] = {
        "run": run.name,
        "checkpoint": point,
        "source": str(summary_path.resolve()),
        "source_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "tasks": len(records),
        "raw_full_successes": raw_full,
        "raw_full_success_rate": raw_full / len(records),
        "raw_mean_test_pass_rate": sum(float(record["score"]) for record in records) / len(records),
    }
    session_files = sorted(
        (run / "sessions").glob(f"*test_cp{point:03}_appworld_????_appworld/sessions/*/benchmark/scores.json")
    )
    if len(session_files) != len(records):
        result.update({"status": "unverifiable_sessions", "session_files": len(session_files)})
        return result

    scores_by_id = {}
    for path in session_files:
        scores = read_json(path)
        task_id = str(scores["task_id"])
        if task_id in scores_by_id:
            raise ValueError(f"Duplicate session for task {task_id}: {path}")
        scores_by_id[task_id] = (path, scores)

    corrections = []
    corrected_scores = []
    for record in records:
        task_id = str(record["task_id"])
        path, scores = scores_by_id[task_id]
        score = float(record["score"])
        if abs(score - float(scores["pass_percentage"]) / 100) > 1e-9:
            raise ValueError(f"Evaluation and tracker disagree for {task_id}: {path}")
        tracker = scores["test_tracker"]
        if len(tracker["passes"]) + len(tracker["failures"]) != int(scores["num_tests"]):
            raise ValueError(f"Incomplete test tracker for {task_id}: {path}")
        trajectory_path = path.parent.parent / "trajectory.jsonl"
        events = [json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines()]
        if unnecessary_finish_answer(tracker, events):
            corrected = round(100 * (int(scores["pass_count"]) + 1) / int(scores["num_tests"]), 1) / 100
            if corrected <= score or corrected > 1:
                raise ValueError(f"Invalid correction for {task_id}: {path}")
            corrections.append({
                "task_id": task_id,
                "raw_score": score,
                "corrected_score": corrected,
                "new_full_success": len(tracker["failures"]) == 1,
            })
            corrected_scores.append(corrected)
        else:
            corrected_scores.append(score)

    new_full = sum(item["new_full_success"] for item in corrections)
    result.update({
        "status": "audited",
        "corrected_sessions": len(corrections),
        "new_full_successes": new_full,
        "corrected_full_successes": raw_full + new_full,
        "corrected_full_success_rate": (raw_full + new_full) / len(records),
        "corrected_mean_test_pass_rate": sum(corrected_scores) / len(records),
        "corrections": corrections,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="PREMiSE experiment root containing s*_t3 runs")
    parser.add_argument("--output", type=Path, help="Separate correction artifact; defaults to root")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "appworld_unnecessary_finish_answer_correction.json"
    groups = [
        correct_group(run, path)
        for run in sorted(root.glob("s*_t3"))
        for path in sorted((run / "test_evaluation").glob("cp*_appworld.json"))
    ]
    payload = {
        "method": "omit only a nonempty answer from a valid, executed finish when the official answer is null",
        "kind": "counterfactual_audit_not_official_score",
        "groups": groups,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for group in groups:
        if group["status"] == "audited":
            print(
                f"{group['run']} cp{group['checkpoint']:03}: "
                f"{group['raw_full_successes']}/{group['tasks']} -> "
                f"{group['corrected_full_successes']}/{group['tasks']} "
                f"({group['corrected_sessions']} corrections)"
            )
        else:
            print(f"{group['run']} cp{group['checkpoint']:03}: {group['status']}")
    print(output)


if __name__ == "__main__":
    main()
