# SPDX-License-Identifier: Apache-2.0
"""Officially evaluate a saved AppWorld turn-40 state without generating or replaying actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def rescore(source_result, output, appworld_root):
    source = json.loads(source_result.read_text())
    if (source.get("budget_failure") or {}).get("reason") != "max_turns":
        raise ValueError("Only original max-turn terminations may be rescored")
    saved = Path(source["session_dir"]) / "benchmark/task_output"
    before = hashes(saved)
    if not before or not (saved / "dbs").is_dir():
        raise ValueError("Missing saved final task state")
    local_root = output / "appworld_root"
    local_root.mkdir(parents=True, exist_ok=True)
    data = local_root / "data"
    if not data.exists():
        data.symlink_to(appworld_root / "data", target_is_directory=True)
    experiment = "saved_turn40"
    target = local_root / "experiments/outputs" / experiment / "tasks" / source["task_id"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copytree(saved, target)
    for relative, expected in before.items():
        if hashlib.sha256((target / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Copied final state differs from the saved actor state")

    from appworld import update_root
    from appworld.evaluator import evaluate_task

    update_root(str(local_root.resolve()))
    tracker = evaluate_task(task_id=source["task_id"], experiment_name=experiment,
                            suppress_errors=True, save_report=True)
    evaluation = tracker.to_dict(stats_only=False)
    passed, failed, total = len(evaluation["passes"]), len(evaluation["failures"]), evaluation["num_tests"]
    if total <= 0 or passed + failed != total:
        raise ValueError("Official evaluation did not produce complete test outcomes")
    if before != hashes(saved):
        raise ValueError("Original saved state was modified")
    result = {
        "task_id": source["task_id"], "source_repeat": source["source_repeat"], "source_id": source["source_id"],
        "pct": passed / total, "success": bool(evaluation["success"]),
        "passed_tests": passed, "total_tests": total, "difficulty": evaluation.get("difficulty"),
        "evaluation": evaluation, "source_result": str(source_result.resolve()),
        "source_task_output": str(saved), "source_state_sha256": before,
        "original_pct": source["pct"], "turn_cap_details": source["budget_failure"],
        "scoring_rule": "official_pct_at_turn_limit_v1", "llm_calls": 0, "actions_replayed": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("source_id", "pct", "passed_tests", "total_tests", "success")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--appworld-root", type=Path, default=Path.home() / ".exgentic/appworld")
    args = parser.parse_args()
    rescore(args.source_result, args.output, args.appworld_root)


if __name__ == "__main__":
    main()
