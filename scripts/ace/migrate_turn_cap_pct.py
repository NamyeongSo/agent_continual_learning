# SPDX-License-Identifier: Apache-2.0
"""Create a traced successor cohort from saved rollouts and official turn-cap rescoring."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ace.run_context_proposal_bottleneck import digest, now, read, summarize, write  # noqa: E402


def rescore_feedback(old_capture, outcome, protocol):
    capture = copy.deepcopy(old_capture)
    limits = {
        "version": protocol["actor_limits"]["policy_version"], "input_tokens": 100000,
        "output_tokens": None, "max_turns": 40, "token_scope": "environment_turn",
        "turn_limit_scoring": "official_pct_at_limit",
    }
    capture["training_feedback"] = {
        "score": outcome["pct"], "success": outcome["success"], "is_finished": False,
        "session_metrics": {
            "pass_percentage": 100 * outcome["pct"], "pass_count": outcome["passed_tests"],
            "fail_count": outcome["total_tests"] - outcome["passed_tests"],
            "num_tests": outcome["total_tests"], "difficulty": outcome["difficulty"], "success": outcome["success"],
        },
        "session_metadata": {
            "test_tracker": outcome["evaluation"], "turn_limit_reached": True,
            "turn_cap_details": outcome["turn_cap_details"], "experiment_limits": limits,
            "turn_limit_scoring": "official_pct_at_limit",
        },
    }
    return capture


def migrate(source, output):
    source, output = source.resolve(), output.resolve()
    previous, protocol = read(source / "protocol.json"), read(output / "protocol.json")
    old_pid = read(source / "controller.json")["pid"]
    if Path(f"/proc/{old_pid}").exists():
        raise ValueError("Predecessor controller must finish stopping before migration")
    if protocol.get("turn_limit_scoring") != "official_pct_at_limit":
        raise ValueError("Successor must explicitly pin official turn-cap PCT scoring")
    for key in ("task_ids", "model", "model_settings", "appworld_policy_sha256", "initial_playbook"):
        if protocol[key] != previous[key]:
            raise ValueError(f"Unmatched cohort setting: {key}")
    for relative in ("src/exgentic/agents/ace/ace_instance.py", "src/exgentic/agents/ace/prompts/reflector.py",
                     "src/exgentic/agents/ace/prompts/curator.py", "src/exgentic/agents/ace/playbook_utils.py"):
        if (source / "source_snapshot" / relative).read_bytes() != (output / "source_snapshot" / relative).read_bytes():
            raise ValueError(f"Native proposal implementation changed: {relative}")
    original_rows = [read(p) for p in sorted((source / "stage1").glob("*/result.json"))]
    if len(original_rows) != 250:
        raise ValueError("Expected the entire initial trajectory bank")
    changed, provenance = set(), []
    for row in original_rows:
        directory = output / "stage1" / row["id"]
        if (directory / "result.json").exists():
            raise FileExistsError("Refusing to overwrite an already migrated bank")
        directory.mkdir(parents=True, exist_ok=True)
        original_result = source / "stage1" / row["id"] / "result.json"
        updated = {**row, "initial_rollout_reused": True, "source_result_path": str(original_result),
                   "source_result_sha256": digest(original_result.read_bytes())}
        if (row.get("budget_failure") or {}).get("reason") == "max_turns":
            changed.add(row["source_id"])
            report_path = output / "turn_cap_rescoring" / row["id"] / "result.json"
            report = read(report_path)
            if report["source_id"] != row["source_id"] or report["task_id"] != row["task_id"]:
                raise ValueError("Rescored state/source pairing mismatch")
            session_dir = directory / "rescored_session"
            capture_path = session_dir / "agent/ace_capture.json"
            capture = rescore_feedback(read(row["capture_path"]), report, protocol)
            write(capture_path, capture)
            write(session_dir / "benchmark/scores.json", {
                "task_id": row["task_id"], "test_tracker": report["evaluation"],
                "turn_limit_reached": True, "turn_cap_details": report["turn_cap_details"],
                "turn_limit_scoring": "official_pct_at_limit", "rescore_report": str(report_path),
            })
            write(session_dir / "results.json", {
                "task_id": row["task_id"], "score": report["pct"], "success": report["success"],
                "is_finished": False, "details": capture["training_feedback"],
                "source_session_dir": row["session_dir"], "rescore_report": str(report_path),
            })
            for name in ("task_output", "logs"):
                (session_dir / "benchmark" / name).symlink_to(
                    Path(row["session_dir"]) / "benchmark" / name, target_is_directory=True,
                )
            (session_dir / "trajectory.jsonl").symlink_to(Path(row["session_dir"]) / "trajectory.jsonl")
            updated.pop("budget_failure", None)
            updated.update({k: report[k] for k in ("pct", "success", "passed_tests", "total_tests", "difficulty")})
            updated.update(
                official_score=report["pct"], session_status="success" if report["success"] else "unsuccessful",
                session_dir=str(session_dir), capture_path=str(capture_path), turn_limit_reached=True,
                turn_cap_details=report["turn_cap_details"], turn_limit_scoring="official_pct_at_limit",
                rescore_report=str(report_path), rescored_at=now(),
            )
        write(directory / "result.json", updated)
        provenance.append({"source_id": row["id"], "original_result": str(original_result),
                           "original_sha256": digest(original_result.read_bytes()),
                           "migrated_sha256": digest((directory / "result.json").read_bytes()),
                           "rescored": row["id"] in changed})
    reused = []
    for path in sorted((source / "stage2").glob("*/result.json")):
        row = read(path)
        if row["source_id"] in changed:
            continue
        if row["appworld_policy_sha256"] != protocol["appworld_policy_sha256"]:
            raise ValueError("Proposal policy mismatch")
        if digest(Path(row["updated_context_path"]).read_bytes()) != row["updated_context_sha256"]:
            raise ValueError("Source proposal context hash mismatch")
        target = output / "stage2" / row["id"]
        target.mkdir(parents=True)
        shutil.copy2(row["updated_context_path"], target / "updated_context.txt")
        shutil.copy2(row["reflection_path"], target / "reflection.json")
        for attempt in path.parent.glob("attempt_*"):
            (target / attempt.name).symlink_to(attempt, target_is_directory=True)
        write(target / "result.json", {
            **row, "updated_context_path": str(target / "updated_context.txt"),
            "reflection_path": str(target / "reflection.json"), "reused_from": str(path),
        })
        reused.append(row["id"])
    report = {"created_at": now(), "source_cohort": str(source), "initial_rollouts_reused": 250,
              "new_initial_actor_calls": 0, "rescored_sources": sorted(changed),
              "proposal_chains_to_regenerate_for_changed_grading": len(changed) * 2,
              "completed_proposals_reused": reused, "initial_bank": provenance}
    write(output / "migration_manifest.json", report)
    protocol.update(source_initial_bank=str(source), protocol_revision=2,
                    initial_trajectories_regenerated=0, rescored_initial_rollouts=len(changed),
                    completed_proposals_reused=len(reused))
    write(output / "protocol.json", protocol)
    summarize(output, 1)
    status = read(source / "status.json")
    write(source / "status.json", {**status, "state": "superseded", "successor_cohort": str(output),
                                  "reason": "User changed turn-limit scoring to official PCT", "updated_at": now()})
    with (source / "progress.log").open("a") as stream:
        stream.write(f"PROTOCOL SUPERSEDED: official turn-cap PCT successor: {output}\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"initial_bank", "completed_proposals_reused"}},
                     ensure_ascii=False, indent=2))
    print("Completed proposals reused:", len(reused))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    migrate(args.source, args.output)


if __name__ == "__main__":
    main()
