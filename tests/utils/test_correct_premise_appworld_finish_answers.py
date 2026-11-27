# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""The counterfactual must change only completed action tasks with an extra answer."""

from __future__ import annotations

import json

from scripts.utils.correct_premise_appworld_finish_answers import correct_group


def test_correct_group_excludes_questions_and_missing_finish(tmp_path):
    run = tmp_path / "s44_sequential_t3"
    summary_path = run / "test_evaluation" / "cp100_appworld.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps({
        "complete": True,
        "results": [
            {"task_id": "action_done", "score": 0.5},
            {"task_id": "action_other_failure", "score": 0.0},
            {"task_id": "question", "score": 0.5},
            {"task_id": "missing_finish", "score": 0.5},
        ],
    }), encoding="utf-8")

    def session(index, task_id, expected, extra_failure=False, finish=True):
        session_dir = (
            run / "sessions" / f"run_test_cp100_appworld_{index:04}_appworld" / "sessions" / task_id
        )
        benchmark_dir = session_dir / "benchmark"
        benchmark_dir.mkdir(parents=True)
        failures = [{
            "requirement": "assert answers match.",
            "trace": f"AssertionError:\n'wrong answer'\n==\n'{expected}'",
        }]
        if extra_failure:
            failures.append({"requirement": "assert model changes match.", "trace": "AssertionError"})
        passes = [] if extra_failure else [{"requirement": "assert state matches."}]
        benchmark_dir.joinpath("scores.json").write_text(json.dumps({
            "task_id": task_id,
            "pass_percentage": 0.0 if extra_failure else 50.0,
            "pass_count": len(passes),
            "num_tests": 2,
            "test_tracker": {"passes": passes, "failures": failures},
        }), encoding="utf-8")
        events = []
        if finish:
            events = [
                {"event": "action", "action": {
                    "name": "finish", "id": "finish-1", "arguments": {"answer": "wrong answer"},
                    "validation": {"valid": True},
                }},
                {"event": "observation", "observation": {
                    "invoking_actions": [{"id": "finish-1"}],
                    "result": {"message": "Marked the active task complete."},
                }},
            ]
        session_dir.joinpath("trajectory.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
        )

    session(0, "action_done", "null")
    session(1, "action_other_failure", "null", extra_failure=True)
    session(2, "question", "42")
    session(3, "missing_finish", "null", finish=False)

    result = correct_group(run, summary_path)

    assert result["status"] == "audited"
    assert result["corrected_sessions"] == 2
    assert result["new_full_successes"] == 1
    assert result["corrected_full_success_rate"] == 0.25
    assert result["corrected_mean_test_pass_rate"] == 0.625
    assert {item["task_id"] for item in result["corrections"]} == {"action_done", "action_other_failure"}
