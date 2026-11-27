# SPDX-License-Identifier: Apache-2.0
"""Check that BWT uses each task's own accumulated checkpoint, with complete coverage."""

import hashlib
import json

import pytest

from scripts.ace import analyze_reflection_accumulation as analysis


def phase_rows(tmp_path):
    tasks = [f"task_{i}" for i in range(50)]
    protocol = {"task_orders": {"42": {"train": tasks, "val": ["val_task"]}}, "appworld_policy_sha256": "policy"}
    rows = []
    for condition in analysis.conditions(42):
        for index, task in enumerate(tasks, 1):
            text = f"Accumulated through {task}"
            digest = hashlib.sha256(text.encode()).hexdigest()
            path = tmp_path / "conditions" / condition["id"] / "checkpoints" / f"{index:03d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"task_id": task, "playbook": text, "context_sha256": digest}))
            rows.append({"condition_id": condition["id"], "task_id": task, "phase": "bwt_checkpoint",
                         "seed": 42, "evaluation_repeat": 0, "appworld_policy_sha256": "policy",
                         "checkpoint_step": index, "initial_context_sha256": digest,
                         "passed_tests": 1, "total_tests": 2, "pct": 0.5, "success": False})
    return protocol, rows


def test_bwt_validates_each_task_checkpoint_and_keeps_300_evaluations(tmp_path):
    protocol, rows = phase_rows(tmp_path)
    lookup = analysis.validate_phase(tmp_path, protocol, 42, "bwt_checkpoint", rows)
    assert len(lookup) == 300
    assert {r["checkpoint_step"] for r in lookup.values()} == set(range(1, 51))


@pytest.mark.parametrize("problem", ["wrong_checkpoint", "missing", "duplicate", "wrong_split"])
def test_bwt_cannot_substitute_final_or_independent_candidate_scores(tmp_path, problem):
    protocol, rows = phase_rows(tmp_path)
    if problem == "wrong_checkpoint":
        rows[0]["checkpoint_step"] = 50
    elif problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows.append(dict(rows[0]))
    else:
        rows[0]["task_id"] = "val_task"
    with pytest.raises(ValueError):
        analysis.validate_phase(tmp_path, protocol, 42, "bwt_checkpoint", rows)


def test_tgc_uses_success_flag_and_excludes_missing_outcomes():
    rows = [{"pct": 1.0, "success": False}, {"pct": 0.9999, "success": True},
            {"pct": 0.0, "success": None}]
    result = analysis.tgc_summary(rows)
    assert result["tgc_rate"] == 0.5
    assert result["tgc_recorded_outcomes"] == 2
    assert result["tgc_missing_outcomes"] == 1
    with pytest.raises(ValueError):
        analysis.official_tgc({"pct": 1.0, "success": "false"})
