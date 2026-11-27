# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import json

import pytest

from scripts.utils.audit_training_analysis import audit_training, has_diagnostics, jsonl_records


def test_analysis_failures_are_separate_from_unique_skipped_updates(tmp_path):
    error = "BadRequestError: This model's maximum context length is 131072 tokens"
    (tmp_path / "analysis_errors.jsonl").write_text("\n".join(json.dumps(row) for row in [
        {"stage": "instruction_proposal", "error": error},
        {"stage": "instruction_proposal", "error": "ReadTimeout"},
    ]) + '\n{"stage":')
    (tmp_path / "skipped_updates.jsonl").write_text("\n".join(json.dumps(row) for row in [
        {"batch": 2, "error": error}, {"batch": 2, "error": error}, {"batch": 4, "error": "ReadTimeout"},
    ]))
    result = audit_training({"id": "gepa", "method": "gepa", "train_dir": str(tmp_path)})
    assert result["analysis_failed_attempts"] == 2
    assert result["analysis_context_failed_attempts"] == 1
    assert result["analysis_other_failed_attempts"] == 1
    assert result["gepa_skipped_batches"] == 2
    assert result["gepa_context_skipped_batches"] == 1
    assert has_diagnostics(result)
    skillopt = audit_training({"id": "skillopt", "method": "skillopt", "train_dir": str(tmp_path)})
    assert skillopt["analysis_failed_attempts"] == 2
    assert skillopt["gepa_skipped_batches"] == 0


def test_malformed_history_is_not_silently_ignored(tmp_path):
    path = tmp_path / "errors.jsonl"
    path.write_text('broken\n{"stage": "analyst"}\n')
    with pytest.raises(json.JSONDecodeError):
        jsonl_records(path)


def test_task_skips_and_successful_iterations_are_distinct(tmp_path):
    (tmp_path / "iterations").mkdir()
    (tmp_path / "iterations/iter_0001.json").write_text(json.dumps({"status": "skipped"}, indent=2))
    (tmp_path / "iterations/iter_0002.json").write_text(json.dumps({"status": "completed"}, indent=2))
    result = audit_training({"id": "premise", "method": "premise", "train_dir": str(tmp_path)})
    assert result["premise_skipped_iterations"] == 1
    assert result["analysis_failed_attempts"] == 0
    (tmp_path / "run/sessions/s1/agent").mkdir(parents=True)
    (tmp_path / "run/sessions/s1/agent/task_skip.json").write_text(json.dumps({
        "reason": "context_length_exceeded", "playbook_updated": False,
    }))
    ace = audit_training({"id": "ace", "method": "ace", "train_dir": str(tmp_path)})
    assert ace["ace_context_skipped_tasks"] == 1
