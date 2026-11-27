# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Protect frozen checkpoint selection and diagnostic score aggregation."""

import copy
import json
from types import SimpleNamespace

import pytest

from exgentic.agents.premise.pipeline import write_json
from scripts.premise import reevaluate_terminalbench2_without_timeouts as reevaluation


@pytest.fixture
def frozen_source(tmp_path):
    source = tmp_path / "source"
    root = source / "s42/run"
    manifest = tmp_path / "split.json"
    write_json(manifest, {})
    snapshot = {
        "cursor": 30,
        "config": {"manifest": str(manifest), "model": "openai/test", "max_steps": {"terminalbench2": 100},
                   "benchmark_configs": {"terminalbench2": {"actor": "terminus2"}}},
        "states": {"terminalbench2": {"beam": [{"playbook": "frozen"}]}},
    }
    cp = root / "checkpoints/checkpoint_030.json"
    write_json(cp, snapshot)
    write_json(root / "state.json", snapshot)
    rows = [
        {"task_id": "scored-deadline", "score": 1.0, "status": "completed"},
        {"task_id": "failed-deadline", "score": 0.0, "status": "error", "error": "TaskTimeLimitError: expired"},
        {"task_id": "context-error", "score": 0.0, "status": "error", "error": "maximum context length"},
        {"task_id": "unsolved", "score": 0.0, "status": "completed"},
    ]
    write_json(root / "test_evaluation/cp030_terminalbench2.json", {
        "complete": True, "identity": {"checkpoint_sha256": reevaluation.sha256(cp), "learning": False,
                                       "manifest_sha256": reevaluation.sha256(manifest),
                                       "task_ids": [r["task_id"] for r in rows]}, "results": rows,
    })
    for index in range(len(rows)):
        write_json(root / f"sessions/test_test_cp030_terminalbench2_{index:04d}_terminalbench2"
                   "/sessions/session/results.json", {
                       "details": {"session_metadata": {"agent_timed_out": index == 0}},
                   })
    return source


def test_selects_scored_deadline_and_explicit_error_without_including_unsolved_tasks(frozen_source):
    sources, jobs = reevaluation.collect_sources(frozen_source, [42], 30)
    assert {job["task_id"] for job in jobs} == {"scored-deadline", "failed-deadline"}
    cp = frozen_source / "s42/run/checkpoints/checkpoint_030.json"
    snapshot = sources[42]["snapshot"]
    snapshot["states"]["terminalbench2"]["beam"][0]["playbook"] = "changed"
    write_json(cp, snapshot)
    with pytest.raises(ValueError, match="checkpoint hash differs"):
        reevaluation.collect_sources(frozen_source, [42], 30)


def test_report_keeps_all_tasks_in_denominator_and_does_not_publish_partial_score(frozen_source, tmp_path):
    sources, jobs = reevaluation.collect_sources(frozen_source, [42], 30)
    before = copy.deepcopy(sources)
    partial = reevaluation.report(tmp_path / "out", sources, jobs, {}, "running")
    assert partial["groups"]["42"]["mean_official_score"] is None
    assert partial["groups"]["42"]["complete"] is False
    records = {
        job["id"]: {"task_id": job["task_id"], "score": 0.0 if job["index"] == 0 else 1.0,
                    "status": "error" if job["index"] == 0 else "completed"} for job in jobs
    }
    final = reevaluation.report(tmp_path / "out", sources, jobs, records, "completed")
    group = final["groups"]["42"]
    assert group["mean_official_score"] == 0.25
    assert group["execution_errors"] == 2  # Includes the original context error.
    assert group["total_tasks"] == 4
    assert final["complete"] is True
    assert sources == before


def test_native_partial_score_is_not_counted_after_acting_context_error(frozen_source, tmp_path, monkeypatch):
    sources, jobs = reevaluation.collect_sources(frozen_source, [42], 30)
    source_before = copy.deepcopy(sources)
    captured = []

    def execute(*, config):
        captured.append(config)
        return SimpleNamespace(session_results=[SimpleNamespace(
            session_id="session", status=SimpleNamespace(value="success"), score=1.0,
            details={"session_metadata": {"enforce_session_timeouts": False, "agent_error": "context exceeded"}},
        )])

    def trajectory(*_args):
        raise RuntimeError("Terminus-2 failed: maximum context length exceeded")

    monkeypatch.setattr(reevaluation, "execute", execute)
    monkeypatch.setattr(reevaluation, "trajectory_from_session", trajectory)
    record = reevaluation.evaluate_job(jobs[0], sources[42], tmp_path / "out", 30)
    assert record["status"] == "error"
    assert record["score"] == 0.0
    assert record["native_score"] == 1.0
    assert record["context_error"] is True
    assert captured[0].benchmark_kwargs["enforce_session_timeouts"] is False
    assert captured[0].agent_kwargs["playbook"] == "frozen"
    assert sources == source_before
    saved = json.loads((tmp_path / "out/records" / f"{jobs[0]['id']}.json").read_text())
    assert saved == record
