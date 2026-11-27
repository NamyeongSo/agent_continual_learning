# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Monitor evidence must match saved selection IDs without exposing private trace fields."""

import json

import pytest

from scripts.research_journal.live_progress import LiveProgress
from scripts.research_journal.store import JournalError, JournalStore


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def monitor(tmp_path):
    root = tmp_path / "outputs/batch"
    train = root / "run/train"
    store = JournalStore(tmp_path / "journal", [tmp_path])
    live = LiveProgress(store, base_dir=tmp_path)
    write(root / "launch_manifest.json", {"jobs": [{"id": "ab42", "proposal_retrieval": "ab",
                                                   "output": str(train.parent), "train_dir": str(train)}]})
    parent = {"task_id": "old", "benchmark": "bfcl", "task": "original task", "success": False,
              "actions": [{"action": {"name": "lookup", "arguments": {"id": 3}}}],
              "context": "PRIVATE_CONTEXT", "details": "HIDDEN_EVALUATOR", "session_dir": "PRIVATE_PATH"}
    reflection = {"id": "reflection_3", "parent_rank": 2, "analysis": "actual source reflection"}
    write(train / "iterations/iter_0001.json", {"iteration": 1, "parents": [None, parent],
                                               "reflections": [reflection]})
    example = {"id": "0001:reflection_3", "iteration": 1, "task_id": "old", "benchmark": "bfcl",
               "retrieval_role": "B", "action_f1": 0.0, "reflection_cosine": 0.7}
    write(train / "iterations/iter_0002.json", {"iteration": 2, "task_id": "now", "parents": [parent],
                                               "reflections": [{**reflection, "parent_rank": 1}],
                                               "candidates": [{"id": "candidate_1", "parent_rank": 1,
                                                               "reflection": reflection, "playbook": "PRIVATE_PLAYBOOK",
                                                               "proposal_retrieval": {"mode": "ab", "requested": 2,
                                                                                      "selected": 1,
                                                                                      "examples": [example]}}]})
    write(train / "iterations/iter_0003.json", {"iteration": 3, "status": "skipped", "error_message": "timeout"})
    return live, train


def test_latest_retrieval_survives_skipped_task_and_cache_stays_compact(monitor):
    live, train = monitor
    summary = live.proposal_summary(train, "ab", 3)
    assert summary["iteration"] == 2 and summary["processed"] == 3
    assert summary["latest_status"] == "skipped" and summary["latest_error"] == "timeout"
    assert summary["candidates"][0]["examples"][0]["action_f1"] == 0
    cached = json.dumps(list(live.files.entries.values()))
    assert "PRIVATE" not in cached and "actions" not in cached


def test_details_resolve_exact_reflection_and_parent_without_private_fields(monitor):
    live, _ = monitor
    details = live.proposal_details("batch", "ab42", "2")
    source = details["sources"]["0001:reflection_3"]
    assert source["reflection"]["parent_rank"] == 2
    assert source["reflection"]["analysis"] == "actual source reflection"
    assert source["actions"][0]["action"]["arguments"] == {"id": 3}
    assert source["success"] is False
    assert "PRIVATE" not in json.dumps(details) and "HIDDEN_EVALUATOR" not in json.dumps(details)


@pytest.mark.parametrize("batch,run,iteration,status", [
    ("../batch", "ab42", "2", 404), ("batch", "../run", "2", 404),
    ("batch", "ab42", "../1", 400), ("batch", "ab42", "0", 400), ("batch", "ab42", "99", 404),
])
def test_endpoint_rejects_unknown_runs_and_untrusted_paths(monitor, batch, run, iteration, status):
    live, _ = monitor
    with pytest.raises(JournalError) as error:
        live.proposal_details(batch, run, iteration)
    assert error.value.status == status


def test_evidence_symlink_outside_allowed_roots_is_rejected(monitor, tmp_path):
    live, train = monitor
    path = train / "iterations/iter_0001.json"
    path.unlink()
    path.symlink_to("/etc/passwd")
    with pytest.raises(JournalError) as error:
        live.proposal_details("batch", "ab42", 2)
    assert error.value.status == 403
