# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import hashlib
import json

import pytest

from scripts.premise.analyze_bfcl_context_construction import analyze, entries


def test_context_entries_preserve_multiline_rules_and_ignore_headers():
    text = "## INSIGHTS\n[a-00001] helpful=1 harmful=0 :: First line\nContinued rule.\n" \
           "[a-00002] helpful=0 harmful=1 :: Second rule.\n## OTHER\n"
    assert entries(text) == {"a-00001": "First line\nContinued rule.", "a-00002": "Second rule."}


def test_evaluated_file_identity_and_noop_content_are_checked_separately(tmp_path):
    folder = tmp_path / "frozen" / "cp10"
    folder.mkdir(parents=True)
    parent = "## INSIGHTS\n[a-00001] helpful=0 harmful=0 :: Existing rule.\n"
    (folder / "parent.json").write_text(json.dumps({"playbook": {"playbook": parent}}))
    (folder / "proposal_trace.json").write_text(json.dumps({"score": 0, "actions": []}))
    hashes = []
    for index in range(1, 6):
        record = {"id": f"candidate{index}", "playbook": {"playbook": parent}, "valid": True, "applied": []}
        path = folder / f"candidate_{index:02d}.json"
        path.write_text(json.dumps(record))
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    point = {"key": "cp10", "seed": 42, "checkpoint": 10, "source": str(tmp_path), "context_sha256": hashes,
             "scores": {"full_train": [0.2] * 5, "full_validation": [0.4, 0.4, 0.4, 0.4, 0.3]}}
    raw = {"models": [{"label": "Synthetic", "points": [point]}]}
    result, candidates, _ = analyze(raw)
    assert len({row["context_sha256"] for row in candidates}) == 5
    assert len({row["playbook_sha256"] for row in candidates}) == 1
    assert result["models"][0]["no_op_candidates"] == 5
    assert result["models"][0]["mean_unique_contexts_per_checkpoint"] == 1
    assert result["duplicate_context_scores"][0]["full_validation_range"] == pytest.approx(0.1)
    (folder / "candidate_01.json").write_text("{}")
    with pytest.raises((KeyError, ValueError)):
        analyze(raw)
