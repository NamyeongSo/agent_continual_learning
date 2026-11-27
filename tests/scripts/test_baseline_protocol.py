# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.baselines.audit_premise_budget import audit_run
from scripts.baselines.protocol import DEFAULT_PROTOCOL, ROOT, build_plan, validation_pools

MANIFEST = ROOT / "scripts/utils/task_splits/bfcl_appworld_terminalbench2_seed42_44.json"
QUOTAS = {"bfcl": 17, "appworld": 17, "terminalbench2": 16}


def test_fixed_validation_is_method_independent_and_excludes_train_and_test():
    manifest = json.loads(MANIFEST.read_text())
    pools = validation_pools(manifest, 42, QUOTAS)
    assert Counter(item["benchmark"] for item in pools["interleaved"]) == QUOTAS
    for benchmark in QUOTAS:
        selected = {item["task_id"] for item in pools["interleaved"] if item["benchmark"] == benchmark}
        section = manifest[f"{benchmark}_42"]
        assert selected <= set(section["val"])
        assert not selected & set(section["train"] + section["test"])
    sequential = build_plan(DEFAULT_PROTOCOL, MANIFEST, 42, "sequential")
    interleaved = build_plan(DEFAULT_PROTOCOL, MANIFEST, 42, "interleaved")
    assert sequential["validation"] == interleaved["validation"] == pools
    assert sequential["validation_sha256"] == interleaved["validation_sha256"]
    assert {key: len(value) for key, value in pools["by_benchmark"].items()} == {
        "bfcl": 50, "appworld": 50, "terminalbench2": 30,
    }


def test_validation_membership_is_independent_of_manifest_list_order():
    manifest = json.loads(MANIFEST.read_text())
    original = validation_pools(manifest, 42, QUOTAS)["interleaved"]
    for benchmark in QUOTAS:
        manifest[f"{benchmark}_42"]["val"].reverse()
    assert validation_pools(manifest, 42, QUOTAS)["interleaved"] == original
    assert validation_pools(manifest, 43, QUOTAS)["interleaved"] != original


def test_validation_refuses_contamination_or_insufficient_quota():
    manifest = json.loads(MANIFEST.read_text())
    with pytest.raises(ValueError, match="quota"):
        validation_pools(manifest, 42, {"terminalbench2": 31})
    manifest["bfcl_42"]["val"][0] = manifest["bfcl_42"]["train"][0]
    with pytest.raises(ValueError, match="overlapping"):
        validation_pools(manifest, 42, QUOTAS)


@pytest.mark.parametrize("mode", ["isolated", "sequential", "interleaved"])
def test_controller_plan_preserves_unique_encounters_and_pending_choices(mode):
    plan = build_plan(DEFAULT_PROTOCOL, MANIFEST, 42, mode)
    order = [(item["benchmark"], item["task_id"]) for item in plan["controller_train_order"]]
    assert len(order) == len(set(order)) == plan["unique_encounters"] == 130
    assert not plan["runnable"]
    assert plan["role"] == "controller_plan_not_optimizer_trainset"
    assert plan["accepted"]["gepa"]["reflection_minibatch_size"] == 5
    assert plan["accepted"]["gepa"]["candidate_selection_strategy"] == "pareto"
    assert plan["accepted"]["skillopt"]["accumulation"] == 1
    assert plan["pending_proposals"] == {}
    assert plan["accepted"]["budget"]["total_training_limit"] == "unlimited"
    assert plan["accepted"]["budget"]["premise_reference"] == "measurement_only"
    assert "budget" not in plan["pending_proposals"]
    assert plan["accepted"]["stream"]["passes"] == 1
    if mode == "sequential":
        assert [benchmark for benchmark, _ in order] == ["bfcl"] * 50 + ["appworld"] * 50 + ["terminalbench2"] * 30


def write_run(tmp_path, parents, candidates, *, status=None):
    run = tmp_path / "reference"
    (run / "iterations").mkdir(parents=True)
    (run / "sessions").mkdir()
    (run / "state.json").write_text(json.dumps({
        "cursor": 1, "config": {"mode": "sequential", "seed": 42, "past_count": 3},
    }))
    artifact = {"iteration": 1, "benchmark": "bfcl", "task_id": "task1", "parents": parents,
                "candidates": candidates, "status": status}
    (run / "iterations/iter_0001.json").write_text(json.dumps(artifact))
    return run


def trace(root, name, prefix_length=0):
    return {"session_dir": str(root / "sessions" / name / "sessions/id"), "prefix_length": prefix_length}


def test_audit_recovers_failed_candidate_sessions_and_excludes_test_evaluation(tmp_path):
    expected = tmp_path / "reference"
    parent = "premise_ns_i0001_parent1_bfcl"
    candidate1 = "premise_ns_i0001_candidate_1_current_try1_bfcl"
    candidate2 = "premise_ns_i0001_candidate_1_current_try2_bfcl"
    failed = "premise_ns_i0001_candidate_2_current_try1_bfcl"
    run = write_run(tmp_path, [trace(expected, parent), None], [{"current_result": {
        "replay": trace(expected, candidate2, prefix_length=3), "replay_attempts": 2,
    }}, {"valid": False}])
    for name in (parent, candidate1, candidate2, failed, "premise_ns_test_cp130_bfcl_001_bfcl"):
        (run / "sessions" / name).mkdir()
    report = audit_run(run)
    assert report["counts"]["artifact_attempt_lower_bound"] == 3
    assert report["counts"]["recorded_prefix_replays"] == 1
    assert report["lineage_observed_sessions"] == report["retained_attempt_lower_bound"] == 4
    assert not report["exact_total_known"]


def test_audit_follows_archived_prefix_paths_and_deduplicates_copied_runs(tmp_path):
    archive = tmp_path / "archived"
    old = archive / "old/reference"
    name = "premise_old_i0001_parent1_bfcl"
    (old / "sessions" / name).mkdir(parents=True)
    historical = Path("/unavailable/outputs/old/reference")
    run = write_run(tmp_path, [trace(historical, name)], [])
    (run / "sessions" / name).mkdir()
    report = audit_run(run, archive)
    assert report["lineage_observed_sessions"] == report["retained_attempt_lower_bound"] == 1
    assert report["missing_final_session_dirs"] == 0


def test_audit_keeps_missing_sessions_as_a_lower_bound_and_excludes_abandoned_namespace(tmp_path):
    expected = tmp_path / "reference"
    run = write_run(tmp_path, [trace(expected, "premise_kept_i0001_parent1_bfcl")], [], status="skipped")
    (run / "sessions/premise_abandoned_i0001_parent1_bfcl").mkdir()
    report = audit_run(run)
    assert report["retained_attempt_lower_bound"] == report["missing_final_session_dirs"] == 1
    assert report["counts"]["skipped_updates"] == 1
    assert report["lineage_observed_sessions"] == 0
    assert report["source_observed_sessions_including_abandoned_attempts"] == 1


def test_audit_rejects_duplicate_encounters_and_different_premise_validation(tmp_path):
    run = write_run(tmp_path, [], [])
    artifact = json.loads((run / "iterations/iter_0001.json").read_text())
    artifact["iteration"] = 2
    (run / "iterations/iter_0002.json").write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="Duplicate encounter"):
        audit_run(run)
    state = json.loads((run / "state.json").read_text())
    state["config"]["candidate_validation"] = "random_val4"
    (run / "state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="current/past"):
        audit_run(run)
