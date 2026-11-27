# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Verify audit position reversal, input isolation, finite calls and tied rankings."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json

import pytest

from scripts.premise import bfcl_trajectory_reward_judge as judge
from scripts.premise import run_bfcl_trajectory_judge_reliability as audit


def task_and_inputs():
    task = {"user_turns": [{"text": "Read the note."}], "tool_classes": ["Notes"]}
    traces = [{"key": context, "events": [{"type": "request", "turn": 0, "text": "Read the note."},
                                         {"type": "observation", "turn": 0, "result": {"text": context}}],
               "lost_evidence": False, "truncated_fields": 0}
              for context in ["parent", *[f"candidate_{i:02d}" for i in range(1, 6)]]]
    return task, traces


def parsed(mapping):
    reply = {"schema_version": 1, "items": [
        {"id": identity, "a": [1], "ae": [[1]], "v": [0, 0, 0, 0], "e": [[], [], [], []],
         "h": 0, "he": [], "u": []} for identity in mapping]}
    result = judge.parse_judgment(json.dumps(reply), mapping)
    assert result["ok"]
    return result


@pytest.mark.parametrize("original_seed,audit_seed", [(0, 1), (42, 20261021), (777, 777)])
def test_exact_reverse_inverts_builder_shuffle_without_changing_evidence(original_seed, audit_seed):
    task, traces = task_and_inputs()
    before = copy.deepcopy(traces)
    _, original_map = judge.build_judge_request(task, traces, seed=original_seed, model="judge")
    reversed_traces = audit.reversed_inputs(traces, original_map, seed=audit_seed)
    _, reversed_map = judge.build_judge_request(task, reversed_traces, seed=audit_seed, model="judge")
    assert [reversed_map[key]["key"] for key in sorted(reversed_map)] == list(reversed(
        [original_map[key]["key"] for key in sorted(original_map)]))
    assert traces == before
    assert {id(trace) for trace in reversed_traces} == {id(trace) for trace in traces}


def group_rows():
    return [{"id": f"{model.lower()}_s{seed}_cp{cp:03d}_q{position:02d}",
             "model": model, "seed": seed, "checkpoint": cp, "query_position": position,
             "task_id": f"task_{position}",
             "official_success": (cp + position) % 2}
            for model in ("Qwen", "Gemma") for seed in (42, 43, 44)
            for cp in (10, 15) for position in (0, 1, 2)]


def test_balanced_selection_uses_planned_metadata_and_excludes_other_models():
    groups = [*group_rows(), {"model": "OSS"}]
    selected = audit.select_audit_groups(groups)
    assert len(selected) == 12
    assert all(group["checkpoint"] == 10 and group["query_position"] in {0, 1} for group in selected)
    for group in groups[:-1]:
        group["official_success"] = 1 - group["official_success"]
    assert [group["id"] for group in audit.select_audit_groups(groups)] == [group["id"] for group in selected]


def test_comparison_retains_constant_correlation_as_undefined_and_unknown_transitions():
    task, traces = task_and_inputs()
    _, mapping = judge.build_judge_request(task, traces, seed=42, model="judge")
    original = parsed(mapping)["judgments"]
    repeated = copy.deepcopy(original)
    repeated["candidate_01"]["metrics"]["material_violation"]["value"] = None
    comparison = audit.compare_judgments(original, repeated)
    assert comparison["candidate_fulfillment_spearman"] is None
    assert comparison["candidate_spearman_undefined_due_to_ties"]
    assert comparison["candidate_top_set_jaccard"] == 1
    row = next(row for row in comparison["attempts"] if row["context"] == "candidate_01")
    assert row["material_unknown_transition"] and not row["material_known_flip"]


def test_comparison_measures_reversed_rankings_without_any_performance_target():
    task, traces = task_and_inputs()
    _, mapping = judge.build_judge_request(task, traces, seed=42, model="judge")
    original = parsed(mapping)["judgments"]
    repeated = copy.deepcopy(original)
    for index in range(1, 6):
        original[f"candidate_{index:02d}"]["metrics"]["fulfillment"]["lower_bound"] = index / 5
        repeated[f"candidate_{index:02d}"]["metrics"]["fulfillment"]["lower_bound"] = (6 - index) / 5
    comparison = audit.compare_judgments(original, repeated)
    assert comparison["candidate_fulfillment_spearman"] == pytest.approx(-1)
    assert comparison["candidate_top_set_jaccard"] == 0


def test_observable_inputs_strip_official_labels_and_reproduce_recorded_compaction():
    task, inputs = task_and_inputs()
    group = {"public_task": task, "traces": [
        {"context": trace["key"], "available": True, "official_success": "PRIVATE_SUCCESS",
         "result_path": "PRIVATE_PATH", "features": {"score": "PRIVATE_SCORE"},
         "compact_trajectory": {"events": trace["events"], "lost_evidence": False, "truncated_fields": 0}}
        for trace in inputs]}
    before = copy.deepcopy(group)
    public = audit.observable_inputs(group, 3)
    assert all(set(trace) == {"key", "events", "lost_evidence", "truncated_fields"} for trace in public)
    text = json.dumps(public)
    assert all(value not in text for value in ("PRIVATE_SUCCESS", "PRIVATE_PATH", "PRIVATE_SCORE"))
    assert public[0]["lost_evidence"]
    assert public[0]["truncated_fields"] > 0
    assert group == before


def args_for(folder, prepare_only=False):
    return argparse.Namespace(folder=folder, base_url="http://unused/v1", model="judge", concurrency=2,
                              timeout=2, health_wait_seconds=0, health_poll_seconds=.01,
                              max_sent_attempts=1, prepare_only=prepare_only)


def write_cache(folder):
    task, inputs = task_and_inputs()
    groups = audit.select_audit_groups(group_rows())
    for group in groups:
        group["public_task"] = task
        group["traces"] = [{"context": trace["key"], "available": True, "official_success": 0,
                            "compact_trajectory": {"events": trace["events"],
                                                   "lost_evidence": False, "truncated_fields": 0}}
                           for trace in inputs]
    cache = folder / "group_features.jsonl"
    cache.write_text("\n".join(json.dumps(group) for group in groups))
    return groups, hashlib.sha256(cache.read_bytes()).hexdigest()


def test_prepare_only_makes_no_health_tokenization_or_completion_call(tmp_path, monkeypatch):
    write_cache(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Preparation must not contact any model endpoint")

    monkeypatch.setattr(audit, "model_ready", forbidden)
    monkeypatch.setattr(audit, "judge_group", forbidden)
    asyncio.run(audit.run(args_for(tmp_path, prepare_only=True)))
    manifest = json.loads((tmp_path / "reliability_audit/manifest.json").read_text())
    assert len(manifest["group_ids"]) == 12
    assert manifest["max_sent_chat_calls_per_group"] == 1


def test_default_audit_makes_at_most_twelve_sent_calls_and_keeps_failures_unknown(tmp_path, monkeypatch):
    groups, cache_hash = write_cache(tmp_path)
    original_folder = tmp_path / "judge_results"
    original_folder.mkdir()
    for group in groups:
        public = audit.observable_inputs(group, None)
        _, mapping = judge.build_judge_request(group["public_task"], public, seed=42, model="judge")
        result = {**parsed(mapping), "position_map": mapping, "request_sha256": "original_request"}
        original = {"ok": True, "source_cache_sha256": cache_hash, "judge_model": "judge", "result": result,
                    "attempts": [{"result": result, "text_limit": None}]}
        (original_folder / f"{group['id']}.json").write_text(json.dumps(original))
    sent_calls = []

    async def invalid_completion(task, inputs, **kwargs):
        _, mapping = judge.build_judge_request(task, inputs, seed=kwargs["seed"], model=kwargs["model"])
        original_map = json.loads(next(original_folder.glob("*.json")).read_text())["result"]["position_map"]
        assert [mapping[key]["key"] for key in sorted(mapping)] == list(reversed(
            [original_map[key]["key"] for key in sorted(original_map)]))
        sent_calls.append(mapping)
        return {"ok": False, "status": "invalid_judgment", "judgments": None,
                "chat_request_started": True, "position_map": mapping}

    monkeypatch.setattr(audit, "model_ready", lambda *args: True)
    monkeypatch.setattr(audit, "judge_group", invalid_completion)
    asyncio.run(audit.run(args_for(tmp_path)))
    assert len(sent_calls) == 12
    summary = json.loads((tmp_path / "reliability_audit/summary.json").read_text())
    assert summary["accepted_groups"] == 0 and summary["fulfillment_mean_absolute_difference"] is None
    assert summary["status_counts"] == {"invalid_judgment": 12}
