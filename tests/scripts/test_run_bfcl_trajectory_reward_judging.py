# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Checks for judge data isolation and bounded evidence compaction."""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "premise"))
queue = importlib.import_module("run_bfcl_trajectory_reward_judging")


def test_compaction_keeps_original_length_and_marks_missing_evidence():
    original = {"result": "x" * 80, "number": 7, "nested": ["kept", "y" * 60]}
    shortened, count = queue.compact_value(original, 20)
    assert count == 2
    assert shortened["result"] == {"text_prefix": "x" * 20, "truncated": True, "original_chars": 80}
    assert shortened["number"] == 7
    assert shortened["nested"][0] == "kept"
    assert original["result"] == "x" * 80


def test_recompaction_retains_original_observation_length():
    old = {"text_prefix": "a" * 40, "truncated": True, "original_chars": 10000}
    result, count = queue.compact_value(old, 10)
    assert result["original_chars"] == 10000
    assert result["text_prefix"] == "a" * 10
    assert count == 1


def test_official_success_and_identity_cannot_enter_judge_trace():
    group = {"traces": [{
        "available": True, "context": "candidate_01", "official_success": 1,
        "identity": {"private": "not public"}, "features": {"not": "prompt data"},
        "compact_trajectory": {"events": [{"type": "action", "name": "write", "arguments": "a" * 80}],
                               "lost_evidence": False, "truncated_fields": 0},
    }]}
    traces = queue.judge_inputs(group, 20)
    assert set(traces[0]) == {"key", "events", "lost_evidence", "truncated_fields"}
    assert traces[0]["lost_evidence"] is True
    assert traces[0]["truncated_fields"] == 1
    assert len(traces[0]["events"]) == 1
    assert group["traces"][0]["compact_trajectory"]["lost_evidence"] is False


def test_missing_trajectory_is_never_invented_as_empty_behavior():
    with pytest.raises(ValueError, match="Missing trajectories"):
        queue.judge_inputs({"traces": [{"available": False}]})


def test_interleaving_completes_each_prefix_before_advancing_checkpoint():
    groups = [
        {"id": f"{model}/{seed}/{cp}/{n}", "model": model, "seed": seed, "checkpoint": cp, "query_position": n}
        for model in ("Qwen", "Gemma") for seed in (42, 43) for cp in (10, 20) for n in (1, 0)
    ]
    ordered = queue.interleave_checkpoints(groups)
    assert len({g["id"] for g in ordered}) == 16
    assert [g["checkpoint"] for g in ordered[:8]] == [10] * 8
    for position in range(0, 16, 2):
        assert [g["query_position"] for g in ordered[position:position + 2]] == [0, 1]
        assert ordered[position]["model"] == ordered[position + 1]["model"]


def test_pre_tokenization_connection_refusals_do_not_exhaust_inference_budget():
    attempts = [{"result": {"status": "judge_transport_error", "error": "Connection refused"}}] * 3
    assert queue.counted_chat_attempts(attempts) == 0
    assert queue.retryable_infrastructure({"ok": False, "status": "judge_transport_error", "attempts": attempts})


def test_sent_but_unknown_completions_conservatively_consume_budget():
    old = {"result": {"status": "judge_transport_error", "prompt_tokens_counted": 15000}}
    new = {"result": {"status": "judge_transport_error", "chat_request_started": True}}
    assert queue.counted_chat_attempts([old, new, new]) == 3
    assert not queue.retryable_infrastructure({
        "ok": False, "status": "judge_transport_error", "attempts": [old, new, new],
    })


def test_budget_precheck_failure_does_not_count_as_an_inference():
    assert queue.counted_chat_attempts([{"result": {
        "status": "input_budget_exceeded", "prompt_tokens_counted": 25000,
    }}]) == 0
