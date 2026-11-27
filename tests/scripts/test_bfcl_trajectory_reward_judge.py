# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Test grounded judge isolation, evidence validation, and independent token budgets."""

from __future__ import annotations

import asyncio
import copy
import json
import urllib.error

import pytest

from scripts.premise import bfcl_trajectory_reward_judge as judge


def group():
    task = {"user_turns": [{"turn_id": 0, "text": "Read the note."}, {"turn_id": 1, "text": "List notes."}],
            "tool_classes": ["Notes"], "score": "PRIVATE_SCORE_SENTINEL", "path": "PRIVATE_REFERENCE_SENTINEL"}
    traces = [{"key": f"private_{index}", "context": "PRIVATE_CONTEXT_SENTINEL", "model": "PRIVATE_MODEL_SENTINEL",
               "events": [{"type": "request", "turn": 0, "text": "Read the note."},
                          {"type": "action", "turn": 0, "name": "read_note", "arguments": {"title": "draft"}},
                          {"type": "observation", "turn": 0, "result": {"text": "Hello"}}],
               "lost_evidence": False, "truncated_fields": 0} for index in range(6)]
    return task, traces


def response(mapping):
    return {"schema_version": 1, "items": [{"id": identity, "a": [1, 0], "ae": [[2], []],
                                           "v": [0, 0, 0, 0], "e": [[], [], [], []], "h": 0,
                                           "he": [], "u": []} for identity in mapping]}


def request_and_map(seed=42):
    task, traces = group()
    return judge.build_judge_request(task, traces, seed=seed, model="test_model")


def test_prompt_uses_only_public_inputs_and_observable_events_with_anonymous_positions():
    payload, mapping = request_and_map()
    text = json.dumps(payload["messages"])
    assert all(sentinel not in text for sentinel in (
        "PRIVATE_SCORE_SENTINEL", "PRIVATE_REFERENCE_SENTINEL", "PRIVATE_CONTEXT_SENTINEL", "PRIVATE_MODEL_SENTINEL"))
    assert all(entry["key"] not in text for entry in mapping.values())
    assert set(mapping) == set("ABCDEF")
    assert "read_note" in text and "Hello" in text and "Notes" in text
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["temperature"] == 0


def test_anonymization_is_reproducible_but_randomizes_original_positions():
    payload, mapping = request_and_map(42)
    repeated, repeated_map = request_and_map(42)
    different, different_map = request_and_map(43)
    assert payload == repeated and mapping == repeated_map
    assert mapping != different_map
    assert {row["key"] for row in mapping.values()} == {row["key"] for row in different_map.values()}
    # The requests can be textually equal when identical traces are deliberately used.
    assert different["messages"][0] == payload["messages"][0]


def test_shared_inventory_counts_unreached_turns_as_zero_without_easy_denominator():
    _, mapping = request_and_map()
    parsed = judge.parse_judgment(json.dumps(response(mapping)), mapping)
    assert parsed["ok"]
    for item in parsed["judgments"].values():
        assert item["metrics"]["fulfillment"] == {
            "value": .5, "lower_bound": .5, "upper_bound": .5, "shared_turn_count": 2, "unknown_turns": 0,
        }


def test_unknown_attainment_preserves_fixed_denominator_and_never_becomes_zero():
    _, mapping = request_and_map()
    raw = response(mapping)
    raw["items"][0]["a"][1] = None
    raw["items"][0]["u"] = [{"field": "a1", "reason": "Relevant final state missing."}]
    parsed = judge.parse_judgment(json.dumps(raw), mapping)
    first = parsed["judgments"][mapping["A"]["key"]]
    assert first["metrics"]["fulfillment"]["value"] is None
    assert first["metrics"]["fulfillment"]["lower_bound"] == .5
    assert first["metrics"]["fulfillment"]["upper_bound"] == 1
    assert first["attainment"][1]["value"] is None


def test_confirmed_material_violation_requires_distinct_grounded_evidence():
    _, mapping = request_and_map()
    raw = response(mapping)
    raw["items"][0]["h"] = 1
    raw["items"][0]["he"] = [1, 2]
    parsed = judge.parse_judgment(json.dumps(raw), mapping)
    assert parsed["judgments"][mapping["A"]["key"]]["metrics"]["material_violation"]["value"] == 1
    raw["items"][0]["he"] = []
    assert not judge.parse_judgment(json.dumps(raw), mapping)["ok"]
    raw["items"][0]["h"] = None
    raw["items"][0]["u"] = [{"field": "h", "reason": "Claimed state cannot be verified."}]
    parsed = judge.parse_judgment(json.dumps(raw), mapping)
    assert parsed["ok"]
    assert parsed["judgments"][mapping["A"]["key"]]["metrics"]["material_violation"]["value"] is None


@pytest.mark.parametrize("case", ("missing_id", "duplicate_id", "extra_item_field", "wrong_denominator",
                                  "out_of_bounds",
                                  "duplicate_evidence", "no_positive_evidence", "boolean_score", "negative_score",
                                  "unknown_without_reason", "spurious_unknown_reason", "nonfinite"))
def test_invalid_judge_outputs_are_unknown_instead_of_zero(case):
    _, mapping = request_and_map()
    raw = response(mapping)
    item = raw["items"][0]
    if case == "missing_id":
        raw["items"].pop()
    elif case == "duplicate_id":
        raw["items"][1]["id"] = item["id"]
    elif case == "extra_item_field":
        item["score"] = 1
    elif case == "wrong_denominator":
        item["a"] = [1]
    elif case == "out_of_bounds":
        item["ae"][0] = [1000]
    elif case == "duplicate_evidence":
        item["ae"][0] = [2, 2]
    elif case == "no_positive_evidence":
        item["ae"][0] = []
    elif case == "boolean_score":
        item["v"][0] = True
    elif case == "negative_score":
        item["v"][0] = -.1
    elif case == "unknown_without_reason":
        item["v"][0] = None
    elif case == "spurious_unknown_reason":
        item["u"] = [{"field": "v0", "reason": "Not actually unknown."}]
    else:
        item["v"][0] = float("nan")
    parsed = judge.parse_judgment(json.dumps(raw), mapping)
    assert parsed["ok"] is False and parsed["judgments"] is None


def test_duplicate_json_keys_and_prose_are_rejected_instead_of_silently_repaired():
    _, mapping = request_and_map()
    raw = json.dumps(response(mapping))
    assert not judge.parse_judgment(raw.replace('"schema_version": 1', '"schema_version": 0,"schema_version": 1'),
                                    mapping)["ok"]
    assert not judge.parse_judgment("Here is my analysis: " + raw, mapping)["ok"]
    assert judge.parse_judgment("```json\n" + raw + "\n```", mapping)["ok"]
    assert not judge.parse_judgment(None, mapping)["ok"]


def test_source_compaction_marks_missing_evidence_without_adding_grading_metadata():
    task, traces = group()
    traces[0]["lost_evidence"] = True
    traces[0]["truncated_fields"] = 3
    traces[0]["events"][1]["validation"] = {"name_valid": False}
    payload, mapping = judge.build_judge_request(task, traces, seed=42, model="test_model")
    attempt_id = next(key for key, entry in mapping.items() if entry["key"] == traces[0]["key"])
    text = payload["messages"][1]["content"]
    data = json.loads(text.removeprefix("PUBLIC_DATA\n").removesuffix("\nEND_PUBLIC_DATA"))
    attempt = next(item for item in data["attempts"] if item["id"] == attempt_id)
    assert attempt["observations_incomplete"] is True and attempt["truncated_fields"] == 3
    assert attempt["events"][1]["validation"]["name_valid"] is False


def test_independent_input_budget_is_exactly_counted_before_any_chat_call(monkeypatch):
    calls = []

    def post(url, payload, timeout, api_key):
        calls.append(url)
        return {"count": 24001}

    monkeypatch.setattr(judge, "_post_json", post)
    task, traces = group()
    result = asyncio.run(judge.judge_group(task, traces, seed=42, base_url="http://test/v1", model="test_model"))
    assert result["status"] == "input_budget_exceeded" and result["judgments"] is None
    assert calls == ["http://test/tokenize"]


def test_client_parses_one_completion_and_records_usage_without_hidden_retries(monkeypatch):
    task, traces = group()
    _, mapping = request_and_map()
    calls = []

    def post(url, payload, timeout, api_key):
        calls.append(url)
        if url.endswith("/tokenize"):
            return {"count": 100}
        return {"usage": {"prompt_tokens": 100, "completion_tokens": 500},
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(response(mapping)),
                                                                     "reasoning_content": ""}}]}

    monkeypatch.setattr(judge, "_post_json", post)
    result = asyncio.run(judge.judge_group(task, traces, seed=42, base_url="http://test/v1", model="test_model"))
    assert result["ok"] and result["reasoning_characters"] == 0 and result["prompt_tokens_counted"] == 100
    assert calls == ["http://test/tokenize", "http://test/v1/chat/completions"]
    assert len(result["request_sha256"]) == 64


@pytest.mark.parametrize("case", ("transport", "truncated", "measured_prompt_over_budget",
                                  "measured_output_over_budget"))
def test_judge_transport_and_budget_failures_remain_unknown(monkeypatch, case):
    calls = []

    def post(url, payload, timeout, api_key):
        calls.append(url)
        if case == "transport":
            raise urllib.error.URLError("temporarily unavailable")
        if url.endswith("/tokenize"):
            return {"count": 100}
        return {"usage": {"prompt_tokens": 24001 if case == "measured_prompt_over_budget" else 100,
                          "completion_tokens": 2049 if case == "measured_output_over_budget" else 500},
                "choices": [{"finish_reason": "length", "message": {"content": "{"}}]}

    monkeypatch.setattr(judge, "_post_json", post)
    task, traces = group()
    original = copy.deepcopy(traces)
    result = asyncio.run(judge.judge_group(task, traces, seed=42, base_url="http://test", model="test_model"))
    assert result["ok"] is False and result["judgments"] is None
    assert len(calls) <= 2 and original == traces
