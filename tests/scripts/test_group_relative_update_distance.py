# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Check distance bins, independent-repeat controls and matched aggregation."""

import json

import pytest

from scripts.premise.audit_group_relative_update_distance import analyze, context_name, pair_scores, request_audit


def action(value):
    return [{"name": "act", "arguments": {"value": value}}]


def fixture(cases):
    points = [0, 10, 20]
    prepared = {"protocol": "test", "model": "test", "temperature": 1, "repeats": 2,
                "distance_definition": "steps", "aggregation": "equal tasks",
                "checkpoints": {"bfcl": points}, "cases": []}
    draws = []
    for identifier, task, banks in cases:
        prepared["cases"].append({"id": identifier, "benchmark": "bfcl", "task": task,
                                  "context_sha256": {context_name(p): str(p) for p in points}})
        draws.extend({"case": identifier, "context": context_name(p), "repeat": k,
                      "status": "completed", "actions": action(value)}
                     for p, bank in zip(points, banks, strict=True) for k, value in enumerate(bank))
    return prepared, draws


def strict_distances(summary):
    return {r["distance"]: r for r in summary["distances"] if r["mode"] == "strict"}


def test_all_checkpoint_pairs_and_no_self_draw_bias():
    prepared, draws = fixture([("one", "task", [["a", "b"], ["a", "a"], ["c", "c"]])])
    result = analyze(prepared, draws)
    distances = strict_distances(result)
    assert {d: r["checkpoint_pairs"] for d, r in distances.items()} == {0: 3, 10: 2, 20: 1}
    assert distances[0]["f1"] == pytest.approx(2 / 3)
    assert distances[10]["f1"] == .25
    assert distances[20]["f1"] == 0
    assert distances[0]["sampling_adjusted_gap"] == 0
    assert pair_scores([action("a"), action("b")], [action("a"), action("b")], "strict", True)["f1"] == 0


def test_precision_is_later_prediction_and_zero_distance_is_direction_neutral():
    a = action("a")
    b = [*a, {"name": "other", "arguments": {}}]
    scores = pair_scores([a, a], [b, b], "strict")
    assert scores["precision"] == .5 and scores["recall"] == 1
    scores = pair_scores([a, b], [a, b], "strict", True)
    assert scores["precision"] == scores["recall"] == .75


def test_equal_task_weights_and_global_complete_prefix_mask():
    prepared, draws = fixture([
        ("one", "taskA", [["a", "a"], ["a", "a"], ["a", "a"]]),
        ("two", "taskA", [["a", "a"], ["a", "a"], ["a", "a"]]),
        ("three", "taskB", [["b", "c"], ["b", "c"], ["b", "c"]]),
        ("four", "taskB", [["a", "a"], ["a", "a"], ["a", "a"]]),
    ])
    draws[-1].update(status="error", error_type="TimeoutError", error="test")
    result = analyze(prepared, draws)
    assert result["coverage"]["bfcl"]["included_cases"] == ["one", "two", "three"]
    assert strict_distances(result)[0]["f1"] == .5
    assert strict_distances(result)[10]["f1"] == .75
    assert all(r["prefixes"] == 3 and r["tasks"] == 2 for r in result["distances"])
    assert result["coverage"]["bfcl"]["error_draws"] == 1
    with pytest.raises(ValueError, match="Incomplete"):
        analyze(prepared, draws[:-1])
    interim = analyze(prepared, draws[:-1], require_complete=False)
    assert interim["coverage"]["bfcl"]["missing_draws"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        analyze(prepared, [*draws, draws[0]])


def test_request_audit_counts_failed_attempts_and_checks_actual_output_cap(tmp_path):
    prepared = {"model": "openai/test", "model_call_settings": {"temperature": 1, "max_tokens": None,
                "timeout": 600, "num_retries": 1}, "cases": []}
    request = {"model": "test", "temperature": 1}
    (tmp_path / "calls").mkdir()
    call = tmp_path / "calls/one.jsonl"
    call.write_text(json.dumps({"request": request, "error_type": "TimeoutError"}) + "\n"
                    + json.dumps({"request": request, "response": {"usage": {"completion_tokens": 3000,
                                 "prompt_tokens": 5000}}}) + "\n")
    audit = request_audit(prepared, tmp_path)
    assert audit["all_requested_settings_match"] and audit["new_physical_calls"] == 2
    assert audit["new_error_attempts"] == 1 and audit["measured_output_tokens"] == 3000
    call.write_text(json.dumps({"request": {**request, "max_tokens": 2048}}) + "\n")
    assert not request_audit(prepared, tmp_path)["all_requested_settings_match"]


def test_content_match_distinguishes_native_duration_from_command_changes():
    prepared, draws = fixture([("one", "task", [["a", "a"], ["a", "a"], ["b", "b"]])])
    prepared["checkpoints"] = {"terminalbench2": [0, 10, 20]}
    prepared["cases"][0]["benchmark"] = "terminalbench2"
    for row in draws:
        point = int(row["context"][2:])
        row["actions"] = [{"name": "terminal_command", "arguments": {
            "keystrokes": "ls\n" if point < 20 else "pwd\n", "duration": 1 if point == 0 else .1}}]
    result = analyze(prepared, draws)
    pair = next(r for r in result["checkpoint_pairs"]
                if r["mode"] == "strict" and r["earlier"] == 0 and r["later"] == 10)
    assert pair["f1"] == 0 and pair["content_action_match"] == 1
    pair = next(r for r in result["checkpoint_pairs"]
                if r["mode"] == "loose" and r["earlier"] == 10 and r["later"] == 20)
    assert pair["f1"] == 1 and pair["content_action_match"] == 0
