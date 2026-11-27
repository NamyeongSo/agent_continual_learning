# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Validate F1 reliability statistics independently of model calls."""

import pytest

from scripts.premise.audit_group_relative_f1 import action_prf, analyze, content_signature, strict_f1


def action(value):
    return [{"name": "act", "arguments": {"value": value}}]


def case_rows(benchmark, initial, learned):
    case = {"id": benchmark, "benchmark": benchmark, "task": "task"}
    rows = [{"case": benchmark, "benchmark": benchmark, "task": "task", "context": "initial",
             "repeat": -1, "status": "completed", "actions": action("a")}]
    for context, values in (("initial", initial), ("learned", learned)):
        rows.extend({"case": benchmark, "benchmark": benchmark, "task": "task", "context": context,
                     "repeat": i, "status": "completed", "actions": action(value)} for i, value in enumerate(values))
    return case, rows


def test_statistics_separate_same_context_noise_from_context_effect_and_keep_benchmarks_separate():
    stable, stable_rows = case_rows("bfcl", ["a"] * 4, ["b"] * 4)
    noisy, noisy_rows = case_rows("appworld", ["a", "b", "a", "b"], ["b"] * 4)
    prepared = {"model": "test", "temperature": 1, "repeats": 4, "cases": [stable, noisy]}
    result = analyze(prepared, stable_rows + noisy_rows)
    bfcl, appworld = result["benchmarks"]["bfcl"], result["benchmarks"]["appworld"]
    assert bfcl["same_context_f1"] == 1 and bfcl["changed_context_f1"] == 0
    assert bfcl["fixed_reference_repeat_sd"] == 0 and bfcl["perfectly_repeated_prefix_fraction"] == 1
    assert appworld["same_context_f1"] == pytest.approx(1 / 3)
    assert appworld["changed_context_f1"] == 0.5 and appworld["context_gap"] < 0
    assert appworld["fixed_reference_repeat_sd"] == 0.5
    assert appworld["learned_same_context_f1"] == 1 and appworld["learned_same_context_exact_match"] == 1
    assert appworld["initial_fixed_reference_f1_distribution"]["zero_fraction"] == 0.5
    assert len(result["action_examples"]) == 3


def test_stable_zero_f1_does_not_imply_stable_actions_and_errors_are_not_zero_filled():
    floor, rows = case_rows("appworld", ["b", "c", "d", "e"], ["f"] * 4)
    prepared = {"model": "test", "temperature": 1, "repeats": 4, "cases": [floor]}
    metrics = analyze(prepared, rows)["benchmarks"]["appworld"]
    assert metrics["fixed_reference_repeat_sd"] == 0
    assert metrics["zero_reference_f1_prefix_fraction"] == 1
    assert metrics["same_context_f1"] == 0 and metrics["perfectly_repeated_prefix_fraction"] == 0
    rows[-1].update(status="error", error_type="truncated")
    metrics = analyze(prepared, rows)["benchmarks"]["appworld"]
    assert metrics["complete_prefixes"] == 0 and metrics["errors"] == {"truncated": 1}
    assert "same_context_f1" not in metrics


def test_content_diagnostic_ignores_only_python_formatting_or_terminal_duration():
    a = [{"name": "execute_python", "arguments": {"code": "x=1 # comment"}}]
    b = [{"name": "execute_python", "arguments": {"code": "x = 1"}}]
    c = [{"name": "execute_python", "arguments": {"code": "x = 2"}}]
    assert content_signature(a, "appworld") == content_signature(b, "appworld")
    assert content_signature(a, "appworld") != content_signature(c, "appworld")
    assert a[0]["arguments"]["code"] == "x=1 # comment"
    a = [{"name": "terminal_command", "arguments": {"keystrokes": "ls\n", "duration": 1}}]
    b = [{"name": "terminal_command", "arguments": {"keystrokes": "ls\n", "duration": .1}}]
    c = [{"name": "terminal_command", "arguments": {"keystrokes": "pwd\n", "duration": 1}}]
    assert content_signature(a, "terminalbench2") == content_signature(b, "terminalbench2")
    assert content_signature(a, "terminalbench2") != content_signature(c, "terminalbench2")


def test_precision_recall_use_multiset_counts_and_loose_ignores_arguments():
    a, b = action("a")[0], action("b")[0]
    strict = action_prf([a, a, b], [a, a])
    assert strict == {"precision": 2 / 3, "recall": 1, "f1": 0.8}
    assert strict["f1"] == strict_f1([a, a, b], [a, a])
    assert action_prf([a], [b]) == {"precision": 0, "recall": 0, "f1": 0}
    assert action_prf([a], [b], "loose") == {"precision": 1, "recall": 1, "f1": 1}
    assert action_prf([], []) == {"precision": 1, "recall": 1, "f1": 1}
    for predicted, reference in (([], [a]), ([a], [])):
        assert action_prf(predicted, reference) == {"precision": 0, "recall": 0, "f1": 0}


def test_precision_recall_direction_and_macro_f1_are_explicit():
    case, rows = case_rows("bfcl", ["a", "a"], ["a", "a"])
    extra = {"name": "extra", "arguments": {}}
    rows[-1]["actions"].append(extra)
    prepared = {"model": "test", "temperature": 1, "repeats": 2, "cases": [case]}
    result = analyze(prepared, rows)
    metrics = result["benchmarks"]["bfcl"]["agreement_metrics"]
    for mode in ("strict", "loose"):
        fixed = metrics[mode]["fixed_reference"]["learned"]
        assert fixed["precision"]["mean"] == 0.75
        assert fixed["recall"]["mean"] == 1
        assert fixed["f1"]["mean"] == pytest.approx((1 + 2 / 3) / 2)
        assert fixed["f1"]["mean"] != pytest.approx(2 * 0.75 / 1.75)
        cross = metrics[mode]["pairwise"]["cross_context"]
        assert cross["precision"] == 0.75 and cross["recall"] == 1
        learned = metrics[mode]["pairwise"]["learned_self"]
        assert learned["precision"] == learned["recall"] == 0.75
    assert metrics["strict"]["fixed_reference"]["initial"]["f1"]["mean"] == result[
        "benchmarks"]["bfcl"]["initial_aggregate_f1_mean"]


def test_loose_native_actions_compare_types_not_command_text_or_duration():
    a = [{"name": "terminal_command", "arguments": {"keystrokes": "ls\n", "duration": 1}}]
    b = [{"name": "terminal_command", "arguments": {"keystrokes": "pwd\n", "duration": .1}}]
    assert action_prf(a, b, "strict")["f1"] == 0
    assert action_prf(a, b, "loose")["f1"] == 1
