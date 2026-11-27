# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Guard paired temperature coverage and distinguish overlap from action order."""

import copy

import pytest

from scripts.premise.compare_regenerated_temperatures import diagnostics, ordered_similarity, shared_rows


def action(value):
    return {"name": "act", "arguments": {"value": value}}


def test_ordered_similarity_preserves_order_and_content_normalization_is_only_syntactic():
    a, b = action("a"), action("b")
    assert ordered_similarity([b, a], [a, b]) == 0
    assert ordered_similarity([a], [a, b]) == .5
    assert ordered_similarity([], []) == 1
    assert ordered_similarity([], [a]) == 0
    assert ordered_similarity([a], [b], "loose") == 1
    first = [[{"name": "python", "arguments": {"code": "x = 1 # formatting"}}]]
    second = [[{"name": "python", "arguments": {"code": "x=1"}}]]
    d = diagnostics(first, second, "appworld", "strict")
    assert d["prefix_exact_fraction"] == 0
    assert d["content_f1"] == d["content_ordered_action_similarity"] == 1
    second[0][0]["arguments"]["code"] = "x=2"
    assert diagnostics(first, second, "appworld", "strict")["content_f1"] == 0


def test_shared_sequences_use_identical_pair_ids_and_constant_task_cohort_over_checkpoints():
    def row(task, point, pairs):
        return {"benchmark": "bfcl", "task": task, "kind": "cp0_complete_sequences", "mode": "strict",
                "earlier": 0, "later": point, "repeats": [{"reference_repeat": a, "prediction_repeat": b,
                                                           "f1": score} for a, b, score in pairs]}

    root = [(-1, 0, .1), (-1, 1, .2), (0, -1, .3), (0, 1, .4), (1, -1, .5), (1, 0, .6)]
    first = {"task_comparisons": [row("a", 0, root), row("a", 10, [(-1, 0, .2), (0, 0, .7), (1, 0, .8)]),
                                  row("b", 0, root), row("b", 10, [(0, 0, .2)])]}
    other = copy.deepcopy(first)
    other["task_comparisons"][0]["repeats"] = [other["task_comparisons"][0]["repeats"][i] for i in (3, 5)]
    other["task_comparisons"][1]["repeats"] = other["task_comparisons"][1]["repeats"][-1:]
    other["task_comparisons"][-1]["repeats"] = [{"reference_repeat": 0, "prediction_repeat": 1, "f1": .9}]
    a, b = shared_rows(first, other)
    assert len(a) == len(b) == 2 and {r["task"] for r in a} == {"a"}
    later = next(r for r in a if r["later"] == 10)
    assert later["sample_pairs"] == 1 and later["f1"] == pytest.approx(.8)
    assert {(s["reference_repeat"], s["prediction_repeat"]) for s in later["repeats"]} == {(1, 0)}
    assert len(first["task_comparisons"][0]["repeats"]) == 6  # Preserve source analysis.
