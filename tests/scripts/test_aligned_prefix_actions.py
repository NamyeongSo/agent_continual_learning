# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Check prefix alignment, validity, independent controls and task aggregation."""

import copy
from statistics import fmean

from scripts.premise.analyze_aligned_prefix_actions import (
    METRICS,
    aggregate_prefixes,
    independent_pairs,
    score_prefix_pair,
    valid_prefixes,
)
from scripts.premise.audit_group_relative_f1 import action_prf


def action(value):
    return [{"name": "act", "arguments": {"value": value}}]


class Scorer:
    def score(self, prediction, reference, mode):
        assert len(prediction) == len(reference) == 1
        return {"bleu1": 0, "bleu4": 0, "meteor": 0}


def test_actions_at_different_prefixes_cannot_match_and_lexical_scoring_keeps_unit():
    a, b = action("a"), action("b")
    assert action_prf([*a, *b], [*b, *a])["f1"] == 1
    scores = [score_prefix_pair(p, r, "bfcl", "strict", Scorer()) for p, r in [(a, b), (b, a)]]
    assert fmean(r["f1"] for r in scores) == 0
    assert all(r["exact_action_list"] == 0 for r in scores)


def test_ast_diagnostic_preserves_different_constants():
    a = [{"name": "python", "arguments": {"code": "x=1 # comment"}}]
    b = [{"name": "python", "arguments": {"code": "x = 1"}}]
    row = score_prefix_pair(a, b, "appworld", "strict", Scorer())
    assert row["f1"] == 0 and row["content_f1"] == 1
    b[0]["arguments"]["code"] = "x=2"
    assert score_prefix_pair(a, b, "appworld", "strict", Scorer())["content_f1"] == 0


def test_independent_controls_never_compare_same_draw_with_itself():
    assert len(independent_pairs(0, 0, bank=True)) == 6
    assert all(i != j for i, j in independent_pairs(0, 0, bank=True))
    assert independent_pairs(0, 0, reference=True) == [(-1, 0), (-1, 1)]
    assert len(independent_pairs(0, 10, bank=True)) == 6


def test_validity_intersects_temperatures_repeats_and_all_checkpoints():
    cases = [{"id": "one", "benchmark": "bfcl", "task": "task"},
             {"id": "two", "benchmark": "bfcl", "task": "task"}]
    prepared = {"cases": cases, "checkpoints": {"bfcl": [0, 10]}}
    index = {(c["id"], p, k): {"status": "completed"} for c in cases for p in [0, 10]
             for k in ([-1, 0, 1] if p == 0 else [0, 1])}
    other = copy.deepcopy(index)
    other[("two", 10, 1)] = {"status": "error", "error_type": "timeout"}
    included, excluded = valid_prefixes(prepared, [index, other])
    assert [c["id"] for c in included] == ["one"]
    assert excluded[0]["failures"] == [{"temperature": 1, "checkpoint": 10, "repeat": 1, "error_type": "timeout"}]
    del other[("one", 0, -1)]
    assert valid_prefixes(prepared, [index, other])[0] == []


def test_mean_prefixes_then_equal_tasks_instead_of_length_weighting():
    rows = []
    for task, count, value in [("short", 1, 1), ("long", 3, 0)]:
        for _ in range(count):
            scores = dict.fromkeys(METRICS, value)
            scores.update({f"{prefix}_{k}": value for prefix in ("self", "gap") for k in METRICS[:11]})
            rows.append({"temperature": 0, "cohort": "valid_prefixes", "view": "all_pairs", "benchmark": "bfcl",
                         "task": task, "mode": "strict", "earlier": 0, "later": 0, "distance": 0, **scores})
    _, benches, curves = aggregate_prefixes(rows)
    assert benches[0]["tasks"] == 2 and benches[0]["prefixes"] == 4
    assert benches[0]["f1"] == curves[0]["f1"] == .5
