# SPDX-License-Identifier: Apache-2.0
"""Verify statistical outlier counts and selection-seed deduplication for rule/context candles."""

import numpy as np
import pytest

from scripts.ace.plot_rule_generality_candles import analyze, build_groups, tukey_stats


def test_tukey_whiskers_and_strict_fences():
    stats = tukey_stats([-3, 1, 2, 3, 7])
    assert stats["q1"] == 1 and stats["median"] == 2 and stats["q3"] == 3
    assert stats["lower_fence"] == -2 and stats["upper_fence"] == 6
    assert stats["lower_whisker"] == 1 and stats["upper_whisker"] == 3
    assert stats["low_count"] == stats["high_count"] == 1
    edges = tukey_stats([-2, 1, 2, 3, 6])
    assert edges["outlier_count"] == 0


def test_zero_iqr_and_no_nonfinite_imputation():
    assert tukey_stats([1, 1, 1, 1, 2])["high_count"] == 1
    for data in ([], [1, np.nan]):
        with pytest.raises(ValueError):
            tukey_stats(data)


def test_same_context_and_rules_selected_in_three_seeds_count_once_per_strategy():
    base = {"model": "qwen38", "mode": "thinking", "strategy": "best", "context_id": "c1", "task_id": "t1",
            "context_mean_cosine": ".5", "source_repeat": "0", "rules": "2"}
    empty = {**base, "context_id": "c2", "task_id": "t2", "context_mean_cosine": "", "rules": "0"}
    selections = [{**row, "seed": str(seed)} for row in (base, empty) for seed in (42, 43, 44)]
    selections += [{**base, "seed": "42", "strategy": "worst"}]
    contexts = {"c1": base, "c2": empty}
    rules = [{"rule_id": f"r{i}", "context_id": "c1", "rule_text": str(value),
              "mean_context_max_cosine": str(value)} for i, value in enumerate((.3, .7))]
    groups = build_groups(selections, contexts, rules)
    best = groups["qwen38", "thinking", "best"]
    assert best["selected_unique_contexts"] == 2 and best["empty_unique_contexts"] == 1
    assert len(best["context"]) == 1 and len(best["rule"]) == 2
    assert best["context"][0]["selected_seeds"] == "42;43;44"
    assert len(groups["qwen38", "thinking", "worst"]["rule"]) == 2


def test_outlier_rules_can_be_hidden_inside_normal_context_means():
    context_items, rule_items = [], []
    for i in range(5):
        common = {"model": "qwen38", "mode": "thinking", "strategy": "best", "context_id": f"c{i}"}
        context_items.append({**common, "item_id": f"c{i}", "score": .5})
        for j, value in enumerate((.1, .9) if i == 4 else (.5, .5)):
            rule_items.append({**common, "item_id": f"r{i}_{j}", "score": value})
    groups = {("qwen38", "thinking", "best"): {"context": context_items, "rule": rule_items,
                                               "selected_unique_contexts": 5, "empty_unique_contexts": 0}}
    summaries, outliers = analyze(groups)
    assert len(summaries) == 2 and outliers["context"] == []
    assert {r["tail"] for r in outliers["rule"]} == {"low", "high"}
    assert all(r["parent_context_outlier_tail"] == "none" for r in outliers["rule"])
