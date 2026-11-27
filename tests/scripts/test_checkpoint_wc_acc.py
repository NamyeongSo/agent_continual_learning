# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import copy

import pytest

from scripts.utils.measure_checkpoint_wc_acc import stage_score


def fixture():
    order = ["a1", "a2", "b1", "b2", "c1", "c2"]
    outcomes = {cp: dict.fromkeys(order, 0) for cp in (2, 4, 6)}
    outcomes[4].update(a1=1, a2=0, b1=1, b2=0)
    outcomes[6].update(a1=0, a2=1, b1=1, b2=1, c1=0, c2=0)
    return order, outcomes


def test_minimum_is_of_block_means_and_excludes_its_own_learning_endpoint():
    order, outcomes = fixture()
    result, blocks = stage_score(order, outcomes, 3, block_size=2)
    assert blocks[0]["minimum_score"] == .5
    assert blocks[0]["observed_checkpoints"] == "[4, 6]"
    assert result["wc_acc"] == .5
    assert result["matched_current_accuracy"] == .5


def test_recovery_does_not_erase_an_earlier_observed_drop():
    order, outcomes = fixture()
    outcomes[4].update(a1=0, a2=0)
    outcomes[6] = dict.fromkeys(order, 1)
    result, blocks = stage_score(order, outcomes, 3, block_size=2)
    assert blocks[0]["minimum_cp"] == 4
    assert result["wc_acc"] == pytest.approx(2 / 3)
    assert result["matched_current_accuracy"] == 1


def test_missing_outcomes_use_a_common_task_cohort_and_empty_blocks_are_undefined():
    order, outcomes = fixture()
    outcomes[4]["a2"] = None
    result, blocks = stage_score(order, outcomes, 3, block_size=2)
    assert blocks[0]["common_task_ids"] == '["a1"]'
    assert blocks[0]["excluded_task_ids"] == '["a2"]'
    assert result["common_tasks"] == 5
    assert result["strict_full_task_wc_acc"] is None
    assert result["wc_acc"] <= result["matched_current_accuracy"]
    outcomes[4]["a1"] = None
    result, _ = stage_score(order, outcomes, 3, block_size=2)
    assert result["wc_acc"] is None and result["status"] == "missing_block"


def test_first_stage_includes_zero_outcomes_and_later_observations_do_not_leak_back():
    order, outcomes = fixture()
    outcomes[2].update(a1=1, a2=0)
    result, _ = stage_score(order, outcomes, 1, block_size=2)
    assert result["wc_acc"] == .5 and result["past_min_accuracy"] is None
    original, _ = stage_score(order, outcomes, 2, block_size=2)
    changed = copy.deepcopy(outcomes)
    changed[6] = dict.fromkeys(order, 1)
    later, _ = stage_score(order, changed, 2, block_size=2)
    assert later == original
