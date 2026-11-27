# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import copy

import numpy as np
import pytest

from scripts.premise.bfcl_full_pool_selection import STRATEGIES
from scripts.premise.bfcl_full_pool_statistics import REFERENCE, leave_one_seed_out, statistical_tests

REWARDS = ("mean", *(f"reward{i}" for i in range(1, 12)))


def _rows(models=("Qwen",), counts=None):
    counts = counts or {42: 1, 43: 1, 44: 1}
    rows = []
    for model in models:
        for seed in (42, 43, 44):
            for checkpoint in range(counts[seed]):
                for sampler_index, sampler in enumerate(STRATEGIES):
                    for reward_index, reward in enumerate(REWARDS):
                        rows.append({"model": model, "seed": seed, "checkpoint": checkpoint,
                                     "key": f"seed{seed}_cp{checkpoint}", "sampler": sampler, "reward": reward,
                                     "n": 10, "validation_lift": (sampler_index * 12 + reward_index) / 1000})
    return rows


def test_all_prespecified_methods_are_paired_and_holm_family_has_47_comparisons_per_model_and_block():
    tests = statistical_tests(_rows(("Qwen", "Gemma")), 20261021)
    assert len(tests) == 2 * 4 * 47
    expected = {f"{sampler}:{reward}" for sampler in STRATEGIES for reward in REWARDS} - {REFERENCE}
    for model in ("Qwen", "Gemma"):
        for block in (2, 3, 4, 5):
            family = [row for row in tests if row["model"] == model and row["block_size"] == block]
            assert {row["method"] for row in family} == expected
            assert all(row["family_size"] == 47 and row["reference"] == REFERENCE for row in family)
            assert all(row["checkpoints"] == 3 and row["exact"] and row["permutations"] == 8 for row in family)
            assert all(0 <= row["p_two_sided"] <= row["p_holm"] <= 1 for row in family)
    # These contrasts are fixed positive shifts, so seed-block sign flips have exactly two extreme signs.
    assert all(row["p_two_sided"] == 0.25 and row["p_holm"] == 1 for row in tests)


def test_effect_is_a_paired_checkpoint_average_in_reward_units_not_a_rho_or_percent_scaled_difference():
    rows = _rows(counts={42: 2, 43: 1, 44: 1})
    for row in rows:
        if row["sampler"] == "representative" and row["reward"] == "mean":
            row["validation_lift"] = {42: 0.1, 43: 0.2, 44: -0.4}[row["seed"]]
    tests = statistical_tests(rows, 7)
    effect = next(row for row in tests if row["method"] == "representative:mean" and row["block_size"] == 2)
    assert effect["difference"] == pytest.approx((0.1 + 0.1 + 0.2 - 0.4) / 4)
    assert effect["blocks"] == 3  # Blocks never combine different seeds.
    assert "mean_rho_difference" not in effect


def test_altering_held_seed_validation_cannot_choose_a_different_method_for_that_fold():
    rows = _rows()
    preferred = "representative:reward3"
    for row in rows:
        row["validation_lift"] = 0.5 if f"{row['sampler']}:{row['reward']}" == preferred else 0
    before = leave_one_seed_out(rows)
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["seed"] == 42:
            row["validation_lift"] = -100 if f"{row['sampler']}:{row['reward']}" == preferred else 100
    after = leave_one_seed_out(changed)
    original = next(row for row in before if row["held_out_seed"] == 42)
    updated = next(row for row in after if row["held_out_seed"] == 42)
    assert original["method"] == updated["method"] == preferred
    assert original["development_lift"] == updated["development_lift"] == 0.5
    assert original["held_out_lift"] != updated["held_out_lift"]
    assert updated["retrospective"] is True


def test_development_selection_weights_seeds_equally_even_when_checkpoint_counts_differ():
    rows = _rows(counts={42: 8, 43: 1, 44: 1})
    for row in rows:
        method = f"{row['sampler']}:{row['reward']}"
        row["validation_lift"] = 0
        if method == "representative:mean":
            row["validation_lift"] = 0.8 if row["seed"] == 42 else 0
        if method == "cluster_stratified:mean":
            row["validation_lift"] = 0.6
    held44 = next(row for row in leave_one_seed_out(rows) if row["held_out_seed"] == 44)
    assert held44["method"] == "cluster_stratified:mean"
    assert held44["development_lift"] == pytest.approx(0.6)
    assert held44["held_out_lift"] == pytest.approx(0.6)


def test_ties_use_sampler_then_reward_first_appearance_order_without_viewing_held_performance():
    rows = _rows(("Qwen", "Gemma"))
    for row in rows:
        row["validation_lift"] = 0
    ordinary = leave_one_seed_out(rows)
    assert len(ordinary) == 6 and all(row["method"] == REFERENCE for row in ordinary)
    # Place another sampler and reward first to explicitly fix their priority on ties.
    rows.sort(key=lambda row: (row["sampler"] != "representative", row["reward"] != "reward4"))
    reordered = leave_one_seed_out(rows)
    assert all(row["method"] == "representative:reward4" for row in reordered)


@pytest.mark.parametrize("function", [leave_one_seed_out, lambda rows: statistical_tests(rows, 7)])
def test_missing_method_checkpoint_is_rejected_instead_of_silently_changing_the_comparison_cohort(function):
    rows = _rows()
    rows = [row for row in rows if not (row["key"] == "seed42_cp0" and row["sampler"] == "representative"
                                      and row["reward"] == "reward2")]
    with pytest.raises(ValueError, match="same checkpoint keys"):
        function(rows)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [({"n": 9}, "n=10"), ({"validation_lift": np.nan}, "finite"),
     ({"model": "OSS"}, "Qwen/Gemma"), ({"checkpoint": 99}, "identity differs")],
)
def test_mismatched_scope_invalid_outcomes_and_mispaired_checkpoint_identity_are_rejected(mutation, message):
    rows = _rows()
    rows[1].update(mutation)
    with pytest.raises(ValueError, match=message):
        leave_one_seed_out(rows)


def test_duplicate_rows_and_incomplete_prespecified_method_families_are_rejected():
    rows = _rows()
    with pytest.raises(ValueError, match="Duplicate"):
        leave_one_seed_out([*rows, rows[0]])
    with pytest.raises(ValueError, match="twelve"):
        leave_one_seed_out([row for row in rows if row["reward"] != "reward11"])
