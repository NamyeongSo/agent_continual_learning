# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Budget and pairing invariants for the matched uniform training reference."""

import numpy as np
import pytest

from scripts.premise.analyze_bfcl_uniform_parent_reference import uniform_prefixes


def test_uniform_pool_contains_all_fifty_chronological_indices():
    queries = uniform_prefixes("s42_cp020", 1000, 20261022)
    assert queries.shape == (1000, 10)
    assert set(queries[:, 0]) == set(range(50))
    assert np.all(np.diff(np.sort(queries, axis=1), axis=1) > 0)


def test_shared_nested_prefixes_do_not_depend_on_model_or_reward():
    first = uniform_prefixes("s42_cp020", 100, 20261022)
    second = uniform_prefixes("s42_cp020", 100, 20261022)
    assert np.array_equal(first, second)
    for n in range(1, 11):
        assert np.array_equal(first[:, :n], second[:, :n])


def test_checkpoint_draws_are_distinct_and_empty_trials_are_rejected():
    assert not np.array_equal(uniform_prefixes("s42_cp020", 100, 71), uniform_prefixes("s43_cp020", 100, 71))
    with pytest.raises(ValueError, match="Positive"):
        uniform_prefixes("s42_cp020", 0, 71)
