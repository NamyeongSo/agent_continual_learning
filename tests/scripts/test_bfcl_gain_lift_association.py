# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.
"""Separate between-seed association from within-seed checkpoint association."""

import numpy as np
import pytest

from scripts.premise.analyze_bfcl_gain_lift_association import center_by_group, correlation


def test_seed_centering_removes_between_seed_confounding():
    x = [0., 1., 2., 10., 11., 12.]
    y = [2., 1., 0., 12., 11., 10.]
    seeds = [42, 42, 42, 43, 43, 43]
    assert correlation(x, y) > .9
    assert correlation(center_by_group(x, seeds), center_by_group(y, seeds)) == pytest.approx(-1)


def test_constant_and_roundoff_vectors_have_no_defined_correlation():
    assert correlation([1, 2, 3], [0, 1e-16, -1e-16]) is None
    assert correlation([1, 1, 1], [1, 2, 3], rank=True) is None


def test_spearman_keeps_average_ranks_for_ties():
    assert correlation([0, 0, 1, 1], [1, 1, 0, 0], rank=True) == pytest.approx(-1)


@pytest.mark.parametrize("x,y", [([1], [2]), ([1, 2], [1]), ([1, np.nan], [1, 2])])
def test_invalid_pairs_are_rejected(x, y):
    with pytest.raises(ValueError, match="paired finite"):
        correlation(x, y)
