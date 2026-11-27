# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

import numpy as np
import pytest

from scripts.premise.plot_bfcl_sample_size_contours import ellipse_points, fit_models, gaussian_moments


@pytest.mark.parametrize("rho", [-0.7, 0, 0.6])
def test_gaussian_contour_has_the_requested_probability_mass_mahalanobis_radius(rho):
    fit = gaussian_moments(rho)
    covariance = np.asarray(fit["covariance"])
    for mass in (0.5, 0.9):
        contour = ellipse_points(covariance, mass)
        squared_distance = np.einsum("ni,ij,nj->n", contour, np.linalg.inv(covariance), contour)
        assert squared_distance == pytest.approx(np.full(len(contour), -2 * np.log(1 - mass)))
        assert contour[0] == pytest.approx(contour[-1])


def test_unmeasured_and_singular_correlations_do_not_produce_invented_ellipses():
    assert not gaussian_moments(None)["defined"]
    assert not gaussian_moments(1)["defined"]
    assert not gaussian_moments(-1)["defined"]
    with pytest.raises(ValueError, match="between -1 and 1"):
        gaussian_moments(1.1)
    with pytest.raises(ValueError, match="positive eigenvalues"):
        ellipse_points(np.ones((2, 2)), 0.9)


def test_contour_parameters_preserve_each_measured_n_target_and_evaluator_correlation():
    sizes = []
    for n in range(1, 11):
        sizes.append({"n": n, "targets": {target: {
            "paired_checkpoint_count_mean": n, "defined_trials": 100, "total_trials": 100,
            "train10": {"mean_pearson": n / 20}, "val10": {"mean_pearson": -n / 40},
        } for target in ("full_train", "full_validation")}})
    summary = {"models": [{"label": "Synthetic", "errors": 0, "evaluations": 550,
                           "checkpoints": [{"key": "cp"}], "n10_matches_existing_summary": True, "sizes": sizes}]}
    fitted = fit_models(summary)[0]
    for entry in fitted["sizes"]:
        for target in ("full_train", "full_validation"):
            moments = entry["targets"][target]
            for selector, rho in (("train10", entry["n"] / 20), ("val10", -entry["n"] / 40)):
                assert moments[selector]["mean"] == [0, 0]
                assert moments[selector]["covariance"] == [[1, rho], [rho, 1]]
            assert moments["paired_checkpoint_count_mean"] == entry["n"]


def test_nonstandardized_covariance_changes_ellipse_scale_as_well_as_orientation():
    covariance = np.array([[4.0, 0], [0, 1.0]])
    points = ellipse_points(covariance, 0.5)
    radius = np.sqrt(-2 * np.log(0.5))
    assert np.max(np.abs(points[:, 0])) == pytest.approx(2 * radius)
    assert np.max(np.abs(points[:, 1])) == pytest.approx(radius)
