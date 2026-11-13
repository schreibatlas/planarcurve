from __future__ import annotations

import numpy as np
import pytest

from planarcurve.fitting.policy import SplineEndpointPolicy
from planarcurve.fitting.spline import (
    SplineFitOptions,
    SplineObservation,
    fit_spline_control_points,
)
from planarcurve.spline import BSplineCurve, spline_design_matrix


def basis_curve() -> BSplineCurve:
    return BSplineCurve(
        knots=(0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0, 1.0),
        control_points=(
            (0.0, 0.0),
            (1.0, 1.8),
            (2.3, 2.2),
            (3.8, 1.1),
            (5.0, 0.4),
            (6.0, 0.7),
        ),
    )


def observations_for(curve: BSplineCurve) -> list[SplineObservation]:
    rng = np.random.default_rng(73)
    controls = np.asarray(curve.control_points)
    observations = []
    for count, matrix, offset in (
        (51, np.array([[1.2, 0.15], [0.0, 0.9]]), np.array([12.0, 5.0])),
        (43, np.array([[0.85, -0.1], [0.0, 1.1]]), np.array([-4.0, 8.0])),
    ):
        u = np.linspace(0.0, 1.0, count)
        xy = spline_design_matrix(u, curve) @ controls
        observed = xy @ matrix.T + offset + rng.normal(0.0, 0.005, xy.shape)
        if count == 51:
            observed[count // 2] += [8.0, -6.0]
        canonical = (observed - offset) @ np.linalg.inv(matrix).T
        observations.append(SplineObservation(u=u, xy=canonical))
    return observations


def test_sparse_irls_recovers_shared_spline_and_locked_endpoints():
    basis = basis_curve()
    truth = np.asarray(basis.control_points)
    seed = truth.copy()
    seed[1, 1] += 0.35
    seed[2, 0] -= 0.25
    seed[3, 1] += 0.2
    options = SplineFitOptions(
        smoothness=1e-5,
        seed_weight=1e-6,
        robust_scale=0.1,
        irls_iterations=4,
    )
    result = fit_spline_control_points(
        basis, observations_for(basis), seed_control_points=seed, options=options
    )
    repeated = fit_spline_control_points(
        basis, observations_for(basis), seed_control_points=seed, options=options
    )
    fitted = np.asarray(result.curve.control_points)
    np.testing.assert_array_equal(fitted[[0, -1]], truth[[0, -1]])
    np.testing.assert_allclose(fitted, repeated.curve.control_points, atol=0.0)
    assert np.max(np.linalg.norm(fitted[1:-1] - truth[1:-1], axis=1)) < 0.1
    u = np.linspace(0.0, 1.0, 201)
    design = spline_design_matrix(u, basis)
    curve_error = np.linalg.norm(design @ fitted - design @ truth, axis=1)
    assert np.percentile(curve_error, 95) < 0.05
    assert result.residual_rms < 1.1
    assert result.iterations == 4


def test_observation_weights_and_endpoint_fit_policy():
    basis = basis_curve()
    u = np.linspace(0.0, 1.0, 31)
    xy = spline_design_matrix(u, basis) @ np.asarray(basis.control_points)
    observation = SplineObservation(u=u, xy=xy, weight=np.ones(len(u)))
    seed = np.asarray(basis.control_points).copy()
    seed[0] += [1.0, -1.0]
    seed[-1] += [-1.0, 1.0]
    options = SplineFitOptions(
        smoothness=0.0,
        seed_weight=0.0,
        endpoint_policy=SplineEndpointPolicy(start="free", end="anchored"),
        endpoint_sigma=0.1,
    )
    result = fit_spline_control_points(
        basis, [observation], seed_control_points=seed, options=options
    )
    fitted = np.asarray(result.curve.control_points)
    assert not np.allclose(fitted[0], seed[0])
    assert np.linalg.norm(fitted[-1] - seed[-1]) < np.linalg.norm(
        np.asarray(basis.control_points)[-1] - seed[-1]
    )


def test_spline_fit_rejects_invalid_observations_and_options():
    basis = basis_curve()
    bad = SplineObservation(u=np.array([0.0, 1.0]), xy=np.array([[0.0, 0.0], [np.nan, 1.0]]))
    with pytest.raises(ValueError, match="finite"):
        fit_spline_control_points(basis, [bad])
    with pytest.raises(ValueError, match="robust scale"):
        SplineFitOptions(robust_scale=0.0)
