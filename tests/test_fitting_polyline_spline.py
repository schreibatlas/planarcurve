from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import savgol_filter

from planarcurve.fitting.policy import SplineEndpointPolicy
from planarcurve.fitting.spline import (
    PolylineSplineFitOptions,
    fit_polyline_spline,
)
from planarcurve.polyline import distance_to_polyline, normalized_arclength, unit_tangents
from planarcurve.spline import BSplineCurve


def _options(
    *,
    max_control_points: int = 12,
    candidate_control_counts: tuple[int, ...] = (4, 6, 8, 10, 12),
    **kwargs: object,
) -> PolylineSplineFitOptions:
    return PolylineSplineFitOptions(
        max_control_points=max_control_points,
        candidate_control_counts=candidate_control_counts,
        **kwargs,
    )


def _cubic_bezier(parameters: np.ndarray) -> np.ndarray:
    controls = np.array([[0.0, 0.0], [30.0, 60.0], [70.0, -60.0], [100.0, 0.0]])
    t = parameters[:, None]
    return (
        (1.0 - t) ** 3 * controls[0]
        + 3.0 * (1.0 - t) ** 2 * t * controls[1]
        + 3.0 * (1.0 - t) * t**2 * controls[2]
        + t**3 * controls[3]
    )


def _angle_error(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.abs((first - second + np.pi) % (2.0 * np.pi) - np.pi)


def test_fit_known_smooth_curves_and_select_minimum_for_a_line():
    parameters = np.linspace(0.0, 1.0, 161)
    rng = np.random.default_rng(23)
    fixtures = (
        (
            "arc",
            lambda t: np.column_stack(
                (50.0 * np.cos(-0.9 + 1.8 * t), 50.0 * np.sin(-0.9 + 1.8 * t))
            ),
        ),
        ("cubic S", _cubic_bezier),
        (
            "open loop-like arc",
            lambda t: np.column_stack(
                (
                    30.0 * np.cos((-0.85 + 1.7 * t) * np.pi),
                    30.0 * np.sin((-0.85 + 1.7 * t) * np.pi),
                )
            ),
        ),
    )

    line = np.column_stack((120.0 * parameters, np.zeros_like(parameters)))
    line_result = fit_polyline_spline(line)
    assert isinstance(line_result.curve, BSplineCurve)
    assert line_result.diagnostics.control_count == 4
    assert line_result.diagnostics.met_target_error
    np.testing.assert_allclose(
        line_result.curve.evaluate([0.0, 1.0], derivatives=False).xy,
        line[[0, -1]],
        atol=1e-10,
    )

    for name, make_curve in fixtures:
        observed = make_curve(parameters)
        if name != "open loop-like arc":
            observed = observed + rng.normal(0.0, 0.02, observed.shape)
        result = fit_polyline_spline(observed, options=_options())
        dense_parameters = np.linspace(0.0, 1.0, 3001)
        truth = make_curve(dense_parameters)
        sampled = result.curve.sample_by_arclength(180).xy
        errors = distance_to_polyline(sampled, truth)
        assert np.percentile(errors, 95) < 0.015 * np.sum(
            np.linalg.norm(np.diff(observed, axis=0), axis=1)
        ), name
        assert result.diagnostics.control_count >= 4


def test_pixel_staircase_fit_improves_position_heading_and_curvature():
    x = np.linspace(0.0, 200.0, 201)
    quantized = np.rint(np.column_stack((x, 0.0015 * (x - 100.0) ** 2)))
    points = quantized[np.concatenate(([True], np.any(np.diff(quantized, axis=0) != 0.0, axis=1)))]
    result = fit_polyline_spline(points)

    spline_samples = result.curve.sample_by_arclength(201)
    progress = np.linspace(0.0, 1.0, len(spline_samples.u))
    dense_x = np.linspace(0.0, 200.0, 10001)
    truth = np.column_stack((dense_x, 0.0015 * (dense_x - 100.0) ** 2))
    truth_progress = normalized_arclength(truth)
    truth_heading = np.arctan(0.003 * (dense_x - 100.0))
    expected_heading = np.interp(progress, truth_progress, truth_heading)
    spline_heading = np.arctan2(spline_samples.tangent[:, 1], spline_samples.tangent[:, 0])
    guard = (progress > 0.08) & (progress < 0.92)
    spline_error = np.rad2deg(_angle_error(spline_heading, expected_heading))

    position_error = distance_to_polyline(spline_samples.xy, truth)
    assert np.median(position_error) <= 0.20
    assert np.percentile(position_error, 95) <= 0.45
    assert np.sqrt(np.mean(spline_error[guard] ** 2)) <= 2.5
    assert np.percentile(spline_error[guard], 95) <= 6.0
    assert np.all(spline_samples.curvature[guard] > 0.0)

    raw_progress = normalized_arclength(points)
    raw_heading = np.arctan2(unit_tangents(points)[:, 1], unit_tangents(points)[:, 0])
    raw_truth_heading = np.interp(raw_progress, truth_progress, truth_heading)
    raw_error = np.rad2deg(_angle_error(raw_heading, raw_truth_heading))
    raw_guard = (raw_progress > 0.08) & (raw_progress < 0.92)

    smoothed = savgol_filter(points, window_length=11, polyorder=3, axis=0)
    smooth_progress = normalized_arclength(smoothed)
    smooth_heading = np.arctan2(unit_tangents(smoothed)[:, 1], unit_tangents(smoothed)[:, 0])
    smooth_truth_heading = np.interp(smooth_progress, truth_progress, truth_heading)
    smooth_error = np.rad2deg(_angle_error(smooth_heading, smooth_truth_heading))
    smooth_guard = (smooth_progress > 0.08) & (smooth_progress < 0.92)

    spline_rmse = np.sqrt(np.mean(spline_error[guard] ** 2))
    raw_rmse = np.sqrt(np.mean(raw_error[raw_guard] ** 2))
    smooth_rmse = np.sqrt(np.mean(smooth_error[smooth_guard] ** 2))
    assert spline_rmse < raw_rmse * 0.25
    assert spline_rmse < smooth_rmse * 0.25


def test_scale_rotation_and_translation_invariance():
    parameters = np.linspace(0.0, 1.0, 101)
    points = _cubic_bezier(parameters)
    points += np.column_stack((np.zeros_like(parameters), 0.01 * np.sin(17.0 * parameters)))
    options = _options(max_control_points=10, candidate_control_counts=(4, 6, 8, 10))
    base = fit_polyline_spline(points, options=options)
    evaluation_parameters = np.linspace(0.0, 1.0, 151)
    base_evaluation = base.curve.evaluate(evaluation_parameters)

    for scale in (0.5, 1.0, 4.0, 20.0):
        scaled = fit_polyline_spline(points * scale, options=options)
        assert scaled.diagnostics.control_count == base.diagnostics.control_count
        np.testing.assert_allclose(
            scaled.curve.evaluate(evaluation_parameters).xy / scale,
            base_evaluation.xy,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            scaled.curve.evaluate(evaluation_parameters).tangent,
            base_evaluation.tangent,
            atol=1e-10,
        )
        assert scaled.diagnostics.residual_rms == pytest.approx(
            base.diagnostics.residual_rms, abs=1e-12
        )
        assert scaled.diagnostics.residual_p95 == pytest.approx(
            base.diagnostics.residual_p95, abs=1e-12
        )

    angle = np.deg2rad(37.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    offset = np.array([1000.0, -500.0])
    transformed = fit_polyline_spline(points @ rotation.T + offset, options=options)
    transformed_evaluation = transformed.curve.evaluate(evaluation_parameters)
    np.testing.assert_allclose(
        (transformed_evaluation.xy - offset) @ rotation,
        base_evaluation.xy,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        transformed_evaluation.tangent,
        base_evaluation.tangent @ rotation.T,
        atol=1e-10,
    )
    assert transformed.diagnostics.control_count == base.diagnostics.control_count
    assert transformed.diagnostics.residual_rms == pytest.approx(
        base.diagnostics.residual_rms, abs=1e-12
    )
    assert transformed.diagnostics.residual_p95 == pytest.approx(
        base.diagnostics.residual_p95, abs=1e-12
    )


def test_low_weights_reduce_outlier_influence_without_hiding_unweighted_error():
    points = np.column_stack((np.linspace(0.0, 100.0, 101), np.zeros(101)))
    points[50, 1] = 20.0
    low_weights = np.ones(len(points))
    low_weights[50] = 0.001
    options = _options(
        max_control_points=8,
        candidate_control_counts=(4, 6, 8),
        robust_scale=100.0,
        smoothness=0.0001,
    )
    downweighted = fit_polyline_spline(points, weights=low_weights, options=options)
    fully_weighted = fit_polyline_spline(points, weights=np.ones(len(points)), options=options)
    low_curve = downweighted.curve.sample_by_arclength(201).xy
    full_curve = fully_weighted.curve.sample_by_arclength(201).xy

    assert np.max(np.abs(low_curve[:, 1])) < 0.2
    assert np.max(np.abs(full_curve[:, 1])) > 1.0
    assert downweighted.diagnostics.weighted_residual_rms is not None
    assert downweighted.diagnostics.residual_rms > options.target_rms_error_fraction
    assert not downweighted.diagnostics.met_target_error


def test_high_level_endpoint_policies_are_respected():
    points = np.column_stack((np.linspace(0.0, 10.0, 51), np.zeros(51)))
    points[0, 1] = 4.0
    points[-1, 1] = -3.0
    weights = np.ones(len(points))
    weights[[0, -1]] = 0.0
    fits = {}
    for policy in ("free", "anchored", "locked"):
        options = PolylineSplineFitOptions(
            min_control_points=4,
            max_control_points=4,
            candidate_control_counts=(4,),
            robust_scale=10.0,
            smoothness=0.1,
            endpoint_policy=SplineEndpointPolicy(start=policy, end=policy),
            endpoint_sigma=0.05,
        )
        fits[policy] = fit_polyline_spline(points, weights=weights, options=options)

    free_endpoints = fits["free"].curve.evaluate([0.0, 1.0], derivatives=False).xy
    anchored_endpoints = fits["anchored"].curve.evaluate([0.0, 1.0], derivatives=False).xy
    locked_endpoints = fits["locked"].curve.evaluate([0.0, 1.0], derivatives=False).xy
    np.testing.assert_allclose(locked_endpoints, points[[0, -1]])
    assert np.linalg.norm(anchored_endpoints - points[[0, -1]]) < np.linalg.norm(
        free_endpoints - points[[0, -1]]
    )


def test_model_selection_prefers_small_models_and_reports_fallback():
    parameters = np.linspace(0.0, 1.0, 121)
    line = np.column_stack((100.0 * parameters, np.zeros_like(parameters)))
    line_result = fit_polyline_spline(
        line,
        options=_options(max_control_points=16, candidate_control_counts=(4, 8, 12, 16)),
    )
    assert line_result.diagnostics.control_count == 4

    s_curve = np.column_stack((100.0 * parameters, 60.0 * (2.0 * parameters - 1.0) ** 3))
    s_result = fit_polyline_spline(s_curve, options=_options())
    assert 4 < s_result.diagnostics.control_count < 12

    alternating = np.column_stack((np.arange(101, dtype=float), np.arange(101, dtype=float) % 2.0))
    staircase_options = PolylineSplineFitOptions(
        min_control_points=4,
        max_control_points=16,
        candidate_control_counts=(4, 8, 12, 16),
        target_rms_error_fraction=0.006,
        target_p95_error_fraction=0.01,
    )
    staircase = fit_polyline_spline(alternating, options=staircase_options)
    assert staircase.diagnostics.met_target_error
    assert staircase.diagnostics.control_count < staircase_options.max_control_points

    strict_options = PolylineSplineFitOptions(
        min_control_points=4,
        max_control_points=8,
        candidate_control_counts=(4, 6, 8),
        target_rms_error_fraction=1e-12,
        target_p95_error_fraction=1e-12,
    )
    fallback = fit_polyline_spline(s_curve, options=strict_options)
    assert not fallback.diagnostics.met_target_error
    best_candidate = min(
        fallback.diagnostics.candidate_scores,
        key=lambda candidate: candidate.score,
    )
    assert fallback.diagnostics.control_count == best_candidate.control_count


def test_duplicate_removal_weight_alignment_and_invalid_inputs():
    points = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    result = fit_polyline_spline(
        points,
        weights=np.array([1.0, 0.0, 1.0, 1.0]),
        options=PolylineSplineFitOptions(
            min_control_points=4,
            max_control_points=4,
            candidate_control_counts=(4,),
        ),
    )
    assert result.diagnostics.normalized_scale == pytest.approx(2.0)
    assert result.diagnostics.weighted_residual_rms is not None
    np.testing.assert_allclose(
        result.curve.evaluate([0.0, 1.0], derivatives=False).xy,
        points[[0, -1]],
        atol=1e-10,
    )

    baseline = np.column_stack((np.linspace(0.0, 10.0, 51), np.zeros(51)))
    trailing_duplicate = np.vstack((baseline, [10.0, 5.0], [10.0, 5.0]))
    trailing_weights = np.ones(len(trailing_duplicate))
    trailing_weights[-1] = 0.0
    endpoint_options = PolylineSplineFitOptions(
        min_control_points=4,
        max_control_points=4,
        candidate_control_counts=(4,),
        robust_scale=100.0,
        smoothness=0.0001,
        endpoint_policy=SplineEndpointPolicy(start="free", end="free"),
    )
    ignored_endpoint = fit_polyline_spline(
        trailing_duplicate, weights=trailing_weights, options=endpoint_options
    )
    weighted_endpoint = fit_polyline_spline(
        trailing_duplicate,
        weights=np.ones(len(trailing_duplicate)),
        options=endpoint_options,
    )
    assert np.max(np.abs(ignored_endpoint.curve.sample_by_arclength(101).xy[:, 1])) < 2.0
    assert np.max(np.abs(weighted_endpoint.curve.sample_by_arclength(101).xy[:, 1])) > 4.0

    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
        fit_polyline_spline([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="one value per input point"):
        fit_polyline_spline(points, weights=[1.0, 1.0])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        fit_polyline_spline(points, weights=[1.0, -1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        fit_polyline_spline(points, weights=[1.0, np.nan, 1.0, 1.0])
    with pytest.raises(ValueError, match="positive total weight"):
        fit_polyline_spline(points, weights=np.zeros(len(points)))
    with pytest.raises(ValueError, match="distinct points"):
        fit_polyline_spline([[1.0, 1.0], [1.0, 1.0]])
