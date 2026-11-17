from __future__ import annotations

import numpy as np
import pytest

from planarcurve.fitting.parameters import HeadingRateParameterization
from planarcurve.fitting.trajectory import (
    TrajectoryFitOptions,
    estimate_heading_rate,
    estimate_polyline_heading,
    fit_shared_trajectory_profile,
    fit_trajectory,
    initialize_change_positions,
    resample_by_arclength,
)
from planarcurve.trajectory import (
    HeadingRateChange,
    HeadingRateProfile,
    Trajectory2D,
    evaluate_trajectory,
    trajectory_total_turn,
)
from planarcurve.transforms import Similarity2D


def true_trajectory() -> Trajectory2D:
    return Trajectory2D(
        length=1.0,
        start_heading_deg=40.0,
        heading_rate=HeadingRateProfile(
            start_rate_deg_per_u=30.0,
            changes=(
                HeadingRateChange(0.25, 260.0),
                HeadingRateChange(0.55, 210.0),
                HeadingRateChange(0.8, -40.0),
            ),
            end_rate_deg_per_u=15.0,
        ),
    )


def transformed_points(
    trajectory: Trajectory2D,
    transform: Similarity2D,
    *,
    seed: int,
    noise: float = 0.0,
) -> np.ndarray:
    points = transform.apply(evaluate_trajectory(trajectory, count=257).xy)
    return points + np.random.default_rng(seed).normal(0.0, noise, points.shape)


def test_sampled_curve_helpers_validate_and_estimate_heading():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert resample_by_arclength(points, samples=5).shape == (5, 2)
    headings = estimate_polyline_heading(points)
    np.testing.assert_allclose(headings, 0.0, atol=1e-12)
    rates = estimate_heading_rate(np.linspace(0.0, 1.0, len(headings)), headings)
    np.testing.assert_allclose(rates, 0.0, atol=1e-12)
    positions = initialize_change_positions(np.zeros(32), 3)
    assert np.all(np.diff(positions) >= 0.02 - 1e-12)
    with pytest.raises(ValueError, match="finite"):
        resample_by_arclength(np.array([[0.0, 0.0], [np.nan, 1.0]]))


def test_single_trajectory_fit_recovers_noisy_synthetic_shape():
    target = transformed_points(
        true_trajectory(),
        Similarity2D(scale=2.0, rotation_deg=0.0, dx=3.0, dy=-1.0),
        seed=7,
        noise=0.002,
    )
    result = fit_trajectory(
        target,
        TrajectoryFitOptions(change_count=3, restarts=1, seed=0, max_nfev=250),
    )
    assert result.success
    assert result.metrics["median_position_error"] <= 0.015
    assert result.metrics["p95_position_error"] <= 0.04
    assert result.metrics["endpoint_tangent_error_deg"] <= 10.0
    assert result.metrics["total_turn_error_deg"] <= 5.0
    assert result.transform.scale == pytest.approx(2.0, rel=0.08)

    assert result.trajectory.start_heading_deg == 0.0
    assert result.transform.rotation_deg == pytest.approx(40.0, abs=10.0)


def test_shared_profile_fits_transformed_occurrences():
    trajectory = true_trajectory()
    transforms = (
        Similarity2D(scale=0.7, rotation_deg=-20.0, dx=3.0, dy=-1.5),
        Similarity2D(scale=1.3, rotation_deg=35.0, dx=-2.0, dy=4.0),
        Similarity2D(scale=1.0, rotation_deg=5.0, dx=0.0, dy=0.0),
    )
    occurrences = [
        transformed_points(trajectory, transform, seed=index, noise=0.001)
        for index, transform in enumerate(transforms)
    ]
    result = fit_shared_trajectory_profile(
        occurrences,
        TrajectoryFitOptions(change_count=3, restarts=0, seed=0, max_nfev=300),
    )
    assert result.success
    assert result.trajectory.start_heading_deg == 0.0
    pointwise_errors = []
    fitted_xy = evaluate_trajectory(result.trajectory, count=64).xy
    for occurrence, transform in zip(occurrences, result.transforms, strict=True):
        target_points = resample_by_arclength(occurrence, samples=64)
        pointwise_errors.extend(np.linalg.norm(transform.apply(fitted_xy) - target_points, axis=1))
    assert result.metrics["p95_position_error"] == pytest.approx(
        np.percentile(pointwise_errors, 95)
    )
    assert len(result.transforms) == len(occurrences)
    assert abs(trajectory_total_turn(result.trajectory) - trajectory_total_turn(trajectory)) < 12.0
    assert result.metrics["median_position_error"] < 0.02
    scales = [transform.scale for transform in result.transforms]
    assert scales[1] / scales[0] == pytest.approx(1.3 / 0.7, rel=0.12)


def test_trajectory_fit_options_reject_invalid_configuration():
    with pytest.raises(ValueError, match="change_count"):
        TrajectoryFitOptions(change_count=-1)
    with pytest.raises(ValueError, match="weights"):
        TrajectoryFitOptions(change_count=1, position_weight=-1.0)

    with pytest.raises(ValueError, match="samples"):
        TrajectoryFitOptions(change_count=1, samples=3.5)
    with pytest.raises(ValueError, match="restarts"):
        TrajectoryFitOptions(change_count=1, restarts=1.5)
    with pytest.raises(ValueError, match="max_nfev"):
        TrajectoryFitOptions(change_count=1, max_nfev=2.5)
    with pytest.raises(ValueError, match="change_count"):
        TrajectoryFitOptions(change_count=1.5)
    with pytest.raises(ValueError, match="smoothing_window"):
        TrajectoryFitOptions(change_count=1, smoothing_window=4.5)


def test_even_smoothing_windows_preserve_length_and_one_is_unsmoothed():
    points = np.column_stack((np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 10) ** 2))
    headings = estimate_polyline_heading(points, smooth_window=4)
    unsmoothed = estimate_polyline_heading(points, smooth_window=1)
    rates = estimate_heading_rate(np.linspace(0.0, 1.0, 10), headings, smooth_window=4)

    assert headings.shape == (10,)
    assert rates.shape == (10,)
    np.testing.assert_allclose(
        unsmoothed,
        np.rad2deg(
            np.unwrap(
                np.arctan2(np.gradient(points, axis=0)[:, 1], np.gradient(points, axis=0)[:, 0])
            )
        ),
    )


def test_change_position_initialization_stays_strictly_inside_interval():
    positions = initialize_change_positions(np.array([100.0, 0.0, 0.0, 0.0, 0.0]), 1)

    assert positions[0] > 0.0
    assert positions[-1] < 1.0
    assert np.all(np.diff(positions) >= 0.02)


def test_public_profile_parameterization_uses_identifiable_position_logits():
    for change_count in (0, 2):
        positions = np.linspace(0.3, 0.7, change_count)
        rates = np.linspace(-20.0, 20.0, change_count + 2)
        changes = tuple(
            HeadingRateChange(float(at), float(rate))
            for at, rate in zip(positions, rates[1:-1], strict=True)
        )
        for interpolation in ("linear", "cubic-hermite"):
            profile = HeadingRateProfile(
                start_rate_deg_per_u=float(rates[0]),
                changes=changes,
                end_rate_deg_per_u=float(rates[-1]),
                interpolation=interpolation,
            )
            codec = HeadingRateParameterization(
                change_count=change_count,
                min_change_spacing=0.02,
                max_heading_rate_deg_per_u=100.0,
                interpolation=interpolation,
            )
            encoded = codec.encode(profile)
            decoded = codec.decode(encoded)

            assert codec.version == 1
            assert encoded.size == 2 * change_count + 2
            assert decoded.interpolation == interpolation
            np.testing.assert_allclose(
                [change.at for change in decoded.changes], positions, atol=1e-12
            )
            np.testing.assert_allclose(
                [
                    decoded.start_rate_deg_per_u,
                    *(change.heading_rate_deg_per_u for change in decoded.changes),
                    decoded.end_rate_deg_per_u,
                ],
                rates,
                atol=1e-9,
            )


def test_fitted_profile_is_invariant_to_coordinate_scale_and_rotation():
    base_points = evaluate_trajectory(true_trajectory(), count=257).xy
    options = TrajectoryFitOptions(change_count=3, samples=48, restarts=0, max_nfev=150)
    signatures = []
    for scale, rotation in ((0.01, 0.0), (1.0, 35.0), (100.0, 120.0), (10_000.0, -70.0)):
        points = Similarity2D(
            scale=scale, rotation_deg=rotation, dx=3.0 * scale, dy=-2.0 * scale
        ).apply(base_points)
        result = fit_trajectory(points, options)
        assert result.trajectory.start_heading_deg == 0.0
        signatures.append(
            np.array(
                [
                    *(change.at for change in result.trajectory.heading_rate.changes),
                    result.trajectory.heading_rate.start_rate_deg_per_u,
                    *(
                        change.heading_rate_deg_per_u
                        for change in result.trajectory.heading_rate.changes
                    ),
                    result.trajectory.heading_rate.end_rate_deg_per_u,
                ]
            )
        )

    for signature in signatures[1:]:
        np.testing.assert_allclose(signature, signatures[0], rtol=0.1, atol=3.0)


def test_fitting_package_exports_stable_trajectory_api():
    import importlib

    public = importlib.import_module("planarcurve.fitting")
    implementation = importlib.import_module("planarcurve.fitting.trajectory")
    public_names = (
        "SharedTrajectoryFitResult",
        "TrajectoryFitOptions",
        "TrajectoryFitResult",
        "fit_shared_trajectory_profile",
        "fit_trajectory",
    )

    for name in public_names:
        assert getattr(public, name) is getattr(implementation, name)


def test_heading_rate_parameterization_is_exported_from_fitting_package():
    from planarcurve.fitting import HeadingRateParameterization as exported
    from planarcurve.fitting.parameters import HeadingRateParameterization as implementation

    assert exported is implementation
