from __future__ import annotations

import numpy as np
import pytest

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
        true_trajectory(), Similarity2D(scale=2.0, rotation_deg=0.0, dx=3.0, dy=-1.0),
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
