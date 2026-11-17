import numpy as np
import pytest

from planarcurve.fitting import (
    TrajectoryFitOptions,
    TrajectoryVariationPolicy,
    fit_trajectory_family,
)
from planarcurve.trajectory import HeadingRateProfile, Trajectory2D, evaluate_trajectory
from planarcurve.transforms import Similarity2D


def _curve(rate: float, *, scale: float, rotation: float, offset: float) -> np.ndarray:
    trajectory = Trajectory2D(
        length=1.0,
        start_heading_deg=0.0,
        heading_rate=HeadingRateProfile(rate, rate),
    )
    points = evaluate_trajectory(trajectory, count=129).xy
    return Similarity2D(scale, rotation, offset, -offset / 2.0).apply(points)


def _options() -> TrajectoryFitOptions:
    return TrajectoryFitOptions(change_count=0, samples=40, restarts=0, max_nfev=150)


def test_rate_variation_is_bounded_explicit_and_deterministic():
    occurrences = [
        _curve(85.0, scale=0.8, rotation=-30.0, offset=1.0),
        _curve(90.0, scale=1.2, rotation=15.0, offset=-2.0),
        _curve(95.0, scale=1.5, rotation=75.0, offset=3.0),
    ]
    policy = TrajectoryVariationPolicy(rate_sigma=12.0)

    first = fit_trajectory_family(occurrences, fit_options=_options(), variation_policy=policy)
    second = fit_trajectory_family(occurrences, fit_options=_options(), variation_policy=policy)

    assert first.success and second.success
    assert len(first.occurrences) == len(occurrences)
    deviations = np.asarray([item.rate_deviation for item in first.occurrences])
    assert deviations.shape == (len(occurrences), 2)
    assert np.all(np.abs(deviations) <= policy.rate_sigma + 1e-10)
    assert np.any(np.abs(deviations) > 0.1)
    assert first.model.parameter_scale["rate_0"] > 0.0
    assert first.model.parameter_scale["rate_1"] > 0.0
    for first_item, second_item in zip(first.occurrences, second.occurrences, strict=True):
        np.testing.assert_allclose(
            first_item.rate_deviation, second_item.rate_deviation, atol=1e-10
        )
        assert first_item.robust_cost == pytest.approx(second_item.robust_cost)


def test_variation_policy_validates_scales_and_keeps_positions_fixed():
    with pytest.raises(ValueError, match="rate_sigma"):
        TrajectoryVariationPolicy(rate_sigma=0.0)
    with pytest.raises(ValueError, match="position_sigma"):
        TrajectoryVariationPolicy(position_sigma=-1.0)

    result = fit_trajectory_family(
        [_curve(90.0, scale=1.0, rotation=0.0, offset=0.0)],
        fit_options=_options(),
        variation_policy=TrajectoryVariationPolicy(rate_sigma=10.0),
    )
    assert result.model.change_count == 0
    assert result.model.trajectory.heading_rate.changes == ()
    assert len(result.occurrences[0].rate_deviation) == 2
