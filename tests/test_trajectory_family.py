import numpy as np
import pytest

from planarcurve.fitting import (
    TrajectoryFitOptions,
    TrajectoryVariationPolicy,
    fit_trajectory_family,
)
from planarcurve.trajectory import (
    HeadingRateProfile,
    Trajectory2D,
    evaluate_trajectory,
    trajectory_total_turn,
)
from planarcurve.transforms import Similarity2D


def _arc(rate: float = 90.0) -> Trajectory2D:
    return Trajectory2D(
        length=1.0,
        start_heading_deg=0.0,
        heading_rate=HeadingRateProfile(rate, rate),
    )


def _occurrence(
    trajectory: Trajectory2D,
    *,
    scale: float,
    rotation: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    points = evaluate_trajectory(trajectory, count=129).xy
    return Similarity2D(scale, rotation, dx, dy).apply(points)


def _options() -> TrajectoryFitOptions:
    return TrajectoryFitOptions(change_count=0, samples=40, restarts=0, max_nfev=150)


def test_shared_family_reports_transforms_and_occurrence_error_breakdown():
    trajectory = _arc()
    occurrences = [
        _occurrence(trajectory, scale=0.5, rotation=-35.0, dx=2.0, dy=-1.0),
        _occurrence(trajectory, scale=2.0, rotation=70.0, dx=-4.0, dy=3.0),
        _occurrence(trajectory, scale=1.2, rotation=10.0, dx=0.0, dy=0.0),
    ]
    snapshots = [points.copy() for points in occurrences]

    result = fit_trajectory_family(occurrences, fit_options=_options())

    assert result.success
    assert result.model.training_occurrences == len(occurrences)
    assert result.model.change_count == 0
    assert len(result.occurrences) == len(occurrences)
    assert result.validation_cost is None
    assert result.score == pytest.approx(result.training_cost + result.complexity_cost)
    assert trajectory_total_turn(result.model.trajectory) > 60.0
    assert all(item.accepted and item.optimizer_success for item in result.occurrences)
    assert all(item.optimizer_evaluations > 0 for item in result.occurrences)
    assert all(item.rate_deviation == (0.0, 0.0) for item in result.occurrences)
    assert all(item.position_rmse < 0.01 for item in result.occurrences)
    assert all(item.heading_rmse_deg < 2.0 for item in result.occurrences)
    assert all(item.endpoint_error < 0.01 for item in result.occurrences)
    assert all(item.turn_error_deg < 5.0 for item in result.occurrences)
    assert all(item.robust_cost >= 0.0 for item in result.occurrences)
    for points, snapshot in zip(occurrences, snapshots, strict=True):
        np.testing.assert_array_equal(points, snapshot)


def test_reversed_occurrence_is_not_silently_reoriented_and_is_diagnosed():
    trajectory = _arc()
    occurrences = [
        _occurrence(trajectory, scale=scale, rotation=rotation, dx=dx, dy=dy)
        for scale, rotation, dx, dy in (
            (0.8, -20.0, 1.0, -2.0),
            (1.0, 15.0, 0.0, 1.0),
            (1.4, 90.0, -3.0, 0.5),
            (0.6, -75.0, 2.0, 4.0),
        )
    ]
    occurrences.append(occurrences[0][::-1].copy())

    result = fit_trajectory_family(occurrences, fit_options=_options())

    assert trajectory_total_turn(result.model.trajectory) > 45.0
    assert all(item.accepted for item in result.occurrences[:-1])
    assert not result.occurrences[-1].accepted
    assert result.occurrences[-1].robust_cost > 3.0 * np.median(
        [item.robust_cost for item in result.occurrences[:-1]]
    )


def test_family_requires_observations_and_rejects_zero_length_curves():
    with pytest.raises(ValueError, match="at least one"):
        fit_trajectory_family([], fit_options=_options())
    with pytest.raises(ValueError, match="zero-length"):
        fit_trajectory_family([np.ones((5, 2))], fit_options=_options())


def test_family_variation_policy_does_not_allow_position_offsets_yet():
    with pytest.raises(NotImplementedError, match="change-position"):
        fit_trajectory_family(
            [_occurrence(_arc(), scale=1.0, rotation=0.0, dx=0.0, dy=0.0)],
            fit_options=_options(),
            variation_policy=TrajectoryVariationPolicy(allow_change_position_offsets=True),
        )


def test_very_short_but_nonzero_observations_are_fit():
    occurrences = [
        np.array([[0.0, 0.0], [2.0, 0.0]]),
        np.array([[3.0, -1.0], [3.0, 4.0]]),
    ]

    result = fit_trajectory_family(occurrences, fit_options=_options())

    assert result.success
    assert len(result.occurrences) == 2
    assert all(item.accepted for item in result.occurrences)
    assert all(item.position_rmse < 1e-6 for item in result.occurrences)
