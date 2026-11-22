from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from planarcurve import insert_heading_rate_change
from planarcurve.fitting import (
    TrajectoryFitOptions,
    TrajectoryFitSeed,
    TrajectoryRefinementOptions,
    fit_trajectory,
    refine_trajectory_model,
)
from planarcurve.trajectory import (
    HeadingRateChange,
    HeadingRateProfile,
    Trajectory2D,
    evaluate_heading_rate,
    evaluate_trajectory,
)
from planarcurve.transforms import Similarity2D


def _seed() -> TrajectoryFitSeed:
    trajectory = Trajectory2D(
        length=1.0,
        start_heading_deg=0.0,
        heading_rate=HeadingRateProfile(0.0, 0.0),
    )
    return TrajectoryFitSeed(trajectory, Similarity2D())


def test_zero_change_profile_promotes_at_half_progress():
    profile = HeadingRateProfile(10.0, 30.0)

    promoted = insert_heading_rate_change(profile, 0.5)

    assert len(promoted.changes) == 1
    assert promoted.changes[0].at == 0.5
    assert promoted.changes[0].heading_rate_deg_per_u == pytest.approx(20.0)
    assert promoted.interpolation == profile.interpolation


def test_profile_insertion_preserves_interpolation_and_original_profile():
    profile = HeadingRateProfile(
        start_rate_deg_per_u=20.0,
        changes=(HeadingRateChange(0.75, 80.0),),
        end_rate_deg_per_u=-10.0,
        interpolation="linear",
    )
    original_changes = profile.changes
    expected_rate = evaluate_heading_rate(profile, np.array([0.4]))[0]

    promoted = insert_heading_rate_change(profile, 0.4)

    assert [change.at for change in promoted.changes] == [0.4, 0.75]
    assert promoted.changes[0].heading_rate_deg_per_u == pytest.approx(expected_rate)
    assert promoted.interpolation == profile.interpolation
    assert profile.changes == original_changes


def test_profile_insertion_rejects_invalid_or_duplicate_locations():
    profile = HeadingRateProfile(
        0.0,
        0.0,
        changes=(HeadingRateChange(0.5, 10.0),),
    )
    for at in (0.0, 1.0, -0.1, 1.1, np.nan):
        with pytest.raises(ValueError, match="strictly inside"):
            insert_heading_rate_change(profile, at)
    with pytest.raises(ValueError, match="duplicates"):
        insert_heading_rate_change(profile, 0.5)


def test_fit_trajectory_default_and_explicit_none_are_identical():
    points = np.column_stack((np.linspace(0.0, 1.0, 17), np.zeros(17)))
    options = TrajectoryFitOptions(change_count=0, samples=16, restarts=0, max_nfev=30)

    default = fit_trajectory(points, options)
    explicit_none = fit_trajectory(points, options, initial=None)

    assert explicit_none == default
    assert default.optimizer_evaluations > 0


def test_fit_trajectory_accepts_a_valid_warm_seed():
    points = np.column_stack((np.linspace(0.0, 1.0, 17), np.zeros(17)))
    options = TrajectoryFitOptions(change_count=0, samples=16, restarts=0, max_nfev=30)

    result = fit_trajectory(points, options, initial=_seed())

    assert result.success
    assert result.optimizer_evaluations > 0


def test_fit_trajectory_rejects_seed_topology_mismatch():
    points = np.column_stack((np.linspace(0.0, 1.0, 17), np.zeros(17)))
    options = TrajectoryFitOptions(change_count=1, samples=16, restarts=0, max_nfev=30)

    with pytest.raises(ValueError, match="change count"):
        fit_trajectory(points, options, initial=_seed())


def test_warm_seed_is_attempt_zero_and_restarts_can_improve(monkeypatch):
    module = importlib.import_module("planarcurve.fitting.trajectory")
    points = np.column_stack((np.linspace(0.0, 1.0, 17), np.zeros(17)))
    options = TrajectoryFitOptions(change_count=0, samples=16, restarts=1, seed=7)
    seed = _seed()
    calls = []

    def fake_least_squares(_residuals, initial_values, **_kwargs):
        values = np.asarray(initial_values).copy()
        calls.append(values)
        index = len(calls) - 1
        return SimpleNamespace(
            x=values,
            cost=float(10 - index),
            success=True,
            nfev=index + 1,
        )

    monkeypatch.setattr(module, "least_squares", fake_least_squares)
    result = fit_trajectory(points, options, initial=seed)

    assert len(calls) == 2
    np.testing.assert_allclose(calls[0], 0.0)
    assert not np.array_equal(calls[0], calls[1])
    assert result.cost == 9.0
    assert result.optimizer_evaluations == 3


def _points_for_profile(profile: HeadingRateProfile) -> np.ndarray:
    trajectory = Trajectory2D(1.0, 0.0, profile)
    return evaluate_trajectory(trajectory, count=65).xy


def _s_curve_points() -> np.ndarray:
    return _points_for_profile(
        HeadingRateProfile(
            270.0,
            270.0,
            changes=(HeadingRateChange(0.5, -270.0),),
        )
    )


def _two_regime_points() -> np.ndarray:
    return _points_for_profile(
        HeadingRateProfile(
            240.0,
            240.0,
            changes=(
                HeadingRateChange(0.33, -240.0),
                HeadingRateChange(0.67, 240.0),
            ),
        )
    )


def _refinement_fit_options(*, min_change_spacing: float = 0.05) -> TrajectoryFitOptions:
    return TrajectoryFitOptions(
        change_count=0,
        samples=32,
        restarts=0,
        min_change_spacing=min_change_spacing,
        max_nfev=100,
    )


@pytest.mark.parametrize(
    ("profile", "expected_changes"),
    [
        (HeadingRateProfile(0.0, 0.0), 0),
        (HeadingRateProfile(90.0, 90.0), 0),
    ],
)
def test_refinement_keeps_straight_and_constant_turn_trajectories_simple(
    profile: HeadingRateProfile, expected_changes: int
):
    result = refine_trajectory_model(
        _points_for_profile(profile),
        fit_options=_refinement_fit_options(),
        refinement_options=TrajectoryRefinementOptions(max_change_count=2),
    )

    assert result.selected_change_count == expected_changes
    assert result.steps[0].accepted
    assert result.stop_reason in {"insufficient_gain", "max_change_count_reached"}


def test_refinement_promotes_an_s_curve_from_zero_changes():
    result = refine_trajectory_model(
        _s_curve_points(),
        fit_options=_refinement_fit_options(),
        refinement_options=TrajectoryRefinementOptions(
            max_change_count=1,
            candidate_insertions=3,
            complexity_weight=0.001,
        ),
    )

    assert result.selected_change_count == 1
    assert any(step.accepted and step.candidate_change_count == 1 for step in result.steps)
    assert result.stop_reason == "max_change_count_reached"


def test_refinement_can_reach_two_separated_steering_regimes():
    result = refine_trajectory_model(
        _two_regime_points(),
        fit_options=_refinement_fit_options(),
        refinement_options=TrajectoryRefinementOptions(
            max_change_count=2,
            candidate_insertions=4,
            residual_samples=64,
            minimum_absolute_gain=1e-5,
            minimum_relative_gain=0.001,
            complexity_weight=0.0001,
        ),
    )

    assert result.selected_change_count == 2
    accepted_counts = [step.candidate_change_count for step in result.steps if step.accepted]
    assert accepted_counts == [0, 1, 2]


def test_complexity_penalty_stops_earlier_and_maximum_does_not_force_changes():
    points = _s_curve_points()
    fit_options = _refinement_fit_options()
    low_penalty = refine_trajectory_model(
        points,
        fit_options=fit_options,
        refinement_options=TrajectoryRefinementOptions(
            max_change_count=3,
            complexity_weight=0.001,
        ),
    )
    high_penalty = refine_trajectory_model(
        points,
        fit_options=fit_options,
        refinement_options=TrajectoryRefinementOptions(
            max_change_count=3,
            complexity_weight=1.0,
        ),
    )
    line = np.column_stack((np.linspace(0.0, 1.0, 65), np.zeros(65)))
    short_limit = refine_trajectory_model(
        line,
        fit_options=fit_options,
        refinement_options=TrajectoryRefinementOptions(max_change_count=1),
    )
    long_limit = refine_trajectory_model(
        line,
        fit_options=fit_options,
        refinement_options=TrajectoryRefinementOptions(max_change_count=4),
    )

    assert low_penalty.selected_change_count > high_penalty.selected_change_count
    assert short_limit.selected_change_count == long_limit.selected_change_count == 0


def test_refinement_is_deterministic_and_accepted_steps_improve_score():
    points = _s_curve_points()
    options = TrajectoryRefinementOptions(
        max_change_count=2,
        complexity_weight=0.001,
        seed=17,
    )
    fit_options = _refinement_fit_options()
    first = refine_trajectory_model(points, fit_options=fit_options, refinement_options=options)
    second = refine_trajectory_model(points, fit_options=fit_options, refinement_options=options)

    assert first == second
    assert first.stop_reason == second.stop_reason
    assert first.stop_reason
    accepted = [step for step in first.steps if step.accepted and step.candidate_change_count > 0]
    assert accepted
    assert all(step.absolute_gain > 0.0 for step in accepted)
    assert all(step.relative_gain > 0.0 for step in accepted)


def test_refinement_rejects_nonzero_start_order_and_reports_no_legal_locations():
    points = np.column_stack((np.linspace(0.0, 1.0, 65), np.zeros(65)))
    with pytest.raises(ValueError, match="change_count must be zero"):
        refine_trajectory_model(
            points,
            fit_options=TrajectoryFitOptions(change_count=1),
        )

    result = refine_trajectory_model(
        points,
        fit_options=_refinement_fit_options(min_change_spacing=0.6),
        refinement_options=TrajectoryRefinementOptions(max_change_count=1),
    )
    assert result.selected_change_count == 0
    assert result.stop_reason == "no_legal_insertion_location"


def test_nonfinite_residual_signal_stops_as_numerical_failure(monkeypatch):
    module = importlib.import_module("planarcurve.fitting.model_selection")
    points = _s_curve_points()
    monkeypatch.setattr(
        module,
        "_mapped_state",
        lambda _trajectory, _transform, samples: (
            np.full((samples, 2), np.nan),
            np.zeros(samples),
        ),
    )

    result = refine_trajectory_model(
        points,
        fit_options=_refinement_fit_options(),
        refinement_options=TrajectoryRefinementOptions(max_change_count=1),
    )

    assert result.selected_change_count == 0
    assert result.stop_reason == "numerical_failure"


def test_candidate_insertions_exclude_endpoints_and_duplicate_knots_at_zero_spacing():
    from planarcurve.fitting.model_selection import _residual_insertion_candidates

    grid = np.linspace(0.0, 1.0, 11)
    signal = np.zeros_like(grid)
    signal[[0, 5, 10]] = [10.0, 20.0, 9.0]
    locations = _residual_insertion_candidates(
        signal,
        grid,
        (0.5,),
        min_spacing=0.0,
        limit=3,
    )

    assert locations
    assert all(0.0 < location < 1.0 for location in locations)
    assert 0.5 not in locations


def test_candidate_insertions_respect_spacing_and_nonmaximum_suppression():
    from planarcurve.fitting.model_selection import _residual_insertion_candidates

    grid = np.linspace(0.0, 1.0, 21)
    signal = np.zeros_like(grid)
    signal[[4, 5, 6, 15, 16]] = [2.0, 5.0, 3.0, 4.0, 3.0]
    locations = _residual_insertion_candidates(
        signal,
        grid,
        (0.5,),
        min_spacing=0.15,
        limit=4,
    )

    assert locations
    assert all(0.15 <= location <= 0.85 for location in locations)
    assert all(abs(location - 0.5) >= 0.15 for location in locations)
    assert all(
        abs(left - right) >= 0.15
        for index, left in enumerate(locations)
        for right in locations[index + 1 :]
    )


def test_refinement_evaluation_count_is_reported_and_bounded():
    fit_options = _refinement_fit_options()
    refinement_options = TrajectoryRefinementOptions(
        max_change_count=3,
        candidate_insertions=2,
        complexity_weight=0.001,
    )
    result = refine_trajectory_model(
        _s_curve_points(),
        fit_options=fit_options,
        refinement_options=refinement_options,
    )
    maximum_fits = 1 + refinement_options.max_change_count * refinement_options.candidate_insertions
    evaluation_bound = maximum_fits * (fit_options.restarts + 1) * fit_options.max_nfev

    assert result.optimizer_evaluations == sum(
        step.fit.optimizer_evaluations for step in result.steps
    )
    assert 0 < result.optimizer_evaluations <= evaluation_bound


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_change_count": -1},
        {"candidate_insertions": 0},
        {"residual_samples": 7},
        {"minimum_absolute_gain": -1.0},
        {"minimum_relative_gain": np.inf},
        {"complexity_weight": -1.0},
        {"parameter_precision": 0.0},
        {"parameter_precision": 1.1},
    ],
)
def test_refinement_options_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        TrajectoryRefinementOptions(**kwargs)
