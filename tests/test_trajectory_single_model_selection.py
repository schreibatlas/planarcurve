import numpy as np
import pytest

from planarcurve.fitting import (
    HeadingRateParameterization,
    TrajectoryFitOptions,
    TrajectoryModelCandidate,
    TrajectoryModelSelectionOptions,
    TrajectoryModelSelectionResult,
    model_selection,
    select_trajectory_model,
)
from planarcurve.fitting.trajectory import TrajectoryFitResult
from planarcurve.trajectory import (
    HeadingRateChange,
    HeadingRateProfile,
    Trajectory2D,
    evaluate_heading_rate,
    evaluate_trajectory,
    trajectory_total_turn,
)
from planarcurve.transforms import Similarity2D


def _selection_options(
    change_counts: tuple[int, ...] = (0, 1, 2, 3), *, complexity_weight: float = 0.01
) -> TrajectoryModelSelectionOptions:
    return TrajectoryModelSelectionOptions(
        change_counts=change_counts,
        validation_fraction=0.0,
        complexity_weight=complexity_weight,
        parameter_precision=1e-3,
        seed=7,
    )


def _fit_options() -> TrajectoryFitOptions:
    return TrajectoryFitOptions(change_count=0, samples=32, restarts=0, max_nfev=100)


def _points(profile: HeadingRateProfile, *, count: int = 65) -> np.ndarray:
    trajectory = Trajectory2D(length=1.0, start_heading_deg=0.0, heading_rate=profile)
    return evaluate_trajectory(trajectory, count=count).xy


def _s_curve_points() -> np.ndarray:
    return _points(
        HeadingRateProfile(
            270.0,
            270.0,
            changes=(HeadingRateChange(0.5, -270.0),),
        )
    )


def test_straight_line_selects_minimal_profile_and_succeeds():
    points = np.column_stack((np.linspace(0.0, 1.0, 65), np.zeros(65)))

    result = select_trajectory_model(
        points, fit_options=_fit_options(), selection_options=_selection_options()
    )

    assert result.selected_change_count == 0
    assert result.selected.success
    assert abs(trajectory_total_turn(result.selected.trajectory)) < 1.0
    assert result.selected.trajectory.length == 1.0
    assert result.selected.trajectory.start_heading_deg == 0.0
    assert result.selected.transform.scale > 0.0


def test_constant_curvature_arc_uses_minimal_profile_and_fits_within_tolerance():
    points = _points(HeadingRateProfile(90.0, 90.0))

    result = select_trajectory_model(
        points, fit_options=_fit_options(), selection_options=_selection_options()
    )

    assert result.selected_change_count == 0
    assert result.selected.success
    assert result.selected.metrics["p95_position_error"] < 0.02
    assert abs(trajectory_total_turn(result.selected.trajectory) - 90.0) < 12.0


def test_s_curve_selects_turning_phases_in_the_known_order():
    result = select_trajectory_model(
        _s_curve_points(), fit_options=_fit_options(), selection_options=_selection_options()
    )
    rates = evaluate_heading_rate(
        result.selected.trajectory.heading_rate, np.array([0.05, 0.5, 0.95])
    )

    assert result.selected_change_count > 0
    assert result.selected.success
    assert rates[0] > 0.0
    assert rates[1] < 0.0
    assert rates[2] > 0.0


def test_small_coordinate_noise_preserves_selected_order_and_total_turn():
    points = _s_curve_points()
    noisy = points + np.random.default_rng(15).normal(0.0, 0.002, points.shape)

    exact = select_trajectory_model(
        points, fit_options=_fit_options(), selection_options=_selection_options()
    )
    perturbed = select_trajectory_model(
        noisy, fit_options=_fit_options(), selection_options=_selection_options()
    )

    assert abs(perturbed.selected_change_count - exact.selected_change_count) <= 1
    assert (
        abs(
            trajectory_total_turn(perturbed.selected.trajectory)
            - trajectory_total_turn(exact.selected.trajectory)
        )
        < 10.0
    )


def test_similarity_transform_preserves_canonical_profile_and_moves_placement():
    points = _s_curve_points()
    placement = Similarity2D(scale=2.8, rotation_deg=57.0, dx=3.0, dy=-1.0)
    original = select_trajectory_model(
        points, fit_options=_fit_options(), selection_options=_selection_options()
    )
    transformed = select_trajectory_model(
        placement.apply(points),
        fit_options=_fit_options(),
        selection_options=_selection_options(),
    )

    grid = np.linspace(0.0, 1.0, 21)
    original_rates = evaluate_heading_rate(original.selected.trajectory.heading_rate, grid)
    transformed_rates = evaluate_heading_rate(transformed.selected.trajectory.heading_rate, grid)
    expected_translation = placement.apply(
        np.array([[original.selected.transform.dx, original.selected.transform.dy]])
    )[0]

    assert transformed.selected_change_count == original.selected_change_count
    np.testing.assert_allclose(transformed_rates, original_rates, atol=1.0)
    assert transformed.selected.transform.scale == pytest.approx(
        placement.scale * original.selected.transform.scale, rel=1e-3
    )
    assert transformed.selected.transform.rotation_deg == pytest.approx(
        placement.rotation_deg + original.selected.transform.rotation_deg, abs=0.1
    )
    np.testing.assert_allclose(
        [transformed.selected.transform.dx, transformed.selected.transform.dy],
        expected_translation,
        atol=0.02,
    )


def test_selection_is_deterministic_and_returns_candidates_in_requested_order():
    points = _s_curve_points()
    options = _selection_options((2, 0, 1))

    first = select_trajectory_model(points, fit_options=_fit_options(), selection_options=options)
    second = select_trajectory_model(points, fit_options=_fit_options(), selection_options=options)

    assert [candidate.change_count for candidate in first.candidates] == [2, 0, 1]
    assert first.selected_change_count == second.selected_change_count
    np.testing.assert_allclose(
        [candidate.score for candidate in first.candidates],
        [candidate.score for candidate in second.candidates],
    )
    np.testing.assert_allclose(
        evaluate_heading_rate(first.selected.trajectory.heading_rate, np.linspace(0, 1, 21)),
        evaluate_heading_rate(second.selected.trajectory.heading_rate, np.linspace(0, 1, 21)),
    )


def test_failed_candidates_do_not_hide_successes_and_all_failed_is_deterministic(monkeypatch):
    trajectory = Trajectory2D(1.0, 0.0, HeadingRateProfile(0.0, 0.0))
    outcomes = {0: (False, 0.0), 1: (True, 10.0), 2: (True, 10.0)}

    def fake_fit(points, options):
        success, cost = outcomes[options.change_count]
        return TrajectoryFitResult(trajectory, Similarity2D(), success, cost, {})

    monkeypatch.setattr(model_selection, "fit_trajectory", fake_fit)
    options = _selection_options((0, 1, 2), complexity_weight=0.0)
    result = select_trajectory_model(
        [[0.0, 0.0], [1.0, 0.0]], fit_options=_fit_options(), selection_options=options
    )

    assert isinstance(result, TrajectoryModelSelectionResult)
    assert result.selected_change_count == 1
    assert all(isinstance(candidate, TrajectoryModelCandidate) for candidate in result.candidates)

    outcomes.update({0: (False, 10.0), 1: (False, 10.0), 2: (False, 0.0)})
    failed = select_trajectory_model(
        [[0.0, 0.0], [1.0, 0.0]], fit_options=_fit_options(), selection_options=options
    )
    repeated = select_trajectory_model(
        [[0.0, 0.0], [1.0, 0.0]], fit_options=_fit_options(), selection_options=options
    )

    assert not failed.selected.success
    assert failed.selected_change_count == repeated.selected_change_count == 2


def test_complexity_penalty_uses_codec_size_and_increases_with_change_count(monkeypatch):
    trajectory = Trajectory2D(1.0, 0.0, HeadingRateProfile(0.0, 0.0))
    monkeypatch.setattr(
        model_selection,
        "fit_trajectory",
        lambda points, options: TrajectoryFitResult(trajectory, Similarity2D(), True, 2.0, {}),
    )
    options = _selection_options((0, 1, 2), complexity_weight=0.1)

    result = select_trajectory_model(
        [[0.0, 0.0], [1.0, 0.0]], fit_options=_fit_options(), selection_options=options
    )
    factor = options.complexity_weight * np.log1p(1.0 / options.parameter_precision)
    parameter_counts = [
        HeadingRateParameterization(
            change_count=count, min_change_spacing=0.02, max_heading_rate_deg_per_u=36000.0
        ).size
        + 4
        for count in options.change_counts
    ]

    assert [candidate.score for candidate in result.candidates] == pytest.approx(
        [2.0 + factor * count for count in parameter_counts]
    )
    assert all(
        first.complexity_cost < second.complexity_cost
        for first, second in zip(result.candidates, result.candidates[1:], strict=False)
    )


def test_nonzero_validation_fraction_is_not_accepted_for_one_curve():
    options = TrajectoryModelSelectionOptions(change_counts=(0,), validation_fraction=0.2)

    with pytest.raises(ValueError, match="validation_fraction must be zero"):
        select_trajectory_model(
            [[0.0, 0.0], [1.0, 0.0]],
            fit_options=_fit_options(),
            selection_options=options,
        )
