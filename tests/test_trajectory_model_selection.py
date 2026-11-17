import numpy as np

from planarcurve.fitting import (
    TrajectoryFitOptions,
    TrajectoryModelSelectionOptions,
    select_shared_trajectory_model,
)
from planarcurve.trajectory import (
    HeadingRateChange,
    HeadingRateProfile,
    Trajectory2D,
    evaluate_trajectory,
)
from planarcurve.transforms import Similarity2D


def _observations(trajectory: Trajectory2D, *, count: int = 4, noise: float = 0.0):
    base = evaluate_trajectory(trajectory, count=81).xy
    rng = np.random.default_rng(15)
    return [
        Similarity2D(
            scale=0.6 + 0.3 * index,
            rotation_deg=-35.0 + 25.0 * index,
            dx=2.0 * index,
            dy=-index,
        ).apply(base)
        + rng.normal(0.0, noise, base.shape)
        for index in range(count)
    ]


def _fit_options() -> TrajectoryFitOptions:
    return TrajectoryFitOptions(change_count=0, samples=24, restarts=0, max_nfev=100)


def _selection_options(
    change_counts: tuple[int, ...], *, validation_fraction: float = 0.0, seed: int = 0
) -> TrajectoryModelSelectionOptions:
    return TrajectoryModelSelectionOptions(
        change_counts=change_counts,
        validation_fraction=validation_fraction,
        complexity_weight=0.01,
        parameter_precision=1e-3,
        seed=seed,
    )


def test_straight_and_noisy_families_prefer_zero_unnecessary_changes():
    line = Trajectory2D(1.0, 0.0, HeadingRateProfile(0.0, 0.0))
    options = _selection_options((0, 1, 2, 3, 4))
    exact = select_shared_trajectory_model(
        _observations(line), fit_options=_fit_options(), selection_options=options
    )
    noisy = select_shared_trajectory_model(
        _observations(line, noise=0.004),
        fit_options=_fit_options(),
        selection_options=options,
    )

    assert exact.success and noisy.success
    assert exact.model.change_count == 0
    assert noisy.model.change_count < max(options.change_counts)


def test_constant_turn_arc_does_not_grow_arbitrary_knots():
    arc = Trajectory2D(1.0, 0.0, HeadingRateProfile(90.0, 90.0))
    options = _selection_options((0, 1, 2, 3))
    result = select_shared_trajectory_model(
        _observations(arc), fit_options=_fit_options(), selection_options=options
    )

    assert result.success
    assert result.model.change_count == 0


def test_s_curve_selects_additional_profile_complexity():
    s_curve = Trajectory2D(
        1.0,
        0.0,
        HeadingRateProfile(
            180.0,
            180.0,
            changes=(HeadingRateChange(0.5, -180.0),),
        ),
    )
    options = _selection_options((0, 1, 2, 3))
    result = select_shared_trajectory_model(
        _observations(s_curve), fit_options=_fit_options(), selection_options=options
    )

    assert result.success
    assert 0 < result.model.change_count < max(options.change_counts)


def test_validation_split_and_selection_are_deterministic_and_decomposed():
    line = Trajectory2D(1.0, 0.0, HeadingRateProfile(0.0, 0.0))
    observations = _observations(line, count=5, noise=0.002)
    options = _selection_options((0, 1), validation_fraction=0.2, seed=42)

    first = select_shared_trajectory_model(
        observations, fit_options=_fit_options(), selection_options=options
    )
    second = select_shared_trajectory_model(
        observations, fit_options=_fit_options(), selection_options=options
    )

    assert first.model.training_occurrences == 4
    assert first.validation_cost is not None and np.isfinite(first.validation_cost)
    assert first.score == first.training_cost + first.complexity_cost + first.validation_cost
    assert first.model.change_count == second.model.change_count
    assert first.score == second.score


def test_selection_options_reject_invalid_candidates_and_scales():
    import pytest

    with pytest.raises(ValueError, match="nonempty"):
        TrajectoryModelSelectionOptions(change_counts=())
    with pytest.raises(ValueError, match="unique"):
        TrajectoryModelSelectionOptions(change_counts=(0, 0))
    with pytest.raises(ValueError, match="validation_fraction"):
        TrajectoryModelSelectionOptions(validation_fraction=1.0)
    with pytest.raises(ValueError, match="complexity_weight"):
        TrajectoryModelSelectionOptions(complexity_weight=-1.0)
    with pytest.raises(ValueError, match="parameter_precision"):
        TrajectoryModelSelectionOptions(parameter_precision=0.0)
