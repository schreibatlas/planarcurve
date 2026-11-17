"""Compare selected trajectory model orders on synthetic line and S-curve families."""

from __future__ import annotations

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


def _observations(trajectory: Trajectory2D) -> list[np.ndarray]:
    base = evaluate_trajectory(trajectory, count=97).xy
    rng = np.random.default_rng(13)
    return [
        Similarity2D(
            scale=0.7 + index * 0.25,
            rotation_deg=-40.0 + index * 20.0,
            dx=float(index),
            dy=-0.5 * index,
        ).apply(base)
        + rng.normal(0.0, 0.001, base.shape)
        for index in range(5)
    ]


def _select(trajectory: Trajectory2D):
    return select_shared_trajectory_model(
        _observations(trajectory),
        fit_options=TrajectoryFitOptions(
            change_count=0,
            samples=32,
            restarts=0,
            max_nfev=150,
        ),
        selection_options=TrajectoryModelSelectionOptions(
            change_counts=(0, 1, 2, 3, 4),
            validation_fraction=0.2,
            complexity_weight=0.01,
            parameter_precision=1e-3,
            seed=13,
        ),
    )


def _report(name: str, result) -> None:
    print(f"{name}: selected change count={result.model.change_count}")
    print(f"  training cost:   {result.training_cost:.6f}")
    print(f"  complexity cost: {result.complexity_cost:.6f}")
    print(f"  validation cost: {result.validation_cost:.6f}")
    print(f"  total score:     {result.score:.6f}")


def main() -> None:
    line = Trajectory2D(1.0, 0.0, HeadingRateProfile(0.0, 0.0))
    s_curve = Trajectory2D(
        1.0,
        0.0,
        HeadingRateProfile(
            180.0,
            180.0,
            changes=(HeadingRateChange(0.5, -180.0),),
        ),
    )
    line_result = _select(line)
    s_curve_result = _select(s_curve)
    assert line_result.success and line_result.model.change_count == 0
    assert 0 < s_curve_result.model.change_count < 4
    _report("straight family", line_result)
    _report("S-shaped family", s_curve_result)


if __name__ == "__main__":
    main()
