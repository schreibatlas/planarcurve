"""Fit a directed shared family from transformed synthetic arc observations."""

from __future__ import annotations

import numpy as np

from planarcurve import descriptor_distance, trajectory_descriptor
from planarcurve.fitting import TrajectoryFitOptions, fit_trajectory_family
from planarcurve.trajectory import HeadingRateProfile, Trajectory2D, evaluate_trajectory
from planarcurve.transforms import Similarity2D


def main() -> None:
    arc = Trajectory2D(1.0, 0.0, HeadingRateProfile(90.0, 90.0))
    base = evaluate_trajectory(arc, count=129).xy
    rng = np.random.default_rng(7)
    transforms = (
        Similarity2D(0.6, -35.0, 2.0, -1.0),
        Similarity2D(1.5, 20.0, -3.0, 2.5),
        Similarity2D(2.0, 75.0, 1.0, 4.0),
        Similarity2D(0.9, 140.0, -1.0, -2.0),
    )
    occurrences = [
        transform.apply(base) + rng.normal(0.0, 0.0005, base.shape) for transform in transforms
    ]

    signatures = [trajectory_descriptor(points) for points in occurrences]
    assert max(descriptor_distance(signatures[0], item) for item in signatures[1:]) < 0.03

    result = fit_trajectory_family(
        occurrences,
        fit_options=TrajectoryFitOptions(
            change_count=0,
            samples=48,
            restarts=1,
            seed=7,
            max_nfev=200,
        ),
    )
    assert result.success
    assert all(item.accepted for item in result.occurrences)
    print(f"shared change count: {result.model.change_count}")
    print(f"training cost: {result.training_cost:.6f}")
    print(f"family score: {result.score:.6f}")
    print(f"rate dispersion: {dict(result.model.parameter_scale)}")
    for index, occurrence in enumerate(result.occurrences):
        print(
            f"occurrence {index}: scale={occurrence.transform.scale:.3f}, "
            f"position_rmse={occurrence.position_rmse:.5f}, "
            f"heading_rmse_deg={occurrence.heading_rmse_deg:.3f}, "
            f"accepted={occurrence.accepted}"
        )


if __name__ == "__main__":
    main()
