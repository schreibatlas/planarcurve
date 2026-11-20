"""Select a compact heading-rate model for one smooth planar trajectory."""

from __future__ import annotations

import numpy as np

from planarcurve.fitting import (
    TrajectoryFitOptions,
    TrajectoryModelSelectionOptions,
    select_trajectory_model,
)
from planarcurve.trajectory import (
    HeadingRateChange,
    HeadingRateProfile,
    Trajectory2D,
    evaluate_trajectory,
    trajectory_total_turn,
)


def main() -> None:
    source = Trajectory2D(
        length=1.0,
        start_heading_deg=0.0,
        heading_rate=HeadingRateProfile(
            270.0,
            270.0,
            changes=(HeadingRateChange(0.5, -270.0),),
        ),
    )
    points = evaluate_trajectory(source, count=97).xy
    result = select_trajectory_model(
        points,
        fit_options=TrajectoryFitOptions(
            change_count=0,
            samples=40,
            restarts=0,
            max_nfev=150,
        ),
        selection_options=TrajectoryModelSelectionOptions(
            change_counts=(0, 1, 2, 3, 4),
            validation_fraction=0.0,
            seed=7,
        ),
    )
    assert result.selected.success

    profile = result.selected.trajectory.heading_rate
    knot_positions = (0.0, *(change.at for change in profile.changes), 1.0)
    knot_rates = (
        profile.start_rate_deg_per_u,
        *(change.heading_rate_deg_per_u for change in profile.changes),
        profile.end_rate_deg_per_u,
    )

    print(f"Selected change count: {result.selected_change_count}")
    print(f"Knot positions: {np.asarray(knot_positions)}")
    print(f"Knot rates (degrees per normalized progress): {np.asarray(knot_rates)}")
    print(f"Total turn (degrees): {trajectory_total_turn(result.selected.trajectory):.3f}")
    print(f"Fit cost: {result.selected.cost:.6f}")
    print(f"Fit error metrics: {dict(result.selected.metrics)}")


if __name__ == "__main__":
    main()
