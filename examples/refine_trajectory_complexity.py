"""Progressively refine a synthetic S-curve with smooth steering controls."""

from __future__ import annotations

from planarcurve.fitting import (
    TrajectoryFitOptions,
    TrajectoryRefinementOptions,
    refine_trajectory_model,
)
from planarcurve.trajectory import (
    HeadingRateChange,
    HeadingRateProfile,
    Trajectory2D,
    evaluate_trajectory,
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
    result = refine_trajectory_model(
        points,
        fit_options=TrajectoryFitOptions(
            change_count=0,
            samples=32,
            restarts=0,
            max_nfev=100,
        ),
        refinement_options=TrajectoryRefinementOptions(
            max_change_count=2,
            candidate_insertions=3,
            complexity_weight=0.001,
        ),
    )

    print(f"changes=0 score={result.steps[0].score:.6f}")
    for step in result.steps[1:]:
        status = "accepted" if step.accepted else "rejected"
        print(
            f"try +1 at u={step.insertion_progress:.3f} "
            f"score={step.score:.6f} {status} ({step.reason})"
        )
    print(f"selected_changes={result.selected_change_count}")
    print(f"optimizer_evaluations={result.optimizer_evaluations}")
    print("Change points are smooth steering controls, not geometric corners.")
    assert result.selected_change_count == 1


if __name__ == "__main__":
    main()
