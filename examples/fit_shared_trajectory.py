from pathlib import Path

import numpy as np

from planarcurve.fitting.trajectory import (
    TrajectoryFitOptions,
    fit_shared_trajectory_profile,
)
from planarcurve.trajectory import (
    HeadingRateChange,
    HeadingRateProfile,
    Trajectory2D,
    evaluate_trajectory,
)
from planarcurve.transforms import Similarity2D

TRAJECTORY = Trajectory2D(
    length=1.0,
    start_heading_deg=8.0,
    heading_rate=HeadingRateProfile(
        start_rate_deg_per_u=0.0,
        changes=(HeadingRateChange(0.32, 250.0), HeadingRateChange(0.68, -110.0)),
        end_rate_deg_per_u=10.0,
    ),
)
TRANSFORMS = (
    Similarity2D(scale=72.0, rotation_deg=-5.0, dx=70.0, dy=130.0),
    Similarity2D(scale=66.0, rotation_deg=22.0, dx=280.0, dy=110.0),
    Similarity2D(scale=78.0, rotation_deg=-18.0, dx=505.0, dy=130.0),
)
COLORS = ("#1769aa", "#d97900", "#388e3c")


def polyline(points: np.ndarray, color: str, *, opacity: float = 1.0) -> str:
    coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-opacity="{opacity}" points="{coordinates}"/>'
    )


def main() -> None:
    rng = np.random.default_rng(4)
    base_points = evaluate_trajectory(TRAJECTORY, count=81).xy
    occurrences = [
        transform.apply(base_points) + rng.normal(0.0, 0.25, base_points.shape)
        for transform in TRANSFORMS
    ]
    fit = fit_shared_trajectory_profile(
        occurrences,
        TrajectoryFitOptions(
            change_count=2,
            samples=48,
            restarts=0,
            seed=0,
            max_nfev=100,
        ),
    )
    assert fit.success
    assert len(fit.transforms) == len(occurrences)
    assert fit.metrics["p95_position_error"] < 1.0

    fitted_base = evaluate_trajectory(fit.trajectory, count=81).xy
    rows = []
    for observed, transform, color in zip(occurrences, fit.transforms, COLORS, strict=True):
        rows.append(polyline(observed, color, opacity=0.35))
        rows.append(polyline(transform.apply(fitted_base), color))
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="680" height="260" '
        'viewBox="0 0 680 260">' + "".join(rows) + "</svg>"
    )
    Path("fit_shared_trajectory.svg").write_text(svg, encoding="utf-8")
    print(f"fit success: {fit.success}")
    print(f"p95 position error: {fit.metrics['p95_position_error']:.3f}")
    for index, transform in enumerate(fit.transforms):
        print(
            f"occurrence {index}: scale={transform.scale:.3f}, "
            f"rotation={transform.rotation_deg:.2f}, "
            f"offset=({transform.dx:.2f}, {transform.dy:.2f})"
        )
    print("wrote fit_shared_trajectory.svg")


if __name__ == "__main__":
    main()
