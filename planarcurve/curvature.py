"""Compatibility imports for the former curvature-module API.

New code should import the neutral trajectory API from :mod:`planarcurve.trajectory`.
"""

from __future__ import annotations

import numpy as np

from .trajectory import (
    HeadingRateChange,
    HeadingRateProfile,
    Pose2D,
    Trajectory2D,
    TrajectorySample,
    evaluate_heading,
    evaluate_heading_rate,
    evaluate_trajectory,
    evaluate_trajectory_at,
    integrated_heading_rate,
    sample_trajectory_by_arclength,
    trajectory_end_pose,
    trajectory_to_cubic_path,
    trajectory_total_turn,
)

__all__ = [
    "CurvatureChange",
    "CurvatureProfile",
    "CurvatureSample",
    "CurvatureStrokeGeometry",
    "CurvatureStrokeSample",
    "CurvatureTrajectory",
    "Pose2D",
    "StrokePose",
    "evaluate_curvature_stroke",
    "evaluate_heading",
    "evaluate_heading_rate",
    "evaluate_stroke_at",
    "integrated_heading_rate",
    "sample_by_arclength",
    "sample_curvature_stroke",
    "to_cubic_beziers",
    "trajectory_end_pose",
    "trajectory_total_turn",
]


CurvatureChange = HeadingRateChange
CurvatureProfile = HeadingRateProfile
CurvatureTrajectory = Trajectory2D
CurvatureStrokeGeometry = Trajectory2D
CurvatureSample = TrajectorySample
CurvatureStrokeSample = TrajectorySample
StrokePose = Pose2D
evaluate_curvature_stroke = evaluate_trajectory
evaluate_stroke_at = evaluate_trajectory_at


def curvature_to_cubic_segments(
    trajectory: Trajectory2D, *, max_error: float = 1e-4
):
    return list(trajectory_to_cubic_path(trajectory, max_error=max_error).segments)


def sample_curvature_stroke(
    trajectory: Trajectory2D, *, count: int = 256, samples: int = 64
) -> np.ndarray:
    return sample_trajectory_by_arclength(
        trajectory, samples, integration_count=count
    )


def sample_by_arclength(
    trajectory: Trajectory2D, count: int, *, integration_count: int = 256
) -> np.ndarray:
    return sample_trajectory_by_arclength(
        trajectory, count, integration_count=integration_count
    )


to_cubic_beziers = trajectory_to_cubic_path
