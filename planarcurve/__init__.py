"""Planar curve and trajectory mathematics."""

from .trajectory import (
    HeadingRateChange,
    HeadingRateProfile,
    Pose2D,
    Trajectory2D,
    TrajectorySample,
    curvature,
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
from .transforms import Affine2D, Similarity2D
from .types import CubicBezier, CubicPath, CubicSegment, Point2D, QuadraticBezier

try:
    from ._version import __version__
except ImportError:  # source tree before setuptools_scm generated _version.py
    __version__ = "0+unknown"

__all__ = [
    "Affine2D",
    "CubicBezier",
    "CubicPath",
    "QuadraticBezier",
    "CubicSegment",
    "HeadingRateChange",
    "HeadingRateProfile",
    "Point2D",
    "Pose2D",
    "Similarity2D",
    "Trajectory2D",
    "TrajectorySample",
    "__version__",
    "curvature",
    "evaluate_heading",
    "evaluate_heading_rate",
    "evaluate_trajectory",
    "evaluate_trajectory_at",
    "integrated_heading_rate",
    "sample_trajectory_by_arclength",
    "trajectory_end_pose",
    "trajectory_to_cubic_path",
    "trajectory_total_turn",
]
