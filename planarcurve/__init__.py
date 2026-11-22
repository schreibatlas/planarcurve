"""Planar curve and trajectory mathematics."""

from .descriptors import (
    TrajectoryDescriptor,
    descriptor_distance,
    reverse_descriptor,
    trajectory_descriptor,
)
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
    insert_heading_rate_change,
    integrated_heading_rate,
    sample_trajectory_by_arclength,
    trajectory_end_pose,
    trajectory_to_cubic_path,
    trajectory_total_turn,
)
from .trajectory_chain import (
    DirectedTrajectory,
    TrajectoryChain,
    TrajectoryChainSample,
    evaluate_trajectory_chain,
    reverse_trajectory,
    trajectory_chain_to_cubic_path,
)
from .transforms import (
    Affine2D,
    Similarity2D,
    fit_similarity_transform,
    transform_cubic_path,
)
from .types import (
    CubicBezier,
    CubicPath,
    CubicSegment,
    FloatArray,
    Point2D,
    PointLike,
    QuadraticBezier,
)

try:
    from ._version import __version__
except ImportError:  # source tree before setuptools_scm generated _version.py
    __version__ = "0+unknown"

__all__ = [
    "Affine2D",
    "CubicBezier",
    "CubicPath",
    "FloatArray",
    "QuadraticBezier",
    "CubicSegment",
    "HeadingRateChange",
    "HeadingRateProfile",
    "Point2D",
    "PointLike",
    "Pose2D",
    "Similarity2D",
    "fit_similarity_transform",
    "Trajectory2D",
    "TrajectorySample",
    "__version__",
    "curvature",
    "evaluate_heading",
    "evaluate_heading_rate",
    "insert_heading_rate_change",
    "evaluate_trajectory",
    "evaluate_trajectory_at",
    "integrated_heading_rate",
    "sample_trajectory_by_arclength",
    "trajectory_end_pose",
    "trajectory_to_cubic_path",
    "trajectory_total_turn",
    "DirectedTrajectory",
    "TrajectoryChain",
    "TrajectoryChainSample",
    "evaluate_trajectory_chain",
    "reverse_trajectory",
    "trajectory_chain_to_cubic_path",
    "TrajectoryDescriptor",
    "transform_cubic_path",
    "descriptor_distance",
    "reverse_descriptor",
    "trajectory_descriptor",
]
