"""Planar curve and trajectory mathematics."""

from .curvature import (
    CurvatureChange,
    CurvatureProfile,
    CurvatureSample,
    CurvatureTrajectory,
    Pose2D,
)
from .transforms import Affine2D, Similarity2D
from .types import CubicBezier, CubicSegment, Point2D

try:
    from ._version import __version__
except ImportError:  # source tree before setuptools_scm generated _version.py
    __version__ = "0+unknown"

__all__ = [
    "Affine2D",
    "CubicBezier",
    "CubicSegment",
    "CurvatureChange",
    "CurvatureProfile",
    "CurvatureSample",
    "CurvatureTrajectory",
    "Point2D",
    "Pose2D",
    "Similarity2D",
    "__version__",
]
