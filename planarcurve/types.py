from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

Point2D: TypeAlias = tuple[float, float]
PointLike: TypeAlias = ArrayLike
FloatArray: TypeAlias = NDArray[np.float64]


def _finite_point(value: ArrayLike, *, name: str) -> Point2D:
    point = np.asarray(value, dtype=float)
    if point.shape != (2,):
        raise ValueError(f"{name} must have exactly two coordinates")
    if not np.isfinite(point).all():
        raise ValueError(f"{name} must contain finite coordinates")
    return float(point[0]), float(point[1])


def _require_integer(value: int, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


@dataclass(frozen=True)
class CubicSegment:
    """Cubic Bezier chain segment; the start point is the previous endpoint."""

    c1: PointLike
    c2: PointLike
    end: PointLike

    def __post_init__(self) -> None:
        object.__setattr__(self, "c1", _finite_point(self.c1, name="c1"))
        object.__setattr__(self, "c2", _finite_point(self.c2, name="c2"))
        object.__setattr__(self, "end", _finite_point(self.end, name="end"))


@dataclass(frozen=True)
class CubicPath:
    """A cubic chain with an explicit starting point."""

    start: PointLike
    segments: tuple[CubicSegment, ...]
    segments: Sequence[CubicSegment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _finite_point(self.start, name="start"))
        object.__setattr__(self, "segments", tuple(self.segments))
        if not all(isinstance(segment, CubicSegment) for segment in self.segments):
            raise TypeError("segments must contain CubicSegment values")

    def to_beziers(self) -> tuple[CubicBezier, ...]:
        """Return the cubic Bezier segments in this path."""
        point = self.start
        beziers = []
        for segment in self.segments:
            beziers.append(CubicBezier(point, segment.c1, segment.c2, segment.end))
            point = segment.end
        return tuple(beziers)


@dataclass(frozen=True)
class CubicBezier:
    """Cubic Bezier curve defined by four finite 2D control points."""

    p0: PointLike
    p1: PointLike
    p2: PointLike
    p3: PointLike

    def __post_init__(self) -> None:
        for name in ("p0", "p1", "p2", "p3"):
            object.__setattr__(self, name, _finite_point(getattr(self, name), name=name))


@dataclass(frozen=True)
class QuadraticBezier:
    """Quadratic Bezier curve defined by three finite 2D control points."""

    p0: PointLike
    p1: PointLike
    p2: PointLike

    def __post_init__(self) -> None:
        for name in ("p0", "p1", "p2"):
            object.__setattr__(self, name, _finite_point(getattr(self, name), name=name))
