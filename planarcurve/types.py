from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias

Point2D: TypeAlias = tuple[float, float]


def _finite_point(value: Point2D, *, name: str) -> Point2D:
    point = (float(value[0]), float(value[1]))
    if not all(isfinite(component) for component in point):
        raise ValueError(f"{name} must contain finite coordinates")
    return point


@dataclass(frozen=True)
class CubicSegment:
    """Cubic Bezier chain segment; the start point is the previous endpoint."""

    c1: Point2D
    c2: Point2D
    end: Point2D

    def __post_init__(self) -> None:
        object.__setattr__(self, "c1", _finite_point(self.c1, name="c1"))
        object.__setattr__(self, "c2", _finite_point(self.c2, name="c2"))
        object.__setattr__(self, "end", _finite_point(self.end, name="end"))


@dataclass(frozen=True)
class CubicPath:
    """A cubic chain with an explicit starting point."""

    start: Point2D
    segments: tuple[CubicSegment, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _finite_point(self.start, name="start"))
        object.__setattr__(self, "segments", tuple(self.segments))
        if not all(isinstance(segment, CubicSegment) for segment in self.segments):
            raise TypeError("segments must contain CubicSegment values")

    def to_beziers(self) -> tuple[CubicBezier, ...]:
        point = self.start
        beziers = []
        for segment in self.segments:
            beziers.append(CubicBezier(point, segment.c1, segment.c2, segment.end))
            point = segment.end
        return tuple(beziers)


@dataclass(frozen=True)
class CubicBezier:
    p0: Point2D
    p1: Point2D
    p2: Point2D
    p3: Point2D

    def __post_init__(self) -> None:
        for name in ("p0", "p1", "p2", "p3"):
            object.__setattr__(self, name, _finite_point(getattr(self, name), name=name))




@dataclass(frozen=True)
class QuadraticBezier:
    p0: Point2D
    p1: Point2D
    p2: Point2D

    def __post_init__(self) -> None:
        for name in ("p0", "p1", "p2"):
            object.__setattr__(self, name, _finite_point(getattr(self, name), name=name))
