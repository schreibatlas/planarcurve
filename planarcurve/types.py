from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, TypeAlias

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
class CubicBezier:
    p0: Point2D
    p1: Point2D
    p2: Point2D
    p3: Point2D

    def __post_init__(self) -> None:
        for name in ("p0", "p1", "p2", "p3"):
            object.__setattr__(self, name, _finite_point(getattr(self, name), name=name))


@dataclass(frozen=True)
class SplineEndpointPolicy:
    start: Literal["locked", "free", "anchored"] = "locked"
    end: Literal["locked", "free", "anchored"] = "locked"

    def __post_init__(self) -> None:
        allowed = {"locked", "free", "anchored"}
        if self.start not in allowed or self.end not in allowed:
            raise ValueError("endpoint policy must be locked, free, or anchored")
