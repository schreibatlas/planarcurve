"""Optional SVG path adapter based on FontTools."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import hypot, isfinite

import numpy as np

try:
    from fontTools.pens.recordingPen import RecordingPen
    from fontTools.svgLib.path import parse_path
except ImportError as exc:  # pragma: no cover
    raise ImportError("planarcurve.svg requires FontTools; install planarcurve[svg]") from exc

from .transforms import Affine2D
from .types import Point2D


@dataclass(frozen=True)
class PathInfo:
    start: Point2D
    end: Point2D
    tangent_start: Point2D | None
    tangent_end: Point2D | None
    length: float
    bbox: tuple[float, float, float, float]
    move_count: int
    closed: bool


def _norm(vector: Point2D) -> Point2D | None:
    length = hypot(vector[0], vector[1])
    if length <= 1e-12:
        return None
    return vector[0] / length, vector[1] / length


def _lerp(a: Point2D, b: Point2D, t: float) -> Point2D:
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def _quad(p0: Point2D, p1: Point2D, p2: Point2D, t: float) -> Point2D:
    return _lerp(_lerp(p0, p1, t), _lerp(p1, p2, t), t)


def _cubic(p0: Point2D, p1: Point2D, p2: Point2D, p3: Point2D, t: float) -> Point2D:
    a = _lerp(p0, p1, t)
    b = _lerp(p1, p2, t)
    c = _lerp(p2, p3, t)
    return _lerp(_lerp(a, b, t), _lerp(b, c, t), t)


def _segment_length(points: Iterable[Point2D]) -> float:
    values = list(points)
    return sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(values, values[1:], strict=False))


def _record(path: str):
    pen = RecordingPen()
    parse_path(path, pen)
    if not pen.value:
        raise ValueError("path produced no operations")
    return pen.value


def analyze_path(path: str, samples_per_curve: int = 24) -> PathInfo:
    operations = _record(path)
    move_count = sum(op == "moveTo" for op, _ in operations)
    closed = any(op == "closePath" for op, _ in operations)
    if move_count == 0:
        raise ValueError("path has no moveTo")
    current: Point2D | None = None
    start: Point2D | None = None
    first_direction: Point2D | None = None
    last_direction: Point2D | None = None
    sampled: list[Point2D] = []
    length = 0.0
    for operator, args in operations:
        if operator == "moveTo":
            current = tuple(map(float, args[0]))
            if start is None:
                start = current
            sampled.append(current)
        elif operator in {"endPath", "closePath"}:
            continue
        elif current is None:
            raise ValueError("drawing operation before moveTo")
        elif operator == "lineTo":
            end = tuple(map(float, args[0]))
            direction = end[0] - current[0], end[1] - current[1]
            first_direction = first_direction or direction
            last_direction = direction
            length += hypot(*direction)
            sampled.append(end)
            current = end
        elif operator == "curveTo":
            if len(args) != 3:
                raise ValueError("only ordinary cubic curves are supported")
            c1, c2, end = (tuple(map(float, point)) for point in args)
            direction = c1[0] - current[0], c1[1] - current[1]
            if hypot(*direction) <= 1e-12:
                direction = end[0] - current[0], end[1] - current[1]
            first_direction = first_direction or direction
            last_direction = end[0] - c2[0], end[1] - c2[1]
            points = [
                _cubic(current, c1, c2, end, i / samples_per_curve)
                for i in range(samples_per_curve + 1)
            ]
            length += _segment_length(points)
            sampled.extend(points[1:])
            current = end
        elif operator == "qCurveTo":
            if len(args) != 2 or args[-1] is None:
                raise ValueError("only ordinary quadratic curves are supported")
            control, end = (tuple(map(float, point)) for point in args)
            direction = control[0] - current[0], control[1] - current[1]
            first_direction = first_direction or direction
            last_direction = end[0] - control[0], end[1] - control[1]
            points = [
                _quad(current, control, end, i / samples_per_curve)
                for i in range(samples_per_curve + 1)
            ]
            length += _segment_length(points)
            sampled.extend(points[1:])
            current = end
        else:
            raise ValueError(f"unsupported path operation: {operator}")
    if start is None or current is None:
        raise ValueError("path has no usable geometry")
    if not all(isfinite(value) for point in sampled for value in point):
        raise ValueError("path contains non-finite coordinates")
    xs = [point[0] for point in sampled]
    ys = [point[1] for point in sampled]
    return PathInfo(
        start=start,
        end=current,
        tangent_start=_norm(first_direction) if first_direction else None,
        tangent_end=_norm(last_direction) if last_direction else None,
        length=length,
        bbox=(min(xs), min(ys), max(xs), max(ys)),
        move_count=move_count,
        closed=closed,
    )


def sample_path(path: str, samples_per_curve: int = 24) -> list[Point2D]:
    operations = _record(path)
    current: Point2D | None = None
    sampled: list[Point2D] = []
    for operator, args in operations:
        if operator == "moveTo":
            current = tuple(map(float, args[0]))
            sampled.append(current)
        elif operator in {"endPath", "closePath"}:
            continue
        elif current is None:
            raise ValueError("drawing operation before moveTo")
        elif operator == "lineTo":
            current = tuple(map(float, args[0]))
            sampled.append(current)
        elif operator == "curveTo":
            c1, c2, end = (tuple(map(float, point)) for point in args)
            points = [
                _cubic(current, c1, c2, end, i / samples_per_curve)
                for i in range(samples_per_curve + 1)
            ]
            sampled.extend(points[1:])
            current = end
        elif operator == "qCurveTo":
            control, end = (tuple(map(float, point)) for point in args)
            points = [
                _quad(current, control, end, i / samples_per_curve)
                for i in range(samples_per_curve + 1)
            ]
            sampled.extend(points[1:])
            current = end
        else:
            raise ValueError(f"unsupported path operation: {operator}")
    return sampled


def transform_path(path: str, transform: Affine2D) -> str:
    operations = _record(path)
    chunks: list[str] = []

    def apply(point: Point2D) -> Point2D:
        x, y = transform.apply(np.asarray([point], dtype=float))[0]
        return float(x), float(y)

    def fmt(point: Point2D) -> str:
        return f"{point[0]:.6g} {point[1]:.6g}"

    for operator, args in operations:
        if operator == "moveTo":
            chunks.append(f"M {fmt(apply(tuple(map(float, args[0]))))}")
        elif operator == "lineTo":
            chunks.append(f"L {fmt(apply(tuple(map(float, args[0]))))}")
        elif operator == "curveTo":
            points = [apply(tuple(map(float, point))) for point in args]
            chunks.append("C " + " ".join(fmt(point) for point in points))
        elif operator == "qCurveTo":
            points = [apply(tuple(map(float, point))) for point in args if point is not None]
            chunks.append("Q " + " ".join(fmt(point) for point in points))
        elif operator in {"endPath", "closePath"}:
            continue
        else:
            raise ValueError(f"unsupported transform operation: {operator}")
    return " ".join(chunks)


def path_without_initial_move(path: str) -> str:
    text = path.strip()
    if not text.startswith("M"):
        raise ValueError("path must start with M")
    operations = _record(path)
    chunks: list[str] = []

    def fmt(point: Point2D) -> str:
        return f"{point[0]:.6g} {point[1]:.6g}"

    skipped = False
    for operator, args in operations:
        if operator == "moveTo" and not skipped:
            skipped = True
            continue
        if operator == "lineTo":
            chunks.append(f"L {fmt(tuple(map(float, args[0])))}")
        elif operator == "curveTo":
            chunks.append("C " + " ".join(fmt(tuple(map(float, point))) for point in args))
        elif operator == "qCurveTo":
            chunks.append(
                "Q "
                + " ".join(fmt(tuple(map(float, point))) for point in args if point is not None)
            )
        elif operator in {"endPath", "closePath"}:
            continue
        elif operator == "moveTo":
            raise ValueError("component contains multiple move operations")
    return " ".join(chunks)


def cubic_bridge(
    p0: Point2D,
    t0: Point2D | None,
    p3: Point2D,
    t3: Point2D | None,
    tension: float,
) -> str:
    t0 = t0 or (1.0, 0.0)
    t3 = t3 or (1.0, 0.0)
    c1 = p0[0] + t0[0] * tension, p0[1] + t0[1] * tension
    c2 = p3[0] - t3[0] * tension, p3[1] - t3[1] * tension
    return (
        f"M {p0[0]:.6g} {p0[1]:.6g} "
        f"C {c1[0]:.6g} {c1[1]:.6g} "
        f"{c2[0]:.6g} {c2[1]:.6g} "
        f"{p3[0]:.6g} {p3[1]:.6g}"
    )
