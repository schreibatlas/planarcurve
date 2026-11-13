"""Optional SVG path adapter based on FontTools."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import hypot

import numpy as np

try:
    from fontTools.pens.recordingPen import RecordingPen
    from fontTools.svgLib.path import parse_path
except ImportError as exc:  # pragma: no cover
    raise ImportError("planarcurve.svg requires FontTools; install planarcurve[svg]") from exc

from .bezier import evaluate_cubic, evaluate_quadratic_bezier
from .transforms import Affine2D
from .types import CubicBezier, FloatArray, Point2D, PointLike, QuadraticBezier, _require_integer


@dataclass(frozen=True)
class PathInfo:
    """Aggregate sampled geometry metadata across SVG subpaths.

    Attributes
    ----------
    start, end : 2-tuples of float
        First point of the first subpath and last point of the last subpath.
    tangent_start, tangent_end : 2-tuples of float or None
        Unit endpoint directions for the first and last subpaths, or ``None``
        when no nonzero sampled segment provides a direction.
    length : float
        Sum of sampled subpath chord lengths, in path coordinate units.
    bbox : (xmin, ymin, xmax, ymax)
        Bounds of the sampled path points, in path coordinate units.
    move_count : int
        Number of parsed subpaths.
    closed : bool
        True only when every subpath is closed.
    subpath_closed : tuple of bool
        Closed status for each subpath, in path order.
    """

    start: Point2D
    end: Point2D
    tangent_start: Point2D | None
    tangent_end: Point2D | None
    length: float
    bbox: tuple[float, float, float, float]
    move_count: int
    closed: bool
    subpath_closed: tuple[bool, ...]


def _norm(vector: Point2D) -> Point2D | None:
    length = hypot(vector[0], vector[1])
    if length <= 1e-12:
        return None
    return vector[0] / length, vector[1] / length


def _quad(p0: Point2D, p1: Point2D, p2: Point2D, t: float) -> Point2D:
    curve = QuadraticBezier(p0, p1, p2)
    point = evaluate_quadratic_bezier(curve, np.array([t]))[0]
    return float(point[0]), float(point[1])


def _cubic(p0: Point2D, p1: Point2D, p2: Point2D, p3: Point2D, t: float) -> Point2D:
    curve = CubicBezier(p0, p1, p2, p3)
    point = evaluate_cubic(curve, np.array([t]))[0]
    return float(point[0]), float(point[1])


def _segment_length(points: Iterable[Point2D]) -> float:
    values = list(points)
    return sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(values, values[1:], strict=False))


def _record(path: str):
    pen = RecordingPen()
    parse_path(path, pen)
    if not pen.value:
        raise ValueError("path produced no operations")
    return pen.value


def _validate_samples_per_curve(samples_per_curve: int) -> int:
    return _require_integer(samples_per_curve, name="samples_per_curve", minimum=1)


def _sample_subpaths(
    path: str, samples_per_curve: int
) -> tuple[tuple[FloatArray, ...], tuple[bool, ...]]:
    samples_per_curve = _validate_samples_per_curve(samples_per_curve)
    operations = _record(path)
    subpaths: list[FloatArray] = []
    closed_subpaths: list[bool] = []
    current: Point2D | None = None
    subpath_start: Point2D | None = None
    sampled: list[Point2D] | None = None
    closed = False

    def finish_subpath() -> None:
        nonlocal sampled
        if sampled is not None:
            subpaths.append(np.asarray(sampled, dtype=float))
            closed_subpaths.append(closed)
            sampled = None

    for operator, args in operations:
        if operator == "moveTo":
            finish_subpath()
            current = tuple(map(float, args[0]))
            subpath_start = current
            sampled = [current]
            closed = False
        elif operator == "endPath":
            finish_subpath()
            current = None
            subpath_start = None
            closed = False
        elif operator == "closePath":
            if current is None or subpath_start is None or sampled is None:
                raise ValueError("closePath before moveTo")
            if current != subpath_start:
                sampled.append(subpath_start)
            current = subpath_start
            closed = True
        elif current is None or sampled is None:
            raise ValueError("drawing operation before moveTo")
        elif operator == "lineTo":
            current = tuple(map(float, args[0]))
            sampled.append(current)
            closed = False
        elif operator == "curveTo":
            if len(args) != 3:
                raise ValueError("only ordinary cubic curves are supported")
            c1, c2, end = (tuple(map(float, point)) for point in args)
            points = [
                _cubic(current, c1, c2, end, index / samples_per_curve)
                for index in range(samples_per_curve + 1)
            ]
            sampled.extend(points[1:])
            current = end
            closed = False
        elif operator == "qCurveTo":
            if len(args) != 2 or args[-1] is None:
                raise ValueError("only ordinary quadratic curves are supported")
            control, end = (tuple(map(float, point)) for point in args)
            points = [
                _quad(current, control, end, index / samples_per_curve)
                for index in range(samples_per_curve + 1)
            ]
            sampled.extend(points[1:])
            current = end
            closed = False
        else:
            raise ValueError(f"unsupported path operation: {operator}")

    finish_subpath()
    if not subpaths:
        raise ValueError("path has no moveTo")
    if not all(np.isfinite(points).all() for points in subpaths):
        raise ValueError("path contains non-finite coordinates")
    return tuple(subpaths), tuple(closed_subpaths)


def sample_subpaths(path: str, samples_per_curve: int = 24) -> tuple[FloatArray, ...]:
    """Sample each SVG subpath separately, without adding connecting segments."""
    subpaths, _ = _sample_subpaths(path, samples_per_curve)
    return subpaths


def _endpoint_direction(points: FloatArray, *, from_start: bool) -> Point2D | None:
    vectors = np.diff(points, axis=0)
    indices = np.flatnonzero(np.linalg.norm(vectors, axis=1) > 1e-12)
    if indices.size == 0:
        return None
    index = indices[0] if from_start else indices[-1]
    return float(vectors[index, 0]), float(vectors[index, 1])


def analyze_path(path: str, samples_per_curve: int = 24) -> PathInfo:
    """Analyze sampled SVG path geometry and report aggregate metadata.

    Curves are sampled into ``samples_per_curve`` intervals. Length and
    endpoint tangent estimates therefore depend on that sampling density; the
    bounding box is also computed from sampled points.
    """
    subpaths, closed_subpaths = _sample_subpaths(path, samples_per_curve)
    all_points = np.concatenate(subpaths, axis=0)
    first_direction = _endpoint_direction(subpaths[0], from_start=True)
    last_direction = _endpoint_direction(subpaths[-1], from_start=False)
    return PathInfo(
        start=tuple(map(float, subpaths[0][0])),
        end=tuple(map(float, subpaths[-1][-1])),
        tangent_start=_norm(first_direction) if first_direction else None,
        tangent_end=_norm(last_direction) if last_direction else None,
        length=sum(_segment_length(map(tuple, points)) for points in subpaths),
        bbox=(
            float(all_points[:, 0].min()),
            float(all_points[:, 1].min()),
            float(all_points[:, 0].max()),
            float(all_points[:, 1].max()),
        ),
        move_count=len(subpaths),
        closed=all(closed_subpaths),
        subpath_closed=closed_subpaths,
    )


def sample_path(path: str, samples_per_curve: int = 24) -> FloatArray:
    """Sample one SVG subpath, raising if the path contains multiple subpaths.

    Use :func:`sample_subpaths` when the SVG path contains multiple moves.
    """
    subpaths = sample_subpaths(path, samples_per_curve)
    if len(subpaths) != 1:
        raise ValueError("path contains multiple subpaths; use sample_subpaths")
    return subpaths[0]


def transform_path(path: str, transform: Affine2D) -> str:
    """Transform all path coordinates while preserving subpath commands."""
    operations = _record(path)
    chunks: list[str] = []

    def apply(point: Point2D) -> Point2D:
        x, y = transform.apply(np.asarray([point], dtype=float))[0]
        return float(x), float(y)

    def fmt(point: Point2D) -> str:
        return f"{point[0]:.15g} {point[1]:.15g}"

    for operator, args in operations:
        if operator == "moveTo":
            chunks.append(f"M {fmt(apply(tuple(map(float, args[0]))))}")
        elif operator == "lineTo":
            chunks.append(f"L {fmt(apply(tuple(map(float, args[0]))))}")
        elif operator == "curveTo":
            points = [apply(tuple(map(float, point))) for point in args]
            chunks.append("C " + " ".join(fmt(point) for point in points))
        elif operator == "qCurveTo":
            if len(args) != 2 or args[-1] is None:
                raise ValueError("only ordinary quadratic curves are supported")
            points = [apply(tuple(map(float, point))) for point in args]
            chunks.append("Q " + " ".join(fmt(point) for point in points))
        elif operator == "closePath":
            chunks.append("Z")
        elif operator == "endPath":
            continue
        else:
            raise ValueError(f"unsupported transform operation: {operator}")
    return " ".join(chunks)


def path_without_initial_move(path: str) -> str:
    """Return SVG drawing commands after the path's first moveTo."""
    text = path.strip()
    if not text.startswith("M"):
        raise ValueError("path must start with M")
    operations = _record(path)
    chunks: list[str] = []

    def fmt(point: Point2D) -> str:
        return f"{point[0]:.15g} {point[1]:.15g}"

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
    p0: PointLike,
    t0: PointLike | None,
    p3: PointLike,
    t3: PointLike | None,
    tension: float,
) -> str:
    """Create an SVG cubic bridge between two points and endpoint tangents."""
    t0 = (1.0, 0.0) if t0 is None else t0
    t3 = (1.0, 0.0) if t3 is None else t3
    c1 = p0[0] + t0[0] * tension, p0[1] + t0[1] * tension
    c2 = p3[0] - t3[0] * tension, p3[1] - t3[1] * tension
    return (
        f"M {p0[0]:.15g} {p0[1]:.15g} "
        f"C {c1[0]:.15g} {c1[1]:.15g} "
        f"{c2[0]:.15g} {c2[1]:.15g} "
        f"{p3[0]:.15g} {p3[1]:.15g}"
    )
