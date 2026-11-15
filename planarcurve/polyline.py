from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .types import FloatArray, Point2D, _require_integer


def _points(values: ArrayLike) -> FloatArray:
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points must be a finite array with shape (n, 2)")
    return points


def cumulative_arclength(points: ArrayLike) -> FloatArray:
    """Return cumulative chord lengths at each vertex."""
    values = _points(points)
    if len(values) == 0:
        return np.empty(0, dtype=float)
    lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(lengths)))


def normalized_arclength(points: ArrayLike) -> FloatArray:
    """Return cumulative chord lengths normalized to [0, 1]."""
    arc = cumulative_arclength(points)
    if len(arc) < 2 or arc[-1] <= 0.0:
        raise ValueError("normalized arclength requires a nonzero polyline")
    return arc / arc[-1]


def polyline_length(points: ArrayLike) -> float:
    """Return the total chord length of a polyline."""
    arc = cumulative_arclength(points)
    return 0.0 if arc.size == 0 else float(arc[-1])


def resample_polyline_by_arclength(points: ArrayLike, count: int) -> FloatArray:
    """Sample a polyline uniformly by cumulative chord length."""
    values = _points(points)
    count = _require_integer(count, name="count", minimum=2)
    arc = cumulative_arclength(values)
    keep = np.concatenate(([True], np.diff(arc) > 1e-12)) if arc.size else np.empty(0, bool)
    arc = arc[keep]
    values = values[keep]
    if len(arc) < 2 or arc[-1] <= 1e-12:
        raise ValueError("cannot resample a zero-length polyline")
    targets = np.linspace(0.0, arc[-1], count)
    return np.column_stack(
        (np.interp(targets, arc, values[:, 0]), np.interp(targets, arc, values[:, 1]))
    )


def unit_tangents(points: ArrayLike) -> FloatArray:
    """Estimate unit tangents at vertices using neighboring points."""
    values = _points(points)
    if len(values) < 2:
        raise ValueError("at least two points are required")
    tangent = np.empty_like(values)
    tangent[0] = values[1] - values[0]
    tangent[-1] = values[-1] - values[-2]
    if len(values) > 2:
        tangent[1:-1] = values[2:] - values[:-2]
    norms = np.linalg.norm(tangent, axis=1)
    return np.divide(
        tangent,
        norms[:, None],
        out=np.zeros_like(tangent),
        where=norms[:, None] > 1e-12,
    )


def bounding_box(points: ArrayLike) -> tuple[float, float, float, float]:
    """Return the (xmin, ymin, xmax, ymax) bounds of a polyline."""
    values = _points(points)
    if len(values) == 0:
        raise ValueError("cannot compute a bounding box for no points")
    return (
        float(values[:, 0].min()),
        float(values[:, 1].min()),
        float(values[:, 0].max()),
        float(values[:, 1].max()),
    )


@dataclass(frozen=True)
class PolylineProjection:
    """Nearest-point results, one row per query point.

    Attributes
    ----------
    point : (M, 2) float array
        Projected points on the polyline.
    distance : (M,) float array
        Euclidean distance from each query to its projection.
    segment_index : (M,) integer array
        Index of the winning segment in the input polyline.
    segment_t : (M,) float array
        Clamped local parameter in ``[0, 1]`` on the winning segment.
    arc_length : (M,) float array
        Cumulative distance from the polyline start to each projection.
    """

    point: FloatArray
    distance: FloatArray
    segment_index: np.ndarray
    segment_t: FloatArray
    arc_length: FloatArray


def project_points_to_polyline(points: ArrayLike, polyline: ArrayLike) -> PolylineProjection:
    """Project each query onto the nearest segment of a 2D polyline.

    Every query is compared with every segment. For ``M`` queries and ``N``
    segments, both time and temporary memory are ``O(M*N)``. Zero-length
    segments are allowed and contribute their endpoint as a candidate.

    Parameters
    ----------
    points : array_like, shape (M, 2)
        Finite query points.
    polyline : array_like, shape (K, 2)
        Finite polyline vertices, with at least two rows.

    Returns
    -------
    PolylineProjection
        Nearest points, distances, winning segment indices and local
        parameters, and cumulative arc lengths from the polyline start.

    Raises
    ------
    ValueError
        If the inputs are not finite 2D arrays or the polyline has fewer than
        two vertices.
    """
    queries = _points(points)
    vertices = _points(polyline)
    if len(vertices) < 2:
        raise ValueError("polyline projection requires at least two vertices")

    starts = vertices[:-1]
    vectors = np.diff(vertices, axis=0)
    squared_lengths = np.einsum("ij,ij->i", vectors, vectors)
    offsets = queries[:, None, :] - starts[None, :, :]
    numerators = np.einsum("mni,ni->mn", offsets, vectors)
    parameters = np.divide(
        numerators,
        squared_lengths[None, :],
        out=np.zeros_like(numerators),
        where=squared_lengths[None, :] > 0.0,
    )
    parameters = np.clip(parameters, 0.0, 1.0)
    candidates = starts[None, :, :] + parameters[:, :, None] * vectors[None, :, :]
    squared_distances = np.sum((queries[:, None, :] - candidates) ** 2, axis=2)
    segment_index = np.argmin(squared_distances, axis=1)
    rows = np.arange(len(queries))
    segment_t = parameters[rows, segment_index]
    projected = candidates[rows, segment_index]
    distance = np.sqrt(squared_distances[rows, segment_index])
    cumulative = cumulative_arclength(vertices)
    segment_lengths = np.sqrt(squared_lengths)
    arc_length = cumulative[segment_index] + segment_t * segment_lengths[segment_index]
    return PolylineProjection(projected, distance, segment_index, segment_t, arc_length)


def distance_to_polyline(points: ArrayLike, polyline: ArrayLike) -> FloatArray:
    """Return the nearest segment distance for each query point."""
    return project_points_to_polyline(points, polyline).distance


def symmetric_polyline_distance(a: ArrayLike, b: ArrayLike, *, samples: int = 64) -> float:
    """Return the mean bidirectional distance between two polylines.

    Both polylines are sampled uniformly by arc length. The result is an
    average Euclidean distance in the coordinate units of the input arrays.
    """
    samples = _require_integer(samples, name="samples", minimum=2)
    sampled_a = resample_polyline_by_arclength(a, samples)
    sampled_b = resample_polyline_by_arclength(b, int(samples))
    a_to_b = distance_to_polyline(sampled_a, sampled_b)
    b_to_a = distance_to_polyline(sampled_b, sampled_a)
    return 0.5 * float(a_to_b.mean() + b_to_a.mean())


def mean_squared_turn_angle(points: ArrayLike, *, samples: int | None = None) -> float:
    """Return mean squared adjacent-segment turn angle in radians squared.

    When ``samples`` is given, the polyline is first resampled uniformly by
    arc length. The result is the unnormalized mean of squared angles between
    consecutive nonzero segment directions, so it does not depend on scale.
    """
    values = _points(points)
    if samples is not None:
        samples = _require_integer(samples, name="samples", minimum=2)
        values = resample_polyline_by_arclength(values, samples)
    vectors = np.diff(values, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 1e-12
    if not np.any(valid):
        raise ValueError("turning metric requires a nonzero polyline")
    directions = vectors[valid] / lengths[valid, None]
    if len(directions) < 2:
        return 0.0
    turns = np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1.0, 1.0))
    return float(np.mean(turns**2))


def closest_point_on_polyline(point: ArrayLike, polyline: ArrayLike) -> FloatArray:
    """Return the nearest point on a polyline to one query point."""
    query = np.asarray(point, dtype=float)
    if query.shape != (2,):
        raise ValueError("point must have shape (2,)")
    return project_points_to_polyline(query[None, :], polyline).point[0]


@dataclass(frozen=True)
class PolylineIntersection:
    """One self-intersection between two polyline segments.

    Segment indices are zero-based. For a collinear overlap, ``xy`` is the
    first point of overlap along ``first_segment``.
    """

    first_segment: int
    second_segment: int
    xy: Point2D


def _cross_2d(first: FloatArray, second: FloatArray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _project_to_segment(point: FloatArray, start: FloatArray, vector: FloatArray) -> FloatArray:
    squared_length = float(vector @ vector)
    if squared_length == 0.0:
        return start
    parameter = float(np.clip((point - start) @ vector / squared_length, 0.0, 1.0))
    return start + parameter * vector


def _segment_intersection(
    first_start: FloatArray,
    first_end: FloatArray,
    second_start: FloatArray,
    second_end: FloatArray,
    tolerance: float,
) -> tuple[FloatArray, float] | None:
    first_vector = first_end - first_start
    second_vector = second_end - second_start
    first_length = float(np.linalg.norm(first_vector))
    second_length = float(np.linalg.norm(second_vector))

    if first_length <= tolerance:
        if second_length <= tolerance:
            if np.linalg.norm(first_start - second_start) <= tolerance:
                return (first_start + second_start) / 2.0, 0.0
            return None
        projected = _project_to_segment(first_start, second_start, second_vector)
        if np.linalg.norm(first_start - projected) <= tolerance:
            return (first_start + projected) / 2.0, 0.0
        return None
    if second_length <= tolerance:
        projected = _project_to_segment(second_start, first_start, first_vector)
        if np.linalg.norm(second_start - projected) <= tolerance:
            return (second_start + projected) / 2.0, 0.0
        return None

    offset = second_start - first_start
    denominator = _cross_2d(first_vector, second_vector)
    parallel_threshold = tolerance * (first_length + second_length)
    if abs(denominator) > parallel_threshold:
        first_parameter = _cross_2d(offset, second_vector) / denominator
        second_parameter = _cross_2d(offset, first_vector) / denominator
        first_margin = tolerance / first_length
        second_margin = tolerance / second_length
        if not (
            -first_margin <= first_parameter <= 1.0 + first_margin
            and -second_margin <= second_parameter <= 1.0 + second_margin
        ):
            return None
        first_point = first_start + np.clip(first_parameter, 0.0, 1.0) * first_vector
        second_point = second_start + np.clip(second_parameter, 0.0, 1.0) * second_vector
        if np.linalg.norm(first_point - second_point) <= tolerance:
            return (first_point + second_point) / 2.0, 0.0
        return None

    first_line_distance = abs(_cross_2d(offset, first_vector)) / first_length
    second_line_distance = abs(_cross_2d(second_end - first_start, first_vector)) / first_length
    if max(first_line_distance, second_line_distance) > tolerance:
        return None

    first_axis = first_vector / first_length
    second_low = float((second_start - first_start) @ first_axis)
    second_high = float((second_end - first_start) @ first_axis)
    overlap_low = max(0.0, min(second_low, second_high))
    overlap_high = min(first_length, max(second_low, second_high))
    if overlap_high < overlap_low - tolerance:
        return None

    overlap_length = max(0.0, overlap_high - overlap_low)
    if overlap_high >= overlap_low:
        point = first_start + overlap_low * first_axis
    else:
        midpoint = np.clip((overlap_low + overlap_high) / 2.0, 0.0, first_length)
        first_point = first_start + midpoint * first_axis
        second_point = _project_to_segment(first_point, second_start, second_vector)
        point = (first_point + second_point) / 2.0
    return point, overlap_length


def polyline_self_intersections(
    points: ArrayLike,
    *,
    closed: bool = False,
    tolerance: float = 1e-9,
) -> tuple[PolylineIntersection, ...]:
    """Return intersections between nontrivial pairs of polyline segments.

    Consecutive segments that meet only at their shared endpoint are not
    reported. With ``closed=True``, an implicit last-to-first segment is added
    unless the input already repeats its first point, and the first and last
    segments are treated as adjacent. A repeated endpoint is therefore an
    intersection when ``closed=False`` but an ordinary loop join when
    ``closed=True``. Collinear overlap is conservatively reported; ``xy`` is
    the first point of overlap along the first segment. ``tolerance`` is an
    absolute coordinate-space tolerance and may be zero.
    """
    vertices = _points(points)
    if len(vertices) < 2:
        raise ValueError("self-intersection detection requires at least two points")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")

    segments = list(zip(vertices[:-1], vertices[1:], strict=True))
    if closed and not np.array_equal(vertices[0], vertices[-1]):
        segments.append((vertices[-1], vertices[0]))

    intersections = []
    segment_count = len(segments)
    for first_index in range(segment_count):
        first_start, first_end = segments[first_index]
        for second_index in range(first_index + 1, segment_count):
            second_start, second_end = segments[second_index]
            hit = _segment_intersection(
                first_start,
                first_end,
                second_start,
                second_end,
                tolerance,
            )
            if hit is None:
                continue
            point, overlap_length = hit
            adjacent = second_index == first_index + 1 or (
                closed and first_index == 0 and second_index == segment_count - 1
            )
            if adjacent and overlap_length <= tolerance:
                shared = first_end if second_index == first_index + 1 else first_start
                if np.linalg.norm(point - shared) <= tolerance:
                    continue
            intersections.append(
                PolylineIntersection(
                    first_segment=first_index,
                    second_segment=second_index,
                    xy=(float(point[0]), float(point[1])),
                )
            )
    return tuple(intersections)
