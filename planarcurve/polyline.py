from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _points(values: np.ndarray) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points must be a finite array with shape (n, 2)")
    return points


def cumulative_arclength(points: np.ndarray) -> np.ndarray:
    values = _points(points)
    if len(values) == 0:
        return np.empty(0, dtype=float)
    lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(lengths)))


def normalized_arclength(points: np.ndarray) -> np.ndarray:
    arc = cumulative_arclength(points)
    if len(arc) < 2 or arc[-1] <= 0.0:
        raise ValueError("normalized arclength requires a nonzero polyline")
    return arc / arc[-1]


def polyline_length(points: np.ndarray) -> float:
    arc = cumulative_arclength(points)
    return 0.0 if arc.size == 0 else float(arc[-1])


def resample_polyline_by_arclength(points: np.ndarray, count: int) -> np.ndarray:
    values = _points(points)
    if isinstance(count, bool) or count < 2:
        raise ValueError("count must be at least two")
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


def unit_tangents(points: np.ndarray) -> np.ndarray:
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


def bounding_box(points: np.ndarray) -> tuple[float, float, float, float]:
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
    point: np.ndarray
    distance: np.ndarray
    segment_index: np.ndarray
    segment_t: np.ndarray
    arc_length: np.ndarray


def project_points_to_polyline(points: np.ndarray, polyline: np.ndarray) -> PolylineProjection:
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


def distance_to_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    return project_points_to_polyline(points, polyline).distance




def symmetric_polyline_distance(
    a: np.ndarray, b: np.ndarray, *, samples: int = 64
) -> float:
    """Return the mean bidirectional distance between two polylines.

    Both polylines are sampled uniformly by arc length. The result is an
    average Euclidean distance in the coordinate units of the input arrays.
    """
    if isinstance(samples, (bool, np.bool_)) or not isinstance(samples, (int, np.integer)):
        raise ValueError("samples must be an integer of at least two")
    if samples < 2:
        raise ValueError("samples must be an integer of at least two")
    sampled_a = resample_polyline_by_arclength(a, int(samples))
    sampled_b = resample_polyline_by_arclength(b, int(samples))
    a_to_b = distance_to_polyline(sampled_a, sampled_b)
    b_to_a = distance_to_polyline(sampled_b, sampled_a)
    return 0.5 * float(a_to_b.mean() + b_to_a.mean())


def mean_squared_turn_angle(
    points: np.ndarray, *, samples: int | None = None
) -> float:
    """Return mean squared adjacent-segment turn angle in radians squared.

    When ``samples`` is given, the polyline is first resampled uniformly by
    arc length. The result is the unnormalized mean of squared angles between
    consecutive nonzero segment directions, so it does not depend on scale.
    """
    values = _points(points)
    if samples is not None:
        if isinstance(samples, (bool, np.bool_)) or not isinstance(samples, (int, np.integer)):
            raise ValueError("samples must be an integer of at least two")
        if samples < 2:
            raise ValueError("samples must be an integer of at least two")
        values = resample_polyline_by_arclength(values, int(samples))
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

def closest_point_on_polyline(point: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    query = np.asarray(point, dtype=float)
    if query.shape != (2,):
        raise ValueError("point must have shape (2,)")
    return project_points_to_polyline(query[None, :], polyline).point[0]
