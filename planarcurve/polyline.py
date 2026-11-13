from __future__ import annotations

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
