"""Cubic B-spline evaluation and conversion.

SciPy is intentionally isolated to this optional module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

try:
    from scipy.interpolate import BSpline
    from scipy.sparse import csr_array
except ImportError as exc:  # pragma: no cover - exercised in minimal installations
    raise ImportError("planarcurve.spline requires SciPy; install planarcurve[spline]") from exc

from .transforms import Affine2D
from .types import CubicSegment, Point2D


@dataclass(frozen=True)
class BSplineCurve:
    knots: tuple[float, ...]
    control_points: tuple[Point2D, ...]
    degree: int = 3

    def __post_init__(self) -> None:
        knots = tuple(float(value) for value in self.knots)
        control_points = tuple((float(point[0]), float(point[1])) for point in self.control_points)
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "control_points", control_points)
        validate_bspline_geometry(self)

    def evaluate(
        self, u: np.ndarray, *, transform: Affine2D | None = None, derivatives: bool = True
    ) -> CurveEvaluation:
        return evaluate_bspline(self, u, transform=transform, derivatives=derivatives)

    def sample_by_arclength(
        self, count: int, *, transform: Affine2D | None = None
    ) -> CurveEvaluation:
        return sample_curve_by_arclength(self, count, transform=transform)

    def to_cubic_segments(self, *, tolerance: float = 1e-8) -> list[CubicSegment]:
        return spline_to_cubic_segments(self, tolerance=tolerance)


BSplineGeometry = BSplineCurve


@dataclass(frozen=True)
class CurveEvaluation:
    u: np.ndarray
    xy: np.ndarray
    tangent: np.ndarray
    curvature: np.ndarray | None


def _validate_parameters(u: np.ndarray) -> np.ndarray:
    values = np.asarray(u, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("curve parameters must be a finite one-dimensional array")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("curve parameters must lie in [0, 1]")
    return values


def validate_bspline_geometry(geometry: BSplineCurve) -> None:
    if geometry.degree != 3:
        raise ValueError("the MVP supports cubic B-splines only")
    if len(geometry.control_points) < 4:
        raise ValueError("cubic B-splines require at least four control points")
    if len(geometry.knots) != len(geometry.control_points) + geometry.degree + 1:
        raise ValueError("B-spline knot count must equal control point count + degree + 1")
    if not all(isfinite(value) for value in geometry.knots):
        raise ValueError("B-spline knots must be finite")
    if not all(isfinite(value) for point in geometry.control_points for value in point):
        raise ValueError("B-spline control points must be finite")
    if tuple(sorted(geometry.knots)) != geometry.knots:
        raise ValueError("B-spline knots must be nondecreasing")
    if geometry.knots[:4] != (0.0,) * 4 or geometry.knots[-4:] != (1.0,) * 4:
        raise ValueError("cubic B-spline knots must be clamped to [0, 1]")
    interior = geometry.knots[4:-4]
    if any(not 0.0 < value < 1.0 for value in interior) or len(set(interior)) != len(interior):
        raise ValueError("cubic B-spline interior knots must be simple and inside (0, 1)")


def bspline_from_geometry(geometry: BSplineCurve) -> tuple[BSpline, BSpline]:
    validate_bspline_geometry(geometry)
    knots = np.asarray(geometry.knots, dtype=float)
    control_points = np.asarray(geometry.control_points, dtype=float)
    return (
        BSpline(knots, control_points[:, 0], geometry.degree, extrapolate=False),
        BSpline(knots, control_points[:, 1], geometry.degree, extrapolate=False),
    )


def spline_design_matrix(u: np.ndarray, geometry: BSplineCurve) -> csr_array:
    parameters = _validate_parameters(u)
    validate_bspline_geometry(geometry)
    return BSpline.design_matrix(
        parameters,
        np.asarray(geometry.knots, dtype=float),
        geometry.degree,
        extrapolate=False,
    )


def control_roughness_matrix(control_count: int, order: int = 2) -> csr_array:
    if isinstance(control_count, bool) or control_count < 2:
        raise ValueError("control_count must be at least two")
    if isinstance(order, bool) or order < 1 or order >= control_count:
        raise ValueError("difference order must be between one and control_count - 1")
    return csr_array(np.diff(np.eye(control_count), n=order, axis=0))


def evaluate_bspline(
    geometry: BSplineCurve,
    u: np.ndarray,
    *,
    transform: Affine2D | None = None,
    derivatives: bool = True,
) -> CurveEvaluation:
    parameters = _validate_parameters(u)
    x_spline, y_spline = bspline_from_geometry(geometry)
    xy = np.column_stack((x_spline(parameters), y_spline(parameters)))
    transform = transform or Affine2D.identity()
    xy = transform.apply(xy)
    if not derivatives:
        return CurveEvaluation(parameters, xy, np.zeros_like(xy), None)
    first = np.column_stack((x_spline.derivative()(parameters), y_spline.derivative()(parameters)))
    second = np.column_stack(
        (x_spline.derivative(2)(parameters), y_spline.derivative(2)(parameters))
    )
    second = second @ transform.linear.T
    speed = np.linalg.norm(first, axis=1)
    tangent = np.divide(
        first, speed[:, None], out=np.zeros_like(first), where=speed[:, None] > 1e-12
    )
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    curvature = np.divide(cross, speed**3, out=np.zeros_like(cross), where=speed > 1e-12)
    return CurveEvaluation(parameters, xy, tangent, curvature)


# More direct standalone name.
evaluate_curve = evaluate_bspline


def sample_curve_by_arclength(
    geometry: BSplineCurve,
    count: int,
    *,
    transform: Affine2D | None = None,
) -> CurveEvaluation:
    if isinstance(count, bool) or count < 2:
        raise ValueError("arclength sample count must be at least two")
    dense_u = np.linspace(0.0, 1.0, max(1024, count * 32))
    dense = evaluate_bspline(geometry, dense_u, transform=transform, derivatives=False).xy
    segment_lengths = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    keep = np.concatenate(([True], np.diff(cumulative) > 1e-12))
    cumulative = cumulative[keep]
    dense_u = dense_u[keep]
    if len(cumulative) < 2 or cumulative[-1] <= 1e-12:
        raise ValueError("cannot resample a zero-length curve")
    target_lengths = np.linspace(0.0, cumulative[-1], count)
    parameters = np.interp(target_lengths, cumulative, dense_u)
    return evaluate_bspline(geometry, parameters, transform=transform)


def _bezier_values(control_points: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)[:, None]
    one_minus_t = 1.0 - t
    return (
        one_minus_t**3 * control_points[0]
        + 3.0 * one_minus_t**2 * t * control_points[1]
        + 3.0 * one_minus_t * t**2 * control_points[2]
        + t**3 * control_points[3]
    )


def spline_to_cubic_segments(
    geometry: BSplineCurve, *, tolerance: float = 1e-8
) -> list[CubicSegment]:
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("conversion tolerance must be positive and finite")
    validate_bspline_geometry(geometry)
    refined = BSpline(
        np.asarray(geometry.knots, dtype=float),
        np.asarray(geometry.control_points, dtype=float),
        geometry.degree,
        axis=0,
        extrapolate=False,
    )
    for knot in sorted(set(geometry.knots[4:-4])):
        multiplicity = int(np.count_nonzero(np.asarray(refined.t) == knot))
        for _ in range(3 - multiplicity):
            refined = refined.insert_knot(knot)

    refined_knots = np.asarray(refined.t)
    coefficients = np.asarray(refined.c)[: len(refined_knots) - geometry.degree - 1]
    spans = [
        (left, right)
        for left, right in zip(refined_knots[:-1], refined_knots[1:], strict=True)
        if right > left
    ]
    segments: list[CubicSegment] = []
    x_spline, y_spline = bspline_from_geometry(geometry)
    for index, (left, right) in enumerate(spans):
        controls = coefficients[index * 3 : index * 3 + 4]
        segments.append(
            CubicSegment(c1=tuple(controls[1]), c2=tuple(controls[2]), end=tuple(controls[3]))
        )
        local = np.linspace(0.0, 1.0, 33)
        parameters = left + (right - left) * local
        expected = np.column_stack((x_spline(parameters), y_spline(parameters)))
        if np.max(np.linalg.norm(_bezier_values(controls, local) - expected, axis=1)) > tolerance:
            raise ValueError("cubic conversion exceeded the requested tolerance")
    return segments
