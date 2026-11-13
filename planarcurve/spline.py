"""Cubic B-spline evaluation and conversion.

SciPy is intentionally isolated to this optional module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike

try:
    from scipy.interpolate import BSpline
    from scipy.sparse import csr_array
except ImportError as exc:  # pragma: no cover - exercised in minimal installations
    raise ImportError("planarcurve.spline requires SciPy; install planarcurve[spline]") from exc

from .transforms import Affine2D
from .types import CubicPath, CubicSegment, FloatArray, _finite_point, _require_integer


@dataclass(frozen=True)
class BSplineCurve:
    """Clamped cubic B-spline on ``[0, 1]`` with simple interior knots.

    A curve requires degree 3, at least four finite 2D control points, and
    ``n_control_points + 4`` finite, nondecreasing knots. The first and last
    four knots must be 0 and 1 respectively. Interior knots must be distinct
    and lie strictly inside ``(0, 1)``. This supported subset enables
    predictable conversion to cubic Bezier spans; it is not a general SciPy
    ``BSpline`` wrapper. See the spline concepts page for the rationale.
    """

    knots: ArrayLike
    control_points: ArrayLike
    degree: int = 3

    def __post_init__(self) -> None:
        degree = _require_integer(self.degree, name="degree", minimum=0)
        object.__setattr__(self, "degree", degree)
        knots = tuple(float(value) for value in self.knots)
        control_points = tuple(
            _finite_point(point, name="control point") for point in self.control_points
        )
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "control_points", control_points)
        validate_bspline_geometry(self)

    def evaluate(
        self, u: ArrayLike, *, transform: Affine2D | None = None, derivatives: bool = True
    ) -> CurveEvaluation:
        """Evaluate positions and optional derivatives at normalized parameters."""
        return evaluate_bspline(self, u, transform=transform, derivatives=derivatives)

    def sample_by_arclength(
        self, count: int, *, transform: Affine2D | None = None
    ) -> CurveEvaluation:
        """Return samples approximately uniform in transformed arc length."""
        return sample_curve_by_arclength(self, count, transform=transform)

    def to_cubic_segments(self, *, tolerance: float = 1e-8) -> list[CubicSegment]:
        """Convert this spline to cubic Bezier segments."""
        return spline_to_cubic_segments(self, tolerance=tolerance)

    def to_cubic_path(self, *, tolerance: float = 1e-8) -> CubicPath:
        """Convert this spline to an explicit cubic path."""
        return spline_to_cubic_path(self, tolerance=tolerance)


@dataclass(frozen=True)
class CurveEvaluation:
    """Spline evaluation arrays at normalized parameter values.

    Attributes
    ----------
    u : (N,) float array
        Parameters in ``[0, 1]``.
    xy : (N, 2) float array
        Evaluated 2D positions, in the geometry's coordinate units.
    tangent : (N, 2) float array or None
        Unit tangent vectors, or ``None`` when derivatives were not requested.
    curvature : (N,) float array or None
        Signed geometric curvature in inverse coordinate units, or ``None``
        when derivatives were not requested. Degenerate derivative locations
        produce zero values.
    """

    u: FloatArray
    xy: FloatArray
    tangent: FloatArray | None
    curvature: FloatArray | None


def _validate_parameters(u: ArrayLike) -> FloatArray:
    values = np.asarray(u, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("curve parameters must be a finite one-dimensional array")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("curve parameters must lie in [0, 1]")
    return values


def validate_bspline_geometry(geometry: BSplineCurve) -> None:
    """Validate supported clamped cubic B-spline geometry."""
    if geometry.degree != 3:
        raise ValueError("BSplineCurve supports cubic B-splines only")
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
    """Create SciPy spline objects for the x and y coordinates."""
    validate_bspline_geometry(geometry)
    knots = np.asarray(geometry.knots, dtype=float)
    control_points = np.asarray(geometry.control_points, dtype=float)
    return (
        BSpline(knots, control_points[:, 0], geometry.degree, extrapolate=False),
        BSpline(knots, control_points[:, 1], geometry.degree, extrapolate=False),
    )


def spline_design_matrix(u: ArrayLike, geometry: BSplineCurve) -> csr_array:
    """Return the sparse B-spline design matrix for normalized parameters."""
    parameters = _validate_parameters(u)
    validate_bspline_geometry(geometry)
    return BSpline.design_matrix(
        parameters,
        np.asarray(geometry.knots, dtype=float),
        geometry.degree,
        extrapolate=False,
    )


def control_roughness_matrix(control_count: int, order: int = 2) -> csr_array:
    """Return a sparse finite-difference regularizer for control points."""
    control_count = _require_integer(control_count, name="control_count", minimum=2)
    order = _require_integer(order, name="order", minimum=1)
    if order >= control_count:
        raise ValueError("difference order must be less than control_count")
    return csr_array(np.diff(np.eye(control_count), n=order, axis=0))


def evaluate_bspline(
    geometry: BSplineCurve,
    u: ArrayLike,
    *,
    transform: Affine2D | None = None,
    derivatives: bool = True,
) -> CurveEvaluation:
    """Evaluate spline positions and optional tangent and curvature fields."""
    parameters = _validate_parameters(u)
    x_spline, y_spline = bspline_from_geometry(geometry)
    xy = np.column_stack((x_spline(parameters), y_spline(parameters)))
    transform = transform or Affine2D.identity()
    xy = transform.apply(xy)
    if not derivatives:
        return CurveEvaluation(parameters, xy, None, None)
    first = np.column_stack((x_spline.derivative()(parameters), y_spline.derivative()(parameters)))
    first = first @ transform.linear.T
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


def sample_curve_by_arclength(
    geometry: BSplineCurve,
    count: int,
    *,
    transform: Affine2D | None = None,
) -> CurveEvaluation:
    """Sample approximately uniformly in transformed arc length.

    A dense grid of ``max(1024, count * 32)`` parameter values is evaluated,
    chord lengths are accumulated, and target distances are interpolated back
    to parameter values. The optional transform is included in the distance
    calculation. This is a sampled approximation, not exact arc-length
    inversion.
    """
    count = _require_integer(count, name="count", minimum=2)
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


def _bezier_values(control_points: FloatArray, t: ArrayLike) -> FloatArray:
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
    """Convert a supported B-spline to cubic Bezier chain segments."""
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


def spline_to_cubic_path(geometry: BSplineCurve, *, tolerance: float = 1e-8) -> CubicPath:
    """Convert a supported B-spline to a path with explicit start point."""
    start = evaluate_bspline(geometry, np.array([0.0]), derivatives=False).xy[0]
    segments = spline_to_cubic_segments(geometry, tolerance=tolerance)
    return CubicPath(tuple(start), tuple(segments))
