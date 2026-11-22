"""Scale-invariant 2D trajectories defined by normalized heading-rate profiles."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from .types import CubicPath, CubicSegment, FloatArray, _require_integer


@dataclass(frozen=True)
class HeadingRateChange:
    """One heading-rate knot at normalized progress ``at``."""

    at: float
    heading_rate_deg_per_u: float

    def __post_init__(self) -> None:
        if not isfinite(self.at) or not 0.0 < self.at < 1.0:
            raise ValueError("heading-rate change position must lie inside (0, 1)")
        if not isfinite(self.heading_rate_deg_per_u):
            raise ValueError("heading rate must be finite")


@dataclass(frozen=True)
class HeadingRateProfile:
    """Heading-rate knots and interpolation over normalized progress.

    ``start_rate_deg_per_u`` and ``end_rate_deg_per_u`` are rates in degrees
    per unit ``u``. ``changes`` contains strictly ordered interior knots with
    ``at`` values in ``(0, 1)``. ``interpolation`` is ``"linear"`` or
    ``"cubic-hermite"``. The latter computes shape-preserving slopes from the
    rate knots; callers do not supply derivative values.
    """

    start_rate_deg_per_u: float
    end_rate_deg_per_u: float
    changes: tuple[HeadingRateChange, ...] = ()
    interpolation: Literal["linear", "cubic-hermite"] = "cubic-hermite"

    def __post_init__(self) -> None:
        if self.interpolation not in {"linear", "cubic-hermite"}:
            raise ValueError("interpolation must be 'linear' or 'cubic-hermite'")
        if not isfinite(self.start_rate_deg_per_u) or not isfinite(self.end_rate_deg_per_u):
            raise ValueError("heading rates must be finite")
        changes = tuple(self.changes)
        object.__setattr__(self, "changes", changes)
        positions = [change.at for change in changes]
        if any(
            current <= previous for previous, current in zip(positions, positions[1:], strict=False)
        ):
            raise ValueError("heading-rate changes must be strictly ordered")


@dataclass(frozen=True)
class Trajectory2D:
    """Constant-speed planar curve defined by a normalized heading-rate profile.

    The trajectory starts at ``(0, 0)``. ``length`` is its positive geometric
    arc length, ``start_heading_deg`` is measured in degrees, and
    ``heading_rate`` is measured in degrees per unit normalized progress ``u``.
    Theoretical arc length is ``length * u``; positions are obtained by
    numerical integration of the heading-derived tangent.
    """

    length: float
    start_heading_deg: float
    heading_rate: HeadingRateProfile

    def __post_init__(self) -> None:
        if not isfinite(self.length) or self.length <= 0.0:
            raise ValueError("length must be positive and finite")
        if not isfinite(self.start_heading_deg):
            raise ValueError("start_heading_deg must be finite")

    def evaluate(self, *, count: int = 256) -> TrajectorySample:
        """Evaluate the trajectory at uniformly spaced normalized progress."""
        return evaluate_trajectory(self, count=count)


@dataclass(frozen=True)
class Pose2D:
    """One trajectory endpoint pose.

    ``position`` is a two-element vector in coordinate units. ``heading_deg``
    is in degrees; ``heading_rate_deg_per_u`` is in degrees per unit progress.
    """

    position: FloatArray
    heading_deg: float
    heading_rate_deg_per_u: float


@dataclass(frozen=True)
class TrajectorySample:
    """Samples of a trajectory at uniformly spaced normalized progress values.

    Attributes
    ----------
    u : (count,) float array
        Normalized progress in ``[0, 1]`` (dimensionless).
    xy : (count, 2) float array
        Numerically integrated positions in trajectory coordinate units. The
        first position is the origin.
    heading_deg : (count,) float array
        Heading angles in degrees.
    heading_rate_deg_per_u : (count,) float array
        Heading rates in degrees per unit normalized progress.
    tangent : (count, 2) float array
        Dimensionless unit tangent vectors.
    arc_length : (count,) float array
        Theoretical arc lengths in coordinate units, equal to ``trajectory.length * u``.
    end_pose : Pose2D
        At ``u=1``: position shape ``(2,)`` in coordinate units, heading in
        degrees, and heading rate in degrees per unit ``u``.
    """

    u: FloatArray
    xy: FloatArray
    heading_deg: FloatArray
    heading_rate_deg_per_u: FloatArray
    tangent: FloatArray
    arc_length: FloatArray
    end_pose: Pose2D


@dataclass(frozen=True)
class _ProfileKnots:
    positions: FloatArray
    rates: FloatArray
    slopes: FloatArray


@lru_cache(maxsize=128)
def _profile_knots(profile: HeadingRateProfile) -> _ProfileKnots:
    positions = np.asarray([0.0, *(change.at for change in profile.changes), 1.0])
    rates = np.asarray(
        [
            profile.start_rate_deg_per_u,
            *(change.heading_rate_deg_per_u for change in profile.changes),
            profile.end_rate_deg_per_u,
        ]
    )
    return _ProfileKnots(positions, rates, _hermite_slopes(positions, rates))


def _hermite_slopes(knots: FloatArray, values: FloatArray) -> FloatArray:
    deltas = np.diff(values) / np.diff(knots)
    slopes = np.zeros(knots.size)
    if knots.size == 2:
        slopes[:] = deltas[0]
        return slopes
    for index in range(1, knots.size - 1):
        if deltas[index - 1] * deltas[index] <= 0:
            slopes[index] = 0.0
        else:
            weight_left = 2 * (knots[index + 1] - knots[index]) + (knots[index] - knots[index - 1])
            weight_right = (knots[index + 1] - knots[index]) + 2 * (knots[index] - knots[index - 1])
            slopes[index] = (weight_left + weight_right) / (
                weight_left / deltas[index - 1] + weight_right / deltas[index]
            )
    slopes[0] = _endpoint_slope(knots, deltas, 0)
    slopes[-1] = _endpoint_slope(knots, deltas, knots.size - 1)
    return slopes


def _endpoint_slope(knots: FloatArray, deltas: FloatArray, index: int) -> float:
    if knots.size == 2:
        return float(deltas[0])
    if index == 0:
        h0, h1 = knots[1] - knots[0], knots[2] - knots[1]
        d0, d1 = deltas[0], deltas[1]
    else:
        h0, h1 = knots[-1] - knots[-2], knots[-2] - knots[-3]
        d0, d1 = deltas[-1], deltas[-2]
    slope = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if slope * d0 <= 0:
        return 0.0
    if d0 * d1 <= 0 and abs(slope) > abs(3 * d0):
        return float(3 * d0)
    return float(slope)


def _validated_parameters(u: ArrayLike) -> FloatArray:
    values = np.asarray(u, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("u must be a finite 1-D array")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("u must lie in [0, 1]")
    return values


def _span_indices(knots: FloatArray, u: FloatArray) -> np.ndarray:
    return np.clip(np.searchsorted(knots, u, side="right") - 1, 0, knots.size - 2)


def evaluate_heading_rate(profile: HeadingRateProfile, u: ArrayLike) -> FloatArray:
    """Evaluate d(heading in degrees)/du at normalized progress ``u``."""
    values = _validated_parameters(u)
    profile_data = _profile_knots(profile)
    knots, rates = profile_data.positions, profile_data.rates
    indices = _span_indices(knots, values)
    if profile.interpolation == "linear":
        return np.interp(values, knots, rates)

    slopes = profile_data.slopes
    h = knots[indices + 1] - knots[indices]
    t = (values - knots[indices]) / h
    t2, t3 = t * t, t * t * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return (
        h00 * rates[indices]
        + h10 * h * slopes[indices]
        + h01 * rates[indices + 1]
        + h11 * h * slopes[indices + 1]
    )


def insert_heading_rate_change(profile: HeadingRateProfile, at: float) -> HeadingRateProfile:
    """Return a warm-start profile with a new knot at normalized progress ``at``.

    The new rate matches the original profile at the insertion point. Recomputing
    cubic-Hermite slopes can change the interpolated profile elsewhere, so this is
    a warm-start construction rather than an exactly equivalent representation.
    """
    position = float(at)
    if not isfinite(position) or not 0.0 < position < 1.0:
        raise ValueError("insertion progress must lie strictly inside (0, 1)")
    if any(change.at == position for change in profile.changes):
        raise ValueError("insertion progress duplicates an existing change location")
    new_change = HeadingRateChange(
        at=position,
        heading_rate_deg_per_u=float(evaluate_heading_rate(profile, np.array([position]))[0]),
    )
    changes = tuple(sorted((*profile.changes, new_change), key=lambda change: change.at))
    return HeadingRateProfile(
        start_rate_deg_per_u=profile.start_rate_deg_per_u,
        changes=changes,
        end_rate_deg_per_u=profile.end_rate_deg_per_u,
        interpolation=profile.interpolation,
    )


def integrated_heading_rate(profile: HeadingRateProfile, u: ArrayLike) -> FloatArray:
    """Return the exact integral of heading rate from zero to each ``u`` value."""
    values = _validated_parameters(u)
    data = _profile_knots(profile)
    knots, rates, slopes = data.positions, data.rates, data.slopes
    spans = np.diff(knots)
    prefix = np.zeros(knots.size)
    if profile.interpolation == "linear":
        prefix[1:] = np.cumsum(spans * (rates[:-1] + rates[1:]) * 0.5)
    else:
        prefix[1:] = np.cumsum(
            spans * (rates[:-1] + rates[1:]) * 0.5 + spans**2 * (slopes[:-1] - slopes[1:]) / 12.0
        )

    indices = _span_indices(knots, values)
    h = spans[indices]
    t = (values - knots[indices]) / h
    if profile.interpolation == "linear":
        local = h * (rates[indices] * t + 0.5 * (rates[indices + 1] - rates[indices]) * t**2)
        return prefix[indices] + local

    t2, t3, t4 = t * t, t * t * t, t * t * t * t
    h00 = 0.5 * t4 - t3 + t
    h10 = 0.25 * t4 - (2.0 / 3.0) * t3 + 0.5 * t2
    h01 = -0.5 * t4 + t3
    h11 = 0.25 * t4 - (1.0 / 3.0) * t3
    local = h * (h00 * rates[indices] + h01 * rates[indices + 1]) + h**2 * (
        h10 * slopes[indices] + h11 * slopes[indices + 1]
    )
    return prefix[indices] + local


def evaluate_heading(trajectory: Trajectory2D, u: ArrayLike) -> FloatArray:
    """Evaluate heading in degrees, independent of the other requested samples."""
    return trajectory.start_heading_deg + integrated_heading_rate(trajectory.heading_rate, u)


def curvature(trajectory: Trajectory2D, u: ArrayLike) -> FloatArray:
    """Evaluate signed geometric curvature in inverse length units."""
    rates = evaluate_heading_rate(trajectory.heading_rate, u)
    return np.deg2rad(rates) / trajectory.length


def evaluate_trajectory(trajectory: Trajectory2D, *, count: int = 256) -> TrajectorySample:
    """Sample a trajectory at uniformly spaced normalized progress values.

    Parameters
    ----------
    trajectory : Trajectory2D
        The curve to evaluate.
    count : int, optional
        Number of samples, including both endpoints. Must be at least 2 and an
        integer value.

    Returns
    -------
    TrajectorySample
        Sample arrays for progress, position, heading, heading rate, unit
        tangent, theoretical arc length, and the end pose.

    Raises
    ------
    ValueError
        If ``count`` is not an integer of at least 2.

    Notes
    -----
    Positions use adaptive numerical integration of the heading-derived
    tangent. Heading and heading rate are evaluated from the profile, while
    theoretical sample arc length is exactly ``trajectory.length * u`` up to
    floating-point arithmetic.
    """
    count = _require_integer(count, name="count", minimum=2)
    u = np.linspace(0.0, 1.0, count)
    rates = evaluate_heading_rate(trajectory.heading_rate, u)
    heading_deg = evaluate_heading(trajectory, u)
    tangent = np.column_stack((np.cos(np.deg2rad(heading_deg)), np.sin(np.deg2rad(heading_deg))))
    xy = _trajectory_positions_at(trajectory, u)
    arc_length = trajectory.length * u
    end_pose = Pose2D(xy[-1].copy(), float(heading_deg[-1]), float(rates[-1]))
    return TrajectorySample(u, xy, heading_deg, rates, tangent, arc_length, end_pose)


def trajectory_end_pose(trajectory: Trajectory2D, *, count: int = 256) -> Pose2D:
    """Return the trajectory endpoint pose."""
    return evaluate_trajectory(trajectory, count=count).end_pose


def trajectory_total_turn(trajectory: Trajectory2D) -> float:
    """Return the total signed heading change in degrees."""
    return float(integrated_heading_rate(trajectory.heading_rate, np.array([1.0]))[0])


def sample_trajectory_by_arclength(
    trajectory: Trajectory2D, count: int, *, tolerance: float = 1e-10
) -> FloatArray:
    """Sample positions uniformly in theoretical arc length."""
    count = _require_integer(count, name="count", minimum=2)
    u = np.linspace(0.0, 1.0, count)
    return _trajectory_positions_at(trajectory, u, tolerance=tolerance)


def evaluate_trajectory_at(
    trajectory: Trajectory2D, u: ArrayLike, *, tolerance: float = 1e-10
) -> tuple[FloatArray, FloatArray]:
    """Evaluate positions and headings at arbitrary normalized progress values."""
    values = _validated_parameters(u)
    heading = evaluate_heading(trajectory, values)
    xy = _trajectory_positions_at(trajectory, values, tolerance=tolerance)
    return xy, heading


_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(8)


def _gauss_tangent_integral(trajectory: Trajectory2D, low: float, high: float) -> FloatArray:
    u = (low + high) / 2 + (high - low) * _GAUSS_NODES / 2
    heading = np.deg2rad(evaluate_heading(trajectory, u))
    tangent = np.column_stack((np.cos(heading), np.sin(heading)))
    return (high - low) / 2 * (_GAUSS_WEIGHTS @ tangent)


def _adaptive_tangent_integral(
    trajectory: Trajectory2D,
    low: float,
    high: float,
    tolerance: float,
    max_depth: int,
) -> FloatArray:
    whole = _gauss_tangent_integral(trajectory, low, high)
    middle = (low + high) / 2
    left = _gauss_tangent_integral(trajectory, low, middle)
    right = _gauss_tangent_integral(trajectory, middle, high)
    refined = left + right
    if np.linalg.norm(refined - whole) <= tolerance:
        return refined
    if max_depth == 0:
        raise ValueError("trajectory position integration did not meet tolerance")
    return _adaptive_tangent_integral(
        trajectory, low, middle, tolerance / 2, max_depth - 1
    ) + _adaptive_tangent_integral(trajectory, middle, high, tolerance / 2, max_depth - 1)


def _gauss_tangent_integral_many(
    trajectory: Trajectory2D, low: FloatArray, high: FloatArray
) -> FloatArray:
    midpoints = (low + high) / 2.0
    half_widths = (high - low) / 2.0
    parameters = midpoints[:, None] + half_widths[:, None] * _GAUSS_NODES
    headings = np.deg2rad(evaluate_heading(trajectory, parameters.ravel())).reshape(
        parameters.shape
    )
    tangent = np.stack((np.cos(headings), np.sin(headings)), axis=2)
    return half_widths[:, None] * np.sum(_GAUSS_WEIGHTS[None, :, None] * tangent, axis=1)


def _adaptive_tangent_integral_many(
    trajectory: Trajectory2D,
    low: FloatArray,
    high: FloatArray,
    tolerance: float,
    max_depth: int,
) -> FloatArray:
    whole = _gauss_tangent_integral_many(trajectory, low, high)
    middle = (low + high) / 2.0
    refined = _gauss_tangent_integral_many(trajectory, low, middle) + _gauss_tangent_integral_many(
        trajectory, middle, high
    )
    converged = np.linalg.norm(refined - whole, axis=1) <= tolerance
    if np.all(converged):
        return refined
    if max_depth == 0:
        raise ValueError("trajectory position integration did not meet tolerance")
    result = refined.copy()
    unresolved = np.flatnonzero(~converged)
    result[unresolved] = _adaptive_tangent_integral_many(
        trajectory, low[unresolved], middle[unresolved], tolerance / 2.0, max_depth - 1
    ) + _adaptive_tangent_integral_many(
        trajectory, middle[unresolved], high[unresolved], tolerance / 2.0, max_depth - 1
    )
    return result


def _trajectory_positions_at(
    trajectory: Trajectory2D,
    u: FloatArray,
    *,
    tolerance: float = 1e-10,
    max_depth: int = 24,
) -> FloatArray:
    values = _validated_parameters(u)
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("position integration tolerance must be positive and finite")
    max_depth = _require_integer(max_depth, name="max_depth", minimum=0)

    knots = np.array([0.0, *(change.at for change in trajectory.heading_rate.changes), 1.0])
    interval_tolerance = tolerance / (trajectory.length * (knots.size - 1))
    prefix = np.zeros((knots.size, 2), dtype=float)
    for index, (low, high) in enumerate(zip(knots[:-1], knots[1:], strict=True)):
        prefix[index + 1] = prefix[index] + _adaptive_tangent_integral(
            trajectory, low, high, interval_tolerance, int(max_depth)
        )

    segment_indices = np.clip(np.searchsorted(knots, values, side="right") - 1, 0, knots.size - 2)
    positions = np.empty((values.size, 2), dtype=float)
    for index in range(knots.size - 1):
        selected = np.flatnonzero(segment_indices == index)
        if selected.size == 0:
            continue
        lows = np.full(selected.size, knots[index])
        partials = _adaptive_tangent_integral_many(
            trajectory,
            lows,
            values[selected],
            interval_tolerance,
            int(max_depth),
        )
        positions[selected] = trajectory.length * (prefix[index] + partials)
    return positions


def _cubic_value(
    start: FloatArray, c1: FloatArray, c2: FloatArray, end: FloatArray, t: float
) -> FloatArray:
    one = 1.0 - t
    return one**3 * start + 3 * one**2 * t * c1 + 3 * one * t**2 * c2 + t**3 * end


def trajectory_to_cubic_path(
    trajectory: Trajectory2D, *, max_error: float = 1e-4, max_depth: int = 16
) -> CubicPath:
    """Approximate a trajectory with cubic Beziers using sampled error checks.

    Each profile interval is recursively subdivided until the cubic and
    trajectory positions differ by at most ``max_error`` at local parameters
    0.25, 0.5, and 0.75. The criterion is sampled and is not a continuous
    global Hausdorff error bound.

    Parameters
    ----------
    trajectory : Trajectory2D
        The trajectory to approximate.
    max_error : float, optional
        Positive coordinate-space error threshold at the three check points.
    max_depth : int, optional
        Maximum recursive subdivision depth per profile interval.

    Returns
    -------
    CubicPath
        A chain of cubic segments with the trajectory's starting point.

    Raises
    ------
    ValueError
        If the tolerance is invalid or the sampled criterion cannot be met
        within ``max_depth``.
    """
    if not isfinite(max_error) or max_error <= 0.0:
        raise ValueError("max_error must be positive and finite")
    max_depth = _require_integer(max_depth, name="max_depth", minimum=0)

    profile_knots = [0.0, *(change.at for change in trajectory.heading_rate.changes), 1.0]
    position_cache: dict[float, FloatArray] = {0.0: np.zeros(2)}

    def position_at(u: float) -> FloatArray:
        if u not in position_cache:
            position_cache[u] = _trajectory_positions_at(
                trajectory,
                np.array([u]),
                tolerance=max_error * 1e-4,
                max_depth=max_depth + 8,
            )[0]
        return position_cache[u]

    segments: list[CubicSegment] = []

    def fit_interval(low: float, high: float, depth: int) -> None:
        start, end = position_at(low), position_at(high)
        headings = np.deg2rad(evaluate_heading(trajectory, np.array([low, high])))
        tangents = trajectory.length * np.column_stack((np.cos(headings), np.sin(headings)))
        delta = high - low
        control1 = start + delta * tangents[0] / 3
        control2 = end - delta * tangents[1] / 3
        errors = []
        for t in (0.25, 0.5, 0.75):
            expected = position_at(low + t * delta)
            approximated = _cubic_value(start, control1, control2, end, t)
            errors.append(float(np.linalg.norm(expected - approximated)))
        if max(errors) <= max_error:
            segments.append(CubicSegment(c1=tuple(control1), c2=tuple(control2), end=tuple(end)))
            return
        if depth == 0:
            raise ValueError("trajectory cubic conversion did not meet max_error")
        middle = (low + high) / 2
        fit_interval(low, middle, depth - 1)
        fit_interval(middle, high, depth - 1)

    for low, high in zip(profile_knots, profile_knots[1:], strict=False):
        fit_interval(low, high, max_depth)
    return CubicPath(tuple(position_at(0.0)), tuple(segments))
