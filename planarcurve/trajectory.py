"""Scale-invariant 2D trajectories defined by normalized heading-rate profiles."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np

from .types import CubicPath, CubicSegment


@dataclass(frozen=True)
class HeadingRateChange:
    at: float
    heading_rate_deg_per_u: float

    def __post_init__(self) -> None:
        if not isfinite(self.at) or not 0.0 < self.at < 1.0:
            raise ValueError("heading-rate change position must lie inside (0, 1)")
        if not isfinite(self.heading_rate_deg_per_u):
            raise ValueError("heading rate must be finite")


@dataclass(frozen=True)
class HeadingRateProfile:
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
            current <= previous
            for previous, current in zip(positions, positions[1:], strict=False)
        ):
            raise ValueError("heading-rate changes must be strictly ordered")


@dataclass(frozen=True)
class Trajectory2D:
    length: float
    start_heading_deg: float
    heading_rate: HeadingRateProfile

    def __post_init__(self) -> None:
        if not isfinite(self.length) or self.length <= 0.0:
            raise ValueError("length must be positive and finite")
        if not isfinite(self.start_heading_deg):
            raise ValueError("start_heading_deg must be finite")

    def evaluate(self, *, count: int = 256) -> TrajectorySample:
        return evaluate_trajectory(self, count=count)


@dataclass(frozen=True)
class Pose2D:
    position: np.ndarray
    heading_deg: float
    heading_rate_deg_per_u: float


@dataclass(frozen=True)
class TrajectorySample:
    u: np.ndarray
    xy: np.ndarray
    heading_deg: np.ndarray
    heading_rate_deg_per_u: np.ndarray
    tangent: np.ndarray
    arc_length: np.ndarray
    end_pose: Pose2D


@dataclass(frozen=True)
class _ProfileKnots:
    positions: np.ndarray
    rates: np.ndarray
    slopes: np.ndarray


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


def _hermite_slopes(knots: np.ndarray, values: np.ndarray) -> np.ndarray:
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


def _endpoint_slope(knots: np.ndarray, deltas: np.ndarray, index: int) -> float:
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


def _validated_parameters(u: np.ndarray) -> np.ndarray:
    values = np.asarray(u, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("u must be a finite 1-D array")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("u must lie in [0, 1]")
    return values


def _span_indices(knots: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(knots, u, side="right") - 1, 0, knots.size - 2)


def evaluate_heading_rate(profile: HeadingRateProfile, u: np.ndarray) -> np.ndarray:
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


def integrated_heading_rate(profile: HeadingRateProfile, u: np.ndarray) -> np.ndarray:
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
            spans * (rates[:-1] + rates[1:]) * 0.5
            + spans**2 * (slopes[:-1] - slopes[1:]) / 12.0
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
    local = (
        h * (h00 * rates[indices] + h01 * rates[indices + 1])
        + h**2 * (h10 * slopes[indices] + h11 * slopes[indices + 1])
    )
    return prefix[indices] + local


def evaluate_heading(trajectory: Trajectory2D, u: np.ndarray) -> np.ndarray:
    """Evaluate heading in degrees, independent of the other requested samples."""
    return trajectory.start_heading_deg + integrated_heading_rate(trajectory.heading_rate, u)


def curvature(trajectory: Trajectory2D, u: np.ndarray) -> np.ndarray:
    """Evaluate signed geometric curvature in inverse length units."""
    rates = evaluate_heading_rate(trajectory.heading_rate, u)
    return np.deg2rad(rates) / trajectory.length


def evaluate_trajectory(trajectory: Trajectory2D, *, count: int = 256) -> TrajectorySample:
    if isinstance(count, bool) or count < 2:
        raise ValueError("count must be at least 2")
    u = np.linspace(0.0, 1.0, count)
    rates = evaluate_heading_rate(trajectory.heading_rate, u)
    heading_deg = evaluate_heading(trajectory, u)
    tangent = np.column_stack((np.cos(np.deg2rad(heading_deg)), np.sin(np.deg2rad(heading_deg))))
    xy = np.zeros((count, 2), dtype=float)
    xy[1:] = trajectory.length * np.cumsum(
        0.5 * (tangent[:-1] + tangent[1:]) * np.diff(u)[:, None], axis=0
    )
    arc_length = np.zeros(count, dtype=float)
    arc_length[1:] = np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))
    end_pose = Pose2D(xy[-1].copy(), float(heading_deg[-1]), float(rates[-1]))
    return TrajectorySample(u, xy, heading_deg, rates, tangent, arc_length, end_pose)


def trajectory_end_pose(trajectory: Trajectory2D, *, count: int = 256) -> Pose2D:
    return evaluate_trajectory(trajectory, count=count).end_pose


def trajectory_total_turn(trajectory: Trajectory2D) -> float:
    return float(integrated_heading_rate(trajectory.heading_rate, np.array([1.0]))[0])


def sample_trajectory_by_arclength(
    trajectory: Trajectory2D, count: int, *, integration_count: int = 256
) -> np.ndarray:
    if isinstance(count, bool) or count < 2:
        raise ValueError("sample count must be at least two")
    sample = evaluate_trajectory(trajectory, count=integration_count)
    targets = np.linspace(0.0, sample.arc_length[-1], count)
    return np.column_stack(
        (
            np.interp(targets, sample.arc_length, sample.xy[:, 0]),
            np.interp(targets, sample.arc_length, sample.xy[:, 1]),
        )
    )


def evaluate_trajectory_at(
    trajectory: Trajectory2D, u: np.ndarray, *, base_count: int = 257
) -> tuple[np.ndarray, np.ndarray]:
    values = _validated_parameters(u)
    if isinstance(base_count, bool) or base_count < 2:
        raise ValueError("base_count must be at least 2")
    grid = np.unique(np.concatenate([np.linspace(0.0, 1.0, base_count), values]))
    heading = evaluate_heading(trajectory, grid)
    tangent = np.column_stack((np.cos(np.deg2rad(heading)), np.sin(np.deg2rad(heading))))
    xy = np.zeros((grid.size, 2), dtype=float)
    xy[1:] = trajectory.length * np.cumsum(
        0.5 * (tangent[:-1] + tangent[1:]) * np.diff(grid)[:, None], axis=0
    )
    indices = np.searchsorted(grid, values)
    return xy[indices], heading[indices]


_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(8)


def _gauss_tangent_integral(
    trajectory: Trajectory2D, low: float, high: float
) -> np.ndarray:
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
) -> np.ndarray:
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
    ) + _adaptive_tangent_integral(
        trajectory, middle, high, tolerance / 2, max_depth - 1
    )


def _cubic_value(
    start: np.ndarray, c1: np.ndarray, c2: np.ndarray, end: np.ndarray, t: float
) -> np.ndarray:
    one = 1.0 - t
    return one**3 * start + 3 * one**2 * t * c1 + 3 * one * t**2 * c2 + t**3 * end


def trajectory_to_cubic_path(
    trajectory: Trajectory2D, *, max_error: float = 1e-4, max_depth: int = 16
) -> CubicPath:
    """Approximate a trajectory with cubics using quarter-point error checks.

    Each piece is recursively subdivided until the trajectory and cubic differ
    by at most ``max_error`` at its quarter, midpoint, and three-quarter points.
    """
    if not isfinite(max_error) or max_error <= 0.0:
        raise ValueError("max_error must be positive and finite")
    if (
        isinstance(max_depth, bool)
        or not isinstance(max_depth, (int, np.integer))
        or max_depth < 0
    ):
        raise ValueError("max_depth must be a nonnegative integer")

    position_cache: dict[float, np.ndarray] = {0.0: np.zeros(2)}
    profile_knots = [
        0.0,
        *(change.at for change in trajectory.heading_rate.changes),
        1.0,
    ]

    def position_at(u: float) -> np.ndarray:
        if u in position_cache:
            return position_cache[u]
        limits = [value for value in profile_knots if 0.0 < value < u]
        limits.append(u)
        points = [0.0, *limits]
        integral = np.zeros(2)
        tolerance = max_error * 1e-4 / max(1, len(points) - 1)
        for low, high in zip(points, points[1:], strict=False):
            integral += _adaptive_tangent_integral(
                trajectory, low, high, tolerance, max_depth + 8
            )
        position_cache[u] = trajectory.length * integral
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
            segments.append(
                CubicSegment(
                    c1=tuple(control1), c2=tuple(control2), end=tuple(end)
                )
            )
            return
        if depth == 0:
            raise ValueError("trajectory cubic conversion did not meet max_error")
        middle = (low + high) / 2
        fit_interval(low, middle, depth - 1)
        fit_interval(middle, high, depth - 1)

    for low, high in zip(profile_knots, profile_knots[1:], strict=False):
        fit_interval(low, high, max_depth)
    return CubicPath(tuple(position_at(0.0)), tuple(segments))
