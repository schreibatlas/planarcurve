"""Scale-invariant planar trajectories defined by heading-rate profiles.

This module is extracted from InkGrammar's pure numerical curvature kernel.
Angles are expressed in degrees to preserve the existing numerical contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np

from .types import CubicSegment

MIN_CHANGE_SPACING = 0.02
MAX_RATE_DEG = 36000.0


@dataclass(frozen=True)
class CurvatureChange:
    at: float
    rate_deg: float

    def __post_init__(self) -> None:
        if not isfinite(self.at) or not 0.0 < self.at < 1.0:
            raise ValueError("curvature change position must lie inside (0, 1)")
        if not isfinite(self.rate_deg) or abs(self.rate_deg) > MAX_RATE_DEG:
            raise ValueError(f"heading rate must lie in [-{MAX_RATE_DEG}, {MAX_RATE_DEG}]")


@dataclass(frozen=True)
class CurvatureProfile:
    start_rate_deg: float
    end_rate_deg: float
    changes: tuple[CurvatureChange, ...] = ()
    interpolation: Literal["linear", "cubic-hermite"] = "cubic-hermite"

    def __post_init__(self) -> None:
        if self.interpolation not in {"linear", "cubic-hermite"}:
            raise ValueError("interpolation must be 'linear' or 'cubic-hermite'")
        for value in (self.start_rate_deg, self.end_rate_deg):
            if not isfinite(value) or abs(value) > MAX_RATE_DEG:
                raise ValueError(f"heading rate must lie in [-{MAX_RATE_DEG}, {MAX_RATE_DEG}]")
        changes = tuple(self.changes)
        object.__setattr__(self, "changes", changes)
        positions = [change.at for change in changes]
        for previous, current in zip(positions, positions[1:], strict=False):
            if current - previous < MIN_CHANGE_SPACING - 1e-9:
                raise ValueError(
                    "curvature changes must be strictly ordered with "
                    f"spacing >= {MIN_CHANGE_SPACING}"
                )


@dataclass(frozen=True)
class CurvatureTrajectory:
    length: float
    start_angle_deg: float
    profile: CurvatureProfile

    def __post_init__(self) -> None:
        if not isfinite(self.length) or self.length <= 0.0:
            raise ValueError("length must be positive and finite")
        if not isfinite(self.start_angle_deg):
            raise ValueError("start_angle_deg must be finite")

    @property
    def curvature(self) -> CurvatureProfile:
        """Compatibility spelling used by InkGrammar's current implementation."""
        return self.profile

    def evaluate(self, *, count: int = 256) -> CurvatureSample:
        return evaluate_curvature_stroke(self, count=count)


# Migration alias: the underlying concept is a trajectory, not an InkGrammar primitive.
CurvatureStrokeGeometry = CurvatureTrajectory


@dataclass(frozen=True)
class Pose2D:
    position: np.ndarray
    heading_deg: float
    heading_rate_deg: float


StrokePose = Pose2D


@dataclass(frozen=True)
class CurvatureSample:
    u: np.ndarray
    xy: np.ndarray
    heading_deg: np.ndarray
    heading_rate_deg: np.ndarray
    tangent: np.ndarray
    arc_length: np.ndarray
    end_pose: Pose2D

    @property
    def arc(self) -> np.ndarray:
        return self.arc_length


CurvatureStrokeSample = CurvatureSample


def _profile_knots(profile: CurvatureProfile) -> tuple[np.ndarray, np.ndarray]:
    positions = [0.0, *(change.at for change in profile.changes), 1.0]
    rates = [profile.start_rate_deg, *(change.rate_deg for change in profile.changes)]
    rates.append(profile.end_rate_deg)
    return np.asarray(positions, dtype=float), np.asarray(rates, dtype=float)


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


def evaluate_heading_rate(curvature: CurvatureProfile, u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    if u.ndim != 1 or not np.isfinite(u).all():
        raise ValueError("u must be a finite 1-D array")
    if np.any(u < 0.0) or np.any(u > 1.0):
        raise ValueError("u must lie in [0, 1]")
    knots, values = _profile_knots(curvature)
    if curvature.interpolation == "linear":
        return np.interp(u, knots, values)
    slopes = _hermite_slopes(knots, values)
    indices = np.clip(np.searchsorted(knots, u, side="right") - 1, 0, knots.size - 2)
    h = knots[indices + 1] - knots[indices]
    t = (u - knots[indices]) / h
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return (
        h00 * values[indices]
        + h10 * h * slopes[indices]
        + h01 * values[indices + 1]
        + h11 * h * slopes[indices + 1]
    )


def evaluate_heading(geometry: CurvatureTrajectory, u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    if u.ndim != 1 or u.size == 0 or not np.isfinite(u).all():
        raise ValueError("u must be a non-empty finite 1-D array")
    if u[0] != 0.0:
        raise ValueError("u must start at 0.0")
    if np.any(np.diff(u) < 0):
        raise ValueError("u must be ascending")
    rates = evaluate_heading_rate(geometry.profile, u)
    heading = np.empty_like(u)
    heading[0] = geometry.start_angle_deg
    if u.size > 1:
        increments = 0.5 * (rates[:-1] + rates[1:]) * np.diff(u)
        heading[1:] = geometry.start_angle_deg + np.cumsum(increments)
    return heading


def evaluate_curvature_stroke(
    geometry: CurvatureTrajectory, *, count: int = 256
) -> CurvatureSample:
    if isinstance(count, bool) or count < 2:
        raise ValueError("count must be at least 2")
    u = np.linspace(0.0, 1.0, count)
    rates = evaluate_heading_rate(geometry.profile, u)
    heading_deg = evaluate_heading(geometry, u)
    heading = np.deg2rad(heading_deg)
    tangent = np.column_stack((np.cos(heading), np.sin(heading)))
    steps = np.diff(u)
    xy = np.zeros((count, 2), dtype=float)
    xy[1:] = geometry.length * np.cumsum(
        0.5 * (tangent[:-1] + tangent[1:]) * steps[:, None], axis=0
    )
    arc = np.zeros(count, dtype=float)
    arc[1:] = np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))
    end_pose = Pose2D(
        position=xy[-1].copy(),
        heading_deg=float(heading_deg[-1]),
        heading_rate_deg=float(rates[-1]),
    )
    return CurvatureSample(u, xy, heading_deg, rates, tangent, arc, end_pose)


def trajectory_end_pose(geometry: CurvatureTrajectory, *, count: int = 256) -> Pose2D:
    return evaluate_curvature_stroke(geometry, count=count).end_pose


def trajectory_total_turn(geometry: CurvatureTrajectory, *, count: int = 256) -> float:
    u = np.linspace(0.0, 1.0, count)
    rates = evaluate_heading_rate(geometry.profile, u)
    return float(np.sum(0.5 * (rates[:-1] + rates[1:]) * np.diff(u)))


def sample_curvature_stroke(
    geometry: CurvatureTrajectory, *, count: int = 256, samples: int = 64
) -> np.ndarray:
    if isinstance(samples, bool) or samples < 2:
        raise ValueError("samples must be at least 2")
    sample = evaluate_curvature_stroke(geometry, count=count)
    targets = np.linspace(0.0, sample.arc_length[-1], samples)
    return np.column_stack(
        (
            np.interp(targets, sample.arc_length, sample.xy[:, 0]),
            np.interp(targets, sample.arc_length, sample.xy[:, 1]),
        )
    )


def sample_by_arclength(
    geometry: CurvatureTrajectory, count: int, *, integration_count: int = 256
) -> np.ndarray:
    return sample_curvature_stroke(geometry, count=integration_count, samples=count)


def evaluate_stroke_at(
    geometry: CurvatureTrajectory, u: np.ndarray, *, base_count: int = 257
) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=float)
    if u.ndim != 1 or u.size == 0 or not np.isfinite(u).all():
        raise ValueError("u must be a non-empty finite 1-D array")
    if u[0] < 0.0 or u[-1] > 1.0 or np.any(np.diff(u) < 0):
        raise ValueError("u must be ascending inside [0, 1]")
    grid = np.unique(np.concatenate([np.linspace(0.0, 1.0, base_count), u]))
    rates = evaluate_heading_rate(geometry.profile, grid)
    heading = np.empty_like(grid)
    heading[0] = geometry.start_angle_deg
    heading[1:] = geometry.start_angle_deg + np.cumsum(
        0.5 * (rates[:-1] + rates[1:]) * np.diff(grid)
    )
    tangent = np.column_stack((np.cos(np.deg2rad(heading)), np.sin(np.deg2rad(heading))))
    xy = np.zeros((grid.size, 2), dtype=float)
    xy[1:] = geometry.length * np.cumsum(
        0.5 * (tangent[:-1] + tangent[1:]) * np.diff(grid)[:, None], axis=0
    )
    indices = np.searchsorted(grid, u)
    return xy[indices], heading[indices]


def curvature_to_cubic_segments(
    geometry: CurvatureTrajectory, *, max_u_step: float = 0.05
) -> list[CubicSegment]:
    if not isfinite(max_u_step) or max_u_step <= 0:
        raise ValueError("max_u_step must be positive and finite")
    breaks = [0.0, *(change.at for change in geometry.profile.changes), 1.0]
    nodes = [0.0]
    for low, high in zip(breaks, breaks[1:], strict=False):
        steps = max(1, int(np.ceil((high - low) / max_u_step)))
        nodes.extend(low + (high - low) * step / steps for step in range(1, steps + 1))
    u_nodes = np.asarray(nodes)
    xy, heading_deg = evaluate_stroke_at(geometry, u_nodes)
    velocity = (
        np.column_stack((np.cos(np.deg2rad(heading_deg)), np.sin(np.deg2rad(heading_deg))))
        * geometry.length
    )
    segments: list[CubicSegment] = []
    for index in range(u_nodes.size - 1):
        step = u_nodes[index + 1] - u_nodes[index]
        start_point = xy[index]
        end_point = xy[index + 1]
        start_tangent = velocity[index] * step
        end_tangent = velocity[index + 1] * step
        segments.append(
            CubicSegment(
                c1=(
                    float(start_point[0] + start_tangent[0] / 3),
                    float(start_point[1] + start_tangent[1] / 3),
                ),
                c2=(
                    float(end_point[0] - end_tangent[0] / 3),
                    float(end_point[1] - end_tangent[1] / 3),
                ),
                end=(float(end_point[0]), float(end_point[1])),
            )
        )
    return segments


to_cubic_beziers = curvature_to_cubic_segments
