"""Direction-aware normalized descriptors for ordered planar trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike

from .polyline import resample_polyline_by_arclength
from .types import FloatArray, _require_integer

_DESCRIPTOR_FIELDS = (
    "normalized_xy",
    "tangent_xy",
    "heading_rate",
    "total_turn_deg",
    "endpoint_displacement_ratio",
    "signed_turn_energy",
)
_DEFAULT_WEIGHTS = {
    "normalized_xy": 1.0,
    "tangent_xy": 1.0,
    "heading_rate": 1.0,
    "total_turn_deg": 1.0,
    "endpoint_displacement_ratio": 1.0,
    "signed_turn_energy": 1.0,
}


def _readonly_array(values: ArrayLike, *, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _smooth_rows(values: FloatArray, window: int) -> FloatArray:
    if window == 1:
        return values
    left = (window - 1) // 2
    right = window - 1 - left
    padded = np.pad(values, ((left, right), (0, 0)), mode="edge")
    return np.stack(
        [
            np.convolve(padded[:, column], np.ones(window) / window, mode="valid")
            for column in range(2)
        ],
        axis=1,
    )


def _canonical_rotation(angle_rad: float) -> FloatArray:
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[cosine, sine], [-sine, cosine]], dtype=float)


def _canonical_tangents(points: FloatArray, smoothing_window: int) -> tuple[FloatArray, FloatArray]:
    smoothed = _smooth_rows(points, smoothing_window)
    derivatives = np.gradient(smoothed, axis=0)
    norms = np.linalg.norm(derivatives, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("trajectory tangent is undefined at one or more samples")
    tangents = derivatives / norms[:, None]
    start_angle = float(np.arctan2(tangents[0, 1], tangents[0, 0]))
    rotation = _canonical_rotation(start_angle)
    canonical_points = points @ rotation.T
    canonical_tangents = tangents @ rotation.T
    return canonical_points, canonical_tangents


@dataclass(frozen=True)
class TrajectoryDescriptor:
    """A sampled, scale- and pose-normalized descriptor of an ordered curve.

    ``normalized_xy`` has unit input arc length and starts at the origin.
    ``tangent_xy`` contains unit tangents in that same canonical frame.
    ``heading_rate`` is measured in degrees per normalized arc-length unit.
    The signed turn energy is the sum of each sampled heading increment times
    its absolute value, in square degrees.
    """

    sample_count: int
    normalized_xy: FloatArray
    tangent_xy: FloatArray
    heading_rate: FloatArray
    total_turn_deg: float
    endpoint_displacement_ratio: float
    signed_turn_energy: float

    def __post_init__(self) -> None:
        count = _require_integer(self.sample_count, name="sample_count", minimum=2)
        object.__setattr__(self, "sample_count", count)
        object.__setattr__(
            self,
            "normalized_xy",
            _readonly_array(self.normalized_xy, shape=(count, 2), name="normalized_xy"),
        )
        object.__setattr__(
            self,
            "tangent_xy",
            _readonly_array(self.tangent_xy, shape=(count, 2), name="tangent_xy"),
        )
        object.__setattr__(
            self,
            "heading_rate",
            _readonly_array(self.heading_rate, shape=(count,), name="heading_rate"),
        )
        scalars = (self.total_turn_deg, self.endpoint_displacement_ratio, self.signed_turn_energy)
        if not np.isfinite(scalars).all():
            raise ValueError("descriptor metrics must be finite")
        if self.endpoint_displacement_ratio < 0.0:
            raise ValueError("endpoint_displacement_ratio must be nonnegative")


def trajectory_descriptor(
    points: ArrayLike,
    *,
    samples: int = 64,
    smoothing_window: int = 5,
) -> TrajectoryDescriptor:
    """Build a deterministic directed descriptor from ordered 2D points.

    Points are resampled uniformly by chord length, translated to the origin,
    normalized to unit length, and rotated so the starting tangent points along
    +x. Input points are never reversed or mutated.
    """
    samples = _require_integer(samples, name="samples", minimum=2)
    smoothing_window = _require_integer(smoothing_window, name="smoothing_window", minimum=1)
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 2:
        raise ValueError("points must have shape (N, 2) with N >= 2")
    if not np.isfinite(values).all():
        raise ValueError("points must be finite")
    sampled = resample_polyline_by_arclength(values, samples)
    length = float(np.sum(np.linalg.norm(np.diff(sampled, axis=0), axis=1)))
    if not isfinite(length) or length <= 1e-12:
        raise ValueError("trajectory must have positive length")
    normalized = (sampled - sampled[0]) / length
    normalized[0] = 0.0
    canonical_xy, tangents = _canonical_tangents(normalized, smoothing_window)
    tangent_heading = np.unwrap(np.arctan2(tangents[:, 1], tangents[:, 0]))
    progress = np.linspace(0.0, 1.0, samples)
    heading_rate = np.gradient(tangent_heading, progress)
    if smoothing_window > 1:
        padding = (smoothing_window // 2, smoothing_window - 1 - smoothing_window // 2)
        padded = np.pad(heading_rate, padding, mode="edge")
        heading_rate = np.convolve(
            padded,
            np.ones(smoothing_window) / smoothing_window,
            mode="valid",
        )
    heading_changes = np.diff(tangent_heading)
    heading_changes = np.diff(tangent_heading)
    signed_turn_energy = float(
        np.sum(heading_changes * np.abs(heading_changes)) * (180.0 / np.pi) ** 2
    )
    return TrajectoryDescriptor(
        sample_count=samples,
        normalized_xy=canonical_xy,
        tangent_xy=tangents,
        heading_rate=np.rad2deg(heading_rate),
        total_turn_deg=float(np.rad2deg(tangent_heading[-1] - tangent_heading[0])),
        endpoint_displacement_ratio=float(np.linalg.norm(canonical_xy[-1])),
        signed_turn_energy=signed_turn_energy,
    )


def reverse_descriptor(descriptor: TrajectoryDescriptor) -> TrajectoryDescriptor:
    """Return a descriptor of the same geometry traversed in reverse."""
    if not isinstance(descriptor, TrajectoryDescriptor):
        raise TypeError("descriptor must be a TrajectoryDescriptor")
    raw_xy = descriptor.normalized_xy[::-1] - descriptor.normalized_xy[-1]
    raw_tangent = -descriptor.tangent_xy[::-1]
    start_angle = float(np.arctan2(raw_tangent[0, 1], raw_tangent[0, 0]))
    rotation = _canonical_rotation(start_angle)
    return TrajectoryDescriptor(
        sample_count=descriptor.sample_count,
        normalized_xy=raw_xy @ rotation.T,
        tangent_xy=raw_tangent @ rotation.T,
        heading_rate=-descriptor.heading_rate[::-1],
        total_turn_deg=-descriptor.total_turn_deg,
        endpoint_displacement_ratio=descriptor.endpoint_displacement_ratio,
        signed_turn_energy=-descriptor.signed_turn_energy,
    )


def descriptor_distance(
    first: TrajectoryDescriptor,
    second: TrajectoryDescriptor,
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Return a weighted RMS distance between compatible descriptors.

    Feature differences are dimensionless: angles and heading rates are scaled
    by 360 degrees, while signed turn energy is scaled by 360 squared.
    """
    if not isinstance(first, TrajectoryDescriptor) or not isinstance(second, TrajectoryDescriptor):
        raise TypeError("both inputs must be TrajectoryDescriptor values")
    if first.sample_count != second.sample_count:
        raise ValueError("descriptors must have matching sample_count")
    selected = _DEFAULT_WEIGHTS if weights is None else dict(weights)
    if set(selected) - set(_DESCRIPTOR_FIELDS):
        raise ValueError("weights contain an unknown descriptor field")
    normalized_weights = {name: float(selected.get(name, 0.0)) for name in _DESCRIPTOR_FIELDS}
    if not np.isfinite(list(normalized_weights.values())).all() or any(
        weight < 0.0 for weight in normalized_weights.values()
    ):
        raise ValueError("weights must be finite and nonnegative")
    total_weight = sum(normalized_weights.values())
    if total_weight <= 0.0:
        raise ValueError("at least one descriptor weight must be positive")
    differences = {
        "normalized_xy": float(
            np.sqrt(np.mean(np.sum((first.normalized_xy - second.normalized_xy) ** 2, axis=1)))
        ),
        "tangent_xy": float(
            np.sqrt(np.mean(np.sum((first.tangent_xy - second.tangent_xy) ** 2, axis=1)))
        ),
        "heading_rate": float(
            np.sqrt(np.mean((first.heading_rate - second.heading_rate) ** 2)) / 360.0
        ),
        "total_turn_deg": abs(first.total_turn_deg - second.total_turn_deg) / 360.0,
        "endpoint_displacement_ratio": abs(
            first.endpoint_displacement_ratio - second.endpoint_displacement_ratio
        ),
        "signed_turn_energy": abs(first.signed_turn_energy - second.signed_turn_energy) / 360.0**2,
    }
    return float(
        np.sqrt(
            sum(
                normalized_weights[name] * difference**2 for name, difference in differences.items()
            )
            / total_weight
        )
    )
