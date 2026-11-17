"""Generic parameterizations for ordered positions, intervals, and positives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import ArrayLike

from ..trajectory import HeadingRateChange, HeadingRateProfile
from ..types import FloatArray, _require_integer


def ordered_positions_from_logits(
    logits: ArrayLike,
    *,
    min_spacing: float = 0.0,
    start: float = 0.0,
    end: float = 1.0,
) -> FloatArray:
    """Decode interval logits into strictly ordered positions in ``(start, end)``."""
    values = np.asarray(logits, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("logits must be a finite 1-D array")
    _validate_interval(start, end)
    _validate_spacing(min_spacing)
    interval_count = values.size
    position_count = max(interval_count - 1, 0)
    if position_count == 0:
        return np.empty(0, dtype=float)
    span = end - start
    reserved = max(position_count - 1, 0) * min_spacing
    if reserved >= span:
        raise ValueError("interval too small for the requested positions and spacing")
    weights = np.exp(np.maximum(values - values.max(), -27.0))
    fractions = weights / weights.sum()
    gaps = fractions * (span - reserved)
    gaps[1:-1] += min_spacing
    return start + np.cumsum(gaps[:-1])


def ordered_positions_to_logits(
    positions: ArrayLike,
    *,
    min_spacing: float = 0.0,
    start: float = 0.0,
    end: float = 1.0,
) -> FloatArray:
    """Encode ordered positions as interval logits, up to a constant offset."""
    values = np.asarray(positions, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("positions must be a finite 1-D array")
    _validate_interval(start, end)
    _validate_spacing(min_spacing)
    if values.size == 0:
        return np.empty(0, dtype=float)
    if np.any(np.diff(values) <= 0.0) or values[0] <= start or values[-1] >= end:
        raise ValueError("positions must be strictly ordered inside the interval")
    gaps = np.diff(np.concatenate(([start], values, [end])))
    if np.any(gaps[1:-1] < min_spacing - 1e-12):
        raise ValueError(f"positions must respect minimum spacing {min_spacing}")
    available = (end - start) - max(values.size - 1, 0) * min_spacing
    fractions = gaps.copy()
    fractions[1:-1] -= min_spacing
    fractions = np.clip(fractions / available, 1e-12, None)
    return np.log(fractions)


def decode_interval(param: float, *, low: float, high: float) -> float:
    """Map an unconstrained scalar into the open interval ``(low, high)``."""
    _validate_interval(low, high)
    value = float(param)
    if not np.isfinite(value):
        raise ValueError("parameter must be finite")
    if value >= 0.0:
        logistic = 1.0 / (1.0 + np.exp(-value))
    else:
        exponential = np.exp(value)
        logistic = exponential / (1.0 + exponential)
    return float(low + (high - low) * logistic)


def encode_interval(value: float, *, low: float, high: float) -> float:
    """Map an interval value back to its unconstrained logistic parameter."""
    _validate_interval(low, high)
    numeric = float(value)
    if not np.isfinite(numeric) or not low <= numeric <= high:
        raise ValueError("value must be finite and inside the interval")
    fraction = np.clip((numeric - low) / (high - low), 1e-12, 1.0 - 1e-12)
    return float(np.log(fraction / (1.0 - fraction)))


def decode_positive(param: float) -> float:
    """Decode a real value as a positive scalar."""
    value = float(param)
    if not np.isfinite(value):
        raise ValueError("parameter must be finite")
    return float(np.exp(np.clip(value, -50.0, 50.0)))


def encode_positive(value: float) -> float:
    """Encode a positive scalar as its logarithm."""
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError("value must be positive and finite")
    return float(np.log(numeric))


def _validate_interval(low: float, high: float) -> None:
    if not np.isfinite([low, high]).all() or not low < high:
        raise ValueError("interval bounds must be finite and increasing")


def _validate_spacing(min_spacing: float) -> None:
    if not np.isfinite(min_spacing) or min_spacing < 0.0:
        raise ValueError("min_spacing must be finite and nonnegative")


@dataclass(frozen=True)
class HeadingRateParameterization:
    """Version 1 codec for normalized heading-rate profile parameters.

    The encoded vector stores ordered-position logits followed by bounded rate
    logits. Interpolation is configuration, not encoded data, so encode and
    decode require the configured interpolation kind to match the profile.
    Rates decode strictly inside their configured symmetric bound.
    """

    change_count: int
    min_change_spacing: float
    max_heading_rate_deg_per_u: float
    interpolation: Literal["linear", "cubic-hermite"] = "cubic-hermite"

    version: ClassVar[int] = 1

    def __post_init__(self) -> None:
        count = _require_integer(self.change_count, name="change_count", minimum=0)
        object.__setattr__(self, "change_count", count)
        if not np.isfinite(self.min_change_spacing) or self.min_change_spacing < 0.0:
            raise ValueError("min_change_spacing must be finite and nonnegative")
        if (count - 1) * self.min_change_spacing >= 1.0:
            raise ValueError("minimum spacing is too large for the requested changes")
        if (
            not np.isfinite(self.max_heading_rate_deg_per_u)
            or self.max_heading_rate_deg_per_u <= 0.0
        ):
            raise ValueError("maximum heading rate must be positive and finite")
        if self.interpolation not in {"linear", "cubic-hermite"}:
            raise ValueError("interpolation must be 'linear' or 'cubic-hermite'")

    @property
    def size(self) -> int:
        """Number of scalar values in the encoded vector."""
        return 2 * self.change_count + 2

    def encode(self, profile: HeadingRateProfile) -> FloatArray:
        """Encode a profile while validating its topology and interpolation."""
        if not isinstance(profile, HeadingRateProfile):
            raise TypeError("profile must be a HeadingRateProfile")
        if len(profile.changes) != self.change_count:
            raise ValueError("profile change count does not match parameterization")
        if profile.interpolation != self.interpolation:
            raise ValueError("profile interpolation does not match parameterization")
        positions = np.asarray([change.at for change in profile.changes], dtype=float)
        position_logits = ordered_positions_to_logits(
            positions, min_spacing=self.min_change_spacing
        )
        if self.change_count:
            position_logits = position_logits[:-1] - position_logits[-1]
        limit = self.max_heading_rate_deg_per_u
        rates = np.asarray(
            [
                profile.start_rate_deg_per_u,
                *(change.heading_rate_deg_per_u for change in profile.changes),
                profile.end_rate_deg_per_u,
            ],
            dtype=float,
        )
        if np.any(np.abs(rates) > limit):
            raise ValueError("profile heading rates exceed max_heading_rate_deg_per_u")
        rate_logits = np.asarray(
            [
                encode_interval(
                    float(np.clip(rate, -limit * (1.0 - 1e-12), limit * (1.0 - 1e-12))),
                    low=-limit,
                    high=limit,
                )
                for rate in rates
            ],
            dtype=float,
        )
        return np.concatenate((position_logits, rate_logits))

    def decode(self, values: ArrayLike) -> HeadingRateProfile:
        """Decode finite parameter values into an ordered heading-rate profile."""
        encoded = np.asarray(values, dtype=float)
        if encoded.shape != (self.size,) or not np.isfinite(encoded).all():
            raise ValueError(f"values must be a finite 1-D array with length {self.size}")
        count = self.change_count
        positions = ordered_positions_from_logits(
            np.concatenate((encoded[:count], [0.0])),
            min_spacing=self.min_change_spacing,
        )
        limit = self.max_heading_rate_deg_per_u
        rates = np.asarray(
            [decode_interval(float(value), low=-limit, high=limit) for value in encoded[count:]],
            dtype=float,
        )
        changes = tuple(
            HeadingRateChange(float(at), float(rate))
            for at, rate in zip(positions, rates[1:-1], strict=True)
        )
        return HeadingRateProfile(
            start_rate_deg_per_u=float(rates[0]),
            changes=changes,
            end_rate_deg_per_u=float(rates[-1]),
            interpolation=self.interpolation,
        )
