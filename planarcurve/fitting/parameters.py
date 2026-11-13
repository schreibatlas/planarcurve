"""Generic parameterizations for ordered positions, intervals, and positives."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..types import FloatArray


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
