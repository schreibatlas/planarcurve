from __future__ import annotations

import numpy as np

from .types import CubicBezier


def _parameters(t: np.ndarray) -> np.ndarray:
    values = np.asarray(t, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("parameters must be a finite one-dimensional array")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("parameters must lie in [0, 1]")
    return values


def evaluate_quadratic(p0, p1, p2, t: np.ndarray) -> np.ndarray:
    u = _parameters(t)[:, None]
    one = 1.0 - u
    return one**2 * np.asarray(p0) + 2.0 * one * u * np.asarray(p1) + u**2 * np.asarray(p2)


def evaluate_cubic(curve: CubicBezier, t: np.ndarray) -> np.ndarray:
    u = _parameters(t)[:, None]
    one = 1.0 - u
    return (
        one**3 * np.asarray(curve.p0)
        + 3.0 * one**2 * u * np.asarray(curve.p1)
        + 3.0 * one * u**2 * np.asarray(curve.p2)
        + u**3 * np.asarray(curve.p3)
    )


def cubic_tangent(curve: CubicBezier, t: np.ndarray, *, unit: bool = True) -> np.ndarray:
    u = _parameters(t)[:, None]
    one = 1.0 - u
    derivative = 3.0 * (
        one**2 * (np.asarray(curve.p1) - np.asarray(curve.p0))
        + 2.0 * one * u * (np.asarray(curve.p2) - np.asarray(curve.p1))
        + u**2 * (np.asarray(curve.p3) - np.asarray(curve.p2))
    )
    if not unit:
        return derivative
    norms = np.linalg.norm(derivative, axis=1)
    return np.divide(
        derivative,
        norms[:, None],
        out=np.zeros_like(derivative),
        where=norms[:, None] > 1e-12,
    )
