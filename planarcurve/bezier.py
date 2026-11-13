from __future__ import annotations

import numpy as np

from .types import CubicBezier, QuadraticBezier


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


def evaluate_quadratic_bezier(curve: QuadraticBezier, t: np.ndarray) -> np.ndarray:
    return evaluate_quadratic(curve.p0, curve.p1, curve.p2, t)


def quadratic_derivative(curve: QuadraticBezier, t: np.ndarray) -> np.ndarray:
    u = _parameters(t)[:, None]
    return 2.0 * (
        (1.0 - u) * (np.asarray(curve.p1) - np.asarray(curve.p0))
        + u * (np.asarray(curve.p2) - np.asarray(curve.p1))
    )


def evaluate_cubic(curve: CubicBezier, t: np.ndarray) -> np.ndarray:
    u = _parameters(t)[:, None]
    one = 1.0 - u
    return (
        one**3 * np.asarray(curve.p0)
        + 3.0 * one**2 * u * np.asarray(curve.p1)
        + 3.0 * one * u**2 * np.asarray(curve.p2)
        + u**3 * np.asarray(curve.p3)
    )


def cubic_derivative(curve: CubicBezier, t: np.ndarray) -> np.ndarray:
    u = _parameters(t)[:, None]
    one = 1.0 - u
    return 3.0 * (
        one**2 * (np.asarray(curve.p1) - np.asarray(curve.p0))
        + 2.0 * one * u * (np.asarray(curve.p2) - np.asarray(curve.p1))
        + u**2 * (np.asarray(curve.p3) - np.asarray(curve.p2))
    )


def cubic_second_derivative(curve: CubicBezier, t: np.ndarray) -> np.ndarray:
    u = _parameters(t)[:, None]
    return 6.0 * (
        (1.0 - u) * (np.asarray(curve.p2) - 2.0 * np.asarray(curve.p1) + np.asarray(curve.p0))
        + u * (np.asarray(curve.p3) - 2.0 * np.asarray(curve.p2) + np.asarray(curve.p1))
    )


def cubic_tangent(curve: CubicBezier, t: np.ndarray, *, unit: bool = True) -> np.ndarray:
    derivative = cubic_derivative(curve, t)
    if not unit:
        return derivative
    norms = np.linalg.norm(derivative, axis=1)
    return np.divide(
        derivative,
        norms[:, None],
        out=np.zeros_like(derivative),
        where=norms[:, None] > 1e-12,
    )


def _signed_curvature(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    speed = np.linalg.norm(first, axis=1)
    return np.divide(cross, speed**3, out=np.zeros_like(cross), where=speed > 1e-12)


def cubic_curvature(curve: CubicBezier, t: np.ndarray) -> np.ndarray:
    return _signed_curvature(cubic_derivative(curve, t), cubic_second_derivative(curve, t))


def quadratic_tangent(
    curve: QuadraticBezier, t: np.ndarray, *, unit: bool = True
) -> np.ndarray:
    derivative = quadratic_derivative(curve, t)
    if not unit:
        return derivative
    norms = np.linalg.norm(derivative, axis=1)
    return np.divide(
        derivative,
        norms[:, None],
        out=np.zeros_like(derivative),
        where=norms[:, None] > 1e-12,
    )


def quadratic_curvature(curve: QuadraticBezier, t: np.ndarray) -> np.ndarray:
    derivative = quadratic_derivative(curve, t)
    second = np.broadcast_to(
        2.0 * (np.asarray(curve.p2) - 2.0 * np.asarray(curve.p1) + np.asarray(curve.p0)),
        derivative.shape,
    )
    return _signed_curvature(derivative, second)


def split_cubic(curve: CubicBezier, t: float = 0.5) -> tuple[CubicBezier, CubicBezier]:
    parameter = _parameters(np.array([t]))[0]
    p0, p1, p2, p3 = (
        np.asarray(point, dtype=float)
        for point in (curve.p0, curve.p1, curve.p2, curve.p3)
    )
    p01 = (1 - parameter) * p0 + parameter * p1
    p12 = (1 - parameter) * p1 + parameter * p2
    p23 = (1 - parameter) * p2 + parameter * p3
    p012 = (1 - parameter) * p01 + parameter * p12
    p123 = (1 - parameter) * p12 + parameter * p23
    middle = (1 - parameter) * p012 + parameter * p123
    return (
        CubicBezier(tuple(p0), tuple(p01), tuple(p012), tuple(middle)),
        CubicBezier(tuple(middle), tuple(p123), tuple(p23), tuple(p3)),
    )


def _point_line_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = end - start
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        return float(np.linalg.norm(point - start))
    offset = point - start
    cross = direction[0] * offset[1] - direction[1] * offset[0]
    return float(abs(cross) / norm)


def flatten_cubic(
    curve: CubicBezier, *, tolerance: float = 1e-3, max_depth: int = 16
) -> np.ndarray:
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    if isinstance(max_depth, bool) or max_depth < 0:
        raise ValueError("max_depth must be nonnegative")
    points = [np.asarray(curve.p0, dtype=float)]

    def flatten(piece: CubicBezier, depth: int) -> None:
        p0, p1, p2, p3 = (
            np.asarray(point, dtype=float) for point in (piece.p0, piece.p1, piece.p2, piece.p3)
        )
        flatness = max(_point_line_distance(p1, p0, p3), _point_line_distance(p2, p0, p3))
        if flatness <= tolerance:
            points.append(p3)
            return
        if depth == 0:
            raise ValueError("cubic flattening did not meet tolerance")
        left, right = split_cubic(piece)
        flatten(left, depth - 1)
        flatten(right, depth - 1)

    flatten(curve, max_depth)
    return np.asarray(points)


def cubic_length(curve: CubicBezier, *, tolerance: float = 1e-6, max_depth: int = 16) -> float:
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    if isinstance(max_depth, bool) or max_depth < 0:
        raise ValueError("max_depth must be nonnegative")

    def estimate(piece: CubicBezier, local_tolerance: float, depth: int) -> float:
        p0, p1, p2, p3 = (np.asarray(point) for point in (piece.p0, piece.p1, piece.p2, piece.p3))
        chord = float(np.linalg.norm(p3 - p0))
        polygon = float(
            np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2)
        )
        if polygon - chord <= local_tolerance:
            return (polygon + chord) / 2.0
        if depth == 0:
            raise ValueError("cubic length approximation did not meet tolerance")
        left, right = split_cubic(piece)
        return estimate(left, local_tolerance / 2.0, depth - 1) + estimate(
            right, local_tolerance / 2.0, depth - 1
        )

    return estimate(curve, tolerance, max_depth)


def _quadratic_as_cubic(curve: QuadraticBezier) -> CubicBezier:
    p0, p1, p2 = (np.asarray(point, dtype=float) for point in (curve.p0, curve.p1, curve.p2))
    return CubicBezier(
        tuple(p0),
        tuple(p0 + (2.0 / 3.0) * (p1 - p0)),
        tuple(p2 + (2.0 / 3.0) * (p1 - p2)),
        tuple(p2),
    )


def flatten_quadratic(
    curve: QuadraticBezier, *, tolerance: float = 1e-3, max_depth: int = 16
) -> np.ndarray:
    return flatten_cubic(
        _quadratic_as_cubic(curve), tolerance=tolerance, max_depth=max_depth
    )


def quadratic_length(
    curve: QuadraticBezier, *, tolerance: float = 1e-6, max_depth: int = 16
) -> float:
    return cubic_length(_quadratic_as_cubic(curve), tolerance=tolerance, max_depth=max_depth)
