from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

try:
    from scipy.sparse import csr_array, eye, vstack
    from scipy.sparse.linalg import lsqr
except ImportError as exc:  # pragma: no cover - optional fitting extra
    raise ImportError(
        "planarcurve.fitting.spline requires SciPy; install planarcurve[fit]"
    ) from exc

from ..spline import BSplineCurve, control_roughness_matrix, spline_design_matrix
from ..types import _require_integer
from .policy import SplineEndpointPolicy


@dataclass(frozen=True)
class SplineObservation:
    """Observed spline samples with optional per-point weights."""

    u: ArrayLike
    xy: ArrayLike
    weight: ArrayLike | None = None


@dataclass(frozen=True)
class SplineFitOptions:
    """Regularization, robust-fit, and endpoint settings for spline fitting."""

    smoothness: float = 0.02
    seed_weight: float = 0.01
    robust_scale: float = 1.5
    irls_iterations: int = 3
    endpoint_policy: SplineEndpointPolicy = field(default_factory=SplineEndpointPolicy)
    endpoint_sigma: float = 1.0

    def __post_init__(self) -> None:
        values = (self.smoothness, self.seed_weight, self.robust_scale, self.endpoint_sigma)
        if not np.isfinite(values).all():
            raise ValueError("spline fit options must be finite")
        if self.smoothness < 0.0 or self.seed_weight < 0.0:
            raise ValueError("spline regularization weights must be nonnegative")
        if self.robust_scale <= 0.0 or self.endpoint_sigma <= 0.0:
            raise ValueError("robust scale and endpoint sigma must be positive")
        object.__setattr__(
            self,
            "irls_iterations",
            _require_integer(self.irls_iterations, name="irls_iterations", minimum=1),
        )


@dataclass(frozen=True)
class SplineFitResult:
    """Fitted shared spline and its aggregate residual summary."""

    curve: BSplineCurve
    residual_rms: float
    iterations: int


def _validated_observations(
    observations: Sequence[SplineObservation],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    parameters = []
    observed_points = []
    base_weights = []
    for observation in observations:
        u = np.asarray(observation.u, dtype=float)
        xy = np.asarray(observation.xy, dtype=float)
        weights = (
            np.ones(len(u), dtype=float)
            if observation.weight is None
            else np.asarray(observation.weight, dtype=float)
        )
        if u.ndim != 1 or xy.shape != (len(u), 2) or weights.shape != (len(u),):
            raise ValueError("spline observation arrays have inconsistent shapes")
        if (
            not np.isfinite(u).all()
            or np.any((u < 0.0) | (u > 1.0))
            or not np.isfinite(xy).all()
            or not np.isfinite(weights).all()
            or np.any(weights < 0.0)
        ):
            raise ValueError(
                "spline observations must be finite with in-range u and nonnegative weights"
            )
        parameters.append(u)
        observed_points.append(xy)
        base_weights.append(weights)
    if not parameters or sum(map(len, parameters)) == 0:
        raise ValueError("at least one spline observation is required")
    if not np.any(np.concatenate(base_weights) > 0.0):
        raise ValueError("spline observations must have positive total weight")
    return parameters, np.vstack(observed_points), np.concatenate(base_weights)


def _weighted_rows(matrix: csr_array, weights: np.ndarray) -> csr_array:
    return csr_array(matrix.multiply(np.sqrt(weights)[:, None]))


def _solve_coordinate(
    matrix: csr_array,
    target: np.ndarray,
    seed: np.ndarray,
    fixed_indices: np.ndarray,
) -> np.ndarray:
    control_count = len(seed)
    free_indices = np.setdiff1d(np.arange(control_count), fixed_indices, assume_unique=True)
    result = np.empty(control_count, dtype=float)
    result[fixed_indices] = seed[fixed_indices]
    right_hand_side = target - matrix[:, fixed_indices] @ result[fixed_indices]
    if free_indices.size:
        result[free_indices] = lsqr(
            matrix[:, free_indices],
            np.asarray(right_hand_side).ravel(),
            atol=1e-12,
            btol=1e-12,
            iter_lim=max(100, control_count * 20),
        )[0]
    return result


def fit_spline_control_points(
    basis: BSplineCurve,
    observations: Sequence[SplineObservation],
    *,
    seed_control_points: ArrayLike | None = None,
    options: SplineFitOptions | None = None,
) -> SplineFitResult:
    """Fit shared B-spline controls with sparse weighted least squares and IRLS."""
    if options is None:
        options = SplineFitOptions()
    parameter_arrays, observed_xy, base_weights = _validated_observations(observations)
    control_count = len(basis.control_points)
    seed = np.asarray(
        basis.control_points if seed_control_points is None else seed_control_points,
        dtype=float,
    )
    if seed.shape != (control_count, 2) or not np.isfinite(seed).all():
        raise ValueError("seed control points must be finite and match the spline basis")

    data_matrix = vstack([spline_design_matrix(u, basis) for u in parameter_arrays], format="csr")
    roughness = control_roughness_matrix(control_count, order=2)
    roughness_rows = np.sqrt(options.smoothness) * roughness
    seed_rows = np.sqrt(options.seed_weight) * eye(control_count, format="csr")
    endpoint_rows = []
    endpoint_values = []
    for index, policy in (
        (0, options.endpoint_policy.start),
        (-1, options.endpoint_policy.end),
    ):
        if policy == "anchored":
            endpoint_rows.append(eye(control_count, format="csr")[index] / options.endpoint_sigma)
            endpoint_values.append(seed[index] / options.endpoint_sigma)
    fixed_indices = np.asarray(
        [
            index % control_count
            for index, policy in (
                (0, options.endpoint_policy.start),
                (-1, options.endpoint_policy.end),
            )
            if policy == "locked"
        ],
        dtype=int,
    )

    robust_weights = np.ones_like(base_weights)
    control_points = seed.copy()
    for _ in range(options.irls_iterations):
        combined_weights = base_weights * robust_weights
        augmented_matrix = vstack(
            [
                _weighted_rows(data_matrix, combined_weights),
                roughness_rows,
                seed_rows,
                *endpoint_rows,
            ],
            format="csr",
        )
        weighted_observations = observed_xy * np.sqrt(combined_weights)[:, None]
        regularized_values = [
            np.zeros((roughness.shape[0], 2)),
            np.sqrt(options.seed_weight) * seed,
        ]
        regularized_values.extend(endpoint_values)
        augmented_values = np.vstack([weighted_observations, *regularized_values])
        control_points = np.column_stack(
            [
                _solve_coordinate(
                    augmented_matrix,
                    augmented_values[:, axis],
                    seed[:, axis],
                    fixed_indices,
                )
                for axis in range(2)
            ]
        )
        residuals = np.linalg.norm(data_matrix @ control_points - observed_xy, axis=1)
        robust_weights = np.minimum(1.0, options.robust_scale / np.maximum(residuals, 1e-12))

    residuals = np.linalg.norm(data_matrix @ control_points - observed_xy, axis=1)
    curve = BSplineCurve(
        knots=basis.knots,
        control_points=tuple(map(tuple, control_points)),
        degree=basis.degree,
    )
    return SplineFitResult(
        curve=curve,
        residual_rms=float(np.sqrt(np.mean(residuals**2))),
        iterations=options.irls_iterations,
    )
