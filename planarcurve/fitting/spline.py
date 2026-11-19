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

from ..polyline import normalized_arclength
from ..spline import (
    BSplineCurve,
    clamped_uniform_bspline_basis,
    control_roughness_matrix,
    spline_design_matrix,
)
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
    """Regularization, robust-fit, and endpoint settings for spline fitting.

    ``smoothness`` penalizes second differences of control points, and
    ``seed_weight`` regularizes controls toward the seed. IRLS downweights
    residuals larger than ``robust_scale`` for ``irls_iterations`` solves.
    ``endpoint_policy`` independently chooses ``free``, ``locked``, or
    ``anchored`` for each endpoint. Anchored endpoints are softly pulled toward
    the seed with scale ``endpoint_sigma``. See the fitting guide for details.
    """

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
    """Fit shared B-spline controls with sparse weighted least squares and IRLS.

    Observations may have per-point nonnegative weights. Smoothness and seed
    penalties regularize the controls; IRLS reduces the influence of large
    residuals. Endpoint policy determines whether seed endpoints are free,
    fixed, or softly anchored.

    Parameters
    ----------
    basis : BSplineCurve
        Supported cubic spline geometry providing knots and initial controls.
    observations : sequence of SplineObservation
        Samples at normalized parameters, optionally weighted per point.
    seed_control_points : array_like, shape (n_control_points, 2), optional
        Controls used as the regularization seed. Defaults to the basis controls.
    options : SplineFitOptions, optional
        Smoothness, robust-fit, and endpoint settings.

    Returns
    -------
    SplineFitResult
        Fitted curve, residual RMS, and configured IRLS iteration count.
    """
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


@dataclass(frozen=True)
class PolylineSplineFitOptions:
    """Scale-normalized policy for fitting an ordered polyline with a B-spline.

    Residual targets are fractions of total input arc length. If no candidate
    meets both targets, selection minimizes the sum of RMS and p95 residuals
    divided by their targets, plus ``complexity_weight * control_count / max``
    and ``roughness_weight * normalized_control_roughness``.
    """

    min_control_points: int = 4
    max_control_points: int = 16
    target_rms_error_fraction: float = 0.002
    target_p95_error_fraction: float = 0.004
    smoothness: float = 0.02
    robust_scale: float = 1.5
    irls_iterations: int = 4
    endpoint_policy: SplineEndpointPolicy = field(
        default_factory=lambda: SplineEndpointPolicy(start="anchored", end="anchored")
    )
    endpoint_sigma: float = 1.0
    candidate_control_counts: tuple[int, ...] | None = None
    complexity_weight: float = 0.05
    roughness_weight: float = 0.1

    def __post_init__(self) -> None:
        minimum = _require_integer(self.min_control_points, name="min_control_points", minimum=4)
        maximum = _require_integer(self.max_control_points, name="max_control_points", minimum=4)
        if maximum < minimum:
            raise ValueError("max_control_points must be at least min_control_points")
        object.__setattr__(self, "min_control_points", minimum)
        object.__setattr__(self, "max_control_points", maximum)

        values = (
            self.target_rms_error_fraction,
            self.target_p95_error_fraction,
            self.smoothness,
            self.robust_scale,
            self.endpoint_sigma,
            self.complexity_weight,
            self.roughness_weight,
        )
        if not np.isfinite(values).all():
            raise ValueError("polyline spline fit options must be finite")
        if self.target_rms_error_fraction <= 0.0 or self.target_p95_error_fraction <= 0.0:
            raise ValueError("residual target fractions must be positive")
        if self.smoothness < 0.0 or self.complexity_weight < 0.0 or self.roughness_weight < 0.0:
            raise ValueError("smoothness and score weights must be nonnegative")
        if self.robust_scale <= 0.0 or self.endpoint_sigma <= 0.0:
            raise ValueError("robust scale and endpoint sigma must be positive")
        object.__setattr__(
            self,
            "irls_iterations",
            _require_integer(self.irls_iterations, name="irls_iterations", minimum=1),
        )
        if not isinstance(self.endpoint_policy, SplineEndpointPolicy):
            raise ValueError("endpoint_policy must be a SplineEndpointPolicy")

        if self.candidate_control_counts is not None:
            counts = tuple(
                _require_integer(value, name="candidate_control_counts", minimum=4)
                for value in self.candidate_control_counts
            )
            if not counts:
                raise ValueError("candidate_control_counts must not be empty")
            if len(set(counts)) != len(counts):
                raise ValueError("candidate_control_counts must be unique")
            if any(value < minimum or value > maximum for value in counts):
                raise ValueError(
                    "candidate_control_counts must lie within configured control bounds"
                )
            object.__setattr__(self, "candidate_control_counts", tuple(sorted(counts)))


@dataclass(frozen=True)
class PolylineSplineCandidateScore:
    """Normalized metrics and selection score for one control-count candidate."""

    control_count: int
    residual_rms: float
    residual_median: float
    residual_p95: float
    residual_max: float
    roughness: float
    score: float
    met_target_error: bool
    weighted_residual_rms: float | None


@dataclass(frozen=True)
class PolylineSplineFitDiagnostics:
    """Selected-model metrics and metrics for all attempted candidates.

    Residual metrics are normalized by ``normalized_scale``, the input
    polyline's total arc length. Weighted RMS is ``None`` when no weights
    were supplied.
    """

    control_count: int
    residual_rms: float
    residual_median: float
    residual_p95: float
    residual_max: float
    normalized_scale: float
    met_target_error: bool
    candidate_scores: tuple[PolylineSplineCandidateScore, ...]
    weighted_residual_rms: float | None


@dataclass(frozen=True)
class PolylineSplineFitResult:
    """A fitted ordinary B-spline and its model-selection diagnostics."""

    curve: BSplineCurve
    diagnostics: PolylineSplineFitDiagnostics


def _polyline_spline_control_counts(options: PolylineSplineFitOptions) -> tuple[int, ...]:
    if options.candidate_control_counts is not None:
        return options.candidate_control_counts
    span = options.max_control_points - options.min_control_points
    if span <= 12:
        return tuple(range(options.min_control_points, options.max_control_points + 1))
    sampled = np.linspace(options.min_control_points, options.max_control_points, num=9)
    return tuple(sorted({int(value) for value in np.rint(sampled)}))


def _candidate_residuals(
    basis: BSplineCurve,
    parameters: np.ndarray,
    observations: np.ndarray,
    weights: np.ndarray | None,
    *,
    control_count: int,
    options: PolylineSplineFitOptions,
) -> tuple[PolylineSplineCandidateScore, np.ndarray]:
    fit_options = SplineFitOptions(
        smoothness=options.smoothness,
        robust_scale=options.robust_scale,
        irls_iterations=options.irls_iterations,
        endpoint_policy=options.endpoint_policy,
        endpoint_sigma=options.endpoint_sigma,
    )
    fit = fit_spline_control_points(
        basis,
        [SplineObservation(u=parameters, xy=observations, weight=weights)],
        options=fit_options,
    )
    controls = np.asarray(fit.curve.control_points)
    predicted = spline_design_matrix(parameters, basis) @ controls
    residuals = np.linalg.norm(predicted - observations, axis=1)
    residual_rms = float(np.sqrt(np.mean(residuals**2)))
    residual_median = float(np.median(residuals))
    residual_p95 = float(np.percentile(residuals, 95))
    residual_max = float(np.max(residuals))
    second_differences = np.diff(controls, n=2, axis=0)
    roughness = float(np.sqrt(np.mean(np.sum(second_differences**2, axis=1))))
    weighted_rms = (
        None if weights is None else float(np.sqrt(np.average(residuals**2, weights=weights)))
    )
    met_target_error = (
        residual_rms <= options.target_rms_error_fraction
        and residual_p95 <= options.target_p95_error_fraction
    )
    score = (
        residual_rms / options.target_rms_error_fraction
        + residual_p95 / options.target_p95_error_fraction
        + options.complexity_weight * control_count / options.max_control_points
        + options.roughness_weight * roughness
    )
    return (
        PolylineSplineCandidateScore(
            control_count=control_count,
            residual_rms=residual_rms,
            residual_median=residual_median,
            residual_p95=residual_p95,
            residual_max=residual_max,
            roughness=roughness,
            score=float(score),
            met_target_error=met_target_error,
            weighted_residual_rms=weighted_rms,
        ),
        controls,
    )


def fit_polyline_spline(
    points: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    options: PolylineSplineFitOptions | None = None,
) -> PolylineSplineFitResult:
    """Reconstruct an ordered noisy polyline as a scale-normalized cubic spline.

    Consecutive duplicate points are removed before normalized cumulative
    arc-length parameterization. Optional weights align with the original
    observations and are filtered in the same way. Residual diagnostics and
    model-selection targets are fractions of the total input arc length; the
    returned curve is transformed back to the input coordinate frame.
    """
    if options is None:
        options = PolylineSplineFitOptions()
    if not isinstance(options, PolylineSplineFitOptions):
        raise ValueError("options must be a PolylineSplineFitOptions")

    original_points = np.asarray(points, dtype=float)
    if (
        original_points.ndim != 2
        or original_points.shape[1] != 2
        or not np.isfinite(original_points).all()
    ):
        raise ValueError("points must be a finite array with shape (n, 2)")
    if len(original_points) < 2:
        raise ValueError("at least two points are required")

    input_weights = None
    if weights is not None:
        input_weights = np.asarray(weights, dtype=float)
        if input_weights.shape != (len(original_points),):
            raise ValueError("weights must have one value per input point")
        if not np.isfinite(input_weights).all() or np.any(input_weights < 0.0):
            raise ValueError("weights must be finite and nonnegative")

    keep = np.concatenate(([True], np.any(np.diff(original_points, axis=0) != 0.0, axis=1)))
    if not keep[-1]:
        keep[-1] = True
        trailing_run_start = np.flatnonzero(keep[:-1])[-1]
        if trailing_run_start:
            keep[trailing_run_start] = False
    points_array = original_points[keep]
    if len(points_array) < 2 or not np.any(np.diff(points_array, axis=0) != 0.0):
        raise ValueError("polyline must contain at least two distinct points")
    if input_weights is not None:
        input_weights = input_weights[keep]
        if not np.any(input_weights > 0.0):
            raise ValueError("weights must have positive total weight")

    parameters = normalized_arclength(points_array)
    total_length = float(np.linalg.norm(np.diff(points_array, axis=0), axis=1).sum())
    if not np.isfinite(total_length) or total_length <= 0.0:
        raise ValueError("polyline must have finite, nonzero total arc length")
    origin = points_array[0]
    normalized_points = (points_array - origin) / total_length

    candidates = []
    for control_count in _polyline_spline_control_counts(options):
        basis = clamped_uniform_bspline_basis(control_count, normalized_points)
        summary, controls = _candidate_residuals(
            basis,
            parameters,
            normalized_points,
            input_weights,
            control_count=control_count,
            options=options,
        )
        candidates.append((summary, basis, controls))

    satisfactory = [candidate for candidate in candidates if candidate[0].met_target_error]
    if satisfactory:
        selected = min(satisfactory, key=lambda candidate: candidate[0].control_count)
    else:
        selected = min(
            candidates,
            key=lambda candidate: (candidate[0].score, candidate[0].control_count),
        )
    summary, basis, normalized_controls = selected
    curve = BSplineCurve(
        knots=basis.knots,
        control_points=normalized_controls * total_length + origin,
        degree=basis.degree,
    )
    diagnostics = PolylineSplineFitDiagnostics(
        control_count=summary.control_count,
        residual_rms=summary.residual_rms,
        residual_median=summary.residual_median,
        residual_p95=summary.residual_p95,
        residual_max=summary.residual_max,
        normalized_scale=total_length,
        met_target_error=summary.met_target_error,
        candidate_scores=tuple(candidate[0] for candidate in candidates),
        weighted_residual_rms=summary.weighted_residual_rms,
    )
    return PolylineSplineFitResult(curve=curve, diagnostics=diagnostics)
