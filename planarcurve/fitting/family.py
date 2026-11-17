"""Shared directed trajectory family fitting and occurrence diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike

try:
    from scipy.optimize import least_squares
except ImportError as exc:  # pragma: no cover - optional fitting extra
    raise ImportError(
        "planarcurve.fitting.family requires SciPy; install planarcurve[fit]"
    ) from exc

from ..trajectory import HeadingRateChange, HeadingRateProfile, Trajectory2D, trajectory_total_turn
from ..transforms import Similarity2D
from ..types import _require_integer
from .parameters import decode_positive, encode_positive
from .trajectory import (
    TrajectoryFitOptions,
    _mapped_state,
    _profile_residuals,
    _target_state,
    fit_shared_trajectory_profile,
)


def _finite_metrics(values: Mapping[str, float], *, name: str) -> Mapping[str, float]:
    copied = {str(key): float(value) for key, value in values.items()}
    if not all(np.isfinite(value) for value in copied.values()):
        raise ValueError(f"{name} values must be finite")
    return MappingProxyType(copied)


@dataclass(frozen=True)
class TrajectoryVariationPolicy:
    """Optional bounded per-occurrence rate deviations around the family mean.

    ``rate_sigma`` is both the hard absolute deviation bound for each rate knot
    and the scale of its quadratic regularization residual. Change positions and
    model order remain shared. Position offsets are reserved and currently
    rejected when enabled.
    """

    allow_rate_offsets: bool = True
    allow_change_position_offsets: bool = False
    rate_sigma: float = 25.0
    position_sigma: float = 0.02

    def __post_init__(self) -> None:
        if not isinstance(self.allow_rate_offsets, bool) or not isinstance(
            self.allow_change_position_offsets, bool
        ):
            raise ValueError("variation policy flags must be booleans")
        if not np.isfinite(self.rate_sigma) or self.rate_sigma <= 0.0:
            raise ValueError("rate_sigma must be positive and finite")
        if not np.isfinite(self.position_sigma) or self.position_sigma <= 0.0:
            raise ValueError("position_sigma must be positive and finite")


@dataclass(frozen=True)
class TrajectoryFamilyModel:
    """A reusable shared mean trajectory and observed family dispersion."""

    trajectory: Trajectory2D
    change_count: int
    parameter_scale: Mapping[str, float]
    training_occurrences: int
    fit_metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        count = _require_integer(self.change_count, name="change_count", minimum=0)
        occurrences = _require_integer(
            self.training_occurrences, name="training_occurrences", minimum=1
        )
        if len(self.trajectory.heading_rate.changes) != count:
            raise ValueError("change_count must match the trajectory profile")
        scales = _finite_metrics(self.parameter_scale, name="parameter_scale")
        if any(value < 0.0 for value in scales.values()):
            raise ValueError("parameter scales must be nonnegative")
        object.__setattr__(self, "change_count", count)
        object.__setattr__(self, "training_occurrences", occurrences)
        object.__setattr__(self, "parameter_scale", scales)
        object.__setattr__(
            self, "fit_metrics", _finite_metrics(self.fit_metrics, name="fit_metrics")
        )

@dataclass(frozen=True)
class TrajectoryOccurrenceFit:
    """Fit diagnostics for one ordered observation in a trajectory family.

    Position and endpoint errors are normalized by the observation's arc length.
    ``robust_cost`` includes the rate-deviation penalty when variation is used.
    """

    transform: Similarity2D
    position_rmse: float
    heading_rmse_deg: float
    endpoint_error: float
    turn_error_deg: float
    robust_cost: float
    accepted: bool
    optimizer_success: bool
    optimizer_evaluations: int
    rate_deviation: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        metrics = (
            self.position_rmse,
            self.heading_rmse_deg,
            self.endpoint_error,
            self.turn_error_deg,
            self.robust_cost,
            *self.rate_deviation,
        )
        if not np.isfinite(metrics).all():
            raise ValueError("occurrence metrics and rate deviations must be finite")
        if any(value < 0.0 for value in metrics[:5]):
            raise ValueError("occurrence error metrics and robust cost must be nonnegative")
        object.__setattr__(self, "optimizer_evaluations", _require_integer(
            self.optimizer_evaluations, name="optimizer_evaluations", minimum=0
        ))
        object.__setattr__(
            self,
            "rate_deviation",
            tuple(float(value) for value in self.rate_deviation),
        )


@dataclass(frozen=True)
class TrajectoryFamilyFitResult:
    """Family fit with independently inspectable training and complexity costs."""

    model: TrajectoryFamilyModel
    occurrences: tuple[TrajectoryOccurrenceFit, ...]
    success: bool
    training_cost: float
    validation_cost: float | None
    complexity_cost: float
    score: float

    def __post_init__(self) -> None:
        occurrences = tuple(self.occurrences)
        if len(occurrences) != self.model.training_occurrences:
            raise ValueError("occurrence results must match model.training_occurrences")
        if not all(
            np.isfinite(value) and value >= 0.0
            for value in (self.training_cost, self.complexity_cost, self.score)
        ):
            raise ValueError("family costs must be finite and nonnegative")
        if self.validation_cost is not None and (
            not np.isfinite(self.validation_cost) or self.validation_cost < 0.0
        ):
            raise ValueError("validation_cost must be nonnegative and finite")
        object.__setattr__(self, "occurrences", occurrences)


def _trajectory_with_rate_offsets(
    trajectory: Trajectory2D, offsets: np.ndarray
) -> Trajectory2D:
    profile = trajectory.heading_rate
    changes = tuple(
        HeadingRateChange(change.at, change.heading_rate_deg_per_u + float(offset))
        for change, offset in zip(profile.changes, offsets[1:-1], strict=True)
    )
    return Trajectory2D(
        length=trajectory.length,
        start_heading_deg=trajectory.start_heading_deg,
        heading_rate=HeadingRateProfile(
            start_rate_deg_per_u=profile.start_rate_deg_per_u + float(offsets[0]),
            changes=changes,
            end_rate_deg_per_u=profile.end_rate_deg_per_u + float(offsets[-1]),
            interpolation=profile.interpolation,
        ),
    )


def _soft_l1_cost(residuals: np.ndarray, scale: float) -> float:
    scaled = residuals / scale
    return float(scale**2 * np.sum(np.sqrt(1.0 + scaled**2) - 1.0))


def _occurrence_metrics(
    trajectory: Trajectory2D,
    transform: Similarity2D,
    points: ArrayLike,
    options: TrajectoryFitOptions,
) -> tuple[float, float, float, float, np.ndarray]:
    target_points, target_heading, target_length = _target_state(points, options)
    mapped, fitted_heading = _mapped_state(trajectory, transform, len(target_points))
    position_errors = np.linalg.norm(mapped - target_points, axis=1)
    heading_errors = (fitted_heading - target_heading + 180.0) % 360.0 - 180.0
    endpoint_error = max(position_errors[0], position_errors[-1]) / target_length
    position_rmse = float(np.sqrt(np.mean(position_errors**2)) / target_length)
    heading_rmse = float(np.sqrt(np.mean(heading_errors**2)))
    observed_turn = float(target_heading[-1] - target_heading[0])
    turn_error = abs(trajectory_total_turn(trajectory) - observed_turn)
    residuals = _profile_residuals(
        trajectory,
        target_points,
        target_heading,
        target_length,
        transform,
        options,
    )
    return position_rmse, heading_rmse, float(endpoint_error), float(turn_error), residuals


def _fit_occurrence_with_variation(
    trajectory: Trajectory2D,
    points: ArrayLike,
    transform: Similarity2D,
    options: TrajectoryFitOptions,
    policy: TrajectoryVariationPolicy,
    rng: np.random.Generator,
) -> tuple[Trajectory2D, Similarity2D, np.ndarray, bool, int, float]:
    rate_count = options.change_count + 2
    initial = np.concatenate(
        (
            np.zeros(rate_count),
            [
                encode_positive(transform.scale),
                transform.rotation_deg,
                transform.dx,
                transform.dy,
            ],
        )
    )
    base_rates = np.asarray(
        [
            trajectory.heading_rate.start_rate_deg_per_u,
            *(change.heading_rate_deg_per_u for change in trajectory.heading_rate.changes),
            trajectory.heading_rate.end_rate_deg_per_u,
        ]
    )
    rate_lower = np.maximum(
        -policy.rate_sigma,
        -options.max_heading_rate_deg_per_u - base_rates,
    )
    rate_upper = np.minimum(
        policy.rate_sigma,
        options.max_heading_rate_deg_per_u - base_rates,
    )
    lower = np.concatenate((rate_lower, np.full(4, -np.inf)))
    upper = np.concatenate((rate_upper, np.full(4, np.inf)))
    target_points, target_heading, target_length = _target_state(points, options)

    def decode(values: np.ndarray) -> tuple[Trajectory2D, Similarity2D]:
        occurrence_trajectory = _trajectory_with_rate_offsets(trajectory, values[:rate_count])
        base = rate_count
        occurrence_transform = Similarity2D(
            scale=decode_positive(float(values[base])),
            rotation_deg=float(values[base + 1]),
            dx=float(values[base + 2]),
            dy=float(values[base + 3]),
        )
        return occurrence_trajectory, occurrence_transform

    def residuals(values: np.ndarray) -> np.ndarray:
        occurrence_trajectory, occurrence_transform = decode(values)
        fit_residuals = _profile_residuals(
            occurrence_trajectory,
            target_points,
            target_heading,
            target_length,
            occurrence_transform,
            options,
        )
        return np.concatenate((fit_residuals, values[:rate_count] / policy.rate_sigma))

    best = None
    for attempt in range(options.restarts + 1):
        initial_values = initial.copy()
        if attempt:
            initial_values[:rate_count] = np.clip(
                rng.normal(0.0, policy.rate_sigma * 0.1, rate_count),
                lower[:rate_count],
                upper[:rate_count],
            )
            initial_values[rate_count] += rng.normal(0.0, 0.05)
            initial_values[rate_count + 1] += rng.normal(0.0, 2.0)
            initial_values[rate_count + 2 :] += rng.normal(0.0, target_length * 0.01, 2)
        result = least_squares(
            residuals,
            initial_values,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=options.robust_scale,
            max_nfev=options.max_nfev,
        )
        if best is None or result.cost < best.cost:
            best = result

    occurrence_trajectory, occurrence_transform = decode(best.x)
    deviations = best.x[:rate_count].copy()
    return (
        occurrence_trajectory,
        occurrence_transform,
        deviations,
        bool(best.success),
        int(best.nfev),
        float(best.cost),
    )


def fit_trajectory_family(
    occurrences: Sequence[ArrayLike],
    *,
    fit_options: TrajectoryFitOptions,
    variation_policy: TrajectoryVariationPolicy | None = None,
) -> TrajectoryFamilyFitResult:
    """Fit one directed shared trajectory to several ordered observations.

    The default shares an exact profile and fits a positive similarity transform
    per occurrence. When a rate variation policy is supplied, bounded rate
    offsets are fitted per occurrence with the profile topology and knot
    positions fixed. Input orientations are never searched or reversed.
    """
    if not occurrences:
        raise ValueError("at least one occurrence is required")
    if not isinstance(fit_options, TrajectoryFitOptions):
        raise TypeError("fit_options must be TrajectoryFitOptions")
    if variation_policy is not None and not isinstance(
        variation_policy, TrajectoryVariationPolicy
    ):
        raise TypeError("variation_policy must be a TrajectoryVariationPolicy or None")
    if variation_policy is not None and variation_policy.allow_change_position_offsets:
        raise NotImplementedError("per-occurrence change-position offsets are not supported")

    shared = fit_shared_trajectory_profile(occurrences, fit_options)
    trajectory = shared.trajectory
    transforms = shared.transforms
    deviations = [np.zeros(fit_options.change_count + 2) for _ in occurrences]
    optimizer_status = [(shared.success, shared.optimizer_evaluations) for _ in occurrences]
    robust_costs = []

    if variation_policy is not None and variation_policy.allow_rate_offsets:
        rng = np.random.default_rng(fit_options.seed)
        varied = [
            _fit_occurrence_with_variation(
                trajectory, points, transform, fit_options, variation_policy, rng
            )
            for points, transform in zip(occurrences, transforms, strict=True)
        ]
        transforms = tuple(item[1] for item in varied)
        deviations = [item[2] for item in varied]
        optimizer_status = [(item[3], item[4]) for item in varied]
        trajectories = [item[0] for item in varied]
        robust_costs = [item[5] for item in varied]
    else:
        trajectories = [trajectory] * len(occurrences)

    occurrence_results = []
    for index, (points, occurrence_trajectory, transform) in enumerate(
        zip(occurrences, trajectories, transforms, strict=True)
    ):
        position_rmse, heading_rmse, endpoint_error, turn_error, residuals = _occurrence_metrics(
            occurrence_trajectory, transform, points, fit_options
        )
        robust_cost = (
            robust_costs[index]
            if robust_costs
            else _soft_l1_cost(residuals, fit_options.robust_scale)
        )
        occurrence_results.append(
            TrajectoryOccurrenceFit(
                transform=transform,
                position_rmse=position_rmse,
                heading_rmse_deg=heading_rmse,
                endpoint_error=endpoint_error,
                turn_error_deg=turn_error,
                robust_cost=robust_cost,
                accepted=True,
                optimizer_success=optimizer_status[index][0],
                optimizer_evaluations=optimizer_status[index][1],
                rate_deviation=tuple(float(value) for value in deviations[index]),
            )
        )

    costs = np.asarray([item.robust_cost for item in occurrence_results])
    median_cost = float(np.median(costs))
    mad = float(np.median(np.abs(costs - median_cost)))
    acceptance_limit = median_cost + max(3.0 * 1.4826 * mad, 0.05 * max(median_cost, 1e-8))
    occurrence_results = [
        TrajectoryOccurrenceFit(
            transform=item.transform,
            position_rmse=item.position_rmse,
            heading_rmse_deg=item.heading_rmse_deg,
            endpoint_error=item.endpoint_error,
            turn_error_deg=item.turn_error_deg,
            robust_cost=item.robust_cost,
            accepted=item.robust_cost <= acceptance_limit,
            optimizer_success=item.optimizer_success,
            optimizer_evaluations=item.optimizer_evaluations,
            rate_deviation=item.rate_deviation,
        )
        for item in occurrence_results
    ]

    rate_deviation_matrix = np.asarray(deviations)
    parameter_scale = {
        f"rate_{index}": float(np.std(rate_deviation_matrix[:, index]))
        for index in range(rate_deviation_matrix.shape[1])
    }
    position_errors = [item.position_rmse for item in occurrence_results]
    heading_errors = [item.heading_rmse_deg for item in occurrence_results]
    model = TrajectoryFamilyModel(
        trajectory=trajectory,
        change_count=fit_options.change_count,
        parameter_scale=parameter_scale,
        training_occurrences=len(occurrences),
        fit_metrics={
            "mean_position_rmse": float(np.mean(position_errors)),
            "median_position_rmse": float(np.median(position_errors)),
            "mean_heading_rmse_deg": float(np.mean(heading_errors)),
            "accepted_occurrences": float(sum(item.accepted for item in occurrence_results)),
            "optimizer_evaluations": float(
                sum(item.optimizer_evaluations for item in occurrence_results)
            ),
        },
    )
    training_cost = float(sum(item.robust_cost for item in occurrence_results))
    complexity_cost = float(2 * fit_options.change_count + 2 + 4 * len(occurrences))
    return TrajectoryFamilyFitResult(
        model=model,
        occurrences=tuple(occurrence_results),
        success=shared.success and all(item.optimizer_success for item in occurrence_results),
        training_cost=training_cost,
        validation_cost=None,
        complexity_cost=complexity_cost,
        score=training_cost + complexity_cost,
    )
