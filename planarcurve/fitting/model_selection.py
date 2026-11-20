"""Deterministic model-order selection for shared trajectory families."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isfinite, log1p

import numpy as np
from numpy.typing import ArrayLike

try:
    from scipy.optimize import least_squares
except ImportError as exc:  # pragma: no cover - optional fitting extra
    raise ImportError(
        "planarcurve.fitting.model_selection requires SciPy; install planarcurve[fit]"
    ) from exc

from ..trajectory import Trajectory2D, evaluate_trajectory
from ..transforms import Similarity2D
from ..types import _require_integer
from .family import TrajectoryFamilyFitResult, fit_trajectory_family
from .parameters import HeadingRateParameterization, decode_positive, encode_positive
from .trajectory import (
    TrajectoryFitOptions,
    TrajectoryFitResult,
    _profile_residuals,
    _target_state,
    fit_trajectory,
)


@dataclass(frozen=True)
class TrajectoryModelSelectionOptions:
    """Selection settings and explicit numerical complexity-penalty controls.

    Complexity is ``complexity_weight * parameter_count * log1p(1 / parameter_precision)``.
    ``parameter_precision`` is a dimensionless resolution factor in ``(0, 1]``;
    this is a practical regularizer, not a universal MDL score.
    """

    change_counts: tuple[int, ...] = (0, 1, 2, 3, 4)
    validation_fraction: float = 0.2
    complexity_weight: float = 0.01
    parameter_precision: float = 1e-3
    seed: int = 0

    def __post_init__(self) -> None:
        counts = tuple(
            _require_integer(value, name="change_counts", minimum=0) for value in self.change_counts
        )
        if not counts or len(set(counts)) != len(counts):
            raise ValueError(
                "change_counts must be a nonempty tuple of unique nonnegative integers"
            )
        object.__setattr__(self, "change_counts", counts)
        if not isfinite(self.validation_fraction) or not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in [0, 1)")
        if not isfinite(self.complexity_weight) or self.complexity_weight < 0.0:
            raise ValueError("complexity_weight must be finite and nonnegative")
        if not isfinite(self.parameter_precision) or not 0.0 < self.parameter_precision <= 1.0:
            raise ValueError("parameter_precision must lie in (0, 1]")
        object.__setattr__(self, "seed", _require_integer(self.seed, name="seed"))


@dataclass(frozen=True)
class TrajectoryModelCandidate:
    """One fitted model-order candidate and its regularized score."""

    change_count: int
    fit: TrajectoryFitResult
    complexity_cost: float
    score: float


@dataclass(frozen=True)
class TrajectoryModelSelectionResult:
    """Selected single-trajectory fit together with all candidate diagnostics."""

    selected: TrajectoryFitResult
    selected_change_count: int
    candidates: tuple[TrajectoryModelCandidate, ...]


def _split_indices(
    count: int, options: TrajectoryModelSelectionOptions
) -> tuple[np.ndarray, np.ndarray]:
    if count < 3 or options.validation_fraction == 0.0:
        indices = np.arange(count)
        return indices, np.empty(0, dtype=int)
    validation_count = min(
        count - 2,
        max(1, int(round(count * options.validation_fraction))),
    )
    shuffled = np.random.default_rng(options.seed).permutation(count)
    return shuffled[validation_count:], shuffled[:validation_count]


def _validation_cost(
    trajectory: Trajectory2D,
    occurrences: Sequence[ArrayLike],
    options: TrajectoryFitOptions,
) -> float:
    per_occurrence_costs = []
    for points in occurrences:
        target_points, target_heading, target_length = _target_state(points, options)

        trajectory_xy = evaluate_trajectory(trajectory, count=options.samples).xy
        unit_end_length = float(np.linalg.norm(trajectory_xy[-1]))
        initial_transform = Similarity2D(
            scale=target_length / max(unit_end_length, 1e-9),
            rotation_deg=float(target_heading[0]),
            dx=float(target_points[0, 0]),
            dy=float(target_points[0, 1]),
        )
        initial = np.asarray(
            [
                encode_positive(initial_transform.scale),
                initial_transform.rotation_deg,
                initial_transform.dx,
                initial_transform.dy,
            ]
        )

        def residuals(
            values: np.ndarray,
            target_points: np.ndarray = target_points,
            target_heading: np.ndarray = target_heading,
            target_length: float = target_length,
        ) -> np.ndarray:
            transform = Similarity2D(
                scale=decode_positive(float(values[0])),
                rotation_deg=float(values[1]),
                dx=float(values[2]),
                dy=float(values[3]),
            )
            return _profile_residuals(
                trajectory,
                target_points,
                target_heading,
                target_length,
                transform,
                options,
            )

        result = least_squares(
            residuals,
            initial,
            loss="soft_l1",
            f_scale=options.robust_scale,
            max_nfev=options.max_nfev,
        )
        per_occurrence_costs.append(float(result.cost))
    return float(sum(per_occurrence_costs))


def select_shared_trajectory_model(
    occurrences: Sequence[ArrayLike],
    *,
    fit_options: TrajectoryFitOptions,
    selection_options: TrajectoryModelSelectionOptions,
) -> TrajectoryFamilyFitResult:
    """Fit every requested change count and return the lowest-scoring family.

    Validation is split by occurrence using ``selection_options.seed``. For
    fewer than three observations, or when validation is disabled, all inputs
    are used for training and ``validation_cost`` is ``None``.
    """
    if not occurrences:
        raise ValueError("at least one occurrence is required")
    if not isinstance(fit_options, TrajectoryFitOptions):
        raise TypeError("fit_options must be TrajectoryFitOptions")
    if not isinstance(selection_options, TrajectoryModelSelectionOptions):
        raise TypeError("selection_options must be TrajectoryModelSelectionOptions")
    train_indices, validation_indices = _split_indices(len(occurrences), selection_options)
    training = [occurrences[index] for index in train_indices]
    validation = [occurrences[index] for index in validation_indices]
    complexity_factor = selection_options.complexity_weight * log1p(
        1.0 / selection_options.parameter_precision
    )

    candidates = []
    for change_count in selection_options.change_counts:
        candidate_options = replace(
            fit_options,
            change_count=change_count,
            seed=selection_options.seed,
        )
        result = fit_trajectory_family(training, fit_options=candidate_options)
        validation_cost = (
            _validation_cost(result.model.trajectory, validation, candidate_options)
            if validation
            else None
        )
        parameter_count = 2 * change_count + 2 + 4 * len(training)
        complexity_cost = complexity_factor * parameter_count
        score = result.training_cost + complexity_cost
        if validation_cost is not None:
            score += validation_cost
        candidates.append(
            replace(
                result,
                validation_cost=validation_cost,
                complexity_cost=complexity_cost,
                score=score,
            )
        )

    converged = [candidate for candidate in candidates if candidate.success]
    pool = converged or candidates
    return min(pool, key=lambda candidate: (candidate.score, candidate.model.change_count))


def select_trajectory_model(
    points: ArrayLike,
    *,
    fit_options: TrajectoryFitOptions,
    selection_options: TrajectoryModelSelectionOptions = TrajectoryModelSelectionOptions(
        validation_fraction=0.0
    ),
) -> TrajectoryModelSelectionResult:
    """Fit requested profile complexities and select the best single-curve model.

    The score is fit cost plus the shared selector's configurable complexity
    penalty. Occurrence-level validation is not meaningful for one curve, so
    ``selection_options.validation_fraction`` must be zero.
    """
    if not isinstance(fit_options, TrajectoryFitOptions):
        raise TypeError("fit_options must be TrajectoryFitOptions")
    if not isinstance(selection_options, TrajectoryModelSelectionOptions):
        raise TypeError("selection_options must be TrajectoryModelSelectionOptions")
    if selection_options.validation_fraction != 0.0:
        raise ValueError("validation_fraction must be zero for single-trajectory model selection")

    complexity_factor = selection_options.complexity_weight * log1p(
        1.0 / selection_options.parameter_precision
    )
    candidates = []
    for change_count in selection_options.change_counts:
        candidate_options = replace(
            fit_options,
            change_count=change_count,
            seed=selection_options.seed,
        )
        fit = fit_trajectory(points, candidate_options)
        parameterization = HeadingRateParameterization(
            change_count=change_count,
            min_change_spacing=candidate_options.min_change_spacing,
            max_heading_rate_deg_per_u=candidate_options.max_heading_rate_deg_per_u,
        )
        complexity_cost = complexity_factor * (parameterization.size + 4)
        candidates.append(
            TrajectoryModelCandidate(
                change_count=change_count,
                fit=fit,
                complexity_cost=complexity_cost,
                score=fit.cost + complexity_cost,
            )
        )

    converged = [candidate for candidate in candidates if candidate.fit.success]
    pool = converged or candidates
    selected = min(pool, key=lambda candidate: (candidate.score, candidate.change_count))
    return TrajectoryModelSelectionResult(
        selected=selected.fit,
        selected_change_count=selected.change_count,
        candidates=tuple(candidates),
    )
