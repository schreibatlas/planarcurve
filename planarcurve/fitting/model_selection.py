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

from ..trajectory import Trajectory2D, evaluate_trajectory, insert_heading_rate_change
from ..transforms import Similarity2D
from ..types import _require_integer
from .family import TrajectoryFamilyFitResult, fit_trajectory_family
from .parameters import HeadingRateParameterization, decode_positive, encode_positive
from .trajectory import (
    TrajectoryFitOptions,
    TrajectoryFitResult,
    TrajectoryFitSeed,
    _heading_delta,
    _mapped_state,
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
class TrajectoryRefinementOptions:
    """Limits and score thresholds for progressive trajectory refinement."""

    max_change_count: int = 4
    candidate_insertions: int = 3
    residual_samples: int = 96
    minimum_absolute_gain: float = 1e-4
    minimum_relative_gain: float = 0.01
    complexity_weight: float = 0.01
    parameter_precision: float = 1e-3
    seed: int = 0

    def __post_init__(self) -> None:
        for name, minimum in (
            ("max_change_count", 0),
            ("candidate_insertions", 1),
            ("residual_samples", 8),
        ):
            object.__setattr__(
                self,
                name,
                _require_integer(getattr(self, name), name=name, minimum=minimum),
            )
        object.__setattr__(self, "seed", _require_integer(self.seed, name="seed"))
        gains_and_weights = (
            self.minimum_absolute_gain,
            self.minimum_relative_gain,
            self.complexity_weight,
        )
        if not all(isfinite(value) and value >= 0.0 for value in gains_and_weights):
            raise ValueError("gains and complexity_weight must be finite and nonnegative")
        if not isfinite(self.parameter_precision) or not 0.0 < self.parameter_precision <= 1.0:
            raise ValueError("parameter_precision must lie in (0, 1]")


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


@dataclass(frozen=True)
class TrajectoryRefinementStep:
    """Numerical diagnostics for one baseline or proposed refinement fit."""

    parent_change_count: int
    candidate_change_count: int
    insertion_progress: float | None
    fit: TrajectoryFitResult
    fit_cost: float
    complexity_cost: float
    score: float
    absolute_gain: float | None
    relative_gain: float | None
    accepted: bool
    reason: str


@dataclass(frozen=True)
class TrajectoryRefinementResult:
    """Selected progressive fit, accepted model order, and all attempted steps."""

    selected: TrajectoryFitResult
    selected_change_count: int
    steps: tuple[TrajectoryRefinementStep, ...]
    stop_reason: str

    @property
    def optimizer_evaluations(self) -> int:
        """Total SciPy function evaluations across the baseline and child fits."""
        return sum(step.fit.optimizer_evaluations for step in self.steps)


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


def _refinement_complexity_cost(
    change_count: int,
    fit_options: TrajectoryFitOptions,
    refinement_options: TrajectoryRefinementOptions,
) -> float:
    parameterization = HeadingRateParameterization(
        change_count=change_count,
        min_change_spacing=fit_options.min_change_spacing,
        max_heading_rate_deg_per_u=fit_options.max_heading_rate_deg_per_u,
    )
    factor = refinement_options.complexity_weight * log1p(
        1.0 / refinement_options.parameter_precision
    )
    return factor * (parameterization.size + 4)


def _residual_insertion_candidates(
    signal: np.ndarray,
    grid: np.ndarray,
    existing_changes: Sequence[float],
    *,
    min_spacing: float,
    limit: int,
) -> tuple[float, ...]:
    legal = (
        (grid > 0.0)
        & (grid < 1.0)
        & (grid >= min_spacing - 1e-12)
        & (grid <= 1.0 - min_spacing + 1e-12)
    )
    for existing in existing_changes:
        distances = np.abs(grid - existing)
        legal &= (distances >= min_spacing - 1e-12) & (distances > 1e-12)
    legal_indices = np.flatnonzero(legal)
    if legal_indices.size == 0:
        return ()

    peaks = []
    for index in legal_indices:
        left_is_legal = index > 0 and legal[index - 1]
        right_is_legal = index + 1 < len(grid) and legal[index + 1]
        left_is_peak = not left_is_legal or signal[index] >= signal[index - 1]
        right_is_peak = not right_is_legal or signal[index] >= signal[index + 1]
        if left_is_peak and right_is_peak:
            peaks.append(int(index))

    peaks.sort(key=lambda index: (-float(signal[index]), float(grid[index])))
    suppression_distance = max(min_spacing, float(grid[1] - grid[0]))
    selected = []
    for index in peaks:
        if all(
            abs(float(grid[index] - grid[previous])) >= suppression_distance - 1e-12
            for previous in selected
        ):
            selected.append(index)
            if len(selected) == limit:
                break
    return tuple(float(grid[index]) for index in selected)


def _better_refinement_candidate(
    candidate: tuple[int, int, TrajectoryRefinementStep],
    current: tuple[int, int, TrajectoryRefinementStep],
) -> bool:
    candidate_order, _, candidate_step = candidate
    current_order, _, current_step = current
    if not np.isclose(candidate_step.score, current_step.score, rtol=1e-12, atol=1e-12):
        return candidate_step.score < current_step.score
    candidate_key = (
        candidate_step.candidate_change_count,
        candidate_step.fit_cost,
        float(candidate_step.insertion_progress),
        candidate_order,
    )
    current_key = (
        current_step.candidate_change_count,
        current_step.fit_cost,
        float(current_step.insertion_progress),
        current_order,
    )
    return candidate_key < current_key


def refine_trajectory_model(
    points: ArrayLike,
    *,
    fit_options: TrajectoryFitOptions,
    refinement_options: TrajectoryRefinementOptions = TrajectoryRefinementOptions(),
) -> TrajectoryRefinementResult:
    """Progressively add heading-rate changes when regularized gains justify them.

    ``fit_options.change_count`` must be zero. Complexity uses the same parameter
    count and precision penalty as :func:`select_trajectory_model`.
    """
    if not isinstance(fit_options, TrajectoryFitOptions):
        raise TypeError("fit_options must be TrajectoryFitOptions")
    if not isinstance(refinement_options, TrajectoryRefinementOptions):
        raise TypeError("refinement_options must be TrajectoryRefinementOptions")
    if fit_options.change_count != 0:
        raise ValueError("fit_options.change_count must be zero for progressive refinement")

    base_options = replace(fit_options, seed=refinement_options.seed)
    selected = fit_trajectory(points, base_options)
    selected_score = selected.cost + _refinement_complexity_cost(
        0, base_options, refinement_options
    )
    base_valid = isfinite(selected.cost) and isfinite(selected_score)
    base_accepted = selected.success and base_valid
    base_reason = (
        "base_model"
        if base_accepted
        else "base_fit_failed"
        if not selected.success
        else "numerical_failure"
    )
    steps = [
        TrajectoryRefinementStep(
            parent_change_count=0,
            candidate_change_count=0,
            insertion_progress=None,
            fit=selected,
            fit_cost=float(selected.cost),
            complexity_cost=_refinement_complexity_cost(0, base_options, refinement_options),
            score=float(selected_score),
            absolute_gain=None,
            relative_gain=None,
            accepted=base_accepted,
            reason=base_reason,
        )
    ]
    if not selected.success:
        return TrajectoryRefinementResult(selected, 0, tuple(steps), "base_fit_failed")
    if not base_valid:
        return TrajectoryRefinementResult(selected, 0, tuple(steps), "numerical_failure")
    if refinement_options.max_change_count == 0:
        return TrajectoryRefinementResult(selected, 0, tuple(steps), "max_change_count_reached")

    residual_options = replace(base_options, samples=refinement_options.residual_samples)
    grid = np.linspace(0.0, 1.0, refinement_options.residual_samples)
    while True:
        target_points, target_heading, target_length = _target_state(points, residual_options)
        mapped_points, fitted_heading = _mapped_state(
            selected.trajectory, selected.transform, refinement_options.residual_samples
        )
        position_error = np.linalg.norm(mapped_points - target_points, axis=1) / max(
            target_length, np.finfo(float).eps
        )
        heading_error = np.abs(_heading_delta(fitted_heading, target_heading)) / 180.0
        signal = position_error + heading_error
        existing_changes = tuple(change.at for change in selected.trajectory.heading_rate.changes)
        if not np.isfinite(signal).all():
            return TrajectoryRefinementResult(
                selected,
                len(existing_changes),
                tuple(steps),
                "numerical_failure",
            )
        locations = _residual_insertion_candidates(
            signal,
            grid,
            existing_changes,
            min_spacing=base_options.min_change_spacing,
            limit=refinement_options.candidate_insertions,
        )
        if not locations:
            return TrajectoryRefinementResult(
                selected, len(existing_changes), tuple(steps), "no_legal_insertion_location"
            )

        candidate_records: list[tuple[int, int, TrajectoryRefinementStep]] = []
        candidate_change_count = len(existing_changes) + 1
        candidate_options = replace(
            base_options,
            change_count=candidate_change_count,
            seed=refinement_options.seed,
        )
        for candidate_order, location in enumerate(locations):
            promoted_profile = insert_heading_rate_change(
                selected.trajectory.heading_rate, location
            )
            promoted_trajectory = Trajectory2D(
                length=1.0,
                start_heading_deg=0.0,
                heading_rate=promoted_profile,
            )
            fit = fit_trajectory(
                points,
                candidate_options,
                initial=TrajectoryFitSeed(promoted_trajectory, selected.transform),
            )
            fit_cost = float(fit.cost)
            complexity_cost = _refinement_complexity_cost(
                candidate_change_count, candidate_options, refinement_options
            )
            score = fit_cost + complexity_cost
            valid_score = isfinite(fit_cost) and isfinite(score)
            absolute_gain = selected_score - score if valid_score else None
            relative_gain = (
                absolute_gain / max(abs(selected_score), np.finfo(float).eps)
                if absolute_gain is not None
                else None
            )
            reason = (
                "fit_failed"
                if not fit.success
                else "numerical_failure"
                if not valid_score
                else "not_best_candidate"
            )
            step = TrajectoryRefinementStep(
                parent_change_count=len(existing_changes),
                candidate_change_count=candidate_change_count,
                insertion_progress=location,
                fit=fit,
                fit_cost=fit_cost,
                complexity_cost=complexity_cost,
                score=float(score),
                absolute_gain=absolute_gain,
                relative_gain=relative_gain,
                accepted=False,
                reason=reason,
            )
            steps.append(step)
            if fit.success and valid_score:
                candidate_records.append((candidate_order, len(steps) - 1, step))

        if not candidate_records:
            stop_reason = (
                "numerical_failure"
                if any(step.fit.success for step in steps[-len(locations) :])
                else "all_child_fits_failed"
            )
            return TrajectoryRefinementResult(
                selected, len(existing_changes), tuple(steps), stop_reason
            )

        best = candidate_records[0]
        for candidate in candidate_records[1:]:
            if _better_refinement_candidate(candidate, best):
                best = candidate
        best_order, best_step_index, best_step = best
        del best_order
        gain_accepted = (
            best_step.absolute_gain is not None
            and best_step.relative_gain is not None
            and best_step.absolute_gain >= refinement_options.minimum_absolute_gain
            and best_step.relative_gain >= refinement_options.minimum_relative_gain
        )
        for index, step in enumerate(steps[-len(locations) :], start=len(steps) - len(locations)):
            if index == best_step_index:
                steps[index] = replace(
                    step,
                    accepted=gain_accepted,
                    reason=("accepted" if gain_accepted else "insufficient_gain"),
                )

        if not gain_accepted:
            return TrajectoryRefinementResult(
                selected, len(existing_changes), tuple(steps), "insufficient_gain"
            )

        selected = best_step.fit
        selected_score = best_step.score
        if candidate_change_count >= refinement_options.max_change_count:
            return TrajectoryRefinementResult(
                selected, candidate_change_count, tuple(steps), "max_change_count_reached"
            )
