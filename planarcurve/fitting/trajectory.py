"""Generic fitting of normalized heading-rate trajectories to sampled curves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError as exc:  # pragma: no cover - optional fitting extra
    raise ImportError(
        "planarcurve.fitting.trajectory requires SciPy; install planarcurve[fit]"
    ) from exc

from ..polyline import polyline_length, resample_polyline_by_arclength
from ..trajectory import (
    HeadingRateChange,
    HeadingRateProfile,
    Trajectory2D,
    evaluate_trajectory_at,
    trajectory_total_turn,
)
from ..transforms import Similarity2D
from .parameters import (
    decode_interval,
    decode_positive,
    encode_interval,
    encode_positive,
    ordered_positions_from_logits,
    ordered_positions_to_logits,
)


@dataclass(frozen=True)
class TrajectoryFitOptions:
    change_count: int
    samples: int = 64
    restarts: int = 2
    seed: int = 0
    min_change_spacing: float = 0.02
    max_heading_rate_deg_per_u: float = 36000.0
    start_heading_limit_deg: float = 360.0
    position_weight: float = 1.0
    tangent_weight: float = 0.15
    endpoint_weight: float = 2.0
    endpoint_heading_weight: float = 0.2
    turn_weight: float = 0.3
    profile_weight: float = 0.2
    smoothness_weight: float = 0.02
    robust_scale: float = 1.0
    max_nfev: int = 400

    def __post_init__(self) -> None:
        if (
            isinstance(self.change_count, (bool, np.bool_))
            or not isinstance(self.change_count, (int, np.integer))
            or self.change_count < 0
        ):
            raise ValueError("change_count must be a nonnegative integer")
        if (
            isinstance(self.samples, (bool, np.bool_))
            or not isinstance(self.samples, (int, np.integer))
            or self.samples < 2
        ):
            raise ValueError("samples must be an integer of at least two")
        if (
            isinstance(self.restarts, (bool, np.bool_))
            or not isinstance(self.restarts, (int, np.integer))
            or self.restarts < 0
        ):
            raise ValueError("restarts must be a nonnegative integer")
        if not np.isfinite(self.min_change_spacing) or self.min_change_spacing < 0.0:
            raise ValueError("min_change_spacing must be finite and nonnegative")
        if (self.change_count - 1) * self.min_change_spacing >= 1.0:
            raise ValueError("minimum spacing is too large for the requested changes")
        if (
            not np.isfinite(self.max_heading_rate_deg_per_u)
            or self.max_heading_rate_deg_per_u <= 0.0
        ):
            raise ValueError("maximum heading rate must be positive and finite")
        if (
            not np.isfinite(self.start_heading_limit_deg)
            or self.start_heading_limit_deg <= 0.0
        ):
            raise ValueError("start heading limit must be positive and finite")
        weights = (
            self.position_weight,
            self.tangent_weight,
            self.endpoint_weight,
            self.endpoint_heading_weight,
            self.turn_weight,
            self.profile_weight,
            self.smoothness_weight,
        )
        if not np.isfinite(weights).all() or any(weight < 0.0 for weight in weights):
            raise ValueError("trajectory fit weights must be finite and nonnegative")
        if not np.isfinite(self.robust_scale) or self.robust_scale <= 0.0:
            raise ValueError("robust_scale must be positive and finite")
        if (
            isinstance(self.max_nfev, (bool, np.bool_))
            or not isinstance(self.max_nfev, (int, np.integer))
            or self.max_nfev < 1
        ):
            raise ValueError("max_nfev must be a positive integer")


@dataclass(frozen=True)
class TrajectoryFitResult:
    trajectory: Trajectory2D
    transform: Similarity2D
    success: bool
    cost: float
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class SharedTrajectoryFitResult:
    trajectory: Trajectory2D
    transforms: tuple[Similarity2D, ...]
    success: bool
    cost: float
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class _ProfileCodec:
    options: TrajectoryFitOptions

    @property
    def size(self) -> int:
        return 1 + self.options.change_count + 1 + self.options.change_count + 2

    def decode(self, values: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        count = self.options.change_count
        position_logits = values[1 : count + 2]
        positions = ordered_positions_from_logits(
            position_logits, min_spacing=self.options.min_change_spacing
        )
        rate_logits = values[count + 2 :]
        limit = self.options.max_heading_rate_deg_per_u
        rates = np.asarray(
            [decode_interval(value, low=-limit, high=limit) for value in rate_logits]
        )
        angle_limit = self.options.start_heading_limit_deg
        start_heading = decode_interval(
            float(values[0]), low=-angle_limit, high=angle_limit
        )
        return start_heading, positions, rates

    def encode(
        self, start_heading: float, positions: np.ndarray, rates: np.ndarray
    ) -> np.ndarray:
        count = self.options.change_count
        position_logits = ordered_positions_to_logits(
            positions, min_spacing=self.options.min_change_spacing
        )
        if count == 0:
            position_logits = np.zeros(1)
        limit = self.options.max_heading_rate_deg_per_u
        rate_logits = [
            encode_interval(
                float(np.clip(rate, -limit * (1.0 - 1e-12), limit * (1.0 - 1e-12))),
                low=-limit,
                high=limit,
            )
            for rate in rates
        ]
        angle_limit = self.options.start_heading_limit_deg
        angle = float(
            np.clip(start_heading, -angle_limit * (1.0 - 1e-12), angle_limit * (1.0 - 1e-12))
        )
        return np.asarray(
            [
                encode_interval(angle, low=-angle_limit, high=angle_limit),
                *position_logits,
                *rate_logits,
            ],
            dtype=float,
        )

def resample_by_arclength(points: np.ndarray, *, samples: int = 64) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 2:
        raise ValueError("points must have shape (N, 2) with N >= 2")
    if not np.isfinite(values).all():
        raise ValueError("points must be finite")
    if (
        isinstance(samples, (bool, np.bool_))
        or not isinstance(samples, (int, np.integer))
        or samples < 2
    ):
        raise ValueError("samples must be an integer of at least two")
    return resample_polyline_by_arclength(values, int(samples))


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def estimate_polyline_heading(points: np.ndarray, *, smooth_window: int = 5) -> np.ndarray:
    """Estimate continuous heading angles in degrees along ordered samples."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 2:
        raise ValueError("points must have shape (N, 2) with N >= 2")
    if not np.isfinite(values).all():
        raise ValueError("points must be finite")
    if (
        isinstance(smooth_window, (bool, np.bool_))
        or not isinstance(smooth_window, (int, np.integer))
        or smooth_window < 1
    ):
        raise ValueError("smooth_window must be a positive integer")
    if smooth_window > 1:
        values = np.column_stack(
            (_smooth(values[:, 0], smooth_window), _smooth(values[:, 1], smooth_window))
        )
    gradient = np.gradient(values, axis=0)
    heading = np.unwrap(np.arctan2(gradient[:, 1], gradient[:, 0]))
    return np.rad2deg(_smooth(heading, 7))


def estimate_heading_rate(u: np.ndarray, heading_deg: np.ndarray) -> np.ndarray:
    """Estimate smoothed d(heading in degrees)/du from sampled headings."""
    parameters = np.asarray(u, dtype=float)
    headings = np.asarray(heading_deg, dtype=float)
    if parameters.ndim != 1 or headings.shape != parameters.shape or len(parameters) < 2:
        raise ValueError("u and heading_deg must be matching one-dimensional samples")
    if not np.isfinite(parameters).all() or not np.isfinite(headings).all():
        raise ValueError("u and heading_deg must be finite")
    if np.any(np.diff(parameters) <= 0.0):
        raise ValueError("u must be strictly increasing")
    return _smooth(np.gradient(headings, parameters), 7)


def initialize_change_positions(
    observed_rate: np.ndarray, change_count: int, *, min_spacing: float = 0.02
) -> np.ndarray:
    if (
        isinstance(change_count, (bool, np.bool_))
        or not isinstance(change_count, (int, np.integer))
        or change_count < 0
    ):
        raise ValueError("change_count must be a nonnegative integer")
    if not np.isfinite(min_spacing) or min_spacing < 0.0:
        raise ValueError("min_spacing must be finite and nonnegative")
    if change_count == 0:
        return np.empty(0)
    values = np.asarray(observed_rate, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("observed_rate must be a finite one-dimensional array")
    if (change_count - 1) * min_spacing >= 1.0:
        raise ValueError("interval too small for the requested changes and spacing")
    u = np.linspace(0.0, 1.0, values.size)
    scores = np.abs(np.gradient(values, u))
    cumulative = np.cumsum(scores)
    if cumulative[-1] <= 0.0:
        return ordered_positions_from_logits(
            np.zeros(change_count + 1), min_spacing=min_spacing
        )
    fractions = np.arange(1, change_count + 1) / (change_count + 1)
    positions = np.interp(fractions * cumulative[-1], cumulative, u)
    if np.any(np.diff(positions) < min_spacing):
        return ordered_positions_from_logits(
            np.zeros(change_count + 1), min_spacing=min_spacing
        )
    return positions


def _target_state(
    points: np.ndarray, options: TrajectoryFitOptions
) -> tuple[np.ndarray, np.ndarray, float]:
    target_points = resample_by_arclength(points, samples=options.samples)
    target_heading = estimate_polyline_heading(target_points)
    return target_points, target_heading, polyline_length(target_points)


def _initial_profile(
    target_points: np.ndarray, target_heading: np.ndarray, options: TrajectoryFitOptions
) -> tuple[float, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 1.0, target_points.shape[0])
    observed_rate = estimate_heading_rate(u, target_heading)
    positions = ordered_positions_from_logits(
        np.zeros(options.change_count + 1), min_spacing=options.min_change_spacing
    )
    knots = np.concatenate(([0.0], positions, [1.0]))
    rates = np.interp(knots, u, observed_rate)
    limit = options.max_heading_rate_deg_per_u
    rates = np.clip(rates, -limit * (1.0 - 1e-12), limit * (1.0 - 1e-12))
    return float(target_heading[0]), positions, rates


def _trajectory_from(
    start_heading: float, positions: np.ndarray, rates: np.ndarray
) -> Trajectory2D:
    changes = tuple(
        HeadingRateChange(float(at), float(rate))
        for at, rate in zip(positions, rates[1:-1], strict=True)
    )
    profile = HeadingRateProfile(
        start_rate_deg_per_u=float(rates[0]),
        changes=changes,
        end_rate_deg_per_u=float(rates[-1]),
        interpolation="cubic-hermite",
    )
    return Trajectory2D(length=1.0, start_heading_deg=float(start_heading), heading_rate=profile)


def matched_trajectory_state(
    trajectory: Trajectory2D, samples: int
) -> tuple[np.ndarray, np.ndarray]:
    dense_u = np.linspace(0.0, 1.0, 257)
    dense_xy, _ = evaluate_trajectory_at(trajectory, dense_u)
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(dense_xy, axis=0), axis=1))))
    if arc[-1] <= 0.0:
        parameters = dense_u[:samples]
    else:
        parameters = np.interp(np.linspace(0.0, arc[-1], samples), arc, dense_u)
    return evaluate_trajectory_at(trajectory, parameters)


def _heading_delta(fitted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return (fitted - observed + 180.0) % 360.0 - 180.0


def _mapped_state(
    trajectory: Trajectory2D, transform: Similarity2D, samples: int
) -> tuple[np.ndarray, np.ndarray]:
    xy, heading = matched_trajectory_state(trajectory, samples)
    return transform.apply(xy), heading + transform.rotation_deg


def _metrics(
    trajectory: Trajectory2D,
    transform: Similarity2D,
    target_points: np.ndarray,
    target_heading: np.ndarray,
) -> dict[str, float]:
    mapped, heading = _mapped_state(trajectory, transform, len(target_points))
    position_errors = np.linalg.norm(mapped - target_points, axis=1)
    heading_errors = np.abs(_heading_delta(heading, target_heading))
    return {
        "median_position_error": float(np.median(position_errors)),
        "p95_position_error": float(np.percentile(position_errors, 95)),
        "endpoint_error": float(max(position_errors[0], position_errors[-1])),
        "endpoint_tangent_error_deg": float(max(heading_errors[0], heading_errors[-1])),
        "total_turn_error_deg": abs(
            trajectory_total_turn(trajectory)
            - float(target_heading[-1] - target_heading[0])
        ),
    }


def _profile_residuals(
    trajectory: Trajectory2D,
    target_points: np.ndarray,
    target_heading: np.ndarray,
    transform: Similarity2D,
    options: TrajectoryFitOptions,
) -> np.ndarray:
    mapped, heading = _mapped_state(trajectory, transform, len(target_points))
    heading_errors = _heading_delta(heading, target_heading)
    endpoint_values = np.concatenate(
        (
            (mapped[0] - target_points[0]) * options.endpoint_weight,
            (mapped[-1] - target_points[-1]) * options.endpoint_weight,
            [heading_errors[0] * options.endpoint_heading_weight],
            [heading_errors[-1] * options.endpoint_heading_weight],
        )
    )
    rates = np.concatenate(
        (
            [trajectory.heading_rate.start_rate_deg_per_u],
            [change.heading_rate_deg_per_u for change in trajectory.heading_rate.changes],
            [trajectory.heading_rate.end_rate_deg_per_u],
        )
    )
    smoothness = np.diff(rates, 2) * options.smoothness_weight
    return np.concatenate(
        (
            ((mapped - target_points) * options.position_weight).ravel(),
            heading_errors * options.tangent_weight,
            endpoint_values,
            np.array(
                [
                    (
                        trajectory_total_turn(trajectory)
                        - float(target_heading[-1] - target_heading[0])
                    )
                    * options.turn_weight
                ]
            ),
            heading_errors * options.profile_weight,
            smoothness,
        )
    )


def fit_trajectory(
    points: np.ndarray, options: TrajectoryFitOptions
) -> TrajectoryFitResult:
    """Fit one normalized trajectory and a similarity transform to ordered points."""
    target_points, target_heading, target_length = _target_state(points, options)
    codec = _ProfileCodec(options)
    start_heading, positions, rates = _initial_profile(target_points, target_heading, options)
    profile_vector = codec.encode(start_heading, positions, rates)
    initial_trajectory = _trajectory_from(start_heading, positions, rates)
    initial_xy, _ = matched_trajectory_state(initial_trajectory, options.samples)
    unit_end_length = float(np.linalg.norm(initial_xy[-1]))
    initial_transform = Similarity2D(
        scale=target_length / max(unit_end_length, 1e-9),
        dx=float(target_points[0, 0]),
        dy=float(target_points[0, 1]),
    )
    initial = np.concatenate(
        (
            profile_vector,
            [
                encode_positive(initial_transform.scale),
                initial_transform.dx,
                initial_transform.dy,
            ],
        )
    )

    def decode(values: np.ndarray) -> tuple[Trajectory2D, Similarity2D]:
        angle, event_positions, event_rates = codec.decode(values[: codec.size])
        trajectory = _trajectory_from(angle, event_positions, event_rates)
        transform = Similarity2D(
            scale=decode_positive(float(values[codec.size])),
            dx=float(values[codec.size + 1]),
            dy=float(values[codec.size + 2]),
        )
        return trajectory, transform

    def residuals(values: np.ndarray) -> np.ndarray:
        trajectory, transform = decode(values)
        return _profile_residuals(
            trajectory, target_points, target_heading, transform, options
        )

    rng = np.random.default_rng(options.seed)
    best = None
    for attempt in range(options.restarts + 1):
        initial_values = initial if attempt == 0 else initial + rng.normal(0.0, 0.15, initial.shape)
        result = least_squares(
            residuals,
            initial_values,
            loss="soft_l1",
            f_scale=options.robust_scale,
            max_nfev=options.max_nfev,
        )
        if best is None or result.cost < best.cost:
            best = result

    trajectory, transform = decode(best.x)
    return TrajectoryFitResult(
        trajectory=trajectory,
        transform=transform,
        success=bool(best.success),
        cost=float(best.cost),
        metrics=_metrics(trajectory, transform, target_points, target_heading),
    )


def fit_shared_trajectory_profile(
    occurrences: Sequence[np.ndarray], options: TrajectoryFitOptions
) -> SharedTrajectoryFitResult:
    """Fit one normalized trajectory profile shared by transformed occurrences."""
    if not occurrences:
        raise ValueError("at least one occurrence is required")
    states = [_target_state(points, options) for points in occurrences]
    codec = _ProfileCodec(options)
    initial_profiles = [
        _initial_profile(points, heading, options) for points, heading, _ in states
    ]
    initial_headings = np.asarray([item[0] for item in initial_profiles])
    mean_sin = np.sin(np.deg2rad(initial_headings)).mean()
    mean_cos = np.cos(np.deg2rad(initial_headings)).mean()
    circular_mean = float(np.rad2deg(np.arctan2(mean_sin, mean_cos)))
    positions = np.mean([item[1] for item in initial_profiles], axis=0)
    rates = np.mean([item[2] for item in initial_profiles], axis=0)
    profile_vector = codec.encode(circular_mean, positions, rates)
    initial_trajectory = _trajectory_from(circular_mean, positions, rates)
    initial_xy, _ = matched_trajectory_state(initial_trajectory, options.samples)
    unit_end_length = float(np.linalg.norm(initial_xy[-1]))
    transforms = []
    for points, heading, length in states:
        transforms.append(
            Similarity2D(
                scale=length / max(unit_end_length, 1e-9),
                rotation_deg=float(heading[0] - circular_mean),
                dx=float(points[0, 0]),
                dy=float(points[0, 1]),
            )
        )
    initial_transforms = np.asarray(
        [
            value
            for transform in transforms
            for value in (
                encode_positive(transform.scale),
                transform.rotation_deg,
                transform.dx,
                transform.dy,
            )
        ]
    )
    initial = np.concatenate((profile_vector, initial_transforms))

    def decode(values: np.ndarray) -> tuple[Trajectory2D, tuple[Similarity2D, ...]]:
        angle, event_positions, event_rates = codec.decode(values[: codec.size])
        trajectory = _trajectory_from(angle, event_positions, event_rates)
        decoded_transforms = []
        for index in range(len(states)):
            base = codec.size + index * 4
            decoded_transforms.append(
                Similarity2D(
                    scale=decode_positive(float(values[base])),
                    rotation_deg=float(values[base + 1]),
                    dx=float(values[base + 2]),
                    dy=float(values[base + 3]),
                )
            )
        return trajectory, tuple(decoded_transforms)

    def residuals(values: np.ndarray) -> np.ndarray:
        trajectory, decoded_transforms = decode(values)
        blocks = [
            _profile_residuals(trajectory, target_points, heading, transform, options)
            for (target_points, heading, _), transform in zip(
                states, decoded_transforms, strict=True
            )
        ]
        return np.concatenate(blocks)

    rng = np.random.default_rng(options.seed)
    best = None
    for attempt in range(options.restarts + 1):
        initial_values = initial if attempt == 0 else initial + rng.normal(0.0, 0.1, initial.shape)
        result = least_squares(
            residuals,
            initial_values,
            loss="soft_l1",
            f_scale=options.robust_scale,
            max_nfev=options.max_nfev,
        )
        if best is None or result.cost < best.cost:
            best = result

    trajectory, fitted_transforms = decode(best.x)
    all_position_errors = []
    all_turn_errors = []
    for (target_points, heading, _), transform in zip(
        states, fitted_transforms, strict=True
    ):
        metrics = _metrics(trajectory, transform, target_points, heading)
        all_position_errors.append(metrics["median_position_error"])
        all_turn_errors.append(metrics["total_turn_error_deg"])
    return SharedTrajectoryFitResult(
        trajectory=trajectory,
        transforms=fitted_transforms,
        success=bool(best.success),
        cost=float(best.cost),
        metrics={
            "median_position_error": float(np.median(all_position_errors)),
            "p95_position_error": float(np.percentile(all_position_errors, 95)),
            "max_total_turn_error_deg": float(max(all_turn_errors)),
        },
    )
