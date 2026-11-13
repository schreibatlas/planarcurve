"""Generic fitting of normalized heading-rate trajectories to sampled curves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

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
from ..types import FloatArray, _require_integer
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
    """Optimization settings for fitting normalized heading-rate profiles.

    Model complexity is set by ``change_count``, ``min_change_spacing``, and
    ``max_heading_rate_deg_per_u``. Sampling/initialization uses ``samples`` and
    ``smoothing_window``. ``restarts``, ``seed``, ``robust_scale``, and
    ``max_nfev`` configure optimization. The dimensionless ``*_weight`` fields
    scale residual groups, while ``heading_sigma_deg``, ``turn_sigma_deg``, and
    ``smoothness_sigma_deg_per_u`` specify residual scales. See the fitting
    guide for the complete option table and units.
    """

    change_count: int
    samples: int = 64
    restarts: int = 2
    seed: int = 0
    min_change_spacing: float = 0.02
    max_heading_rate_deg_per_u: float = 36000.0
    position_weight: float = 1.0
    tangent_weight: float = 0.15
    endpoint_weight: float = 2.0
    endpoint_heading_weight: float = 0.2
    turn_weight: float = 0.3
    smoothness_weight: float = 0.02
    heading_sigma_deg: float = 15.0
    turn_sigma_deg: float = 15.0
    smoothness_sigma_deg_per_u: float = 100.0
    smoothing_window: int = 7
    robust_scale: float = 1.0
    max_nfev: int = 400

    def __post_init__(self) -> None:
        for name, minimum in (
            ("change_count", 0),
            ("samples", 2),
            ("restarts", 0),
            ("smoothing_window", 1),
            ("max_nfev", 1),
        ):
            object.__setattr__(
                self,
                name,
                _require_integer(getattr(self, name), name=name, minimum=minimum),
            )
        object.__setattr__(self, "seed", _require_integer(self.seed, name="seed"))
        if not np.isfinite(self.min_change_spacing) or self.min_change_spacing < 0.0:
            raise ValueError("min_change_spacing must be finite and nonnegative")
        if (self.change_count - 1) * self.min_change_spacing >= 1.0:
            raise ValueError("minimum spacing is too large for the requested changes")
        if (
            not np.isfinite(self.max_heading_rate_deg_per_u)
            or self.max_heading_rate_deg_per_u <= 0.0
        ):
            raise ValueError("maximum heading rate must be positive and finite")
        weights = (
            self.position_weight,
            self.tangent_weight,
            self.endpoint_weight,
            self.endpoint_heading_weight,
            self.turn_weight,
            self.smoothness_weight,
        )
        if not np.isfinite(weights).all() or any(weight < 0.0 for weight in weights):
            raise ValueError("trajectory fit weights must be finite and nonnegative")
        sigmas = (
            self.heading_sigma_deg,
            self.turn_sigma_deg,
            self.smoothness_sigma_deg_per_u,
        )
        if not np.isfinite(sigmas).all() or any(sigma <= 0.0 for sigma in sigmas):
            raise ValueError("trajectory residual sigmas must be positive and finite")
        if not np.isfinite(self.robust_scale) or self.robust_scale <= 0.0:
            raise ValueError("robust_scale must be positive and finite")


@dataclass(frozen=True)
class TrajectoryFitResult:
    """Fitted trajectory profile, occurrence transform, and metrics."""

    trajectory: Trajectory2D
    transform: Similarity2D
    success: bool
    cost: float
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class SharedTrajectoryFitResult:
    """Shared fitted profile with one transform per observed occurrence."""

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
        count = self.options.change_count
        return count + count + 2

    def decode(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        count = self.options.change_count
        position_logits = np.concatenate((values[:count], [0.0]))
        positions = ordered_positions_from_logits(
            position_logits, min_spacing=self.options.min_change_spacing
        )
        rate_logits = values[count:]
        limit = self.options.max_heading_rate_deg_per_u
        rates = np.asarray(
            [decode_interval(value, low=-limit, high=limit) for value in rate_logits]
        )
        return positions, rates

    def encode(self, positions: np.ndarray, rates: np.ndarray) -> np.ndarray:
        position_logits = ordered_positions_to_logits(
            positions, min_spacing=self.options.min_change_spacing
        )
        if self.options.change_count:
            position_logits = position_logits[:-1] - position_logits[-1]
        limit = self.options.max_heading_rate_deg_per_u
        rate_logits = [
            encode_interval(
                float(np.clip(rate, -limit * (1.0 - 1e-12), limit * (1.0 - 1e-12))),
                low=-limit,
                high=limit,
            )
            for rate in rates
        ]
        return np.asarray([*position_logits, *rate_logits], dtype=float)


def resample_by_arclength(points: ArrayLike, *, samples: int = 64) -> FloatArray:
    """Resample ordered 2D points uniformly by chord length."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 2:
        raise ValueError("points must have shape (N, 2) with N >= 2")
    if not np.isfinite(values).all():
        raise ValueError("points must be finite")
    samples = _require_integer(samples, name="samples", minimum=2)
    return resample_polyline_by_arclength(values, int(samples))


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    left = (window - 1) // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def estimate_polyline_heading(points: ArrayLike, *, smooth_window: int = 5) -> FloatArray:
    """Estimate continuous heading angles in degrees along ordered samples."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 2:
        raise ValueError("points must have shape (N, 2) with N >= 2")
    if not np.isfinite(values).all():
        raise ValueError("points must be finite")
    smooth_window = _require_integer(smooth_window, name="smooth_window", minimum=1)
    if smooth_window > 1:
        values = np.column_stack(
            (_smooth(values[:, 0], smooth_window), _smooth(values[:, 1], smooth_window))
        )
    gradient = np.gradient(values, axis=0)
    heading = np.unwrap(np.arctan2(gradient[:, 1], gradient[:, 0]))
    return np.rad2deg(_smooth(heading, smooth_window))


def estimate_heading_rate(
    u: ArrayLike, heading_deg: ArrayLike, *, smooth_window: int = 7
) -> FloatArray:
    """Estimate d(heading in degrees)/du with an explicit smoothing window."""
    parameters = np.asarray(u, dtype=float)
    headings = np.asarray(heading_deg, dtype=float)
    if parameters.ndim != 1 or headings.shape != parameters.shape or len(parameters) < 2:
        raise ValueError("u and heading_deg must be matching one-dimensional samples")
    if not np.isfinite(parameters).all() or not np.isfinite(headings).all():
        raise ValueError("u and heading_deg must be finite")
    if np.any(np.diff(parameters) <= 0.0):
        raise ValueError("u must be strictly increasing")
    smooth_window = _require_integer(smooth_window, name="smooth_window", minimum=1)
    return _smooth(np.gradient(headings, parameters), smooth_window)


def initialize_change_positions(
    observed_rate: ArrayLike, change_count: int, *, min_spacing: float = 0.02
) -> FloatArray:
    """Initialize ordered change positions from heading-rate variation."""
    change_count = _require_integer(change_count, name="change_count", minimum=0)
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
    cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (scores[:-1] + scores[1:]) * np.diff(u))))
    if cumulative[-1] <= 0.0:
        return ordered_positions_from_logits(np.zeros(change_count + 1), min_spacing=min_spacing)
    fractions = np.arange(1, change_count + 1) / (change_count + 1)
    positions = np.interp(fractions * cumulative[-1], cumulative, u)
    gaps = np.diff(positions)
    if (
        positions[0] <= 0.0
        or positions[-1] >= 1.0
        or np.any(gaps <= 0.0)
        or np.any(gaps < min_spacing - 1e-12)
    ):
        return ordered_positions_from_logits(np.zeros(change_count + 1), min_spacing=min_spacing)
    return positions


def _target_state(
    points: np.ndarray, options: TrajectoryFitOptions
) -> tuple[np.ndarray, np.ndarray, float]:
    target_points = resample_by_arclength(points, samples=options.samples)
    target_heading = estimate_polyline_heading(
        target_points, smooth_window=options.smoothing_window
    )
    return target_points, target_heading, polyline_length(target_points)


def _initial_profile(
    target_points: np.ndarray, target_heading: np.ndarray, options: TrajectoryFitOptions
) -> tuple[np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 1.0, target_points.shape[0])
    observed_rate = estimate_heading_rate(u, target_heading, smooth_window=options.smoothing_window)
    positions = initialize_change_positions(
        observed_rate, options.change_count, min_spacing=options.min_change_spacing
    )
    knots = np.concatenate(([0.0], positions, [1.0]))
    rates = np.interp(knots, u, observed_rate)
    limit = options.max_heading_rate_deg_per_u
    rates = np.clip(rates, -limit * (1.0 - 1e-12), limit * (1.0 - 1e-12))
    return positions, rates


def _trajectory_from(positions: np.ndarray, rates: np.ndarray) -> Trajectory2D:
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
    return Trajectory2D(length=1.0, start_heading_deg=0.0, heading_rate=profile)


def matched_trajectory_state(
    trajectory: Trajectory2D, samples: int
) -> tuple[FloatArray, FloatArray]:
    """Return trajectory positions and headings at a fixed sample count."""
    parameters = np.linspace(0.0, 1.0, samples)
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
            trajectory_total_turn(trajectory) - float(target_heading[-1] - target_heading[0])
        ),
    }


def _profile_residuals(
    trajectory: Trajectory2D,
    target_points: np.ndarray,
    target_heading: np.ndarray,
    target_length: float,
    transform: Similarity2D,
    options: TrajectoryFitOptions,
) -> np.ndarray:
    mapped, heading = _mapped_state(trajectory, transform, len(target_points))
    heading_errors = _heading_delta(heading, target_heading)
    position_scale = target_length
    heading_scale = options.heading_sigma_deg
    endpoint_values = np.concatenate(
        (
            (mapped[0] - target_points[0]) / position_scale * options.endpoint_weight,
            (mapped[-1] - target_points[-1]) / position_scale * options.endpoint_weight,
            [heading_errors[0] / heading_scale * options.endpoint_heading_weight],
            [heading_errors[-1] / heading_scale * options.endpoint_heading_weight],
        )
    )
    rates = np.concatenate(
        (
            [trajectory.heading_rate.start_rate_deg_per_u],
            [change.heading_rate_deg_per_u for change in trajectory.heading_rate.changes],
            [trajectory.heading_rate.end_rate_deg_per_u],
        )
    )
    smoothness = np.diff(rates, 2) / options.smoothness_sigma_deg_per_u * options.smoothness_weight
    turn_error = trajectory_total_turn(trajectory) - float(target_heading[-1] - target_heading[0])
    return np.concatenate(
        (
            ((mapped - target_points) / position_scale * options.position_weight).ravel(),
            heading_errors / heading_scale * options.tangent_weight,
            endpoint_values,
            np.array([turn_error / options.turn_sigma_deg * options.turn_weight]),
            smoothness,
        )
    )


def fit_trajectory(points: ArrayLike, options: TrajectoryFitOptions) -> TrajectoryFitResult:
    """Fit a normalized trajectory profile and a similarity placement to points.

    The fitted profile has length 1 and starting heading 0 degrees. Its
    occurrence is placed by a separate positive-scale ``Similarity2D``. Position
    residuals are normalized by observed polyline length; heading, turn, and
    smoothness residuals use their configured sigma scales. Optimization uses a
    robust least-squares loss.

    Parameters
    ----------
    points : array_like, shape (N, 2)
        Ordered finite samples of one observed curve.
    options : TrajectoryFitOptions
        Model, sampling, and optimization settings.

    Returns
    -------
    TrajectoryFitResult
        Fitted canonical profile, placement transform, optimizer status/cost,
        and position/heading metrics. Inspect ``success`` and ``cost`` rather
        than assuming that a result object implies convergence.
    """
    target_points, target_heading, target_length = _target_state(points, options)
    codec = _ProfileCodec(options)
    positions, rates = _initial_profile(target_points, target_heading, options)
    profile_vector = codec.encode(positions, rates)
    initial_trajectory = _trajectory_from(positions, rates)
    initial_xy, _ = matched_trajectory_state(initial_trajectory, options.samples)
    unit_end_length = float(np.linalg.norm(initial_xy[-1]))
    initial_transform = Similarity2D(
        scale=target_length / max(unit_end_length, 1e-9),
        rotation_deg=float(target_heading[0]),
        dx=float(target_points[0, 0]),
        dy=float(target_points[0, 1]),
    )
    initial = np.concatenate(
        (
            profile_vector,
            [
                encode_positive(initial_transform.scale),
                initial_transform.rotation_deg,
                initial_transform.dx,
                initial_transform.dy,
            ],
        )
    )

    def decode(values: np.ndarray) -> tuple[Trajectory2D, Similarity2D]:
        event_positions, event_rates = codec.decode(values[: codec.size])
        trajectory = _trajectory_from(event_positions, event_rates)
        base = codec.size
        transform = Similarity2D(
            scale=decode_positive(float(values[base])),
            rotation_deg=float(values[base + 1]),
            dx=float(values[base + 2]),
            dy=float(values[base + 3]),
        )
        return trajectory, transform

    def residuals(values: np.ndarray) -> np.ndarray:
        trajectory, transform = decode(values)
        return _profile_residuals(
            trajectory, target_points, target_heading, target_length, transform, options
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
    occurrences: Sequence[ArrayLike], options: TrajectoryFitOptions
) -> SharedTrajectoryFitResult:
    """Fit one normalized profile shared across transformed occurrences.

    The shared profile has length 1 and starting heading 0 degrees. Each input
    occurrence receives its own positive-scale ``Similarity2D`` placement.
    Position residuals are normalized by each occurrence's observed length;
    heading, turn, and smoothness residuals use the configured sigma scales.
    The optimizer uses a robust least-squares loss.

    Parameters
    ----------
    occurrences : sequence of array_like, each shape (N, 2)
        Ordered finite samples for one or more observed curves.
    options : TrajectoryFitOptions
        Shared profile, sampling, and optimization settings.

    Returns
    -------
    SharedTrajectoryFitResult
        The common profile, one transform per occurrence, optimizer
        status/cost, and aggregate metrics. Inspect ``success`` and ``cost``.
    """
    if not occurrences:
        raise ValueError("at least one occurrence is required")
    states = [_target_state(points, options) for points in occurrences]
    codec = _ProfileCodec(options)
    initial_profiles = [_initial_profile(points, heading, options) for points, heading, _ in states]
    positions = np.mean([item[0] for item in initial_profiles], axis=0)
    rates = np.mean([item[1] for item in initial_profiles], axis=0)
    profile_vector = codec.encode(positions, rates)
    initial_trajectory = _trajectory_from(positions, rates)
    initial_xy, _ = matched_trajectory_state(initial_trajectory, options.samples)
    unit_end_length = float(np.linalg.norm(initial_xy[-1]))
    transforms = []
    for points, heading, length in states:
        transforms.append(
            Similarity2D(
                scale=length / max(unit_end_length, 1e-9),
                rotation_deg=float(heading[0]),
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
        event_positions, event_rates = codec.decode(values[: codec.size])
        trajectory = _trajectory_from(event_positions, event_rates)
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
            _profile_residuals(trajectory, target_points, heading, length, transform, options)
            for (target_points, heading, length), transform in zip(
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
    for (target_points, heading, _), transform in zip(states, fitted_transforms, strict=True):
        mapped, _ = _mapped_state(trajectory, transform, len(target_points))
        all_position_errors.extend(np.linalg.norm(mapped - target_points, axis=1))
        all_turn_errors.append(
            abs(trajectory_total_turn(trajectory) - float(heading[-1] - heading[0]))
        )
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
