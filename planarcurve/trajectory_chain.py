"""Direction-aware concatenation and pose propagation for planar trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, radians, sin

import numpy as np
from numpy.typing import ArrayLike

from .trajectory import (
    HeadingRateChange,
    HeadingRateProfile,
    Pose2D,
    Trajectory2D,
    evaluate_heading,
    evaluate_trajectory,
    trajectory_to_cubic_path,
)
from .types import CubicPath, CubicSegment, FloatArray, _require_integer


@dataclass(frozen=True)
class DirectedTrajectory:
    """A trajectory segment and the direction in which it is traversed."""

    trajectory: Trajectory2D
    reversed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, Trajectory2D):
            raise TypeError("trajectory must be a Trajectory2D")


@dataclass(frozen=True)
class TrajectoryChain:
    """An ordered, nonempty sequence of directed trajectory segments."""

    segments: tuple[DirectedTrajectory, ...]

    def __post_init__(self) -> None:
        segments = tuple(self.segments)
        if not segments:
            raise ValueError("a trajectory chain must contain at least one segment")
        if not all(isinstance(segment, DirectedTrajectory) for segment in segments):
            raise TypeError("segments must contain DirectedTrajectory values")
        object.__setattr__(self, "segments", segments)


@dataclass(frozen=True)
class TrajectoryChainSample:
    """Samples and parameter metadata for an evaluated trajectory chain."""

    xy: FloatArray
    tangent: FloatArray
    arc_length: FloatArray
    segment_index: np.ndarray
    segment_u: FloatArray
    end_pose: Pose2D


def reverse_trajectory(trajectory: Trajectory2D) -> Trajectory2D:
    """Return the exact heading-rate representation of reverse traversal.

    For normalized progress ``u``, the reversed heading is
    ``heading(1-u) + 180`` and its heading rate is ``-rate(1-u)``. The new
    trajectory starts at its own origin and ends at the original start in a
    local frame, so its samples agree with the original samples traversed in
    reverse after rigid alignment.
    """
    profile = trajectory.heading_rate
    reversed_profile = HeadingRateProfile(
        start_rate_deg_per_u=-profile.end_rate_deg_per_u,
        end_rate_deg_per_u=-profile.start_rate_deg_per_u,
        changes=tuple(
            HeadingRateChange(1.0 - change.at, -change.heading_rate_deg_per_u)
            for change in reversed(profile.changes)
        ),
        interpolation=profile.interpolation,
    )
    end_heading = float(evaluate_heading(trajectory, np.array([1.0]))[0])
    return Trajectory2D(
        length=trajectory.length,
        start_heading_deg=end_heading + 180.0,
        heading_rate=reversed_profile,
    )


def _traversal(segment: DirectedTrajectory) -> Trajectory2D:
    return reverse_trajectory(segment.trajectory) if segment.reversed else segment.trajectory


def _rotation_matrix(angle_deg: float) -> FloatArray:
    angle = radians(angle_deg)
    cosine, sine = cos(angle), sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=float)


def _place_point(point: ArrayLike, origin: FloatArray, rotation: FloatArray) -> tuple[float, float]:
    transformed = origin + np.asarray(point, dtype=float) @ rotation.T
    return float(transformed[0]), float(transformed[1])


def _initial_pose(start_pose: Pose2D | None) -> tuple[FloatArray, float]:
    if start_pose is None:
        return np.zeros(2, dtype=float), 0.0
    position = np.asarray(start_pose.position, dtype=float)
    if position.shape != (2,) or not np.isfinite(position).all():
        raise ValueError("start_pose.position must contain two finite coordinates")
    if not isfinite(start_pose.heading_deg):
        raise ValueError("start_pose.heading_deg must be finite")
    return position.copy(), float(start_pose.heading_deg)


def evaluate_trajectory_chain(
    chain: TrajectoryChain,
    *,
    samples_per_segment: int = 64,
    start_pose: Pose2D | None = None,
) -> TrajectoryChainSample:
    """Sample a chain, propagating position and local heading frame by segment.

    The first segment is aligned to ``start_pose.heading_deg``. Later segment
    start headings remain offsets in the propagated local frame, so a heading
    discontinuity is preserved rather than smoothed. Both sides of every join
    are returned, with equal positions and separately recorded segment
    parameters and tangents.
    """
    samples_per_segment = _require_integer(
        samples_per_segment, name="samples_per_segment", minimum=2
    )
    position, frame_heading = _initial_pose(start_pose)
    xy_parts = []
    tangent_parts = []
    arc_parts = []
    index_parts = []
    u_parts = []
    arc_offset = 0.0
    end_pose = Pose2D(
        position=position.copy(),
        heading_deg=frame_heading,
        heading_rate_deg_per_u=0.0,
    )

    for index, directed in enumerate(chain.segments):
        trajectory = _traversal(directed)
        sample = evaluate_trajectory(trajectory, count=samples_per_segment)
        rotation_deg = frame_heading - trajectory.start_heading_deg if index == 0 else frame_heading
        rotation = _rotation_matrix(rotation_deg)
        segment_xy = sample.xy @ rotation.T + position
        segment_tangent = sample.tangent @ rotation.T
        xy_parts.append(segment_xy)
        tangent_parts.append(segment_tangent)
        arc_parts.append(arc_offset + sample.arc_length)
        index_parts.append(np.full(samples_per_segment, index, dtype=int))
        u_parts.append(np.linspace(0.0, 1.0, samples_per_segment))

        position = segment_xy[-1].copy()
        frame_heading = rotation_deg + sample.end_pose.heading_deg
        end_pose = Pose2D(
            position=position.copy(),
            heading_deg=float(frame_heading),
            heading_rate_deg_per_u=sample.end_pose.heading_rate_deg_per_u,
        )
        arc_offset += trajectory.length

    return TrajectoryChainSample(
        xy=np.concatenate(xy_parts),
        tangent=np.concatenate(tangent_parts),
        arc_length=np.concatenate(arc_parts),
        segment_index=np.concatenate(index_parts),
        segment_u=np.concatenate(u_parts),
        end_pose=end_pose,
    )


def trajectory_chain_to_cubic_path(
    chain: TrajectoryChain,
    *,
    start_pose: Pose2D | None = None,
    max_error: float = 1e-4,
) -> CubicPath:
    """Convert a connected trajectory chain to one connected ``CubicPath``."""
    position, frame_heading = _initial_pose(start_pose)
    path_start = tuple(float(value) for value in position)
    cubic_segments: list[CubicSegment] = []

    for index, directed in enumerate(chain.segments):
        trajectory = _traversal(directed)
        local_path = trajectory_to_cubic_path(trajectory, max_error=max_error)
        rotation_deg = frame_heading - trajectory.start_heading_deg if index == 0 else frame_heading
        rotation = _rotation_matrix(rotation_deg)

        cubic_segments.extend(
            CubicSegment(
                c1=_place_point(segment.c1, position, rotation),
                c2=_place_point(segment.c2, position, rotation),
                end=_place_point(segment.end, position, rotation),
            )
            for segment in local_path.segments
        )
        position = np.asarray(
            _place_point(local_path.segments[-1].end, position, rotation), dtype=float
        )
        end_heading = float(evaluate_heading(trajectory, np.array([1.0]))[0])
        frame_heading = rotation_deg + end_heading

    return CubicPath(path_start, tuple(cubic_segments))
