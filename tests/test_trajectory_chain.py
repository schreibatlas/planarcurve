from __future__ import annotations

import numpy as np
import pytest

from planarcurve.trajectory import (
    HeadingRateChange,
    HeadingRateProfile,
    Pose2D,
    Trajectory2D,
    evaluate_trajectory,
    trajectory_to_cubic_path,
)
from planarcurve.trajectory_chain import (
    DirectedTrajectory,
    TrajectoryChain,
    evaluate_trajectory_chain,
    reverse_trajectory,
    trajectory_chain_to_cubic_path,
)


def _profile(*, turn: float, interpolation: str = "cubic-hermite") -> HeadingRateProfile:
    return HeadingRateProfile(
        start_rate_deg_per_u=turn,
        end_rate_deg_per_u=turn,
        changes=(HeadingRateChange(0.35, turn), HeadingRateChange(0.72, turn)),
        interpolation=interpolation,
    )


@pytest.mark.parametrize("interpolation", ["linear", "cubic-hermite"])
def test_reverse_trajectory_matches_reversed_samples_after_rigid_alignment(interpolation):
    trajectory = Trajectory2D(
        length=4.0,
        start_heading_deg=-23.0,
        heading_rate=HeadingRateProfile(
            start_rate_deg_per_u=20.0,
            end_rate_deg_per_u=-10.0,
            changes=(HeadingRateChange(0.27, 65.0), HeadingRateChange(0.81, -35.0)),
            interpolation=interpolation,
        ),
    )
    original = evaluate_trajectory(trajectory, count=101)
    reverse = evaluate_trajectory(reverse_trajectory(trajectory), count=101)

    np.testing.assert_allclose(reverse.xy, original.xy[::-1] - original.xy[-1], atol=2e-8)
    np.testing.assert_allclose(reverse.heading_deg, original.heading_deg[::-1] + 180.0)
    np.testing.assert_allclose(
        reverse.heading_rate_deg_per_u, -original.heading_rate_deg_per_u[::-1]
    )


def test_evaluate_chain_propagates_pose_and_preserves_segment_boundaries_and_turns():
    first = Trajectory2D(2.0, 0.0, _profile(turn=90.0, interpolation="linear"))
    second = Trajectory2D(3.0, 35.0, _profile(turn=-20.0, interpolation="linear"))
    start_pose = Pose2D(np.array([10.0, -4.0]), 20.0, 0.0)
    chain = TrajectoryChain((DirectedTrajectory(first), DirectedTrajectory(second)))

    sample = evaluate_trajectory_chain(chain, samples_per_segment=3, start_pose=start_pose)

    np.testing.assert_allclose(sample.xy[0], start_pose.position)
    np.testing.assert_allclose(sample.xy[2], sample.xy[3])
    np.testing.assert_allclose(sample.arc_length, [0.0, 1.0, 2.0, 2.0, 3.5, 5.0])
    np.testing.assert_array_equal(sample.segment_index, [0, 0, 0, 1, 1, 1])
    np.testing.assert_allclose(sample.segment_u, [0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(
        sample.tangent[2],
        [np.cos(np.deg2rad(110.0)), np.sin(np.deg2rad(110.0))],
    )
    np.testing.assert_allclose(
        sample.tangent[3],
        [np.cos(np.deg2rad(145.0)), np.sin(np.deg2rad(145.0))],
    )
    assert sample.end_pose.heading_deg == pytest.approx(125.0)


def test_reversed_directed_segment_is_evaluated_in_reverse():
    trajectory = Trajectory2D(2.0, 0.0, _profile(turn=60.0, interpolation="linear"))
    reversed_trajectory = reverse_trajectory(trajectory)
    chain = TrajectoryChain((DirectedTrajectory(trajectory, reversed=True),))
    pose = Pose2D(np.zeros(2), 0.0, 0.0)

    actual = evaluate_trajectory_chain(chain, samples_per_segment=9, start_pose=pose)
    local = evaluate_trajectory(reversed_trajectory, count=9)
    angle = np.deg2rad(-reversed_trajectory.start_heading_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    np.testing.assert_allclose(actual.xy, local.xy @ rotation.T, atol=2e-8)


def test_trajectory_chain_converts_to_connected_cubic_path():
    first = Trajectory2D(2.0, 0.0, _profile(turn=45.0, interpolation="linear"))
    second = Trajectory2D(1.5, 15.0, _profile(turn=-10.0, interpolation="linear"))
    chain = TrajectoryChain((DirectedTrajectory(first), DirectedTrajectory(second)))
    start_pose = Pose2D(np.array([3.0, 7.0]), 30.0, 0.0)

    sample = evaluate_trajectory_chain(chain, samples_per_segment=5, start_pose=start_pose)
    path = trajectory_chain_to_cubic_path(chain, start_pose=start_pose, max_error=1e-4)
    first_path = trajectory_to_cubic_path(first, max_error=1e-4)

    assert path.start == (3.0, 7.0)
    np.testing.assert_allclose(path.segments[len(first_path.segments) - 1].end, sample.xy[4])
    np.testing.assert_allclose(path.segments[-1].end, sample.xy[-1], atol=2e-8)


def test_trajectory_chain_apis_are_exported_at_package_root():
    import planarcurve

    assert planarcurve.DirectedTrajectory is DirectedTrajectory
    assert planarcurve.TrajectoryChain is TrajectoryChain
    assert planarcurve.evaluate_trajectory_chain is evaluate_trajectory_chain
    assert planarcurve.reverse_trajectory is reverse_trajectory
    assert planarcurve.trajectory_chain_to_cubic_path is trajectory_chain_to_cubic_path
