from __future__ import annotations

import numpy as np
import pytest

from planarcurve.curvature import (
    CurvatureChange,
    CurvatureProfile,
    CurvatureTrajectory,
    curvature_to_cubic_segments,
    evaluate_curvature_stroke,
    evaluate_heading_rate,
    sample_curvature_stroke,
    trajectory_total_turn,
)


def profile() -> CurvatureProfile:
    return CurvatureProfile(
        interpolation="cubic-hermite",
        start_rate_deg=40.0,
        changes=(
            CurvatureChange(0.25, 260.0),
            CurvatureChange(0.55, 200.0),
            CurvatureChange(0.8, -80.0),
        ),
        end_rate_deg=10.0,
    )


def trajectory(length: float = 1.0, start_angle_deg: float = 25.0) -> CurvatureTrajectory:
    return CurvatureTrajectory(length=length, start_angle_deg=start_angle_deg, profile=profile())


def test_scale_changes_size_but_not_normalized_shape():
    reference = evaluate_curvature_stroke(trajectory(1.0), count=257)
    for length in (0.5, 1.0, 2.0, 4.0):
        scaled = evaluate_curvature_stroke(trajectory(length), count=257)
        np.testing.assert_allclose(scaled.xy / length, reference.xy, atol=1e-9)
        np.testing.assert_allclose(scaled.heading_deg, reference.heading_deg, atol=1e-9)


def test_rotation_rotates_full_trajectory():
    base = evaluate_curvature_stroke(trajectory(start_angle_deg=0.0), count=129)
    rotated = evaluate_curvature_stroke(trajectory(start_angle_deg=30.0), count=129)
    angle = np.deg2rad(30.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    np.testing.assert_allclose(rotated.xy, base.xy @ rotation.T, atol=1e-9)
    np.testing.assert_allclose(rotated.heading_deg, base.heading_deg + 30.0, atol=1e-9)


def test_heading_rate_hits_control_points():
    p = profile()
    u = np.array([0.0, 0.25, 0.55, 0.8, 1.0])
    np.testing.assert_allclose(evaluate_heading_rate(p, u), [40, 260, 200, -80, 10])


def test_turn_end_pose_and_arclength_are_consistent():
    t = trajectory()
    sample = t.evaluate(count=257)
    assert sample.end_pose.heading_deg - t.start_angle_deg == pytest.approx(
        trajectory_total_turn(t, count=257)
    )
    np.testing.assert_allclose(sample.end_pose.position, sample.xy[-1])
    assert np.all(np.diff(sample.arc_length) >= 0)


def test_arclength_resampling_and_cubic_conversion_are_deterministic():
    t = trajectory()
    first = sample_curvature_stroke(t, count=257, samples=41)
    second = sample_curvature_stroke(t, count=257, samples=41)
    np.testing.assert_array_equal(first, second)
    assert len(curvature_to_cubic_segments(t, max_u_step=0.05)) > 0


def test_validation_rejects_invalid_change_spacing():
    with pytest.raises(ValueError, match="spacing"):
        CurvatureProfile(
            start_rate_deg=0,
            changes=(CurvatureChange(0.5, 1), CurvatureChange(0.51, 2)),
            end_rate_deg=0,
        )
