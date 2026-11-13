from __future__ import annotations

import numpy as np
import pytest

from planarcurve.trajectory import (
    HeadingRateChange,
    HeadingRateProfile,
    Trajectory2D,
    curvature,
    evaluate_heading,
    evaluate_heading_rate,
    evaluate_trajectory,
    integrated_heading_rate,
    sample_trajectory_by_arclength,
    trajectory_to_cubic_path,
    trajectory_total_turn,
)


def profile() -> HeadingRateProfile:
    return HeadingRateProfile(
        interpolation="cubic-hermite",
        start_rate_deg_per_u=40.0,
        changes=(
            HeadingRateChange(0.25, 260.0),
            HeadingRateChange(0.55, 200.0),
            HeadingRateChange(0.8, -80.0),
        ),
        end_rate_deg_per_u=10.0,
    )


def trajectory(length: float = 1.0, start_heading_deg: float = 25.0) -> Trajectory2D:
    return Trajectory2D(length=length, start_heading_deg=start_heading_deg, heading_rate=profile())


def test_scale_changes_size_but_not_normalized_shape():
    reference = evaluate_trajectory(trajectory(1.0), count=257)
    for length in (0.5, 1.0, 2.0, 4.0):
        scaled = evaluate_trajectory(trajectory(length), count=257)
        np.testing.assert_allclose(scaled.xy / length, reference.xy, atol=1e-9)
        np.testing.assert_allclose(scaled.heading_deg, reference.heading_deg, atol=1e-9)


def test_rotation_rotates_full_trajectory():
    base = evaluate_trajectory(trajectory(start_heading_deg=0.0), count=129)
    rotated = evaluate_trajectory(trajectory(start_heading_deg=30.0), count=129)
    angle = np.deg2rad(30.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    np.testing.assert_allclose(rotated.xy, base.xy @ rotation.T, atol=1e-9)
    np.testing.assert_allclose(rotated.heading_deg, base.heading_deg + 30.0, atol=1e-9)


def test_heading_rate_hits_control_points():
    p = profile()
    u = np.array([0.0, 0.25, 0.55, 0.8, 1.0])
    np.testing.assert_allclose(evaluate_heading_rate(p, u), [40, 260, 200, -80, 10])


def test_linear_heading_integral_includes_sparse_interior_knots():
    p = HeadingRateProfile(
        start_rate_deg_per_u=0.0,
        changes=(HeadingRateChange(0.5, 180.0),),
        end_rate_deg_per_u=0.0,
        interpolation="linear",
    )
    t = Trajectory2D(length=1.0, start_heading_deg=0.0, heading_rate=p)
    np.testing.assert_allclose(evaluate_heading(t, np.array([0.0, 1.0])), [0.0, 90.0])
    assert integrated_heading_rate(p, np.array([1.0]))[0] == pytest.approx(90.0)


def test_heading_evaluation_preserves_arbitrary_query_order():
    t = Trajectory2D(length=1.0, start_heading_deg=15.0, heading_rate=profile())
    query = np.array([0.7, 0.1, 1.0, 0.0, 0.55, 0.25])
    actual = evaluate_heading(t, query)
    sorted_query = np.sort(query)
    sorted_heading = evaluate_heading(t, sorted_query)
    np.testing.assert_allclose(actual, sorted_heading[np.argsort(np.argsort(query))])


def test_integral_matches_dense_reference_at_knots_and_total_is_exact():
    p = profile()
    knots = np.array([0.0, 0.25, 0.55, 0.8, 1.0])
    dense_u = np.linspace(0.0, 1.0, 100_001)
    dense_rate = evaluate_heading_rate(p, dense_u)
    dense_integral = np.concatenate(
        ([0.0], np.cumsum(0.5 * (dense_rate[:-1] + dense_rate[1:]) * np.diff(dense_u)))
    )
    expected = np.interp(knots, dense_u, dense_integral)
    np.testing.assert_allclose(integrated_heading_rate(p, knots), expected, atol=2e-7)
    assert trajectory_total_turn(trajectory()) == pytest.approx(
        integrated_heading_rate(p, [1.0])[0]
    )
    with pytest.raises(TypeError):
        trajectory_total_turn(trajectory(), count=257)


def test_total_turn_is_independent_of_sampling_count():
    t = trajectory()
    turn = trajectory_total_turn(t)
    for count in (4, 5, 8, 257, 4097):
        sample = evaluate_trajectory(t, count=count)
        assert sample.heading_deg[-1] - t.start_heading_deg == pytest.approx(turn)


def test_curvature_is_heading_rate_in_radians_per_length():
    p = HeadingRateProfile(90.0, 90.0, interpolation="linear")
    unit = Trajectory2D(length=1.0, start_heading_deg=0.0, heading_rate=p)
    scaled = Trajectory2D(length=2.0, start_heading_deg=0.0, heading_rate=p)
    u = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(curvature(unit, u), np.pi / 2)
    np.testing.assert_allclose(curvature(scaled, u), curvature(unit, u) / 2)


def test_unconstrained_profile_accepts_close_knots_and_large_finite_rates():
    p = HeadingRateProfile(
        start_rate_deg_per_u=100_000.0,
        changes=(HeadingRateChange(0.5, -100_000.0), HeadingRateChange(0.505, 100_000.0)),
        end_rate_deg_per_u=0.0,
    )
    assert evaluate_heading_rate(p, np.array([0.5, 0.505])).tolist() == [-100_000.0, 100_000.0]
    with pytest.raises(ValueError, match="finite"):
        HeadingRateChange(0.5, float("inf"))


def test_arclength_resampling_and_cubic_path_are_deterministic():
    t = trajectory()
    first = sample_trajectory_by_arclength(t, 41, integration_count=257)
    second = sample_trajectory_by_arclength(t, 41, integration_count=257)
    np.testing.assert_array_equal(first, second)
    path = trajectory_to_cubic_path(t, max_error=1e-4)
    assert path.start == pytest.approx(tuple(evaluate_trajectory(t, count=2).xy[0]))
    assert len(path.segments) > 0
    beziers = path.to_beziers()
    assert len(beziers) == len(path.segments)
    assert beziers[0].p0 == path.start
    assert beziers[-1].p3 == pytest.approx(
        tuple(evaluate_trajectory(t, count=257).xy[-1]), abs=2e-6
    )



def test_cubic_path_conversion_respects_geometric_tolerance():
    p = HeadingRateProfile(360.0, 360.0, interpolation="linear")
    t = Trajectory2D(length=1.0, start_heading_deg=0.0, heading_rate=p)
    path = trajectory_to_cubic_path(t, max_error=1e-4)
    center = np.array([0.0, 1.0 / (2.0 * np.pi)])
    radius = 1.0 / (2.0 * np.pi)
    for curve in path.to_beziers():
        points = [np.asarray(point) for point in (curve.p0, curve.p1, curve.p2, curve.p3)]
        p0, p1, p2, p3 = points
        for u in np.linspace(0.0, 1.0, 9):
            one = 1.0 - u
            point = one**3 * p0 + 3 * one**2 * u * p1 + 3 * one * u**2 * p2 + u**3 * p3
            assert abs(np.linalg.norm(point - center) - radius) < 2e-4
