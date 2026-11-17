import numpy as np
import pytest

from planarcurve.descriptors import (
    TrajectoryDescriptor,
    descriptor_distance,
    reverse_descriptor,
    trajectory_descriptor,
)


def _curve() -> np.ndarray:
    parameter = np.linspace(0.0, 1.0, 200)
    return np.column_stack((parameter, 0.35 * parameter**2 + 0.1 * parameter**3))


def test_descriptor_is_invariant_to_translation_scale_and_rotation():
    points = _curve()
    angle = np.deg2rad(37.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = 12.5 * points @ rotation.T + np.array([40.0, -23.0])

    original = trajectory_descriptor(points)
    actual = trajectory_descriptor(transformed)

    np.testing.assert_allclose(actual.normalized_xy, original.normalized_xy, atol=2e-3)
    np.testing.assert_allclose(actual.tangent_xy, original.tangent_xy, atol=2e-2)
    np.testing.assert_allclose(actual.heading_rate, original.heading_rate, atol=2.0)
    assert actual.total_turn_deg == pytest.approx(original.total_turn_deg, abs=0.1)
    assert descriptor_distance(actual, original) < 0.01


def test_descriptor_does_not_mutate_input_and_exposes_read_only_arrays():
    points = _curve()
    original = points.copy()
    descriptor = trajectory_descriptor(points)

    np.testing.assert_array_equal(points, original)
    for values in (descriptor.normalized_xy, descriptor.tangent_xy, descriptor.heading_rate):
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values[0] = 0.0


def test_descriptor_records_direction_and_reversal_is_explicit_and_involutive():
    descriptor = trajectory_descriptor(_curve())
    reversed_descriptor = reverse_descriptor(descriptor)
    twice_reversed = reverse_descriptor(reversed_descriptor)

    assert descriptor_distance(descriptor, reversed_descriptor) > 0.05
    assert reversed_descriptor.total_turn_deg == pytest.approx(-descriptor.total_turn_deg)
    assert reversed_descriptor.signed_turn_energy == pytest.approx(-descriptor.signed_turn_energy)
    np.testing.assert_allclose(twice_reversed.normalized_xy, descriptor.normalized_xy, atol=1e-12)
    np.testing.assert_allclose(twice_reversed.tangent_xy, descriptor.tangent_xy, atol=1e-12)
    np.testing.assert_allclose(twice_reversed.heading_rate, descriptor.heading_rate, atol=1e-12)
    assert descriptor_distance(twice_reversed, descriptor) == pytest.approx(0.0, abs=1e-12)


def test_line_and_constant_turn_arc_have_stable_signed_metrics():
    line = trajectory_descriptor(np.array([[0.0, 0.0], [2.0, 0.0]]))
    angle = np.linspace(0.0, np.pi / 2.0, 200)
    arc = trajectory_descriptor(np.column_stack((np.cos(angle), np.sin(angle))))
    reversed_arc = reverse_descriptor(arc)

    assert line.total_turn_deg == pytest.approx(0.0, abs=1e-10)
    assert arc.total_turn_deg > 80.0
    assert reversed_arc.total_turn_deg < -80.0
    assert arc.endpoint_displacement_ratio == pytest.approx(np.sqrt(2.0) / (np.pi / 2.0), rel=0.01)


def test_descriptor_distance_weights_and_input_validation():
    points = _curve()
    first = trajectory_descriptor(points)
    second = trajectory_descriptor(points[::-1])

    position_only = descriptor_distance(first, second, weights={"normalized_xy": 1.0})
    assert position_only > 0.05
    assert descriptor_distance(first, first) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="unknown"):
        descriptor_distance(first, second, weights={"unknown": 1.0})
    with pytest.raises(ValueError, match="positive"):
        descriptor_distance(first, second, weights={"normalized_xy": 0.0})
    with pytest.raises(ValueError, match="nonnegative"):
        descriptor_distance(first, second, weights={"normalized_xy": -1.0})
    with pytest.raises(ValueError, match="matching sample_count"):
        descriptor_distance(first, trajectory_descriptor(points, samples=32))

    with pytest.raises(ValueError, match="shape"):
        trajectory_descriptor(np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="finite"):
        trajectory_descriptor(np.array([[0.0, 0.0], [np.nan, 1.0]]))
    with pytest.raises(ValueError, match="zero-length"):
        trajectory_descriptor(np.ones((4, 2)))
    with pytest.raises(ValueError, match="samples"):
        trajectory_descriptor(points, samples=1)
    with pytest.raises(ValueError, match="smoothing_window"):
        trajectory_descriptor(points, smoothing_window=0)
    with pytest.raises(ValueError, match="sample_count"):
        TrajectoryDescriptor(1, np.zeros((1, 2)), np.zeros((1, 2)), np.zeros(1), 0.0, 0.0, 0.0)


def test_descriptor_apis_are_exported_from_package_root():
    import planarcurve

    assert planarcurve.TrajectoryDescriptor is TrajectoryDescriptor
    assert planarcurve.trajectory_descriptor is trajectory_descriptor
    assert planarcurve.reverse_descriptor is reverse_descriptor
    assert planarcurve.descriptor_distance is descriptor_distance
