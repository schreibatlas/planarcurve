from __future__ import annotations

import numpy as np
import pytest

from planarcurve.fitting.spline import SplineEndpointPolicy
from planarcurve.spline import (
    BSplineCurve,
    bspline_from_geometry,
    clamped_uniform_bspline_basis,
    control_roughness_matrix,
    evaluate_bspline,
    sample_curve_by_arclength,
    spline_design_matrix,
    spline_to_cubic_path,
    spline_to_cubic_segments,
)
from planarcurve.transforms import Affine2D
from planarcurve.types import CubicPath


def geometry() -> BSplineCurve:
    return BSplineCurve(
        degree=3,
        knots=(0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0),
        control_points=((0.0, 0.0), (1.0, 2.0), (2.0, 3.0), (3.0, 2.0), (4.0, 0.0)),
    )


def test_spline_evaluation_endpoints_derivatives():
    result = evaluate_bspline(geometry(), np.array([0.0, 1.0]))
    np.testing.assert_allclose(result.xy, [[0.0, 0.0], [4.0, 0.0]], atol=1e-12)
    assert np.linalg.norm(result.tangent, axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert result.curvature is not None
    assert np.isfinite(result.curvature).all()


def test_evaluation_without_derivatives_returns_points_only():
    parameters = np.array([0.0, 0.5, 1.0])
    result = evaluate_bspline(geometry(), parameters, derivatives=False)
    assert result.curvature is None
    assert result.tangent is None
    np.testing.assert_allclose(result.xy[[0, -1]], [[0.0, 0.0], [4.0, 0.0]])


def test_cubic_simple_knot_has_c2_continuity():
    x_spline, y_spline = bspline_from_geometry(geometry())
    left = np.array(
        [[s.derivative(order)(0.5 - 1e-8) for s in (x_spline, y_spline)] for order in range(3)]
    )
    right = np.array(
        [[s.derivative(order)(0.5 + 1e-8) for s in (x_spline, y_spline)] for order in range(3)]
    )
    np.testing.assert_allclose(left, right, atol=1e-5)


def test_design_and_roughness_matrices():
    design = spline_design_matrix(np.array([0.0, 0.25, 0.5, 1.0]), geometry())
    roughness = control_roughness_matrix(5)
    assert design.shape == (4, 5)
    np.testing.assert_allclose(np.asarray(design.sum(axis=1)).ravel(), 1.0)
    np.testing.assert_array_equal(roughness.toarray()[0], [1, -2, 1, 0, 0])


def test_arclength_sampling_line_is_uniform_and_deterministic():
    line = BSplineCurve(
        knots=(0, 0, 0, 0, 1, 1, 1, 1),
        control_points=((0, 0), (1, 0), (2, 0), (3, 0)),
    )
    first = sample_curve_by_arclength(line, 7)
    second = sample_curve_by_arclength(line, 7)
    np.testing.assert_allclose(first.xy[:, 0], np.linspace(0, 3, 7), atol=1e-8)
    np.testing.assert_array_equal(first.u, second.u)


def test_affine_semantics_match_inkgrammar_topology_transform():
    base = evaluate_bspline(geometry(), np.array([0.3]))
    transform = Affine2D.from_scale_shear_translation(
        scale_x=2.0, scale_y=3.0, shear_x=0.25, dx=5.0, dy=-2.0
    )
    transformed = evaluate_bspline(geometry(), np.array([0.3]), transform=transform)
    expected = np.array(
        [[5.0 + 2.0 * (base.xy[0, 0] + 0.25 * base.xy[0, 1]), -2.0 + 3.0 * base.xy[0, 1]]]
    )
    np.testing.assert_allclose(transformed.xy, expected)


def test_affine_rotation_rotates_tangent_and_preserves_curvature():
    parameters = np.array([0.3])
    base = evaluate_bspline(geometry(), parameters)
    rotation = Affine2D.rotation(90.0)
    transformed = evaluate_bspline(geometry(), parameters, transform=rotation)

    np.testing.assert_allclose(transformed.tangent, base.tangent @ rotation.linear.T, atol=1e-12)
    np.testing.assert_allclose(transformed.curvature, base.curvature, atol=1e-12)


def test_affine_uniform_scale_and_translation_preserve_differential_contracts():
    parameters = np.array([0.3])
    base = evaluate_bspline(geometry(), parameters)
    scale = Affine2D.scale(2.5)
    scaled = evaluate_bspline(geometry(), parameters, transform=scale)
    translated = evaluate_bspline(
        geometry(), parameters, transform=Affine2D.translation_only(7.0, -4.0)
    )

    np.testing.assert_allclose(scaled.tangent, base.tangent, atol=1e-12)
    np.testing.assert_allclose(scaled.curvature, base.curvature / 2.5, atol=1e-12)
    np.testing.assert_allclose(translated.tangent, base.tangent, atol=1e-12)
    np.testing.assert_allclose(translated.curvature, base.curvature, atol=1e-12)


def test_affine_shear_derivatives_match_finite_differences():
    parameter = 0.3
    step = 1e-5
    transform = Affine2D.from_scale_shear_translation(
        scale_x=1.7, scale_y=0.8, shear_x=0.4, dx=3.0, dy=-2.0
    )
    transformed = evaluate_bspline(geometry(), np.array([parameter]), transform=transform)
    nearby = evaluate_bspline(
        geometry(), np.array([parameter - step, parameter, parameter + step]), transform=transform
    ).xy
    first = (nearby[2] - nearby[0]) / (2.0 * step)
    second = (nearby[2] - 2.0 * nearby[1] + nearby[0]) / step**2
    speed = np.linalg.norm(first)
    expected_tangent = first / speed
    expected_curvature = (first[0] * second[1] - first[1] * second[0]) / speed**3

    np.testing.assert_allclose(transformed.tangent[0], expected_tangent, atol=1e-8)
    np.testing.assert_allclose(transformed.curvature[0], expected_curvature, atol=1e-6)


def test_spline_to_cubic_segments_reproduces_spans():
    segments = spline_to_cubic_segments(geometry(), tolerance=1e-10)
    assert len(segments) == 2


def test_endpoint_policy_is_fitting_configuration_not_geometry():
    policy = SplineEndpointPolicy(start="free", end="anchored")
    assert (policy.start, policy.end) == ("free", "anchored")


def test_spline_to_cubic_path_uses_shared_path_representation():
    curve = geometry()
    path = spline_to_cubic_path(curve)

    assert isinstance(path, CubicPath)
    assert path.start == curve.control_points[0]
    assert path.segments == tuple(spline_to_cubic_segments(curve))
    assert curve.to_cubic_path() == path


def test_spline_rejects_float_counts_orders_and_extra_dimensions():
    curve = geometry()
    with pytest.raises(ValueError, match="count"):
        sample_curve_by_arclength(curve, 3.5)
    with pytest.raises(ValueError, match="order"):
        control_roughness_matrix(5, order=2.5)
    with pytest.raises(ValueError, match="exactly two"):
        BSplineCurve(
            knots=curve.knots,
            control_points=((0.0, 0.0, 99.0), *curve.control_points[1:]),
        )
    with pytest.raises(ValueError, match="degree"):
        BSplineCurve(knots=curve.knots, control_points=curve.control_points, degree=3.0)


def test_spline_endpoint_policy_defaults_to_free_endpoints():
    policy = SplineEndpointPolicy()

    assert (policy.start, policy.end) == ("free", "free")


def test_clamped_uniform_bspline_basis_seeds_by_arclength_and_preserves_endpoints():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 2.0], [4.0, 2.0]])
    basis = clamped_uniform_bspline_basis(6, points)

    assert basis.degree == 3
    assert basis.knots == pytest.approx((0.0,) * 4 + (1.0 / 3.0, 2.0 / 3.0) + (1.0,) * 4)
    np.testing.assert_allclose(
        basis.control_points,
        [[0.0, 0.0], [1.0, 0.2], [1.0, 1.4], [1.6, 2.0], [2.8, 2.0], [4.0, 2.0]],
    )
    assert basis.control_points[0] == (0.0, 0.0)
    assert basis.control_points[-1] == (4.0, 2.0)


def test_clamped_uniform_bspline_basis_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="control_count"):
        clamped_uniform_bspline_basis(3, [[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="at least two points"):
        clamped_uniform_bspline_basis(4, [[0.0, 0.0]])
    with pytest.raises(ValueError, match="distinct points"):
        clamped_uniform_bspline_basis(4, [[0.0, 0.0], [0.0, 0.0]])
