from __future__ import annotations

import numpy as np
import pytest

from planarcurve.fitting.spline import SplineEndpointPolicy
from planarcurve.spline import (
    BSplineCurve,
    bspline_from_geometry,
    control_roughness_matrix,
    evaluate_bspline,
    sample_curve_by_arclength,
    spline_design_matrix,
    spline_to_cubic_segments,
)
from planarcurve.transforms import Affine2D


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
    np.testing.assert_array_equal(result.tangent, np.zeros((3, 2)))
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


def test_spline_to_cubic_segments_reproduces_spans():
    segments = spline_to_cubic_segments(geometry(), tolerance=1e-10)
    assert len(segments) == 2



def test_endpoint_policy_is_fitting_configuration_not_geometry():
    policy = SplineEndpointPolicy(start="free", end="anchored")
    assert (policy.start, policy.end) == ("free", "anchored")
