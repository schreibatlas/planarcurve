from __future__ import annotations

import numpy as np
import pytest

from planarcurve.bezier import (
    cubic_curvature,
    cubic_derivative,
    cubic_length,
    cubic_second_derivative,
    cubic_tangent,
    evaluate_cubic,
    evaluate_quadratic_bezier,
    flatten_cubic,
    quadratic_length,
    split_cubic,
)
from planarcurve.polyline import (
    bounding_box,
    closest_point_on_polyline,
    cumulative_arclength,
    distance_to_polyline,
    mean_squared_turn_angle,
    normalized_arclength,
    polyline_length,
    polyline_self_intersections,
    project_points_to_polyline,
    resample_polyline_by_arclength,
    symmetric_polyline_distance,
)
from planarcurve.types import CubicBezier, QuadraticBezier


def test_polyline_helpers():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    assert polyline_length(points) == pytest.approx(3.0)
    np.testing.assert_allclose(
        resample_polyline_by_arclength(points, 4),
        [[0, 0], [1, 0], [2, 0], [3, 0]],
    )
    assert bounding_box(points) == (0.0, 0.0, 3.0, 0.0)
    np.testing.assert_allclose(normalized_arclength(points), [0.0, 1.0 / 3.0, 1.0])
    np.testing.assert_allclose(cumulative_arclength(points), [0.0, 1.0, 3.0])


def test_polyline_projection_covers_interiors_endpoints_and_arc_coordinates():
    polyline = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
    queries = np.array([[1.0, 1.0], [3.0, 1.0], [-1.0, 0.0], [2.0, 2.0]])
    result = project_points_to_polyline(queries, polyline)
    np.testing.assert_allclose(result.point, [[1, 0], [2, 1], [0, 0], [2, 2]])
    np.testing.assert_allclose(result.distance, [1, 1, 1, 0])
    np.testing.assert_array_equal(result.segment_index, [0, 1, 0, 1])
    np.testing.assert_allclose(result.segment_t, [0.5, 0.5, 0.0, 1.0])
    np.testing.assert_allclose(result.arc_length, [1.0, 3.0, 0.0, 4.0])
    np.testing.assert_allclose(distance_to_polyline(queries, polyline), result.distance)
    np.testing.assert_allclose(closest_point_on_polyline(queries[0], polyline), [1, 0])


def test_projection_handles_zero_length_duplicate_segments_and_validation():
    polyline = np.array([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    result = project_points_to_polyline(np.array([[1.0, 1.0]]), polyline)
    np.testing.assert_allclose(result.point, [[1.0, 0.0]])
    np.testing.assert_allclose(result.arc_length, [1.0])
    with pytest.raises(ValueError, match="at least two"):
        project_points_to_polyline(np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]]))
    with pytest.raises(ValueError, match="nonzero"):
        normalized_arclength(np.array([[1.0, 1.0]]))


def test_cubic_endpoints_derivatives_and_curvature():
    curve = CubicBezier((0, 0), (1, 0), (2, 1), (3, 1))
    xy = evaluate_cubic(curve, np.array([0.0, 1.0]))
    np.testing.assert_allclose(xy, [[0, 0], [3, 1]])
    tangent = cubic_tangent(curve, np.array([0.0, 1.0]))
    np.testing.assert_allclose(np.linalg.norm(tangent, axis=1), 1.0)
    line = CubicBezier((0, 0), (1, 0), (2, 0), (3, 0))
    np.testing.assert_allclose(cubic_derivative(line, np.array([0.0, 0.5, 1.0])), [[3, 0]] * 3)
    np.testing.assert_allclose(cubic_second_derivative(line, np.array([0.0, 0.5, 1.0])), 0.0)
    np.testing.assert_allclose(cubic_curvature(line, np.array([0.0, 0.5, 1.0])), 0.0)


def test_cubic_split_and_flatten_preserve_curve():
    curve = CubicBezier((0, 0), (1, 0), (1, 1), (0, 1))
    left, right = split_cubic(curve, 0.4)
    np.testing.assert_allclose(left.p3, right.p0)
    np.testing.assert_allclose(
        evaluate_cubic(left, np.array([0.5])), evaluate_cubic(curve, np.array([0.2]))
    )
    np.testing.assert_allclose(
        evaluate_cubic(right, np.array([0.5])), evaluate_cubic(curve, np.array([0.7]))
    )
    coarse = flatten_cubic(curve, tolerance=0.05)
    fine = flatten_cubic(curve, tolerance=0.005)
    assert len(fine) > len(coarse) > 2
    assert cubic_length(CubicBezier((0, 0), (1, 0), (2, 0), (3, 0))) == pytest.approx(3.0)
    assert cubic_length(curve, tolerance=1e-5) > 1.0


def test_quadratic_bezier_type_and_length():
    curve = QuadraticBezier((0, 0), (1, 2), (2, 0))
    np.testing.assert_allclose(
        evaluate_quadratic_bezier(curve, np.array([0.0, 1.0])), [[0, 0], [2, 0]]
    )
    assert quadratic_length(curve, tolerance=1e-4) > 2.0


def test_symmetric_polyline_distance_uses_segments_and_is_direction_invariant():

    horizontal = np.array([[0.0, 0.0], [10.0, 0.0]])
    denser = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    offset = np.array([[0.0, 1.0], [10.0, 1.0]])
    assert symmetric_polyline_distance(horizontal, denser) == pytest.approx(0.0)
    assert symmetric_polyline_distance(horizontal, offset) == pytest.approx(1.0)
    assert symmetric_polyline_distance(horizontal, offset[::-1]) == pytest.approx(1.0)


def test_symmetric_polyline_distance_validates_samples_and_degenerate_curves():

    line = np.array([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="samples"):
        symmetric_polyline_distance(line, line, samples=1)
    with pytest.raises(ValueError, match="zero-length"):
        symmetric_polyline_distance(np.zeros((2, 2)), line)


def test_mean_squared_turn_angle_is_scale_invariant_and_can_resample():

    right_angle = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    scaled = right_angle * 3.0
    assert mean_squared_turn_angle(right_angle) == pytest.approx(np.pi**2 / 4.0)
    assert mean_squared_turn_angle(right_angle, samples=5) == pytest.approx(np.pi**2 / 12.0)
    assert mean_squared_turn_angle(scaled, samples=5) == pytest.approx(np.pi**2 / 12.0)
    assert mean_squared_turn_angle(np.array([[0.0, 0.0], [2.0, 0.0]])) == 0.0
    with pytest.raises(ValueError, match="nonzero"):
        mean_squared_turn_angle(np.zeros((2, 2)))


def test_integer_parameters_and_point_dimensions_are_validated():
    line = np.array([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="count"):
        resample_polyline_by_arclength(line, 3.5)
    with pytest.raises(ValueError, match="max_depth"):
        flatten_cubic(CubicBezier((0, 0), (1, 0), (2, 0), (3, 0)), max_depth=2.5)
    with pytest.raises(ValueError, match="max_depth"):
        cubic_length(CubicBezier((0, 0), (1, 0), (2, 0), (3, 0)), max_depth=2.5)
    with pytest.raises(ValueError, match="exactly two"):
        CubicBezier((0.0, 0.0, 99.0), (1, 0), (2, 0), (3, 0))
    with pytest.raises(ValueError, match="samples"):
        symmetric_polyline_distance(line, line, samples=3.5)


def test_polyline_self_intersections_skip_shared_joins_and_find_crossings():
    ordinary_join = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    assert polyline_self_intersections(ordinary_join) == ()

    crossing = np.array([[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]])
    intersections = polyline_self_intersections(crossing)
    assert len(intersections) == 1
    assert intersections[0].first_segment == 0
    assert intersections[0].second_segment == 2
    np.testing.assert_allclose(intersections[0].xy, [1.0, 1.0])


def test_polyline_self_intersections_observe_closed_loop_semantics():
    repeated_endpoint = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [0.0, 0.0]])

    open_result = polyline_self_intersections(repeated_endpoint)
    assert [(hit.first_segment, hit.second_segment) for hit in open_result] == [(0, 2)]
    assert polyline_self_intersections(repeated_endpoint, closed=True) == ()

    implicit_loop = repeated_endpoint[:-1]
    assert polyline_self_intersections(implicit_loop, closed=True) == ()


def test_polyline_self_intersections_report_collinear_adjacent_overlap():
    retraced = np.array([[0.0, 0.0], [3.0, 0.0], [1.0, 0.0]])

    intersections = polyline_self_intersections(retraced)

    assert len(intersections) == 1
    assert (intersections[0].first_segment, intersections[0].second_segment) == (0, 1)
    np.testing.assert_allclose(intersections[0].xy, [1.0, 0.0])


def test_polyline_self_intersections_validate_input_and_tolerance():
    with pytest.raises(ValueError, match="at least two"):
        polyline_self_intersections([[0.0, 0.0]])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        polyline_self_intersections([[0.0, 0.0], [1.0, 0.0]], tolerance=np.inf)
