import numpy as np
import pytest

from planarcurve.bezier import cubic_tangent, evaluate_cubic
from planarcurve.polyline import bounding_box, polyline_length, resample_polyline_by_arclength
from planarcurve.types import CubicBezier


def test_polyline_helpers():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    assert polyline_length(points) == pytest.approx(3.0)
    np.testing.assert_allclose(
        resample_polyline_by_arclength(points, 4),
        [[0, 0], [1, 0], [2, 0], [3, 0]],
    )
    assert bounding_box(points) == (0.0, 0.0, 3.0, 0.0)


def test_cubic_endpoints_and_tangent():
    curve = CubicBezier((0, 0), (1, 0), (2, 1), (3, 1))
    xy = evaluate_cubic(curve, np.array([0.0, 1.0]))
    np.testing.assert_allclose(xy, [[0, 0], [3, 1]])
    tangent = cubic_tangent(curve, np.array([0.0, 1.0]))
    np.testing.assert_allclose(np.linalg.norm(tangent, axis=1), 1.0)
