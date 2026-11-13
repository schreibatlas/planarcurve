import numpy as np

from planarcurve.bezier import (
    cubic_curvature,
    cubic_length,
    cubic_tangent,
    evaluate_cubic,
    flatten_cubic,
    split_cubic,
)
from planarcurve.types import CubicBezier


def main() -> None:
    curve = CubicBezier((0.0, 0.0), (1.0, 2.0), (3.0, 2.0), (4.0, 0.0))
    parameters = np.linspace(0.0, 1.0, 21)
    points = evaluate_cubic(curve, parameters)
    tangents = cubic_tangent(curve, parameters)
    curvature = cubic_curvature(curve, parameters)
    left, right = split_cubic(curve, 0.4)
    flattened = flatten_cubic(curve, tolerance=0.01)
    length = cubic_length(curve)

    np.testing.assert_allclose(points[[0, -1]], [curve.p0, curve.p3])
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=1), 1.0)
    assert np.isfinite(curvature).all()
    assert left.p3 == right.p0
    np.testing.assert_allclose(
        evaluate_cubic(left, np.array([0.5])), evaluate_cubic(curve, np.array([0.2]))
    )
    np.testing.assert_allclose(flattened[[0, -1]], [curve.p0, curve.p3])
    assert length >= np.linalg.norm(np.asarray(curve.p3) - np.asarray(curve.p0))

    print("evaluated points:", points.shape)
    print("midpoint tangent and curvature:", tangents[10], curvature[10])
    print("split endpoints:", left.p3, right.p0)
    print("flattened vertices:", len(flattened))
    print("curve length:", length)


if __name__ == "__main__":
    main()
