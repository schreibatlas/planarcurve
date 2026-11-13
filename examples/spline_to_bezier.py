import numpy as np

from planarcurve.bezier import evaluate_cubic
from planarcurve.spline import BSplineCurve
from planarcurve.types import CubicPath


def main() -> None:
    curve = BSplineCurve(
        knots=(0, 0, 0, 0, 0.5, 1, 1, 1, 1),
        control_points=((0, 0), (1, 2), (2, 3), (3, 2), (4, 0)),
    )
    sample = curve.sample_by_arclength(21)
    cubic_path = curve.to_cubic_path()

    assert isinstance(cubic_path, CubicPath)
    assert len(sample.xy) == 21
    assert sample.tangent is not None and sample.curvature is not None
    parameters = np.concatenate((np.linspace(0.0, 0.5, 21), np.linspace(0.5, 1.0, 21)))
    expected = curve.evaluate(parameters, derivatives=False).xy
    bezier_parameters = np.linspace(0.0, 1.0, 21)
    actual = np.vstack(
        [evaluate_cubic(bezier, bezier_parameters) for bezier in cubic_path.to_beziers()]
    )
    assert np.max(np.linalg.norm(actual - expected, axis=1)) < 1e-8

    print("uniform arc-length sample shape:", sample.xy.shape)
    print("endpoint tangents:", sample.tangent[[0, -1]])
    print("curvature range:", float(sample.curvature.min()), float(sample.curvature.max()))
    print("cubic segment count:", len(cubic_path.segments))
    for index, segment in enumerate(cubic_path.segments):
        print(index, segment)


if __name__ == "__main__":
    main()
