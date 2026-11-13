import numpy as np

from planarcurve.spline import BSplineCurve
from planarcurve.types import CubicPath

curve = BSplineCurve(
    knots=(0, 0, 0, 0, 0.5, 1, 1, 1, 1),
    control_points=((0, 0), (1, 2), (2, 3), (3, 2), (4, 0)),
)

sample = curve.sample_by_arclength(21)
start = tuple(curve.evaluate(np.array([0.0])).xy[0])
cubic_path = CubicPath(start=start, segments=tuple(curve.to_cubic_segments()))
segments = cubic_path.segments

print("uniform arc-length sample shape:", sample.xy.shape)
print("endpoint tangents:", sample.tangent[[0, -1]])
print("curvature range:", float(sample.curvature.min()), float(sample.curvature.max()))
print("cubic segment count:", len(segments))
for index, segment in enumerate(segments):
    print(index, segment)
