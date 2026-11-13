# Bezier curves

`CubicBezier` and `QuadraticBezier` store finite 2D control points. The `t` parameter lies in `[0, 1]` and is not generally proportional to arc length. This example follows [`bezier_basics.py`](https://github.com/schreibatlas/planarcurve/blob/main/examples/bezier_basics.py):

```python
import numpy as np

from planarcurve.bezier import (
    cubic_curvature,
    cubic_derivative,
    cubic_length,
    cubic_second_derivative,
    cubic_tangent,
    evaluate_cubic,
    flatten_cubic,
    split_cubic,
)
from planarcurve.types import CubicBezier

curve = CubicBezier((0.0, 0.0), (1.0, 2.0), (3.0, 2.0), (4.0, 0.0))
t = np.linspace(0.0, 1.0, 21)
points = evaluate_cubic(curve, t)
first = cubic_derivative(curve, t)
second = cubic_second_derivative(curve, t)
unit = cubic_tangent(curve, t)
curvature = cubic_curvature(curve, t)
left, right = split_cubic(curve, 0.4)
flat = flatten_cubic(curve, tolerance=0.01)
approximate_length = cubic_length(curve)
```

## Evaluation and differential geometry

The evaluators accept a finite one-dimensional array of parameters in `[0, 1]` and return points of shape `(N, 2)`. Derivative functions return the first or second derivative with respect to `t`. Tangent functions return normalized vectors by default; pass `unit=False` to get the raw first derivative. At a degenerate derivative, normalized tangent and signed curvature are zero.

Signed curvature uses the orientation of the control-point order and has inverse-coordinate units. Quadratic curves have parallel evaluation, derivative, tangent, and curvature functions.

## Split and approximate

`split_cubic()` uses de Casteljau subdivision and returns two exact subcurves meeting at the split point. `flatten_cubic()` recursively subdivides until both interior control points are within `tolerance` of the endpoint chord. This is a flatness test based on control-point distance, not an arc-length error tolerance. If the requested flatness cannot be reached within `max_depth`, it raises `ValueError`.

`cubic_length()` is approximate. It compares the endpoint chord with the control-polygon length, recursively subdividing and sharing the absolute tolerance between child pieces. Quadratic flattening and length use the same machinery after exact conversion to a cubic representation. Tolerance and recursion depth must be positive/valid as required by the respective API.
