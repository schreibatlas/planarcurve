# B-splines and cubic paths

This workflow adapts [`spline_to_bezier.py`](https://github.com/schreibatlas/planarcurve/blob/main/examples/spline_to_bezier.py). Install SciPy with `python -m pip install 'planarcurve[spline]'` before importing `planarcurve.spline`.

## Construct a supported curve

A cubic curve with five controls and one simple interior knot at `0.5` uses nine knots: four zeros, `0.5`, and four ones.

```python
import numpy as np

from planarcurve.bezier import evaluate_cubic
from planarcurve.spline import BSplineCurve

curve = BSplineCurve(
    knots=(0, 0, 0, 0, 0.5, 1, 1, 1, 1),
    control_points=((0, 0), (1, 2), (2, 3), (3, 2), (4, 0)),
)
```

Only clamped cubic geometry with simple interior knots is supported. See [spline concepts](../concepts/splines.md) for the full contract. General SciPy spline configurations are not accepted by this API.

## Evaluate and sample

```python
parameters = np.linspace(0.0, 1.0, 21)
evaluation = curve.evaluate(parameters)
arc_samples = curve.sample_by_arclength(21)

assert evaluation.xy.shape == (21, 2)
assert arc_samples.tangent is not None
assert arc_samples.curvature is not None
```

Ordinary evaluation uses the spline parameter, not distance along the curve. Arc-length sampling is approximate and includes differential geometry at the resulting sample parameters. An optional `Affine2D` can be passed to either evaluation or sampling; sampling then measures the transformed curve.

## Convert to Bezier segments

```python
cubic_path = curve.to_cubic_path(tolerance=1e-8)
beziers = cubic_path.to_beziers()
local_t = np.linspace(0.0, 1.0, 21)
converted = np.vstack([evaluate_cubic(bezier, local_t) for bezier in beziers])
```

`to_cubic_path()` returns a `CubicPath` with an explicit starting point and a sequence of cubic segments. `to_beziers()` turns that chain into individual `CubicBezier` values. The conversion inserts interior knots as needed and verifies approximation tolerance at sampled points within each span.
