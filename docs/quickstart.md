# Quickstart

This short workflow moves from sampled points to a transformed trajectory. Each module also has a focused [guide](guides/index.md).

## Measure and resample a polyline

A polyline is an `(N, 2)` array of finite 2D points. Its length is the sum of its segment lengths, and resampling uses equal intervals along that piecewise-linear geometry.

```python
import numpy as np

from planarcurve.polyline import polyline_length, resample_polyline_by_arclength

points = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])
print(polyline_length(points))
samples = resample_polyline_by_arclength(points, count=9)
```

## Evaluate a cubic Bezier

Bezier parameters `t` run from 0 to 1 and are not arc-length parameters.

```python
from planarcurve.bezier import evaluate_cubic
from planarcurve.types import CubicBezier

curve = CubicBezier((0, 0), (1, 2), (3, 2), (4, 0))
curve_points = evaluate_cubic(curve, np.linspace(0.0, 1.0, 25))
```

## Construct a normalized trajectory

Trajectory progress `u` also runs from 0 to 1, but has a distinct meaning: it is normalized progress at constant geometric speed. Heading rates are degrees per unit `u`.

```python
from planarcurve.trajectory import HeadingRateChange, HeadingRateProfile, Trajectory2D

trajectory = Trajectory2D(
    length=12.0,
    start_heading_deg=15.0,
    heading_rate=HeadingRateProfile(
        start_rate_deg_per_u=0.0,
        changes=(HeadingRateChange(0.5, 180.0),),
        end_rate_deg_per_u=0.0,
        interpolation="linear",
    ),
)
sample = trajectory.evaluate(count=129)
```

`sample.xy` contains numerically integrated positions. `sample.arc_length` is theoretical and equals `trajectory.length * sample.u` exactly, up to floating-point arithmetic.

## Place the trajectory with a transform

A trajectory begins at the origin. Apply an affine map to place or orient its sampled points:

```python
from planarcurve.transforms import Affine2D

placement = Affine2D.rotation(30.0).compose(Affine2D.translation_only(5.0, 2.0))
placed_points = placement.apply(sample.xy)
```

Composition is `a.compose(b) == a ∘ b`: `b` is applied first, then `a`.

B-splines and fitting are documented in the [spline](guides/splines.md) and [fitting](guides/fitting.md) guides and require SciPy. SVG adapters in the [SVG guide](guides/svg.md) require FontTools.
