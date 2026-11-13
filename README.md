# planarcurve

`planarcurve` provides numerical building blocks for planar polylines, Bezier curves, cubic B-splines, normalized heading-rate trajectories, transforms, fitting, and SVG paths. Core geometry uses NumPy and remains independent of application-specific models.

## Install

```bash
python -m pip install planarcurve
```

Optional features:

```bash
python -m pip install 'planarcurve[spline]'  # B-spline evaluation
python -m pip install 'planarcurve[fit]'     # spline and trajectory fitting
python -m pip install 'planarcurve[svg]'     # SVG path adapters
python -m pip install 'planarcurve[all]'     # all optional dependencies
```

For development, install `planarcurve[dev]`.

## API map

```text
planarcurve.polyline       lengths, resampling, projection, scoring
planarcurve.bezier         evaluation, derivatives, curvature, flattening
planarcurve.trajectory     normalized heading-rate trajectories
planarcurve.transforms     affine and similarity transforms
planarcurve.spline         cubic B-splines and cubic-path conversion (SciPy)
planarcurve.fitting        spline and trajectory fitting (SciPy)
planarcurve.svg            SVG parsing, sampling, and transformation (FontTools)
```

## Quick start

Resample a polyline by its piecewise-linear arc length:

```python
import numpy as np
from planarcurve.polyline import polyline_length, resample_polyline_by_arclength

points = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 0.0]])
uniform = resample_polyline_by_arclength(points, count=20)
print(polyline_length(points), uniform.shape)
```

Create a normalized heading-rate trajectory:

```python
from planarcurve.trajectory import (
    HeadingRateChange,
    HeadingRateProfile,
    Trajectory2D,
    trajectory_total_turn,
)

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
print(sample.xy[-1], trajectory_total_turn(trajectory))
```

Here `u` is normalized progress in `[0, 1]`. Heading rates are degrees per unit progress. Geometric curvature is signed inverse length and equals `radians(heading_rate) / trajectory.length`.

## Fitting

Trajectory fitting uses a unit-length, zero-heading canonical profile and a separate similarity transform for each occurrence. Position residuals are divided by the observed occurrence's length. Heading, total-turn, and smoothness residuals are scaled by their configured angular or rate sigmas, then multiplied by dimensionless weights. Reported position error metrics use the input coordinate units.

```python
import numpy as np
from planarcurve.fitting.trajectory import TrajectoryFitOptions, fit_trajectory

points = np.array([[0, 0], [1, 0], [2, 1], [3, 2], [4, 2]], dtype=float)
fit = fit_trajectory(points, TrajectoryFitOptions(change_count=1, samples=48, restarts=0))
assert fit.success
print(fit.metrics)
```

Spline fitting uses sparse weighted least squares with iteratively reweighted residuals. `SplineEndpointPolicy` defaults to free endpoints. Choose locked or anchored endpoints explicitly when a fit requires them.

## Supported spline geometry

`BSplineCurve` supports cubic B-splines only. For `n` control points, the knot vector must contain `n + 4` finite values, start with four zeros, end with four ones, and have strictly increasing interior values in `(0, 1)`. At least four finite 2D control points are required. Spline evaluation, derivatives, and conversion require SciPy.

`HeadingRateProfile(interpolation="cubic-hermite")` computes shape-preserving Hermite slopes from its rate knots. Callers provide knot positions and rates, not derivative values. The `linear` option uses piecewise-linear rates.

## Numerical contracts

| Operation | Contract |
| --- | --- |
| Heading and total-turn evaluation | Exact piecewise antiderivative of the selected heading-rate interpolation. |
| Theoretical trajectory arc length | Exactly `length * u`; returned sample arc lengths do not depend on the position integrator. |
| Trajectory positions | Adaptive numerical integration of the heading-derived unit tangent. Accuracy is controlled by integration tolerances and depth. |
| Polyline length and resampling | Exact for the supplied piecewise-linear polyline, up to floating-point arithmetic. |
| Bezier evaluation, derivatives, tangent, and curvature | Analytic evaluation, up to floating-point arithmetic. |
| Bezier flattening and length | Adaptive numerical approximations controlled by tolerance and maximum depth. |
| B-spline evaluation and derivatives | Analytic spline evaluation through SciPy. Arc-length sampling uses a dense sampled approximation. |
| Trajectory-to-cubic conversion | Adaptive pieces are checked at quarter, midpoint, and three-quarter parameters. The tolerance is a sampled criterion, not a formal continuous error bound. |
| SVG Bezier lengths and endpoint tangents | Sampled approximations controlled by `samples_per_curve`. |

Function inputs that accept arrays use NumPy array-like values; public numerical outputs use float arrays. Point values must contain exactly two finite coordinates.

Polyline projection compares every query point with every segment. Its current time and temporary memory costs are `O(M*N)` for `M` queries and `N` segments.

`sample_path()` accepts exactly one SVG subpath. For multiple subpaths, use `sample_subpaths()` to receive separate arrays without artificial connecting segments. `PathInfo.closed` is true only when all subpaths are closed; `subpath_closed` reports each status. Its start and start tangent refer to the first subpath, and its end and end tangent refer to the last.

NumPy arrays stored in frozen result dataclasses remain writable. Frozen dataclasses prevent field reassignment, not mutation of the arrays they contain.

## Optional dependencies and examples

The core package imports with NumPy only. `planarcurve.spline` and the fitting modules use SciPy. `planarcurve.svg` uses FontTools. These optional modules are not imported by the core package.

Install optional features and run examples from the repository root:

```bash
python -m pip install 'planarcurve[all]'
python examples/trajectory_gallery.py
python examples/bezier_basics.py
python examples/spline_to_bezier.py
python examples/svg_pipeline.py
python examples/fit_polyline_to_spline.py
python examples/fit_shared_trajectory.py
```

The synthetic, Matplotlib-free examples are documented in the [examples guide](https://github.com/schreibatlas/planarcurve/blob/main/examples/README.md). The [trajectory gallery](https://github.com/schreibatlas/planarcurve/blob/main/examples/trajectory_gallery.py), [Bezier basics](https://github.com/schreibatlas/planarcurve/blob/main/examples/bezier_basics.py), [spline conversion](https://github.com/schreibatlas/planarcurve/blob/main/examples/spline_to_bezier.py), [SVG pipeline](https://github.com/schreibatlas/planarcurve/blob/main/examples/svg_pipeline.py), [spline fitting](https://github.com/schreibatlas/planarcurve/blob/main/examples/fit_polyline_to_spline.py), and [shared trajectory fitting](https://github.com/schreibatlas/planarcurve/blob/main/examples/fit_shared_trajectory.py) scripts write SVG output where applicable.

## API stability and provenance

The package is pre-release and follows a `0.x` compatibility policy. Public names and numerical behavior may change between minor releases. The generic geometry and fitting functionality was extracted and adapted from InkGrammar. `planarcurve` uses neutral geometry types and does not include handwriting, glyph, dataset, Pydantic, or application orchestration concepts. The project is licensed under Apache-2.0.
