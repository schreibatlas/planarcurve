# planarcurve

`planarcurve` provides numerical building blocks for planar polylines, Bezier curves, cubic B-splines, normalized heading-rate trajectories, transforms, fitting, and SVG paths. Core geometry uses NumPy and remains independent of application-specific models.

**Documentation:** [Guides, concepts, and API reference](https://planarcurve.readthedocs.io/en/latest/).
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
planarcurve.polyline       lengths, resampling, projection, intersections, scoring
planarcurve.bezier         evaluation, derivatives, curvature, flattening
planarcurve.trajectory     normalized heading-rate trajectories
planarcurve.trajectory_chain  directed segments, pose propagation, chain sampling
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

## Directed trajectory chains

`TrajectoryChain` composes generic `Trajectory2D` segments in either direction. Reversal uses the heading-rate profile, and chain samples preserve segment boundaries, tangent changes, and cumulative arc length. `trajectory_chain_to_cubic_path()` provides a connected cubic representation. These primitives contain no application-specific movement semantics.

## Fitting
## Fitting

Fitting requires SciPy. Spline fitting estimates shared control points with weighted sparse least squares and IRLS. Trajectory fitting uses a normalized length-1, zero-heading profile plus a similarity transform for each occurrence. See the [fitting guide](https://planarcurve.readthedocs.io/en/latest/guides/fitting.html) for residual units, endpoint policies, options, and metrics.
## Supported spline geometry

`BSplineCurve` intentionally supports clamped cubic geometry with simple interior knots, not arbitrary SciPy spline configurations. The `cubic-hermite` heading-rate mode computes shape-preserving slopes from rate knots; callers do not supply derivative values. See [spline concepts](https://planarcurve.readthedocs.io/en/latest/concepts/splines.html) for the full contract.
## Numerical contracts

| Operation | Contract |
| --- | --- |
| Heading and theoretical trajectory arc length | Piecewise heading integration is analytic; arc length is exactly `length * u`. |
| Trajectory positions and trajectory-to-cubic conversion | Adaptive numerical integration and sampled error checks, not a continuous global bound. |
| Polyline length and Bezier evaluation/derivatives | Exact for the supplied geometry, up to floating-point arithmetic. |
| Bezier flattening/length and B-spline arc-length sampling | Adaptive or dense-sample approximations. |
| SVG curve lengths and endpoint tangents | Sampled approximations controlled by `samples_per_curve`. |

Public points are finite 2D coordinates. Polyline projection compares every query with every segment, using `O(M*N)` time and temporary memory. SVG `sample_path()` accepts one subpath; use `sample_subpaths()` to preserve multiple subpaths without artificial connectors. Frozen result dataclasses do not make embedded NumPy arrays immutable.

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

See the [examples guide](https://planarcurve.readthedocs.io/en/latest/examples.html) for all six scripts, their optional dependencies, and generated output.

## API stability and provenance

The package is pre-release and follows a `0.x` compatibility policy. Public names and numerical behavior may change between minor releases. The generic geometry and fitting functionality was extracted and adapted from InkGrammar. `planarcurve` uses neutral geometry types and does not include handwriting, glyph, dataset, Pydantic, or application orchestration concepts. The project is licensed under Apache-2.0.
