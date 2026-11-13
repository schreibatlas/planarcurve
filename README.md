# planarcurve

`planarcurve` provides numerical building blocks for planar curves and trajectories: polylines, Bezier curves, B-splines, normalized heading-rate trajectories, affine transforms, SVG paths, and fitting. Its core geometry API requires NumPy and does not depend on application-specific models.

## Install

```bash
python -m pip install planarcurve
```

Install optional features as needed:

```bash
python -m pip install 'planarcurve[spline]'  # B-spline evaluation
python -m pip install 'planarcurve[fit]'     # spline and trajectory fitting
python -m pip install 'planarcurve[svg]'     # SVG path adapters
python -m pip install 'planarcurve[all]'     # SciPy and FontTools
```

For development, install `planarcurve[dev]`.

## Quick start

Resample a polyline by arc length:

```python
import numpy as np
from planarcurve.polyline import polyline_length, resample_polyline_by_arclength

points = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 0.0]])
uniform = resample_polyline_by_arclength(points, count=20)
print(polyline_length(points), uniform.shape)
```

Create and sample a normalized heading-rate trajectory:

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

`u` is normalized progress in `[0, 1]`. Heading-rate values are degrees per unit progress. Geometric curvature is signed inverse length and is exposed separately by `planarcurve.trajectory.curvature`; for a trajectory of length `L`, it equals `radians(heading_rate) / L`.

B-spline operations require SciPy:

```python
import numpy as np
from planarcurve.spline import BSplineCurve

curve = BSplineCurve(
    knots=(0, 0, 0, 0, 0.5, 1, 1, 1, 1),
    control_points=((0, 0), (1, 2), (2, 3), (3, 2), (4, 0)),
)
sample = curve.sample_by_arclength(64)
path_segments = curve.to_cubic_segments(tolerance=1e-7)
```

SVG path parsing and conversion require FontTools. Fitting APIs live in `planarcurve.fitting`; see `examples/fit_polyline_to_spline.py` and `examples/fit_shared_trajectory.py` for complete workflows.

## Concepts and numerical behavior

- A trajectory's heading-rate profile is a function of normalized progress. Supported linear and cubic-Hermite profiles use an exact piecewise antiderivative for heading and total turn.
- Trajectory positions integrate the heading-derived tangent numerically. Increasing the evaluation or integration resolution improves the position approximation.
- Arc-length resampling approximates cumulative distance from sampled geometry. It is useful for uniform output spacing, but its accuracy depends on the sampling resolution.
- `trajectory_to_cubic_path` adaptively subdivides a trajectory and checks the requested positional error at quarter, midpoint, and three-quarter samples of each piece. The tolerance is a sampled approximation criterion, not a formal continuous error bound.
- `Affine2D` applies its linear transform and then translation. `Similarity2D` represents uniform scale, rotation, and translation.

## Optional dependencies

The package core imports with NumPy only. `planarcurve.spline` uses SciPy, as do the fitting modules. `planarcurve.svg` uses FontTools. Installing the core package does not import these optional modules.

## API stability

The package is pre-release and follows a `0.x` compatibility policy. Public names and numerical behavior may change between minor releases. See the API docstrings and focused tests for current contracts.

## Examples

Install all optional features and run the examples from a working directory of your choice:

```bash
python -m pip install 'planarcurve[all]'
python examples/curvature_gallery.py
python examples/spline_to_bezier.py
python examples/svg_pipeline.py
python examples/fit_polyline_to_spline.py
python examples/fit_shared_trajectory.py
```

The gallery and fitting examples write SVG files to the current directory. See [examples/README.md](examples/README.md) for details.

## Provenance

The generic numerical geometry and fitting functionality was extracted and adapted from InkGrammar. `planarcurve` uses neutral geometry types and does not include handwriting, glyph, dataset, Pydantic, or application orchestration concepts. The project is licensed under Apache-2.0.
