# planarcurve

`planarcurve` provides numerical building blocks for 2D polylines, Bezier curves, normalized heading-rate trajectories, transforms, cubic B-splines, fitting, and SVG paths. The core uses NumPy; SciPy and FontTools integrations are optional.

## Choose a module

| Need | Module | Optional dependency |
| --- | --- | --- |
| Polyline lengths, resampling, projection, and scores | `planarcurve.polyline` | None |
| Bezier evaluation, derivatives, flattening, and length | `planarcurve.bezier` and `planarcurve.types` | None |
| Normalized heading-rate trajectories | `planarcurve.trajectory` | None |
| Affine and similarity transforms | `planarcurve.transforms` | None |
| Cubic B-spline evaluation and conversion | `planarcurve.spline` | SciPy |
| Spline and trajectory fitting | `planarcurve.fitting` | SciPy |
| SVG path analysis and adapters | `planarcurve.svg` | FontTools |

## Quick example

Resample a piecewise-linear curve at equal chord-length intervals:

```python
import numpy as np

from planarcurve.polyline import (
    cumulative_arclength,
    normalized_arclength,
    polyline_length,
    resample_polyline_by_arclength,
)

points = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 0.0]])
length = polyline_length(points)
arc = cumulative_arclength(points)
progress = normalized_arclength(points)
resampled = resample_polyline_by_arclength(points, count=5)
print(f"length: {length:.3f}")
print(resampled)
```

Start with the [quickstart](quickstart.md), then choose from [concepts](concepts/index.md), [guides](guides/index.md), [examples](examples.md), or the [API reference](api/index.md). See [installation](installation.md) for optional dependencies and [development](development.md) for local checks. Read the [changelog](changelogs.md) for release history.

:::{admonition} Pre-release status
:class: warning

`planarcurve` is a pre-1.0 `0.x` project. Public names and numerical behavior may change between minor releases.
:::

```{toctree}
:maxdepth: 2
:hidden:

installation
quickstart
concepts/index
guides/index
examples
api/index
development
changelogs
```
