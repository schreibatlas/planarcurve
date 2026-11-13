# SVG path adapters

`planarcurve.svg` uses FontTools. Install it with `python -m pip install 'planarcurve[svg]'`. This guide follows [`svg_pipeline.py`](https://github.com/schreibatlas/planarcurve/blob/main/examples/svg_pipeline.py).

## Analyze and sample one path

```python
import numpy as np

from planarcurve.svg import analyze_path, sample_path, transform_path
from planarcurve.transforms import Affine2D

path = "M 0 0 C 20 0 30 30 50 30 Q 70 30 80 10"
info = analyze_path(path, samples_per_curve=64)
samples = sample_path(path, samples_per_curve=64)
transform = Affine2D.from_scale_shear_translation(
    scale_x=1.5, scale_y=0.75, shear_x=0.2, dx=10.0, dy=-5.0
)
transformed = transform_path(path, transform)
```

`analyze_path()` returns aggregate `PathInfo`: start/end points, endpoint tangents, estimated length, bounding box, move count, aggregate closed status, and per-subpath closed flags. `samples_per_curve` controls the number of subdivisions used to sample each quadratic or cubic segment, and thus affects curve-length and endpoint-tangent estimates.

The adapter supports the ordinary move, line, cubic, quadratic, close, and end-path operations produced by FontTools' parser. It rejects unsupported recorded operations and special compound quadratic forms. It does not claim to implement every rule or command in the full SVG specification.

## Multiple subpaths

`sample_path()` is for exactly one subpath and raises `ValueError` if the parsed path contains multiple `moveTo` operations. Use `sample_subpaths()` to receive a tuple of independent arrays. No artificial connecting segment is inserted between subpaths.

For `PathInfo`:

- `closed` is true only when all subpaths are closed;
- `subpath_closed` records the closed flag for each subpath;
- `start` and `tangent_start` refer to the first subpath;
- `end` and `tangent_end` refer to the last subpath;
- `move_count` is the number of subpaths;
- `length` is the sum of sampled subpath chord lengths;
- `bbox` encloses the sampled points.

Endpoint tangents are unit vectors or `None` when no nonzero segment provides a direction.

## Other adapters

`transform_path()` transforms every supported command coordinate while preserving subpath commands. `path_without_initial_move()` returns the drawing commands after the initial move, for paths containing a single move operation. `cubic_bridge(p0, t0, p3, t3, tension)` constructs an SVG cubic command between endpoints using the supplied endpoint directions; omitted tangents default to `(1, 0)`. Its `tension` is measured in coordinate units.
