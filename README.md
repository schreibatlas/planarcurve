# planarcurve

`planarcurve` is a small numerical Python library for planar curves and trajectories.
It is extracted from the generic geometry kernel in InkGrammar and intentionally has no
handwriting, glyph, optimizer, Pydantic, or dataset concepts.

## Status

MVP / pre-release. The first public release is intended to be `0.1.0`.

## Packaging

The repository uses a **flat package layout** (no `src/` directory) and dynamic VCS
versioning through `setuptools_scm`:

```text
planarcurve/
├── planarcurve/
│   ├── __init__.py
│   ├── bezier.py
│   ├── curvature.py
│   ├── polyline.py
│   ├── spline.py
│   ├── svg.py
│   ├── transforms.py
│   └── types.py
├── tests/
└── pyproject.toml
```

A Git tag such as `v0.1.0` produces version `0.1.0`. Source trees without VCS metadata
fall back to `0.1.0.dev0` for buildability.

## Install

```bash
pip install planarcurve
```

Optional modules:

```bash
pip install "planarcurve[spline]"
pip install "planarcurve[svg]"
```

For development:

```bash
pip install -e ".[dev]"
pytest
```

## Curvature trajectory

```python
from planarcurve.curvature import (
    CurvatureChange,
    CurvatureProfile,
    CurvatureTrajectory,
)

trajectory = CurvatureTrajectory(
    length=1.0,
    start_angle_deg=25.0,
    profile=CurvatureProfile(
        interpolation="cubic-hermite",
        start_rate_deg=40.0,
        changes=(
            CurvatureChange(0.25, 260.0),
            CurvatureChange(0.55, 200.0),
        ),
        end_rate_deg=10.0,
    ),
)

sample = trajectory.evaluate(count=257)
print(sample.xy[-1], sample.end_pose.heading_deg)
```

The public angle contract is degrees, matching InkGrammar during extraction.

## B-spline

```python
import numpy as np
from planarcurve.spline import BSplineCurve

curve = BSplineCurve(
    knots=(0, 0, 0, 0, 0.5, 1, 1, 1, 1),
    control_points=((0, 0), (1, 2), (2, 3), (3, 2), (4, 0)),
)

sample = curve.evaluate(np.linspace(0, 1, 64))
uniform = curve.sample_by_arclength(64)
segments = curve.to_cubic_segments()
```

## Dependency boundary

The core package requires NumPy only. `planarcurve.spline` requires SciPy and
`planarcurve.svg` requires FontTools. The package does not import InkGrammar.

## Provenance

The initial numerical algorithms are extracted/adapted from InkGrammar's generic curve
and trajectory code, preserving its numerical behavior while replacing application
models with neutral frozen dataclasses. Licensed under Apache-2.0.
