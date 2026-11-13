# Trajectories

This guide adapts [`trajectory_gallery.py`](https://github.com/schreibatlas/planarcurve/blob/main/examples/trajectory_gallery.py) and adds evaluation and cubic conversion. A `Trajectory2D` starts at `(0, 0)` and describes heading rate over normalized progress `u ∈ [0, 1]`.

## Straight, constant-rate, and changing profiles

```python
import numpy as np

from planarcurve.trajectory import (
    HeadingRateChange,
    HeadingRateProfile,
    Trajectory2D,
    evaluate_trajectory_at,
    trajectory_to_cubic_path,
    trajectory_total_turn,
)

straight = HeadingRateProfile(0.0, 0.0)
arc = HeadingRateProfile(90.0, 90.0, interpolation="linear")
changing = HeadingRateProfile(
    0.0,
    0.0,
    changes=(HeadingRateChange(0.30, 240.0), HeadingRateChange(0.70, -240.0)),
)
linear = HeadingRateProfile(0.0, 0.0, changes=changing.changes, interpolation="linear")
hermite = HeadingRateProfile(0.0, 0.0, changes=changing.changes, interpolation="cubic-hermite")
trajectory = Trajectory2D(length=140.0, start_heading_deg=0.0, heading_rate=hermite)
sample = trajectory.evaluate(count=513)
```

The default interpolation is `"cubic-hermite"`. It computes shape-preserving slopes from the rates. `"linear"` connects the rate knots with straight segments. In both modes, the heading-rate knots must be strictly ordered interior positions.

## Read a sample and endpoint

`TrajectorySample` provides arrays for `u`, `xy`, `heading_deg`, `heading_rate_deg_per_u`, `tangent`, and `arc_length`, along with `end_pose`. Position samples are numerical integrations of the heading-derived tangent. Heading is obtained from the exact piecewise antiderivative, and theoretical arc length is `trajectory.length * u`.

```python
assert sample.xy.shape == (513, 2)
assert np.allclose(sample.arc_length, trajectory.length * sample.u)
print(sample.xy[-1])
print(sample.end_pose.position, sample.end_pose.heading_deg)
print(sample.end_pose.heading_rate_deg_per_u)
print(trajectory_total_turn(trajectory))  # signed degrees
```

Heading and heading rate are in degrees and degrees per unit `u`, respectively. Curvature is signed and measured in inverse coordinate length. `evaluate_trajectory_at(trajectory, values)` accepts arbitrary finite `u` values in `[0, 1]` and returns `(xy, heading_deg)` arrays.

## Convert to a cubic path

```python
path = trajectory_to_cubic_path(trajectory, max_error=1e-3)
beziers = path.to_beziers()
print(len(path.segments), len(beziers))
```

Conversion adaptively subdivides and checks the trajectory against each cubic at quarter, midpoint, and three-quarter parameters. `max_error` is a sampled acceptance threshold, not a formal global error bound. See [trajectory concepts](../concepts/trajectories.md) for the integration model and parameter semantics.
