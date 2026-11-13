# Working with polylines

A polyline is a finite array of shape `(N, 2)`. Its length is the sum of the Euclidean lengths between consecutive vertices. The operations below share a single example:

```python
import numpy as np

from planarcurve.polyline import (
    bounding_box,
    closest_point_on_polyline,
    cumulative_arclength,
    distance_to_polyline,
    mean_squared_turn_angle,
    normalized_arclength,
    polyline_length,
    project_points_to_polyline,
    resample_polyline_by_arclength,
    symmetric_polyline_distance,
    unit_tangents,
)

line = np.array([[0.0, 0.0], [2.0, 0.0], [3.0, 1.0], [5.0, 1.0]])
```

## Measure and resample

```python
arc = cumulative_arclength(line)  # distance from the first vertex
progress = normalized_arclength(line)  # cumulative distance divided by total
length = polyline_length(line)
even = resample_polyline_by_arclength(line, count=9)
tangents = unit_tangents(line)
bounds = bounding_box(line)  # xmin, ymin, xmax, ymax
```

The returned arc-length values are measured along the supplied piecewise-linear path, including repeated vertices. Normalized arc length and resampling require nonzero total length. Resampling removes consecutive near-duplicate positions before interpolation.

`unit_tangents()` estimates tangents from neighboring vertices. At locations where the estimate degenerates, it returns a zero vector rather than dividing by zero.

## Project queries

```python
queries = np.array([[2.5, 0.7], [4.0, 2.0]])
projection = project_points_to_polyline(queries, line)
nearest_points = projection.point
nearest_distances = distance_to_polyline(queries, line)
one_nearest = closest_point_on_polyline((2.5, 0.7), line)
```

`PolylineProjection` contains one result per query:

| Field | Meaning | Shape |
| --- | --- | --- |
| `point` | Nearest projected point | `(M, 2)` |
| `distance` | Euclidean distance to that point | `(M,)` |
| `segment_index` | Index of the winning segment in the original polyline | `(M,)` |
| `segment_t` | Clamped local parameter on the winning segment, from 0 to 1 | `(M,)` |
| `arc_length` | Distance from the polyline start to the projected point | `(M,)` |

`segment_t` is local to `segment_index`; `arc_length` is cumulative from the start of the polyline. Duplicate zero-length segments are tolerated.

Projection compares every query with every segment. Its time and temporary-memory costs are `O(M*N)` for `M` queries and `N` segments, so it can be expensive for large inputs.

## Compare shape and turning

```python
score = symmetric_polyline_distance(line, even, samples=64)
turn_cost = mean_squared_turn_angle(line, samples=32)
```

The symmetric distance samples both inputs uniformly by arc length and averages the two directions of nearest-segment distances. It is an approximate shape score in input coordinate units. The optional `samples` argument to `mean_squared_turn_angle()` first resamples by arc length. The result is the mean squared turn angle in radians squared and is scale-independent.
