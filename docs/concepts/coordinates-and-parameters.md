# Coordinates and parameters

## Points

The public geometry is explicitly two-dimensional. A point-like value must contain exactly two finite coordinates. Functions accepting a collection of points generally use an `(N, 2)` array. Numerical array results are floating-point NumPy arrays.

Count, order, and recursion-depth arguments require integer values. Floating-point lookalikes and booleans are rejected rather than silently converted.

## Different parameter meanings

Several APIs use parameters in `[0, 1]`, but they do not all represent the same thing.

### Bezier parameter `t`

For a quadratic or cubic Bezier, `t ∈ [0, 1]` is the polynomial curve parameter. Equal increments of `t` generally do not correspond to equal distances along the curve.

### B-spline parameter `u`

A supported `BSplineCurve` uses the clamped interval `u ∈ [0, 1]`. `evaluate(u)` evaluates at those parameter values. `sample_by_arclength()` instead approximates uniform geometric spacing using a dense parameter sample and interpolation.

### Trajectory progress `u`

For `Trajectory2D`, `u ∈ [0, 1]` means normalized progress. Speed with respect to `u` is constant and equals the trajectory's geometric `length`. Therefore its theoretical arc length is

```text
s(u) = length * u
```

The heading-rate profile `q(u)` is measured in degrees per unit `u`, not degrees per coordinate unit. Signed geometric curvature is

```text
κ(u) = radians(q(u)) / length
```

For example, doubling `length` while keeping the same profile halves the curvature while preserving the profile's heading change.

## Transform convention

`Affine2D` stores a row-major 2×2 linear map and a translation:

```text
[x']   [m00 m01] [x]   [dx]
[y'] = [m10 m11] [y] + [dy]
```

`apply()` transforms points. `apply_vectors()` applies only the linear component, without translation. Composition is ordered as function composition:

```python
from planarcurve.transforms import Affine2D

move = Affine2D.translation_only(3.0, 0.0)
scale = Affine2D.scale(2.0)
placed = move.compose(scale).apply([1.0, 0.0])  # [5.0, 0.0]
```

A singular affine map can transform points but has no inverse. `Similarity2D` is restricted to positive uniform scale, rotation in degrees, and translation.
