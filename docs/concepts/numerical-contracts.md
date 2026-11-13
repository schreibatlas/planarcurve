# Numerical contracts

The API distinguishes analytic geometry from operations that deliberately approximate a continuous curve.

## Exact or analytic operations

| Operation | Contract |
| --- | --- |
| Polyline length and cumulative arc length | Sum of supplied segment chord lengths, up to floating-point arithmetic. |
| Bezier evaluation, derivatives, tangent, and curvature | Analytic formulas evaluated in floating-point arithmetic. |
| Heading-rate interpolation and its piecewise antiderivative | Analytic for the selected linear or internally shape-preserving cubic-Hermite profile. |
| Theoretical trajectory arc length | Exactly `length * u`, up to floating-point arithmetic, independent of the position integrator. |
| B-spline evaluation and derivatives | Spline evaluation through SciPy; arc-length sampling remains an approximation. |

## Numerical approximations

| Operation | Approximation |
| --- | --- |
| Trajectory positions | Adaptive numerical integration of the heading-derived tangent. |
| Bezier flattening | Adaptive subdivision using control-point distances from the endpoint chord as a flatness test. |
| Bezier length | Adaptive approximation using chord and control-polygon length estimates. |
| B-spline arc-length sampling | Dense parameter sampling, cumulative chord lengths, and interpolation. |
| Trajectory-to-cubic conversion | Adaptive subdivision with error checks at quarter, midpoint, and three-quarter parameters. This is not a global continuous error proof. |
| SVG curve length and endpoint tangent estimates | Sampled according to `samples_per_curve`. |

## Degenerate geometry

- A zero-length polyline cannot be normalized by arc length or resampled.
- Bezier and spline tangent normalization returns zero vectors where the derivative is degenerate; their curvature routines likewise return zero when speed is too small for the curvature formula.
- Polyline projection tolerates duplicate vertices and zero-length segments.
- A singular `Affine2D` can be applied, but `inverse()` raises because no inverse exists.

## Projection complexity

`project_points_to_polyline()` compares every query with every segment. With `M` query points and `N` segments, the current implementation takes `O(M*N)` time and uses `O(M*N)` temporary memory. Large query/segment combinations can therefore be expensive.

## Mutability

Frozen dataclasses prevent reassignment of their fields. They do not recursively make embedded NumPy arrays immutable; arrays stored in result objects remain writable.
