# Supported B-splines

`BSplineCurve` intentionally supports a narrow, predictable subset of B-spline geometry that can be converted to cubic Bezier spans. It is not a general wrapper for every SciPy `BSpline` configuration.

A valid curve has:

- degree exactly 3;
- at least four finite 2D control points;
- exactly `n_control_points + 4` knots;
- four leading zero knots and four trailing one knots;
- finite, nondecreasing knot values;
- simple, strictly increasing interior knots in `(0, 1)`.

Spline functionality is optional and requires SciPy. Install it with `python -m pip install 'planarcurve[spline]'`.

## Evaluation and differential geometry

`evaluate()` and `evaluate_bspline()` evaluate positions at normalized parameters `u ∈ [0, 1]`. Derivatives can be requested, and the result can include unit tangents and signed geometric curvature. An optional `Affine2D` is applied to positions and to the linear part of the derivatives, so tangent and curvature describe the transformed curve.

Set `derivatives=False` when only positions are needed. `CurveEvaluation` then returns `None` for tangent and curvature.

## Arc-length sampling

`sample_by_arclength()` approximates equal geometric spacing. It evaluates a dense parameter grid of `max(1024, count * 32)` points, accumulates chord lengths, interpolates target arc lengths back to parameter values, then evaluates the spline at those values. This is an approximation, not exact inversion of the spline's arc-length function. When supplied, the transform is included in the sampling distance.

## Cubic-path conversion

`spline_to_cubic_segments()` inserts interior knots as needed and represents each nonzero knot span as a cubic Bezier segment. It verifies the result at 33 uniformly spaced local parameter values per span against the requested tolerance. `to_cubic_path()` additionally stores the explicit start point in a `CubicPath`.
