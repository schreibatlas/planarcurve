# `planarcurve.fitting`

Optional fitting utilities require SciPy. Install with `python -m pip install 'planarcurve[fit]'`.

## Stable public fitting API

The package exports `fit_polyline_spline()` with `PolylineSplineFitOptions`, `PolylineSplineFitResult`, `PolylineSplineFitDiagnostics`, and `PolylineSplineCandidateScore`, in addition to spline and trajectory fitting APIs such as `HeadingRateParameterization`, `TrajectoryFamilyModel`, `TrajectoryModelSelectionOptions`, and `fit_trajectory_family()`. These fitting APIs require SciPy. Install the optional extra with `python -m pip install 'planarcurve[fit]'`. Lower-level initialization helpers remain in their implementation modules.

```{automodule} planarcurve.fitting
:members:
:undoc-members:
:show-inheritance:
:member-order: bysource
```

## High-level polyline reconstruction

`fit_polyline_spline(points, *, weights=None, options=None)` returns an ordinary `BSplineCurve` and a diagnostics record containing the selected control count, normalized geometric residuals, optional weighted RMS, target status, and per-candidate scores. `PolylineSplineFitOptions` controls candidate counts, residual fractions of total arc length, smoothness, robust fitting, and the existing `SplineEndpointPolicy`. The default endpoint policy is anchored, and the lowest-control-count candidate satisfying both RMS and p95 targets wins.

The existing `CurveEvaluation` supplies analytic unit tangents and signed curvature. For normalized arc-length progress `q`, the heading rate is `d(theta)/dq = curvature * total_arc_length`; no separate curvature approximation is needed. See the [fitting guide](../guides/fitting.md#reconstruct-a-polyline-as-a-regularized-spline) for a worked example and metric definitions.

## Endpoint policy

```{automodule} planarcurve.fitting.policy
:members:
:undoc-members:
:show-inheritance:
:member-order: bysource
```

## Optimization parameterization utilities

```{automodule} planarcurve.fitting.parameters
:members:
:undoc-members:
:show-inheritance:
:member-order: bysource
```

## Spline fitting

```{automodule} planarcurve.fitting.spline
:members:
:undoc-members:
:show-inheritance:
:member-order: bysource
```

## Trajectory fitting

```{automodule} planarcurve.fitting.trajectory
:members:
:undoc-members:
:show-inheritance:
:member-order: bysource
```


## Shared trajectory family fitting

```{automodule} planarcurve.fitting.family
:members:
:undoc-members:
:show-inheritance:
:member-order: bysource
```

## Trajectory model selection

```{automodule} planarcurve.fitting.model_selection
:members:
:undoc-members:
:show-inheritance:
:member-order: bysource
```
