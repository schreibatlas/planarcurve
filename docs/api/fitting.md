# `planarcurve.fitting`

Optional fitting utilities require SciPy. Install with `python -m pip install 'planarcurve[fit]'`.

## Stable public fitting API

The package exports `SplineEndpointPolicy`, `SharedTrajectoryFitResult`, `TrajectoryFitOptions`, `TrajectoryFitResult`, `fit_shared_trajectory_profile`, and `fit_trajectory` from `planarcurve.fitting`. These fitting APIs require SciPy. Install the optional extra with `python -m pip install 'planarcurve[fit]'`. Lower-level initialization helpers remain in their implementation modules.

```{automodule} planarcurve.fitting
:members:
:undoc-members:
:show-inheritance:
:member-order: bysource
```

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
