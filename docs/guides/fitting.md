# Fitting

Fitting is optional and requires SciPy. Install it with `python -m pip install 'planarcurve[fit]'`. The package provides two separate workflows: fitting shared B-spline control points, and fitting a normalized heading-rate trajectory with occurrence placement transforms.

## Fit spline control points

A `SplineObservation` pairs parameter values `u` with observed 2D points `xy`. Optional nonnegative per-point weights control each observation's contribution. Multiple observations may be combined to solve one shared control-point set against the same spline basis.

```python
import numpy as np

from planarcurve.fitting.spline import (
    SplineFitOptions,
    SplineObservation,
    fit_spline_control_points,
)
from planarcurve.spline import BSplineCurve

knots = (0.0,) * 4 + (0.5,) + (1.0,) * 4
seed = BSplineCurve(knots, ((0, 0), (1, 2), (2, 2), (3, 0), (4, 0)))
u = np.linspace(0.0, 1.0, 41)
observed = seed.evaluate(u, derivatives=False).xy
fit = fit_spline_control_points(
    seed,
    [SplineObservation(u, observed)],
    options=SplineFitOptions(smoothness=0.02, seed_weight=0.01),
)
print(fit.residual_rms, fit.iterations)
```

`SplineFitOptions` controls:

| Option | Effect |
| --- | --- |
| `smoothness` | Weight for second differences of adjacent control points. Larger values penalize rough control polygons. |
| `seed_weight` | Regularization toward the supplied seed control points. |
| `robust_scale` | Scale for the IRLS residual weights; large residuals are downweighted relative to this scale. |
| `irls_iterations` | Number of iteratively reweighted least-squares solves. |
| `endpoint_policy` | Independent start/end endpoint policies, each `free`, `locked`, or `anchored`. |
| `endpoint_sigma` | Scale of the soft endpoint anchor residual. |

An optional `seed_control_points` argument replaces the basis controls as the regularization seed. Endpoint policies mean:

- `free`: the endpoint control is solved as an ordinary free control;
- `locked`: the endpoint control remains exactly equal to the seed endpoint;
- `anchored`: the endpoint is free but softly regularized toward the seed endpoint using `endpoint_sigma`.

The returned `SplineFitResult.curve` is the fitted curve. `residual_rms` is the root-mean-square Euclidean residual over observed points, and `iterations` reports the configured IRLS iteration count. The [polyline-to-spline example](https://github.com/schreibatlas/planarcurve/blob/main/examples/fit_polyline_to_spline.py) demonstrates noisy observations and outliers.

## Fit a normalized trajectory

Trajectory fitting represents shape canonically:

- the profile trajectory has geometric length 1;
- its starting heading is 0 degrees;
- each observed occurrence gets a separate positive-scale `Similarity2D` transform.

This separates normalized shape from occurrence scale, rotation, and translation. `fit_trajectory()` estimates one profile and one transform. `fit_shared_trajectory_profile()` estimates one shared profile and one transform for each input occurrence.

### Residual units and weights

Position residuals are divided by the observed occurrence length. Heading residuals are divided by `heading_sigma_deg`; total-turn residuals by `turn_sigma_deg`; and second-difference heading-rate smoothness residuals by `smoothness_sigma_deg_per_u`. The configured weights are dimensionless multipliers on those normalized residual groups. Reported position metrics remain in the input coordinate units.

`TrajectoryFitOptions` groups related controls:

| Group | Options | Role |
| --- | --- | --- |
| Model complexity | `change_count`, `min_change_spacing`, `max_heading_rate_deg_per_u` | Number and spacing of profile changes, plus the allowed rate range. |
| Sampling and initialization | `samples`, `smoothing_window` | Resampling density and smoothing used to estimate initial heading information. |
| Optimization | `restarts`, `seed`, `robust_scale`, `max_nfev` | Repeatability, robust least-squares loss, and evaluation budget. |
| Residual weights | `position_weight`, `tangent_weight`, `endpoint_weight`, `endpoint_heading_weight`, `turn_weight`, `smoothness_weight` | Relative influence of position, heading/tangent, endpoint, turn, and smoothness residuals. |
| Residual scales | `heading_sigma_deg`, `turn_sigma_deg`, `smoothness_sigma_deg_per_u` | Units/scales used to normalize angular and heading-rate residuals. |

### Inspect fit results

`TrajectoryFitResult` reports the fitted trajectory, its `Similarity2D`, optimizer `success` and `cost`, and these metrics:

- `median_position_error` and `p95_position_error`, in input coordinate units;
- `endpoint_error`, the larger positional error at either endpoint;
- `endpoint_tangent_error_deg`, the larger endpoint heading error in degrees;
- `total_turn_error_deg`, the absolute signed-turn mismatch in degrees.

The shared-profile result reports aggregate median and 95th-percentile position errors plus `max_total_turn_error_deg` across occurrences. A result object alone does not imply convergence. Callers should inspect `.success`, `.cost`, and the metrics.

The [shared trajectory example](https://github.com/schreibatlas/planarcurve/blob/main/examples/fit_shared_trajectory.py) synthesizes several scaled and rotated occurrences, fits one shared profile, and reports each placement transform.
