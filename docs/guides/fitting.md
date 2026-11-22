# Fitting

Fitting is optional and requires SciPy. Install it with `python -m pip install 'planarcurve[fit]'`. The package provides high-level polyline reconstruction, low-level shared B-spline control-point fitting, and normalized heading-rate trajectory fitting with occurrence placement transforms.

## Reconstruct a polyline as a regularized spline

`fit_polyline_spline()` accepts ordered 2D samples and optional nonnegative per-sample weights. It removes consecutive duplicate points, parameterizes observations by normalized cumulative chord length, normalizes coordinates by total polyline length, and tries clamped cubic bases seeded along the polyline. It returns an ordinary `BSplineCurve`, so the existing analytic tangent and curvature evaluation remains authoritative.

```python
import numpy as np

from planarcurve.fitting import (
    PolylineSplineFitOptions,
    fit_polyline_spline,
)

# `points` are ordered noisy samples and `weights` align with those samples.
fit = fit_polyline_spline(
    points,
    weights=weights,
    options=PolylineSplineFitOptions(
        min_control_points=4,
        max_control_points=16,
        target_rms_error_fraction=0.002,
        target_p95_error_fraction=0.004,
    ),
)
samples = fit.curve.sample_by_arclength(101)
assert samples.tangent is not None
assert samples.curvature is not None
heading = np.arctan2(samples.tangent[:, 1], samples.tangent[:, 0])
curvature = samples.curvature
# For normalized arc-length progress q, d(theta)/dq = curvature * total_length.
dtheta_dq = curvature * fit.diagnostics.normalized_scale
```

Residual diagnostics are normalized by `normalized_scale`, the original polyline's total arc length. Multiply a residual metric by this scale to express it in input coordinate units. The geometric acceptance gate uses unweighted residuals; `weighted_residual_rms` is reported separately when weights are supplied. Zero weights remove samples from the data term, and the endpoint observations remain part of the polyline parameterization and basis seed.

Candidates are considered in increasing control count. The smallest model meeting both RMS and 95th-percentile targets is selected. If none meets both, the minimum score is used: RMS divided by its target plus p95 divided by its target, plus `complexity_weight * control_count / max_control_points`, plus `roughness_weight * normalized_control_roughness`. The result diagnostics expose each candidate's metrics and score, and `met_target_error` reports whether the selected model passed both targets.

The generic endpoint policy is anchored. Pass `SplineEndpointPolicy(start="locked", end="locked")` when both graph-node estimates must remain fixed, or choose `free` or `anchored` independently. The [regularized polyline example](https://github.com/schreibatlas/planarcurve/blob/main/examples/fit_regularized_polyline.py) fits noisy samples of a known cubic curve and reports analytic heading, curvature, and turn rate by normalized progress.

## Stable trajectory fitting imports

The stable trajectory-fitting API is exported from `planarcurve.fitting`:

```python
from planarcurve.fitting import (
    SharedTrajectoryFitResult,
    TrajectoryFitOptions,
    TrajectoryFitResult,
    TrajectoryFitSeed,
    TrajectoryRefinementOptions,
    TrajectoryRefinementResult,
    TrajectoryRefinementStep,
    fit_shared_trajectory_profile,
    fit_trajectory,
    refine_trajectory_model,
)
```

These imports require SciPy. Install it with `python -m pip install 'planarcurve[fit]'`. Low-level initialization helpers remain in `planarcurve.fitting.trajectory` and are not part of this package-level export.

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

| Option            | Effect                                                                                                   |
| ----------------- | -------------------------------------------------------------------------------------------------------- |
| `smoothness`      | Weight for second differences of adjacent control points. Larger values penalize rough control polygons. |
| `seed_weight`     | Regularization toward the supplied seed control points.                                                  |
| `robust_scale`    | Scale for the IRLS residual weights; large residuals are downweighted relative to this scale.            |
| `irls_iterations` | Number of iteratively reweighted least-squares solves.                                                   |
| `endpoint_policy` | Independent start/end endpoint policies, each `free`, `locked`, or `anchored`.                           |
| `endpoint_sigma`  | Scale of the soft endpoint anchor residual.                                                              |

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

`TrajectoryFitSeed` warm-starts a single-curve fit from a canonical trajectory and its `Similarity2D` placement:

```python
from planarcurve.fitting import TrajectoryFitSeed, fit_trajectory

seed = TrajectoryFitSeed(previous_fit.trajectory, previous_fit.transform)
fit = fit_trajectory(points, options, initial=seed)
```

The seed's heading-rate change count must equal `options.change_count`. The seed is the first optimizer attempt; configured restarts still run around it. Passing `initial=None` retains the existing initialization behavior.

### Residual units and weights

Position residuals are divided by the observed occurrence length. Heading residuals are divided by `heading_sigma_deg`; total-turn residuals by `turn_sigma_deg`; and second-difference heading-rate smoothness residuals by `smoothness_sigma_deg_per_u`. The configured weights are dimensionless multipliers on those normalized residual groups. Reported position metrics remain in the input coordinate units.

`TrajectoryFitOptions` groups related controls:

| Group                       | Options                                                                                                               | Role                                                                                       |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Model complexity            | `change_count`, `min_change_spacing`, `max_heading_rate_deg_per_u`                                                    | Number and spacing of profile changes, plus the allowed rate range.                        |
| Sampling and initialization | `samples`, `smoothing_window`                                                                                         | Resampling density and smoothing used to estimate initial heading information.             |
| Optimization                | `restarts`, `seed`, `robust_scale`, `max_nfev`                                                                        | Repeatability, robust least-squares loss, and evaluation budget.                           |
| Residual weights            | `position_weight`, `tangent_weight`, `endpoint_weight`, `endpoint_heading_weight`, `turn_weight`, `smoothness_weight` | Relative influence of position, heading/tangent, endpoint, turn, and smoothness residuals. |
| Residual scales             | `heading_sigma_deg`, `turn_sigma_deg`, `smoothness_sigma_deg_per_u`                                                   | Units/scales used to normalize angular and heading-rate residuals.                         |

### Inspect fit results

`TrajectoryFitResult` reports the fitted trajectory, its `Similarity2D`, optimizer `success` and `cost`, and these metrics:

- `median_position_error` and `p95_position_error`, in input coordinate units;
- `endpoint_error`, the larger positional error at either endpoint;
- `endpoint_tangent_error_deg`, the larger endpoint heading error in degrees;
- `total_turn_error_deg`, the absolute signed-turn mismatch in degrees.

The shared-profile result reports aggregate median and 95th-percentile position errors plus `max_total_turn_error_deg` across occurrences. A result object alone does not imply convergence. Callers should inspect `.success`, `.cost`, and the metrics.

The [shared trajectory example](https://github.com/schreibatlas/planarcurve/blob/main/examples/fit_shared_trajectory.py) synthesizes several scaled and rotated occurrences, fits one shared profile, and reports each placement transform.


### Progressively refine model complexity

`refine_trajectory_model()` requires `fit_options.change_count == 0`. It fits that base model, uses normalized position and heading residuals to propose legal smooth heading-rate knots, warm-starts child fits, and accepts the best child only when both its absolute and relative regularized-score gains meet their thresholds. The complexity term uses the same `parameter_count * complexity_weight * log1p(1 / parameter_precision)` formula as independent model selection.

`TrajectoryRefinementResult.steps` includes the base fit and every proposed child. Each step reports fit and complexity costs, score, gain, acceptance, and a stable reason. `optimizer_evaluations` sums SciPy evaluations across all attempted fits. Change points are smooth steering controls, not geometric corners.

Model complexity is the number of interior heading-rate changes. It is distinct from `samples`, which controls fit resampling density, and `residual_samples`, which controls the grid used to locate candidate changes. Increasing either sampling count does not force a more complex model. Run [`refine_trajectory_complexity.py`](https://github.com/schreibatlas/planarcurve/blob/main/examples/refine_trajectory_complexity.py) for a synthetic S-curve trace.
