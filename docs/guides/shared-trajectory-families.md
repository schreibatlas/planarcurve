# Shared trajectory families

PlanarCurve can describe and fit reusable directed geometric trajectory families. It learns and compares geometric trajectory families. It does not attach semantic meaning to them.

The data flow is:

```text
ordered points
    ↓
directed normalized descriptor
    ↓
Trajectory2D candidate
    ↓
TrajectoryFamilyModel
```

## Directed descriptors

`planarcurve.trajectory_descriptor(points, *, samples=64, smoothing_window=5)` resamples an ordered polyline by chord length, translates its first sample to the origin, divides by arc length, and rotates its starting tangent to +x. It never reverses the points. `normalized_xy`, `tangent_xy`, and `heading_rate` are read-only arrays with `samples` rows. Heading rate is measured in degrees per normalized arc-length unit.

The scalar descriptor fields include signed total turn, endpoint displacement divided by arc length, and signed turn energy. Turn energy is the sum of each sampled heading increment times its absolute value, in square degrees. Use `reverse_descriptor` to request the opposite traversal explicitly. `descriptor_distance` compares equal-sized descriptors with a dimensionless weighted RMS; its optional weight mapping accepts descriptor field names and defaults to equal weights.

```python
from planarcurve import descriptor_distance, trajectory_descriptor

first = trajectory_descriptor(points_a)
second = trajectory_descriptor(points_b)
print(descriptor_distance(first, second))
```

Descriptors provide a fast geometric comparison. Their distances are not probabilities or handwriting classifications.

## Public profile parameterization

`planarcurve.fitting.HeadingRateParameterization` is the public numeric codec for profile parameters. Version 1 encodes ordered change-position logits followed by bounded heading-rate logits. Its `version` class attribute is `1`. The interpolation kind is configured on the codec, rather than encoded as a number, and must match the profile passed to `encode`.

```python
from planarcurve.fitting import HeadingRateParameterization

codec = HeadingRateParameterization(
    change_count=len(profile.changes),
    min_change_spacing=0.02,
    max_heading_rate_deg_per_u=36000.0,
    interpolation=profile.interpolation,
)
values = codec.encode(profile)
restored = codec.decode(values)
```

The codec validates ordered interior positions, configured spacing, finite values, and symmetric rate bounds. Downstream packages should use this API instead of duplicating parameter transforms.

## Fit one shared family

`fit_trajectory_family` fits one normalized shared profile and one positive-scale similarity transform for each observation. Input order is authoritative. Family fitting does not silently search reversed orientations.

```python
from planarcurve.fitting import TrajectoryFitOptions, fit_trajectory_family

result = fit_trajectory_family(
    occurrences,
    fit_options=TrajectoryFitOptions(change_count=2),
)
model = result.model
print(model.change_count, result.score)
for occurrence in result.occurrences:
    print(occurrence.transform, occurrence.position_rmse, occurrence.heading_rmse_deg)
```

`TrajectoryFamilyModel.parameter_scale` reports population standard deviations of the fitted rate offsets, in degrees per normalized progress unit. These are dispersion statistics, not confidence probabilities. Per-occurrence position RMSE and endpoint error are normalized by that observation's arc length. Heading and turn errors are in degrees. The robust cost uses the configured fit residual weights and robust scale. `accepted` flags fits whose robust cost exceeds the threshold `median + max(3 × 1.4826 × MAD, 5% of max(median, 1e-8))`, where MAD is the median absolute deviation. Optimizer convergence and evaluation count are explicit fields.

### Bounded occurrence variation

Pass `TrajectoryVariationPolicy` to enable small per-occurrence rate offsets. Each knot offset is bounded by `rate_sigma` and regularized at that scale. The profile knot positions and model order remain shared, and each occurrence returns its rate-deviation tuple. Arbitrary occurrence-specific splines are not fitted. Change-position offsets are not implemented and are rejected if requested.

```python
from planarcurve.fitting import TrajectoryVariationPolicy

result = fit_trajectory_family(
    occurrences,
    fit_options=fit_options,
    variation_policy=TrajectoryVariationPolicy(rate_sigma=25.0),
)
```

## Select model order

`select_shared_trajectory_model` fits every requested `change_count` and returns the selected model with decomposed costs. The default candidates are `(0, 1, 2, 3, 4)`. Its score is training robust cost, plus complexity cost, plus optional held-out validation cost.

The MVP complexity term is:

```text
complexity_weight × parameter_count × log1p(1 / parameter_precision)
parameter_count = (2 × change_count + 2) + (4 × training occurrences)
```

`parameter_precision` is a dimensionless resolution factor in `(0, 1]`; the default is `1e-3`. This is an explicit configurable regularizer, not a universal MDL interpretation. Validation splits whole occurrences deterministically from `seed`. Fewer than three observations or a zero validation fraction produces no held-out cost.

```python
from planarcurve.fitting import (
    TrajectoryModelSelectionOptions,
    select_shared_trajectory_model,
)

selected = select_shared_trajectory_model(
    occurrences,
    fit_options=fit_options,
    selection_options=TrajectoryModelSelectionOptions(seed=7),
)
print(
    selected.model.change_count,
    selected.training_cost,
    selected.complexity_cost,
    selected.validation_cost,
    selected.score,
)
```

## Synthetic examples

Run the standalone examples from the repository root:

```bash
python examples/fit_shared_trajectory_family.py
python examples/select_trajectory_complexity.py
```

They generate all points in code and require no application-specific assets. PlanarCurve provides geometry and fitting mathematics only. It does not provide gesture, glyph, OCR, handwriting, or other semantic labels.
