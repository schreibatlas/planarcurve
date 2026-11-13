# Planarcurve review and implementation brief

## Purpose of this document

This is a coding-agent brief for turning the current `planarcurve` extraction into a coherent standalone Python package rather than a copy of InkGrammar's geometry kernel.

The review was performed against the supplied reconstructed snapshots of both repositories. The supplied Codecrate unpackers were used in strict mode with machine-header verification and warning failure enabled.

Reconstruction commands used:

```bash
python3 -S context_planarcurve.unpack.py context_planarcurve.md \
  -o review_planarcurve \
  --check-machine-header --strict --fail-on-warning

python3 -S context_inkgrammar.unpack.py context_inkgrammar.md \
  -o review_inkgrammar \
  --check-machine-header --strict --fail-on-warning
```

Validation performed on the reconstructed `planarcurve` snapshot:

```text
pytest:       16 passed
compileall:  passed
curvature parity with InkGrammar fixture: exact
```

`ruff` and `python -m build` could not be run in this sandbox because those tools are not installed. The repository CI is configured to install them through the `dev` extra, so CI still needs to be the final packaging/lint authority.

---

# Executive verdict

## Is planarcurve useful outside InkGrammar?

**Yes, but it is currently more convincing as an extracted kernel than as a finished public geometry library.**

The standalone value is real:

- normalized 2D heading-rate trajectories;
- cubic B-spline evaluation, tangents, curvature, arc-length sampling, and cubic conversion;
- cubic Bezier evaluation;
- polyline length/resampling/tangent helpers;
- affine/similarity transforms;
- SVG path inspection, sampling, transformation, and bridging;
- NumPy-only core with SciPy and FontTools isolated into optional modules.

The most distinctive part is the compact normalized trajectory representation. The rest is a useful coherent layer over NumPy/SciPy/FontTools, especially if the package also owns fitting and correspondence utilities.

However, the current API still exposes several InkGrammar-specific assumptions. The most important work before `0.1.0` is not adding many more features; it is cleaning the mathematical contract and separating pure geometry from fitting policy.

## Recommended positioning

Use a description like:

> `planarcurve` is a small numerical toolkit for 2D curves: polylines, Beziers, normalized heading-rate trajectories, B-splines, arc-length sampling, fitting, transforms, and optional SVG adapters.

Avoid positioning it as a full computational-geometry package. It does not provide intersections, boolean operations, robust offset topology, polygon clipping, or a CAD kernel.

## Main recommendation

Before the first public release:

1. Fix trajectory integration semantics so point evaluation is not dependent on the query grid.
2. Remove InkGrammar validation/fitting policy from pure geometry objects.
3. Move generic spline fitting and trajectory fitting support from InkGrammar into planarcurve.
4. Make InkGrammar depend on planarcurve instead of keeping copied implementations.
5. Add examples and smoke-test them.
6. Clean naming around “curvature” versus normalized heading rate.

The extraction is technically sound. The next step should be **ownership consolidation**, not another parallel copy.

---

# 1. Current package review

## Current module map

| Module | Current responsibility | Dependency | Review |
|---|---|---:|---|
| `planarcurve.types` | point alias, cubic chain segment, cubic Bezier, spline endpoint policy | stdlib | partly pure, partly fitting policy |
| `planarcurve.transforms` | affine and similarity transforms | NumPy | good core, API can be expanded |
| `planarcurve.polyline` | length, cumulative arc, resampling, tangents, bbox | NumPy | good core; add projection/distance helpers |
| `planarcurve.bezier` | quadratic/cubic evaluation and cubic tangent | NumPy | useful but thin |
| `planarcurve.curvature` | normalized heading-rate trajectory model and integration | NumPy | distinctive feature; needs semantic cleanup |
| `planarcurve.spline` | cubic B-spline evaluation, curvature, sampling, conversion | SciPy | good optional module; fitting belongs nearby |
| `planarcurve.svg` | SVG path parsing/analysis/sampling/transforms | FontTools | useful adapter; needs round-trip semantics review |

The flat package layout and dynamic `setuptools_scm` versioning are appropriate and should remain.

## Dependency boundary

The current dependency boundary is good:

- core: NumPy only;
- B-spline module: SciPy only when imported;
- SVG module: FontTools only when imported;
- no InkGrammar/Pydantic/Pillow/Typer/YAML imports.

Preserve this rule. If fitting is extracted, use a SciPy-backed optional fitting module rather than adding SciPy to the mandatory base dependency unless there is a compelling reason.

---

# 2. Critical pre-0.1 findings

## P0: `evaluate_heading()` is not a true evaluator

File: `planarcurve/curvature.py`, `evaluate_heading`, approximately lines 181-195.

The function integrates heading rate by applying trapezoidal integration only over the supplied `u` array. Therefore the answer depends on what sample points the caller happened to request.

Example:

```python
import numpy as np
from planarcurve.curvature import (
    CurvatureChange,
    CurvatureProfile,
    CurvatureTrajectory,
    evaluate_heading,
)

profile = CurvatureProfile(
    start_rate_deg=0.0,
    changes=(CurvatureChange(0.5, 180.0),),
    end_rate_deg=0.0,
    interpolation="linear",
)
trajectory = CurvatureTrajectory(length=1.0, start_angle_deg=0.0, profile=profile)

print(evaluate_heading(trajectory, np.array([0.0, 1.0]))[-1])
```

Current result:

```text
0.0
```

The correct total turn is 90 degrees. The interior change at `u=0.5` is completely missed by the sparse query grid.

This is not just an accuracy issue. It violates normal expectations for a function named `evaluate_heading(trajectory, u)`.

### Required fix

Implement an exact antiderivative for the piecewise heading-rate profile.

For `linear` interpolation, integrate each linear span analytically.

For `cubic-hermite`, the profile is already represented in cubic Hermite basis. Integrate the basis analytically per span. With local span coordinate `t=(u-u0)/h`, the basis antiderivatives are straightforward polynomials:

```text
h00 =  2t^3 - 3t^2 + 1
H00 =  1/2 t^4 - t^3 + t

h10 =  t^3 - 2t^2 + t
H10 =  1/4 t^4 - 2/3 t^3 + 1/2 t^2

h01 = -2t^3 + 3t^2
H01 = -1/2 t^4 + t^3

h11 =  t^3 - t^2
H11 =  1/4 t^4 - 1/3 t^3
```

Remember the outer `du = h dt` factor and the existing `h * slope` factors in the Hermite representation.

Add a function such as:

```python
def integrated_heading_rate(profile: HeadingRateProfile, u: np.ndarray) -> np.ndarray:
    """Integral from 0 to u of d(theta_deg)/du."""
```

Then:

```python
heading = start_angle_deg + integrated_heading_rate(profile, u)
```

The evaluator should not require that `u[0] == 0`. Prefer allowing arbitrary finite `u` values in `[0, 1]`, ideally preserving input order.

### Acceptance tests

- sparse `[0, 1]` query sees interior knots;
- result at a knot equals dense-grid reference;
- result independent of evaluation grid;
- linear profiles match analytic piecewise-linear integrals;
- cubic-Hermite total turn is invariant to `count`.

---

## P0: `trajectory_total_turn()` should be exact and should not accept `count`

File: `planarcurve/curvature.py`, approximately lines 227-230.

The current API numerically integrates a known piecewise polynomial by sampling it on a uniform grid. This makes a basic invariant depend on an arbitrary resolution parameter.

A linear example with an interior knot at `u=0.43` produced:

```text
count=4      81.599347...
count=5      85.373317...
count=8      89.925811...
count=257    89.999587...
count=4097   89.999995...
```

The analytic value is 90 degrees.

Once the profile antiderivative exists, implement:

```python
def trajectory_total_turn(trajectory: Trajectory2D) -> float:
    return float(integrated_heading_rate(trajectory.profile, np.array([1.0]))[0])
```

Remove the numerical `count` parameter before public release.

---

## P0: core geometry contains InkGrammar fitting constraints

### Curvature spacing and rate bounds

File: `planarcurve/curvature.py`, classes `CurvatureChange` and `CurvatureProfile`.

Current constants:

```python
MIN_CHANGE_SPACING = 0.02
MAX_RATE_DEG = 36000.0
```

These are not mathematical requirements for a heading-rate curve. They are model/fitting/schema policy inherited from InkGrammar.

A generic geometry package should allow two valid knots that happen to be 0.005 apart. It should also not impose an arbitrary maximum derivative unless the limit is inherent in a specific public model.

### Recommended change

Pure profile validation should require only:

- finite `u` positions;
- positions strictly increasing;
- positions inside `(0, 1)`;
- finite rate values;
- supported interpolation mode.

Move minimum spacing and optimizer bounds into a fitting configuration or parameterization layer.

For example:

```python
@dataclass(frozen=True)
class TrajectoryFitBounds:
    min_change_spacing: float = 0.02
    max_heading_rate_deg: float = 36000.0
```

InkGrammar can preserve its existing policy by passing those values explicitly.

### B-spline endpoint policy is also fitting policy

`SplineEndpointPolicy` currently lives in `planarcurve.types`, and `BSplineCurve` stores it even though spline evaluation does not use it.

This is another representation/fitting mix.

Change:

```python
BSplineCurve(
    knots=...,
    control_points=...,
    degree=3,
)
```

and move:

```python
SplineEndpointPolicy(start="locked", end="locked")
```

into the spline fitting API.

This also makes the generic `solve_shared_spline_coefficients()` extraction cleaner.

---

## P0: clarify “curvature” terminology

The current profile stores:

```text
q(u) = d(theta_deg) / du
```

where `u` is normalized progress, not physical arc length.

That is **not geometric curvature** in the usual differential-geometry sense.

For a trajectory of length `L`, assuming `s = L*u`, geometric signed curvature is:

```text
kappa(u) = d(theta_rad)/ds
         = radians(q(u)) / L
```

External scientific users will reasonably expect `CurvatureProfile` values to have units `1/length`. The current degree-per-normalized-progress contract can be useful, but it should not be ambiguous.

### Preferred pre-0.1 cleanup

Rename the model to reflect what it actually stores:

```text
CurvatureChange        -> HeadingRateChange
CurvatureProfile       -> HeadingRateProfile
CurvatureTrajectory    -> Trajectory2D or HeadingRateTrajectory
heading_rate_deg       -> heading_rate_deg_per_u
```

Then expose true curvature explicitly:

```python
def curvature(trajectory, u) -> np.ndarray:
    return np.deg2rad(evaluate_heading_rate(trajectory.profile, u)) / trajectory.length
```

If the current names are retained, document the distinction very prominently and add a true-curvature accessor.

Because `0.1.0` has not been released yet, now is the lowest-cost time to fix naming.

---

## P0: make InkGrammar consume planarcurve

The current extraction leaves two implementations alive:

- `planarcurve.curvature` and `inkgrammar.fitting.curvature`;
- `planarcurve.spline` and `inkgrammar.fitting.spline`;
- `planarcurve.svg` and `inkgrammar.geometry`.

There is an optional parity script, but duplicated source will drift.

The correct dependency direction is:

```text
planarcurve  <--  inkgrammar
```

Never the reverse.

### Migration strategy

InkGrammar can keep Pydantic models and application schema. Add explicit adapters from those models to planarcurve's neutral dataclasses.

Example shape:

```python
def to_planar_trajectory(value: CurvatureStrokeGeometry) -> planarcurve.Trajectory2D:
    ...
```

Then replace InkGrammar numerical implementations with calls to planarcurve.

Similarly:

```python
def topology_transform_to_affine(value: TopologyTransform) -> Affine2D:
    return Affine2D.from_scale_shear_translation(...)
```

Do not make planarcurve know about Pydantic models.

### Why this matters

A standalone library is more credible when the original application becomes a real downstream consumer. It also makes the package boundary enforceable in CI.

---

# 3. Generic InkGrammar code that should move into planarcurve

## Move now: spline fitting

Source: `inkgrammar/fitting/spline_fit.py`.

The following code is already generic numerical curve functionality:

- `SplineObservation`;
- `_validated_observations`;
- sparse weighted design-matrix construction;
- IRLS robust weighting;
- `solve_shared_spline_coefficients`.

It depends on SciPy and the spline basis only. It does not fundamentally depend on handwriting.

This is an especially strong candidate because `planarcurve` already contains:

- `spline_design_matrix`;
- `control_roughness_matrix`;
- `BSplineCurve`;
- an endpoint policy that currently has no purpose other than fitting.

### Proposed API

Create `planarcurve/spline_fit.py` or `planarcurve/fitting/spline.py`.

Recommended public shape:

```python
@dataclass(frozen=True)
class SplineObservation:
    u: np.ndarray
    xy: np.ndarray
    weight: np.ndarray | None = None

@dataclass(frozen=True)
class SplineFitOptions:
    smoothness: float = 0.02
    seed_weight: float = 0.01
    robust_scale: float = 1.5
    irls_iterations: int = 3
    endpoint_policy: SplineEndpointPolicy = SplineEndpointPolicy()

@dataclass(frozen=True)
class SplineFitResult:
    curve: BSplineCurve
    residual_rms: float
    iterations: int


def fit_spline_control_points(
    basis: BSplineCurve,
    observations: Sequence[SplineObservation],
    *,
    seed_control_points: np.ndarray | None = None,
    options: SplineFitOptions = SplineFitOptions(),
) -> SplineFitResult:
    ...
```

Do not make users mutate a `BSplineCurve` whose endpoint fitting policy is hidden inside geometry.

### Why this improves external usefulness

This turns planarcurve from “evaluate an already-known spline” into “fit and evaluate a spline from data,” which is a much more complete standalone workflow.

---

## Move now: generic trajectory parameterization

Source: `inkgrammar/fitting/trajectory_parameterization.py`.

These functions are mathematically generic:

- `decode_change_positions`;
- `encode_change_positions`;
- `decode_bounded`;
- `encode_bounded`;
- log-length encoding.

The current rate and spacing defaults are InkGrammar-specific. Extract the algorithms but make limits/configuration explicit.

Proposed module:

```text
planarcurve/fitting/parameters.py
```

Suggested generic names:

```python
ordered_positions_from_logits(...)
ordered_positions_to_logits(...)
decode_interval(...)
encode_interval(...)
decode_positive(...)
encode_positive(...)
```

Trajectory-specific wrappers can live in the trajectory fitter.

---

## Move now: curvature/heading-rate trajectory fitting

Source: `inkgrammar/fitting/curvature_fit.py`.

A large part of this file is generic:

- arc-length resampling;
- heading estimation from an ordered polyline;
- heading-rate estimation;
- event-position initialization;
- normalized profile codec;
- fit-state matching by arc position;
- robust least-squares fitting of one trajectory;
- fitting one shared shape profile across transformed occurrences.

This is arguably the best differentiating capability planarcurve could expose.

### Refactor before moving

Do not copy it verbatim. First remove these InkGrammar assumptions:

- `TrajectoryTransform` model dependency;
- hardcoded `MIN_CHANGE_SPACING` and max rate bounds;
- field name `curvature` for normalized heading rate;
- fixed optimization weights hidden in `_optimization_weights()`;
- fit result metrics with implicit pixel semantics.

### Proposed module

```text
planarcurve/fitting/trajectory.py
```

Suggested public API:

```python
@dataclass(frozen=True)
class TrajectoryFitOptions:
    change_count: int
    samples: int = 64
    restarts: int = 2
    seed: int = 0
    min_change_spacing: float = 0.02
    max_heading_rate_deg_per_u: float = 36000.0
    position_weight: float = 1.0
    tangent_weight: float = 0.15
    endpoint_weight: float = 2.0
    turn_weight: float = 0.3
    smoothness_weight: float = 0.02

@dataclass(frozen=True)
class TrajectoryFitResult:
    trajectory: Trajectory2D
    transform: Similarity2D
    success: bool
    cost: float
    metrics: Mapping[str, float]


def fit_trajectory(points: np.ndarray, options: TrajectoryFitOptions) -> TrajectoryFitResult:
    ...


def fit_shared_trajectory_profile(
    occurrences: Sequence[np.ndarray],
    options: TrajectoryFitOptions,
) -> SharedTrajectoryFitResult:
    ...
```

This is useful for handwriting, vector drawing, gesture analysis, normalized shape families, and curve compression without any InkGrammar dependency.

---

## Move now: polyline projection and distance helpers

Source: `inkgrammar/fitting/residuals.py`, `_distances_to_polyline`.

This is generic and belongs in `planarcurve.polyline`.

Do not expose only a distance scalar. Implement a reusable projection primitive:

```python
@dataclass(frozen=True)
class PolylineProjection:
    point: np.ndarray
    distance: np.ndarray
    segment_index: np.ndarray
    segment_t: np.ndarray
    arc_length: np.ndarray


def project_points_to_polyline(points, polyline) -> PolylineProjection:
    ...
```

Then `distance_to_polyline()` becomes a trivial wrapper.

This will also simplify future fitting and correspondence code.

---

## Move now or soon: heading/tangent estimation from sampled curves

Source: `inkgrammar/fitting/curvature_fit.py`:

- `resample_by_arclength` duplicates `planarcurve.polyline.resample_polyline_by_arclength`;
- `estimate_observed_heading` is generic;
- `estimate_observed_heading_rate` is generic.

Consolidate these into planarcurve rather than keeping a third resampling implementation in InkGrammar.

Possible API:

```python
def polyline_headings(points, *, smooth_window=0, degrees=False): ...
def polyline_curvature(points, *, smoothing=...): ...
```

Be careful with the name `curvature`: compute derivative with respect to arc length if exposed as geometric curvature.

---

# 4. Generic InkGrammar code that may belong later

## Candidate: monotone curve correspondence

Source: `inkgrammar/fitting/correspondence.py`.

The dynamic-programming alignment algorithm is generic, but the current function is coupled to `TargetTrace`, which comes from image centerline extraction and carries tangent/radius/confidence fields.

Do not move it verbatim.

Refactor first into a neutral sample model such as:

```python
@dataclass(frozen=True)
class CurveSamples:
    xy: np.ndarray
    tangent: np.ndarray | None = None
    radius: np.ndarray | None = None
    weight: np.ndarray | None = None
```

Then expose:

```python
def monotone_correspondence(
    source: CurveSamples,
    target: CurveSamples,
    *,
    ...,
) -> Correspondence:
    ...
```

This would be a valuable `planarcurve.matching` feature, but it is not required for `0.1.0`.

---

## Candidate: variable-width SVG/path expansion

Source: `inkgrammar/outline.py`.

Most of `expand_path()` and `expand_tap()` is generic:

- adaptive flattening;
- tangent/normal calculation;
- variable width profile;
- round caps;
- closed contours;
- outline bounds.

Only `expand_shaped_text()` is InkGrammar-specific.

Possible future module:

```text
planarcurve/outline.py
```

or:

```text
planarcurve/svg_outline.py
```

Public functions could be:

```python
stroke_path(path, width, *, config=...)
stroke_point(point, width, *, config=...)
```

### Caution

The current algorithm is a useful stroke expansion utility, not a robust polygon offset/boolean engine. Self-intersections and pathological offset topology need explicit tests and documentation before advertising it as a general offsetter.

Therefore mark it experimental if moved early.

---

## Candidate: coordinate frames

Source: `inkgrammar/dataset/transforms.py`.

The current `ObservationFrame` is application-specific in naming and Pydantic usage, so it should not move directly.

Instead, improve `Affine2D` so InkGrammar can express coordinate frames with generic operations:

```python
transform.inverse()
transform.compose(other)
Affine2D.from_basis(origin, x_axis, y_axis)
```

Then InkGrammar's source-pixel/glyph-unit frame can be a thin domain wrapper around a standard affine map.

---

# 5. Code that should stay out of planarcurve

Do not move code merely because it uses geometry.

Keep the following in InkGrammar or another specialized package:

| Area | Why it should stay out |
|---|---|
| raster preprocessing | image-domain functionality, Pillow/scikit-image |
| skeletonization/centerline extraction | image-analysis pipeline, not curve math |
| `normal_track_ink` | ink-mask/image sampling semantics |
| dataset materialization/review/status | domain workflow |
| glyph/family/topology models | InkGrammar domain schema |
| fitting workspace/checkpoints/reports | application orchestration |
| font building/features/shaping | typography/application layer |
| connection fitting | currently handwriting-specific constraints and metrics |
| generic simulated annealing | useful code, but not curve-specific; would make planarcurve unfocused |
| CLI commands | belong to consuming application unless a small geometry CLI is deliberately designed later |

A good rule is:

> Move an algorithm if its inputs and outputs can be described completely in generic 2D geometry terms without words like glyph, ink, observation, family, topology, handwriting, or dataset.

---

# 6. Core API improvements

## 6.1 Separate geometry objects from fit policy

This should be a package-wide rule.

Pure geometry dataclasses should describe a curve only.

Fitting options should describe:

- locked/anchored/free endpoints;
- minimum event spacing;
- parameter bounds;
- robust loss scale;
- regularization;
- iteration limits;
- restart strategy.

Do not store fit policy in `BSplineCurve` or impose optimizer spacing in `HeadingRateProfile`.

---

## 6.2 Improve `Affine2D`

Current `Affine2D` is useful but incomplete for a geometry package.

Add:

```python
@property
def determinant(self) -> float: ...

def inverse(self) -> Affine2D: ...

def compose(self, other: Affine2D) -> Affine2D: ...

@classmethod
def translation_only(cls, dx=0.0, dy=0.0): ...

@classmethod
def scale(cls, sx, sy=None): ...

@classmethod
def rotation(cls, angle_deg): ...

@classmethod
def from_basis(cls, origin, x_axis, y_axis): ...
```

Document composition order unambiguously.

Add singular-transform tests for `inverse()`.

For vectors/tangents, add a method that applies the linear part without translation:

```python
def apply_vectors(self, vectors): ...
```

For normal transformation, use inverse-transpose if such an API is added.

---

## 6.3 Improve Bezier support

Current Bezier support is enough for extraction parity but thin for a standalone package.

Add, in order:

1. first derivative;
2. second derivative;
3. unit tangent;
4. signed curvature;
5. de Casteljau split;
6. adaptive/fixed-tolerance flattening;
7. approximate length with tolerance;
8. optional quadratic dataclass for API symmetry.

Prefer full `CubicBezier` objects for conversions when possible.

---

## 6.4 Make cubic chain representation coherent

`CubicSegment` is a compact chain element whose start is implicit. That is valid, but conversion APIs currently return a list without a container describing the first start point.

Also, `curvature.py` contains:

```python
to_cubic_beziers = curvature_to_cubic_segments
```

but the function returns `CubicSegment`, not `CubicBezier`.

That alias is misleading.

Choose one of these designs:

### Option A: full Beziers

```python
def trajectory_to_cubic_beziers(...) -> tuple[CubicBezier, ...]:
    ...
```

### Option B: explicit cubic path

```python
@dataclass(frozen=True)
class CubicPath:
    start: Point2D
    segments: tuple[CubicSegment, ...]
```

Then both spline and trajectory conversion return `CubicPath`.

Option B maps naturally to SVG `M` + repeated `C` commands and matches InkGrammar's chain representation without leaving the start point implicit.

---

## 6.5 Improve trajectory-to-cubic conversion accuracy contract

Current `curvature_to_cubic_segments(..., max_u_step=0.05)` uses parameter-step size as its approximation control.

For a public geometry API, a geometric tolerance is easier to reason about.

Eventually prefer:

```python
def to_cubic_path(
    trajectory,
    *,
    max_error: float = 1e-4,
    max_depth: int = 16,
) -> CubicPath:
    ...
```

Use midpoint/quarter-point deviation checks and recursively subdivide until the cubic approximation satisfies the tolerance.

`max_u_step` can remain as a low-level escape hatch if desired.

---

## 6.6 Fix unnecessary derivative work in `evaluate_bspline`

File: `planarcurve/spline.py`, approximately lines 120-146.

Current code computes first and second SciPy spline derivatives before checking:

```python
if not derivatives:
    return ...
```

This means the dense evaluation used by arc-length sampling pays for derivatives it explicitly requested not to compute.

Move derivative construction/evaluation after the early return.

This is a small but clear performance fix.

---

## 6.7 Review B-spline restrictions

Current validation intentionally supports only:

- degree 3;
- normalized `[0,1]` domain;
- clamped endpoints;
- simple interior knots.

This is acceptable for `0.1.0` if clearly documented.

However, the public class is named `BSplineCurve`, not `SimpleClampedCubicBSpline`. Either:

- keep the restrictions and state them prominently; or
- relax interior-knot multiplicity where SciPy and cubic conversion already support it.

Do not broaden degree/domain support just to look generic unless tests and conversion semantics are ready.

---

## 6.8 Expand polyline functionality

The current polyline module is a good foundation.

Add:

```python
normalized_arclength(points)
project_points_to_polyline(points, polyline)
distance_to_polyline(points, polyline)
closest_point_on_polyline(point, polyline)
```

Potential later additions:

```python
signed_area(points)
polyline_normals(points)
simplify_polyline(points, tolerance)
```

Keep algorithms NumPy-only where reasonable.

---

# 7. SVG adapter review

The optional FontTools adapter is useful and should stay optional.

## Good current behavior

- no FontTools dependency is imported from package root;
- path analysis returns endpoints, tangents, approximate length, bbox, move count, and closed flag;
- affine transformation uses the generic `Affine2D` type;
- quadratic and cubic commands are supported for ordinary segments.

## Improvements

### Validate sampling arguments

`analyze_path()` and `sample_path()` should explicitly validate `samples_per_curve` as an integer >= 1 or >= 2, rather than relying on range/division behavior.

### Preserve path closure semantics

A path containing `Z`/`closePath` is currently transformed into explicit line geometry ending at the subpath start, rather than retaining `Z`.

For filled shapes this is often geometrically equivalent, but for SVG stroking `Z` has join/cap semantics that an explicit final line can change.

The transformation adapter should preserve `closePath` if possible.

Add tests for:

```text
M 0 0 L 10 0 L 10 10 Z
```

and multiple subpaths.

### Clarify approximation

`PathInfo.length` is approximate for Beziers because it samples each curve uniformly in parameter.

Either rename/document it as approximate or add a tolerance-based length calculation using the Bezier module.

### Reduce duplication

`planarcurve.svg` has private quadratic/cubic evaluation helpers that overlap `planarcurve.bezier`.

Once the Bezier API is expanded, reuse it.

---

# 8. Naming cleanup before public release

There are still compatibility names inherited from InkGrammar:

```text
CurvatureStrokeGeometry
StrokePose
CurvatureStrokeSample
BSplineGeometry
evaluate_curvature_stroke
sample_curvature_stroke
evaluate_stroke_at
```

They are understandable during extraction, but the package should not carry application migration vocabulary indefinitely.

Because the project is still pre-release, strongly consider cleaning this now.

A neutral target API could look like:

```python
Trajectory2D
TrajectorySample
Pose2D
HeadingRateProfile
HeadingRateChange

evaluate_trajectory(...)
evaluate_trajectory_at(...)
sample_trajectory_by_arclength(...)
trajectory_to_cubic_path(...)
```

If old names are temporarily retained, keep them as undocumented aliases and remove them before `1.0`, but there is little reason to establish them publicly at `0.1`.

---

# 9. Proposed package structure

Recommended near-term layout:

```text
planarcurve/
├── __init__.py
├── _version.py              # generated by setuptools_scm
├── types.py
├── transforms.py
├── polyline.py
├── bezier.py
├── trajectory.py            # preferred rename from curvature.py
├── spline.py                # optional SciPy
├── svg.py                   # optional FontTools
├── fitting/
│   ├── __init__.py
│   ├── parameters.py
│   ├── spline.py            # optional SciPy
│   └── trajectory.py        # optional SciPy
└── py.typed

examples/
├── README.md
├── curvature_gallery.py
├── spline_to_bezier.py
├── svg_pipeline.py
├── fit_polyline_to_spline.py
└── fit_shared_trajectory.py
```

If renaming `curvature.py` to `trajectory.py` is judged too disruptive even pre-release, keep `curvature.py` but still clean class/function names.

---

# 10. Optional dependency design

Suggested extras:

```toml
[project.optional-dependencies]
spline = ["scipy>=1.11"]
fit = ["scipy>=1.11"]
svg = ["fonttools>=4.55"]
all = ["scipy>=1.11", "fonttools>=4.55"]
dev = [
  "pytest>=8",
  "ruff>=0.8",
  "scipy>=1.11",
  "fonttools>=4.55",
  "build>=1.2",
  "twine>=5",
]
```

Do not make FontTools or SciPy mandatory unless the package direction changes substantially.

The base package should remain lightweight.

---

# 11. Examples to add

The following three examples were executed successfully against the supplied reconstructed package. They demonstrate that the package already has standalone value without InkGrammar.

## Example 1: curvature/heading-rate trajectory gallery

File: `examples/curvature_gallery.py`

```python
from pathlib import Path

import numpy as np

from planarcurve.curvature import CurvatureChange, CurvatureProfile, CurvatureTrajectory

PROFILES = [
    ("straight", CurvatureProfile(0.0, 0.0)),
    ("arc", CurvatureProfile(90.0, 90.0, interpolation="linear")),
    (
        "s-curve",
        CurvatureProfile(
            0.0,
            0.0,
            changes=(
                CurvatureChange(0.30, 240.0),
                CurvatureChange(0.70, -240.0),
            ),
        ),
    ),
    (
        "loop-like",
        CurvatureProfile(
            80.0,
            20.0,
            changes=(
                CurvatureChange(0.25, 480.0),
                CurvatureChange(0.65, 220.0),
            ),
        ),
    ),
]

rows = []
for index, (name, profile) in enumerate(PROFILES):
    sample = CurvatureTrajectory(
        length=140.0,
        start_angle_deg=0.0,
        profile=profile,
    ).evaluate(count=513)

    xy = sample.xy.copy()
    xy[:, 1] *= -1
    xy += np.array([40.0, 70.0 + index * 120.0])
    points = " ".join(f"{x:.3f},{y:.3f}" for x, y in xy)

    rows.append(
        f'<text x="15" y="{25 + index * 120}" font-size="14">{name}</text>'
    )
    rows.append(
        f'<polyline fill="none" stroke="black" stroke-width="2" points="{points}"/>'
    )

svg = (
    '<svg xmlns="http://www.w3.org/2000/svg" '
    'width="520" height="520" viewBox="0 0 520 520">'
    + "".join(rows)
    + "</svg>"
)
Path("curvature_gallery.svg").write_text(svg, encoding="utf-8")
```

This example intentionally writes SVG directly and does not depend on matplotlib.

After the naming cleanup, update imports to the final trajectory API.

---

## Example 2: B-spline analysis and cubic conversion

File: `examples/spline_to_bezier.py`

```python
import numpy as np

from planarcurve.spline import BSplineCurve

curve = BSplineCurve(
    knots=(0, 0, 0, 0, 0.5, 1, 1, 1, 1),
    control_points=((0, 0), (1, 2), (2, 3), (3, 2), (4, 0)),
)

sample = curve.sample_by_arclength(21)
segments = curve.to_cubic_segments()

print("uniform arc-length sample shape:", sample.xy.shape)
print("endpoint tangents:", sample.tangent[[0, -1]])
print(
    "curvature range:",
    float(sample.curvature.min()),
    float(sample.curvature.max()),
)
print("cubic segment count:", len(segments))

for index, segment in enumerate(segments):
    print(index, segment)
```

Observed result on the reviewed snapshot:

```text
uniform arc-length sample shape: (21, 2)
endpoint tangents:
[[ 0.4472136   0.89442719]
 [ 0.4472136  -0.89442719]]
cubic segment count: 2
```

This is a strong minimal SciPy example.

---

## Example 3: SVG to transformed uniformly sampled polyline

File: `examples/svg_pipeline.py`

```python
import numpy as np

from planarcurve.polyline import polyline_length, resample_polyline_by_arclength
from planarcurve.svg import analyze_path, sample_path, transform_path
from planarcurve.transforms import Affine2D

path = "M 0 0 C 20 0 30 30 50 30 Q 70 30 80 10"

info = analyze_path(path, samples_per_curve=64)
print(info)

transform = Affine2D.from_scale_shear_translation(
    scale_x=1.5,
    scale_y=0.75,
    shear_x=0.2,
    dx=10,
    dy=-5,
)

transformed = transform_path(path, transform)
polyline = np.asarray(sample_path(transformed, samples_per_curve=64), dtype=float)
uniform = resample_polyline_by_arclength(polyline, 12)

print("sampled length:", polyline_length(polyline))
print(uniform)
```

This demonstrates a useful cross-module workflow rather than isolated functions.

---

## Example 4 to add after spline-fit extraction

File: `examples/fit_polyline_to_spline.py`

Goal:

1. Create noisy samples from a known curve.
2. Define a fixed-knot B-spline basis.
3. Fit robust control points with IRLS.
4. Compare seed RMS and fit RMS.
5. Write an SVG containing observations, seed spline, and fitted spline.

This will communicate more value than a pure evaluation example.

---

## Example 5 to add after trajectory-fit extraction

File: `examples/fit_shared_trajectory.py`

Goal:

1. Generate one normalized trajectory profile.
2. Apply several similarity transforms and small noise.
3. Fit one shared profile to all occurrences.
4. Print recovered profile parameters and transform errors.
5. Write one SVG overlay.

This would demonstrate the feature that is most unique to planarcurve.

---

# 12. README improvements

The current README is a good extraction note but too close to an internal MVP description.

For a public release, structure it around user questions:

```text
# planarcurve
one-paragraph value proposition

## Install
core / spline / fit / svg extras

## Quick examples
- polyline
- Bezier
- trajectory
- spline
- SVG
- fitting

## Concepts
- normalized progress u
- heading rate vs geometric curvature
- arc-length sampling
- transform convention

## Optional dependencies

## Numerical accuracy
explain integration/sampling tolerances

## API stability
0.x policy

## Provenance
InkGrammar extraction and Apache-2.0
```

Do not lead with the fact that it was extracted from InkGrammar. Lead with what the package does; keep provenance later.

---

# 13. Tests to add

## Core trajectory tests

Add tests for:

- exact profile integration independent of query grid;
- arbitrary evaluation positions not starting at zero;
- linear profile analytic solutions;
- constant geometric curvature / circular arc;
- scale relation `kappa ~ 1/L`;
- profile knot locations arbitrarily close when no fit policy is active;
- finite validation only in pure model;
- trajectory cubic conversion geometric tolerance.

## Polyline tests

Add tests for:

- projection onto segment interiors;
- endpoint projection;
- zero-length duplicate segments;
- multiple points vectorized;
- arc coordinate of projection;
- empty/single-point rejection where relevant.

## Transform tests

Add tests for:

- `inverse(apply(p)) == p`;
- composition order;
- vector transform excluding translation;
- singular inverse rejection;
- `Similarity2D.as_affine()` consistency.

## Spline tests

Add tests for:

- `derivatives=False` path;
- fit recovery from noisy observations;
- robust outlier downweighting;
- locked/free/anchored endpoint policy in fitting, not geometry;
- repeated interior knots if support is relaxed;
- arc-length sampling accuracy against a high-resolution reference.

## SVG tests

Add tests for:

- `samples_per_curve` validation;
- closed path transform preserving closure semantics;
- multiple subpaths;
- ordinary quadratic/cubic behavior;
- explicit error for unsupported implied quadratic forms if still unsupported;
- transformed bbox and endpoints.

## Example smoke tests

Add a test that executes every example in a temporary directory.

Examples should not require InkGrammar or local repository data.

---

# 14. CI and release improvements

Current CI tests Python 3.10 and 3.13 and installs all dev dependencies. Keep that matrix, but add a **minimal dependency job**.

Recommended jobs:

### Core-only job

```bash
pip install -e .
python -c "import planarcurve"
pytest tests/core_only ...
```

This proves SciPy and FontTools did not accidentally become required.

### Full job

```bash
pip install -e '.[dev]'
ruff check .
pytest
python -m build
python -m twine check dist/*
```

### Wheel smoke job

Build a wheel, create a fresh environment, install the wheel, and run a small import/evaluation script.

This catches missing package data and editable-install-only mistakes.

### PEP 561 marker

`pyproject.toml` declares:

```toml
[tool.setuptools.package-data]
planarcurve = ["py.typed"]
```

The reconstructed Codecrate snapshot does not contain `planarcurve/py.typed`, but the pack configuration also does not include `py.typed` in its include patterns. Therefore this review cannot determine whether the real Git repository contains it.

**Verify in the actual repository** and add an empty `planarcurve/py.typed` if missing.

Also update `.codecrate.toml` so future context packs include it.

---

# 15. InkGrammar migration plan

Do this after the planarcurve API changes stabilize but before long-term parallel development.

## Step 1: add dependency

InkGrammar:

```toml
dependencies = [
    ...,
    "planarcurve>=0.1,<0.2",
]
```

During unreleased coordinated development, a workspace/editable install is acceptable.

## Step 2: introduce adapters

Keep adapters in InkGrammar, not planarcurve.

Examples:

```text
InkGrammar CurvatureStrokeGeometry -> planarcurve Trajectory2D
InkGrammar CurvatureProfile        -> planarcurve HeadingRateProfile
InkGrammar TopologyTransform       -> planarcurve Affine2D
InkGrammar BSplineGeometry         -> planarcurve BSplineCurve
```

## Step 3: replace implementations

Replace bodies of:

```text
inkgrammar/fitting/curvature.py
inkgrammar/fitting/spline.py
inkgrammar/geometry.py
```

with imports/wrappers where practical.

Do not immediately delete wrappers if many InkGrammar modules import them. First make them delegate, then simplify imports in a later cleanup.

## Step 4: move generic fitting utilities

After planarcurve owns them, update:

```text
inkgrammar/fitting/spline_fit.py
inkgrammar/fitting/trajectory_parameterization.py
inkgrammar/fitting/curvature_fit.py
```

to import planarcurve implementations.

Keep InkGrammar-specific orchestration and result serialization locally.

## Step 5: parity becomes integration testing

Once InkGrammar calls planarcurve, exact parity is no longer a comparison of duplicated algorithms. Replace the current parity script with tests that ensure the InkGrammar adapters preserve schema semantics and numerical outputs.

---

# 16. Suggested implementation phases

## Phase A — correctness and API cleanup (must happen before `0.1.0`)

1. Implement exact heading-rate antiderivative.
2. Make heading and total-turn evaluation query-grid independent.
3. Decide naming: normalized heading rate versus true curvature.
4. Remove `MIN_CHANGE_SPACING` and max-rate constraints from pure curve representation.
5. Move `SplineEndpointPolicy` out of `BSplineCurve` geometry.
6. Remove or rename InkGrammar compatibility aliases before release.
7. Fix `to_cubic_beziers` naming/return type.
8. Fix unnecessary spline derivative work.
9. Add transform inverse/composition.
10. Add missing input validation in SVG sampling.

Definition of done:

```text
pytest passes
ruff passes
wheel/sdist build passes
twine check passes
core imports without SciPy/FontTools
no imports from InkGrammar/Pydantic/Pillow/Typer/YAML
```

## Phase B — standalone usefulness

1. Extract generic spline fitting.
2. Extract neutral trajectory parameterization helpers.
3. Extract trajectory fitting and shared-profile fitting.
4. Add polyline projection/distance.
5. Add three current examples and two fitting examples.
6. Rewrite README around external users.

Definition of done:

- a user can fit a noisy polyline to a spline using only planarcurve;
- a user can fit a compact normalized trajectory using only planarcurve;
- examples execute in CI;
- no InkGrammar terminology appears in public type names or docs except provenance.

## Phase C — make InkGrammar a downstream consumer

1. Add planarcurve dependency.
2. Add model adapters in InkGrammar.
3. Delegate curvature/spline/SVG math.
4. Delete duplicated algorithms after adapter tests pass.

Definition of done:

- only one numerical source of truth;
- InkGrammar tests pass against planarcurve;
- planarcurve tests do not import InkGrammar.

## Phase D — optional expansion

Consider:

- generic monotone correspondence;
- variable-width path outlining;
- adaptive Bezier length/flattening;
- richer curve conversion;
- coordinate-frame conveniences.

Do not block `0.1.0` on these.

---

# 17. Recommended public API direction

A coherent target could look like this:

```python
import numpy as np
from planarcurve import Affine2D, Similarity2D
from planarcurve.polyline import (
    cumulative_arclength,
    resample_polyline_by_arclength,
    project_points_to_polyline,
)
from planarcurve.bezier import CubicBezier
from planarcurve.trajectory import (
    HeadingRateChange,
    HeadingRateProfile,
    Trajectory2D,
)
from planarcurve.spline import BSplineCurve
```

Trajectory example:

```python
trajectory = Trajectory2D(
    length=2.0,
    start_heading_deg=15.0,
    heading_rate=HeadingRateProfile(
        start_rate_deg_per_u=20.0,
        changes=(
            HeadingRateChange(0.3, 180.0),
            HeadingRateChange(0.7, -80.0),
        ),
        end_rate_deg_per_u=0.0,
    ),
)

u = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
state = trajectory.evaluate_at(u)
print(state.xy)
print(state.curvature)  # true signed curvature in 1/length
```

This contract is clearer to people who have never heard of InkGrammar.

---

# 18. What makes the package worth publishing

The package should not try to compete with SciPy as a spline implementation or FontTools as an SVG parser. Its value is the combination of a small set of operations with consistent data contracts:

```text
sampled points
    <-> polylines
    <-> Bezier paths
    <-> normalized trajectories
    <-> B-splines
    <-> SVG path data
```

plus:

```text
transforms
arc-length sampling
curve derivatives
curvature
fitting
conversion
```

That is a coherent problem space.

The strongest differentiator is:

> fit a low-dimensional normalized turning/heading-rate program to an observed 2D curve and reuse it under similarity transforms.

That capability originated in InkGrammar but is not handwriting-specific. It belongs in planarcurve if the API is neutralized.

---

# 19. Final decision matrix

| Item | Move to planarcurve? | Priority | Notes |
|---|---:|---:|---|
| curvature/heading-rate kernel | already there | P0 | fix evaluation semantics and naming |
| B-spline evaluator | already there | P0 | keep optional SciPy |
| SVG geometry adapter | already there | P1 | preserve closure semantics |
| polyline helpers | already there | P1 | add projection/distance |
| spline fitting | **yes** | P1 | clearly generic |
| trajectory parameterization | **yes** | P1 | generic after bounds are configurable |
| trajectory fitting | **yes** | P1 | major standalone value |
| shared trajectory-profile fitting | **yes** | P1 | strong differentiator |
| monotone correspondence | probably | P2 | refactor away from `TargetTrace` |
| variable-width path expansion | probably | P2 | move generic path/tap part only |
| observation coordinate frame | not verbatim | P2 | improve `Affine2D` instead |
| raster/skeleton extraction | no | — | image-domain |
| dataset tooling | no | — | application workflow |
| glyph/family fitting orchestration | no | — | InkGrammar domain |
| simulated annealing | no | — | generic optimizer but off-scope |
| font/shaping | no | — | typography/application domain |

---

# 20. Coding-agent checklist

The coding agent should work in this order and keep each change independently testable:

- [ ] Reconstruct/confirm current snapshot and run baseline tests.
- [ ] Add exact heading-rate integration tests that fail on current code.
- [ ] Implement exact profile antiderivative.
- [ ] Refactor trajectory heading/turn APIs onto the antiderivative.
- [ ] Split geometry validation from fit constraints.
- [ ] Decide/finalize public naming before `0.1.0`.
- [ ] Move spline endpoint policy into fitting.
- [ ] Fix cubic conversion return naming/type.
- [ ] Add `Affine2D.inverse`, composition, vector application, tests.
- [ ] Add polyline projection/distance helpers.
- [ ] Optimize `evaluate_bspline(..., derivatives=False)`.
- [ ] Add SVG sampling validation and close-path tests.
- [ ] Extract spline fit from InkGrammar and port its tests.
- [ ] Extract trajectory parameterization helpers and port tests.
- [ ] Extract trajectory fitting/shared fitting and neutralize terminology/options.
- [ ] Add runnable examples.
- [ ] Rewrite README for standalone users.
- [ ] Add minimal-dependency and wheel-smoke CI jobs.
- [ ] Verify `py.typed` is present in the real repository and included in context packs.
- [ ] Make InkGrammar depend on planarcurve and replace duplicated numerical code with adapters.
- [ ] Run full InkGrammar tests after migration.
- [ ] Tag `v0.1.0` only after both repositories use one numerical source of truth.

---

# Bottom line

`planarcurve` is worth publishing independently. The current package already demonstrates standalone workflows, has a clean lightweight dependency boundary, and has exact parity with the extracted InkGrammar trajectory kernel on the supplied parity fixture.

The main risk is not missing functionality. The main risk is publishing an API that still encodes InkGrammar assumptions and then having to support those assumptions indefinitely.

Use the pre-release window to make the mathematical contract neutral and exact. Then move the genuinely generic fitting pieces into planarcurve and make InkGrammar consume them. That will turn the package from “extracted utilities” into a reusable 2D curve toolkit with a clear reason to exist.
