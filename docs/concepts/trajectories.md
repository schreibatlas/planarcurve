# Trajectories

A `Trajectory2D` describes a planar curve by its geometric length, starting heading, and a profile of heading rate over normalized progress. This separates shape from scale: the profile controls how direction changes, while `length` controls geometric scale.

## Heading-rate profile

`HeadingRateProfile` consists of a start heading rate, zero or more interior `HeadingRateChange` knots, and an end heading rate. Each interior knot has a strictly interior position `at ∈ (0, 1)` and a finite rate. Knot positions must be strictly ordered. Rates are in degrees per unit normalized progress, written `deg/u`.

The interpolation mode is either `"linear"` or `"cubic-hermite"`. The latter computes shape-preserving slopes internally from the knot positions and rates. Callers supply no derivative values; this is not a general Hermite spline with user-selected slopes.

## From profile to positions

The evaluation chain is:

```text
heading rate dθ/du
        ↓ exact piecewise antiderivative
heading θ(u)
        ↓ cos/sin
unit tangent
        ↓ adaptive numerical integration
position xy(u)
```

Heading integration is analytic for the selected piecewise interpolation, up to floating-point arithmetic. Position is obtained by numerically integrating the heading-derived tangent. With geometric length `L`, the position derivative with respect to `u` is `L` times the unit tangent, so theoretical arc length is exactly `L*u`. It is independent of position-integration error and tolerance.

A trajectory's starting position is `(0, 0)`. Apply `Affine2D` or `Similarity2D` to place or orient it in a scene. `trajectory_total_turn()` returns the signed total heading change in degrees.

## Samples and endpoints

`evaluate_trajectory(trajectory, count=...)` samples uniformly in `u`, including 0 and 1. Each `TrajectorySample` includes progress, positions, heading, heading rate, unit tangent, theoretical arc length, and an end pose. Position fields are numerically integrated; the arc-length field follows the exact `length*u` contract.

Use `evaluate_trajectory_at()` for arbitrary `u` values or `sample_trajectory_by_arclength()` for equal intervals in theoretical arc length. Since normalized trajectory speed is constant, uniform `u` and uniform theoretical arc length correspond.

## Conversion to cubic Beziers

`trajectory_to_cubic_path()` adaptively subdivides profile intervals. It compares the cubic approximation with the numerically evaluated trajectory at the quarter, midpoint, and three-quarter parameters. `max_error` is therefore a sampled acceptance criterion, not a proof of a continuous global Hausdorff error bound.


## Reverse traversal and trajectory chains

`reverse_trajectory()` builds the reversed heading-rate profile using `heading_rev(u) = heading(1-u) + 180°` and `rate_rev(u) = -rate(1-u)`. It does not reverse a sampled point array. The resulting trajectory starts at its own origin and matches the original curve traversed backward after rigid alignment.

A `TrajectoryChain` is an ordered sequence of `DirectedTrajectory` values. The first segment is aligned to the requested starting pose. Each following segment starts at the previous endpoint, while its local starting heading remains an offset in the propagated frame. Position joins are exact by construction, and tangent discontinuities are retained rather than smoothed. Chain samples include both sides of each join, along with segment index, segment progress, and global cumulative arc length. Use `trajectory_chain_to_cubic_path()` for a connected cubic representation.
