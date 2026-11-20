# `planarcurve.trajectory`

Normalized-progress heading-rate trajectories, exact reverse traversal, pose propagation, trajectory-chain sampling, and cubic-path approximation.

```{automodule} planarcurve.trajectory
:members:
:undoc-members:
:show-inheritance:
:member-order: bysource
```

## Trajectory chains

`reverse_trajectory()` constructs reverse traversal from the heading-rate profile. `evaluate_trajectory_chain()` propagates poses and returns sampled positions, tangents, cumulative arc length, and per-segment progress. `trajectory_chain_to_cubic_path()` converts the connected chain to one cubic path.

```{automodule} planarcurve.trajectory_chain
:members:
:undoc-members:
:show-inheritance:
:member-order: bysource
```
