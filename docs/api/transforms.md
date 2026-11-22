# `planarcurve.transforms`

Two-dimensional affine and positive-scale similarity transforms.
Two-dimensional affine and positive-scale similarity transforms.

`fit_similarity_transform(target, source, *, weights=None)` maps source points onto target points. It fits a deterministic, orientation-preserving transform with positive scale, rotation, and translation. Reflections are not fitted. Optional finite, nonnegative point weights control each correspondence's contribution.


`transform_cubic_path(path, transform)` returns a new path with its start and every cubic control/end point transformed. It accepts `Affine2D` and `Similarity2D`, preserves segment order, and leaves the input path unchanged.

```{automodule} planarcurve.transforms
:members:
:undoc-members:
:show-inheritance:
:member-order: bysource
```
