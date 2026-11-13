# Examples

These scripts demonstrate the standalone planarcurve API. Install optional dependencies with `python -m pip install 'planarcurve[all]'`, then run them from the repository root or an installed checkout:

```bash
python examples/curvature_gallery.py
python examples/spline_to_bezier.py
python examples/svg_pipeline.py
python examples/fit_polyline_to_spline.py
python examples/fit_shared_trajectory.py
```

The scripts use synthetic inputs and do not depend on InkGrammar or local datasets. `curvature_gallery.py` writes `curvature_gallery.svg`; the spline-fitting and shared-trajectory-fitting examples write `fit_polyline_to_spline.svg` and `fit_shared_trajectory.svg`. The spline, SVG, and fitting examples require the corresponding optional dependencies.
