# Examples

These synthetic scripts demonstrate the standalone planarcurve API without Matplotlib, InkGrammar, or local datasets. Install optional features with `python -m pip install 'planarcurve[all]'`, then run them from the repository root:

```bash
python examples/trajectory_gallery.py
python examples/bezier_basics.py
python examples/spline_to_bezier.py
python examples/svg_pipeline.py
python examples/fit_polyline_to_spline.py
python examples/fit_shared_trajectory.py
```

| Example | Demonstrates | Optional dependency | Output |
| --- | --- | --- | --- |
| [Trajectory gallery](https://github.com/schreibatlas/planarcurve/blob/main/examples/trajectory_gallery.py) | Heading-rate profiles and theoretical arc length | None | `trajectory_gallery.svg` |
| [Bezier basics](https://github.com/schreibatlas/planarcurve/blob/main/examples/bezier_basics.py) | Evaluation, tangent, curvature, subdivision, flattening, and length | None | Console |
| [Spline conversion](https://github.com/schreibatlas/planarcurve/blob/main/examples/spline_to_bezier.py) | B-spline sampling and conversion to `CubicPath` | SciPy | Console |
| [SVG pipeline](https://github.com/schreibatlas/planarcurve/blob/main/examples/svg_pipeline.py) | Parsing, transforming, and resampling one SVG subpath | FontTools | Console |
| [Spline fitting](https://github.com/schreibatlas/planarcurve/blob/main/examples/fit_polyline_to_spline.py) | Robust shared control-point fitting | SciPy | `fit_polyline_to_spline.svg` |
| [Shared trajectory fitting](https://github.com/schreibatlas/planarcurve/blob/main/examples/fit_shared_trajectory.py) | One normalized shape fitted to multiple occurrences | SciPy | `fit_shared_trajectory.svg` |

Assertions in the scripts check the numerical behavior each example demonstrates. SVG outputs are written to the current working directory.
