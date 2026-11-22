# Examples

The repository includes nine synthetic, standalone demonstrations. They do not require Matplotlib, application-specific models, or local datasets. Install optional integrations once with `python -m pip install '.[all]'`, then run scripts from the repository root.

| Script                                                                                                                              | Demonstrates                                                         | Extra    | Output                       |
| ----------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | -------- | ---------------------------- |
| [`trajectory_gallery.py`](https://github.com/schreibatlas/planarcurve/blob/main/examples/trajectory_gallery.py)                     | Straight, arc, and changing heading-rate profiles                    | None     | `trajectory_gallery.svg`     |
| [`bezier_basics.py`](https://github.com/schreibatlas/planarcurve/blob/main/examples/bezier_basics.py)                               | Evaluation, tangents, curvature, subdivision, flattening, and length | None     | Console                      |
| [`spline_to_bezier.py`](https://github.com/schreibatlas/planarcurve/blob/main/examples/spline_to_bezier.py)                         | Arc-length sampling and spline-to-cubic conversion                   | `spline` | Console                      |
| [`svg_pipeline.py`](https://github.com/schreibatlas/planarcurve/blob/main/examples/svg_pipeline.py)                                 | SVG analysis, transform, sampling, and resampling                    | `svg`    | Console                      |
| [`fit_polyline_to_spline.py`](https://github.com/schreibatlas/planarcurve/blob/main/examples/fit_polyline_to_spline.py)             | Robust fitting of shared spline control points                       | `fit`    | `fit_polyline_to_spline.svg` |
| [`fit_shared_trajectory.py`](https://github.com/schreibatlas/planarcurve/blob/main/examples/fit_shared_trajectory.py)               | One normalized profile fitted to transformed occurrences             | `fit`    | `fit_shared_trajectory.svg`  |
| [`fit_shared_trajectory_family.py`](https://github.com/schreibatlas/planarcurve/blob/main/examples/fit_shared_trajectory_family.py) | Directed descriptors and shared family diagnostics                   | `fit`    | Console                      |
| [`select_trajectory_complexity.py`](https://github.com/schreibatlas/planarcurve/blob/main/examples/select_trajectory_complexity.py) | Deterministic model-order selection and decomposed scores            | `fit`    | Console                      |
| [`refine_trajectory_complexity.py`](https://github.com/schreibatlas/planarcurve/blob/main/examples/refine_trajectory_complexity.py) | Warm-started progressive model order, score diagnostics, and smooth steering controls | `fit` | Console |

Run one example or all of them:

```bash
python examples/trajectory_gallery.py
python examples/bezier_basics.py
python examples/spline_to_bezier.py
python examples/svg_pipeline.py
python examples/fit_polyline_to_spline.py
python examples/fit_shared_trajectory.py
python examples/fit_shared_trajectory_family.py
python examples/select_trajectory_complexity.py
python examples/refine_trajectory_complexity.py
```

The scripts contain assertions for the behaviors they demonstrate. The SVG-producing examples write files to the current working directory. They are also used as sources for the task-oriented [guides](guides/index.md).
