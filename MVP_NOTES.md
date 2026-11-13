# MVP validation notes

This MVP was extracted against the supplied InkGrammar repository snapshot.

Validated locally:

- `pytest`: 16 tests passed.
- Curvature trajectory parity with InkGrammar: exact for `u`, `xy`, heading, and heading rate on a 257-point fixture.
- Cubic B-spline parity with InkGrammar: exact for evaluation, tangent, curvature, affine transformation, arc-length parameters, and cubic-segment conversion on representative fixtures.
- Core dependency boundary: no imports of InkGrammar, Pydantic, Pillow, Typer, or YAML.
- SciPy appears only in `planarcurve/spline.py`.
- FontTools appears only in `planarcurve/svg.py`.
- Python bytecode compilation succeeds.

The sandbox did not have `ruff`, `build`, or `setuptools_scm` installed and had no network access, so the wheel/sdist build and Ruff invocation could not be executed here. CI installs these through the `dev` extra before running validation.
