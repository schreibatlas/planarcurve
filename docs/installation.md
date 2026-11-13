# Installation

`planarcurve` requires Python 3.10 or newer. The core geometry package depends on NumPy. Install an extra only when using its integration:

| Capability | Install | Dependency |
| --- | --- | --- |
| Core trajectories, transforms, polylines, and Beziers | `python -m pip install planarcurve` | NumPy |
| B-splines | `python -m pip install 'planarcurve[spline]'` | SciPy |
| Fitting | `python -m pip install 'planarcurve[fit]'` | SciPy |
| SVG adapters | `python -m pip install 'planarcurve[svg]'` | FontTools |
| Everything | `python -m pip install 'planarcurve[all]'` | SciPy and FontTools |

For development, install the package and its test/lint tools in editable mode:

```bash
python -m pip install -e '.[dev]'
```

To build the documentation locally, install all integrations and the documentation toolchain:

```bash
python -m pip install -e '.[all]'
python -m pip install -r docs/requirements.txt
```

Importing `planarcurve` or its core modules does not import SciPy or FontTools. Optional imports live in `planarcurve.spline`, `planarcurve.fitting`, and `planarcurve.svg`. The core-only import boundary is checked in CI.
