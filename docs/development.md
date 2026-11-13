# Development

## Set up

From the repository root, install the development tools and optional integrations:

```bash
python -m pip install -e '.[dev]'
python -m pip install -e '.[all]'
python -m pip install -r docs/requirements.txt
```

The first install provides pytest, Ruff, and build tooling. The `all` extra adds SciPy and FontTools for testing optional modules and building their API reference.

## Checks

Run the test suite and style checks:

```bash
python -m pytest
ruff check .
ruff format --check .
```

Build the documentation with warnings treated as errors. This is the canonical documentation check used in CI:

```bash
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

The helper is also available:

```bash
python docs/make.py html
```

For optional, network-dependent link checking, run `python -m sphinx -b linkcheck docs docs/_build/linkcheck`. Do not treat external link availability as a deterministic pull-request check.
