from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
EXAMPLES = (
    ("trajectory_gallery.py", "trajectory_gallery.svg"),
    ("bezier_basics.py", None),
    ("spline_to_bezier.py", None),
    ("svg_pipeline.py", None),
    ("fit_polyline_to_spline.py", "fit_polyline_to_spline.svg"),
    ("fit_shared_trajectory.py", "fit_shared_trajectory.svg"),
)


@pytest.mark.parametrize(("example", "output"), EXAMPLES)
def test_example_runs_in_an_isolated_working_directory(tmp_path, example, output):
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / example)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if output is not None:
        assert (tmp_path / output).is_file()
    assert result.stdout


@pytest.mark.parametrize(("example", "_output"), EXAMPLES)
def test_importing_example_does_not_run_main(tmp_path, example, _output):
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1], run_name='example_import')",
            str(ROOT / "examples" / example),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert list(tmp_path.iterdir()) == []
