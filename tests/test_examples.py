from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
EXAMPLES = (
    ("curvature_gallery.py", "curvature_gallery.svg"),
    ("spline_to_bezier.py", None),
    ("svg_pipeline.py", None),
    ("fit_polyline_to_spline.py", "fit_polyline_to_spline.svg"),
    ("fit_shared_trajectory.py", "fit_shared_trajectory.svg"),
)


@pytest.mark.parametrize(("example", "output"), EXAMPLES)
def test_example_runs_in_an_isolated_working_directory(tmp_path, example, output):
    subprocess.run(
        [sys.executable, str(ROOT / "examples" / example)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if output is not None:
        assert (tmp_path / output).is_file()
