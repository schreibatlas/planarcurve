import pytest

from planarcurve.svg import analyze_path, cubic_bridge, sample_path, transform_path
from planarcurve.transforms import Affine2D


def test_svg_path_adapter():
    path = "M 0 0 C 1 0 2 1 3 1"
    info = analyze_path(path)
    assert info.start == (0.0, 0.0)
    assert info.end == (3.0, 1.0)
    assert len(sample_path(path)) > 2
    moved = transform_path(path, Affine2D(dx=4, dy=-2))
    assert moved.startswith("M 4 -2")
    assert cubic_bridge((0, 0), None, (1, 1), None, 0.2).startswith("M 0 0 C")


def test_empty_path_rejected():
    with pytest.raises(ValueError):
        analyze_path("")
