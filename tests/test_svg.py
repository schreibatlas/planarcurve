from __future__ import annotations

import pytest

from planarcurve.svg import analyze_path, cubic_bridge, sample_path, transform_path
from planarcurve.transforms import Affine2D


def test_svg_path_adapter():
    path = "M 0 0 C 1 0 2 1 3 1 Q 4 2 5 1"
    info = analyze_path(path)
    assert info.start == (0.0, 0.0)
    assert info.end == (5.0, 1.0)
    assert len(sample_path(path)) > 2
    moved = transform_path(path, Affine2D(dx=4, dy=-2))
    assert moved.startswith("M 4 -2")
    assert analyze_path(moved).end == (9.0, -1.0)
    assert cubic_bridge((0, 0), None, (1, 1), None, 0.2).startswith("M 0 0 C")


def test_empty_path_rejected():
    with pytest.raises(ValueError):
        analyze_path("")


def test_sampling_arguments_are_validated():
    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="samples_per_curve"):
            analyze_path("M 0 0 C 1 0 2 1 3 1", samples_per_curve=value)
        with pytest.raises(ValueError, match="samples_per_curve"):
            sample_path("M 0 0 Q 1 1 2 0", samples_per_curve=value)
    assert len(sample_path("M 0 0 Q 1 1 2 0", samples_per_curve=1)) == 2


def test_close_path_transform_and_sampling_preserve_multiple_subpaths():
    path = "M 0 0 L 10 0 L 10 10 Z M 20 20 L 30 20 Z"
    transformed = transform_path(path, Affine2D.translation_only(5.0, -3.0))
    assert transformed.count("Z") == 2
    info = analyze_path(path)
    assert info.closed
    assert info.move_count == 2
    assert info.length == pytest.approx(10.0 + 10.0 + 2**0.5 * 10.0 + 20.0)
    assert info.end == (20.0, 20.0)
    moved_info = analyze_path(transformed)
    assert moved_info.end == (25.0, 17.0)
    samples = sample_path(path)
    assert samples.count((0.0, 0.0)) == 2
    assert samples.count((20.0, 20.0)) == 2
