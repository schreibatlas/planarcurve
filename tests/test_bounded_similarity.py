from __future__ import annotations

import numpy as np
import pytest

from planarcurve.transforms import (
    Similarity2D,
    SimilarityFitBounds,
    fit_bounded_similarity,
)


def test_bounded_similarity_recovers_allowed_rotation():
    source = np.array([[0.0, 0.0], [2.0, 1.0], [-1.0, 4.0], [5.0, -2.0]])
    target = Similarity2D(scale=1.7, rotation_deg=8.0, dx=4.0, dy=-3.0).apply(source)

    result = fit_bounded_similarity(
        target, source, bounds=SimilarityFitBounds.symmetric_rotation(15.0)
    )

    assert result.transform.rotation_deg == pytest.approx(8.0)
    assert result.mean_squared_error == pytest.approx(0.0, abs=1e-25)
    assert not result.rotation_constrained
    assert not result.scale_constrained


def test_bounded_similarity_recomputes_fit_at_rotation_and_scale_bounds():
    source = np.array([[0.0, 0.0], [2.0, 1.0], [-1.0, 4.0], [5.0, -2.0]])
    target = Similarity2D(scale=3.0, rotation_deg=30.0).apply(source)
    result = fit_bounded_similarity(
        target,
        source,
        bounds=SimilarityFitBounds(
            rotation_min_deg=-15.0,
            rotation_max_deg=15.0,
            scale_min=1.0,
            scale_max=2.0,
        ),
    )
    assert result.transform.rotation_deg == pytest.approx(15.0)
    assert result.transform.scale == pytest.approx(2.0)
    assert result.rotation_constrained
    assert result.scale_constrained
    assert result.mean_squared_error > 0.0


def test_bounded_similarity_cannot_match_upside_down_or_reflected_paths():
    source = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 3.0], [4.0, 1.0]])
    for transform in (Similarity2D(rotation_deg=180.0),):
        target = transform.apply(source)
        result = fit_bounded_similarity(
            target, source, bounds=SimilarityFitBounds.symmetric_rotation(15.0)
        )
        assert result.mean_squared_error > 0.1
    reflected = source.copy()
    reflected[:, 1] *= -1.0
    result = fit_bounded_similarity(
        source, reflected, bounds=SimilarityFitBounds.symmetric_rotation(15.0)
    )
    assert result.transform.scale > 0.0
    assert result.mean_squared_error > 0.1


def test_bounded_similarity_is_stable_for_large_offsets_and_wrapped_bounds():
    source = np.array([[0.0, 0.0], [2.0, 1.0], [-1.0, 4.0], [5.0, -2.0]])
    offset = np.array([1e12, -2e12])
    target = Similarity2D(rotation_deg=179.0, dx=1e12, dy=-2e12).apply(source + offset)
    result = fit_bounded_similarity(
        target,
        source + offset,
        bounds=SimilarityFitBounds(rotation_min_deg=170.0, rotation_max_deg=190.0),
    )
    assert result.transform.rotation_deg == pytest.approx(179.0, abs=1e-3)
    assert result.mean_squared_error == pytest.approx(0.0, abs=1e-6)


def test_similarity_bounds_validate_values():
    with pytest.raises(ValueError, match="finite"):
        SimilarityFitBounds(scale_min=np.inf)
    with pytest.raises(ValueError, match="rotation_min_deg"):
        SimilarityFitBounds(rotation_min_deg=10.0, rotation_max_deg=-10.0)
    with pytest.raises(ValueError, match="positive"):
        SimilarityFitBounds(scale_min=0.0)
    with pytest.raises(ValueError, match="scale_min"):
        SimilarityFitBounds(scale_min=2.0, scale_max=1.0)
