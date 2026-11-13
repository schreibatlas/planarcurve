from __future__ import annotations

import numpy as np
import pytest

from planarcurve.transforms import Affine2D, Similarity2D


def test_affine_inverse_and_composition_order():
    transform = Affine2D.from_scale_shear_translation(
        scale_x=2.0, scale_y=0.75, shear_x=0.2, dx=4.0, dy=-3.0
    )
    points = np.array([[1.0, 2.0], [-4.0, 3.0]])
    np.testing.assert_allclose(transform.inverse().apply(transform.apply(points)), points)

    first = Affine2D.translation_only(3.0, -2.0)
    second = Affine2D.rotation(45.0)
    np.testing.assert_allclose(
        first.compose(second).apply(points), first.apply(second.apply(points))
    )


def test_affine_vectors_and_constructors():
    transform = Affine2D.from_basis((3.0, 4.0), (0.0, 2.0), (-1.0, 0.0))
    np.testing.assert_allclose(transform.apply([[1.0, 1.0]]), [[2.0, 6.0]])
    np.testing.assert_allclose(transform.apply_vectors([[1.0, 1.0]]), [[-1.0, 2.0]])
    assert Affine2D.scale(2.0).determinant == pytest.approx(4.0)
    np.testing.assert_allclose(
        Affine2D.rotation(90.0).apply([[1.0, 0.0]]), [[0.0, 1.0]], atol=1e-15
    )


def test_singular_affine_inverse_rejected():
    with pytest.raises(ValueError, match="singular"):
        Affine2D(1.0, 2.0, 2.0, 4.0).inverse()


def test_similarity_affine_conversion_is_consistent():
    similarity = Similarity2D(scale=1.75, rotation_deg=37.0, dx=5.0, dy=-8.0)
    points = np.array([[0.0, 1.0], [2.0, -3.0]])
    np.testing.assert_allclose(similarity.as_affine().apply(points), similarity.apply(points))
