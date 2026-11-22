from __future__ import annotations

import numpy as np
import pytest

from planarcurve.transforms import (
    Affine2D,
    Similarity2D,
    fit_similarity_transform,
    transform_cubic_path,
)
from planarcurve.types import CubicPath, CubicSegment


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


@pytest.mark.parametrize(
    "transform",
    [
        Similarity2D(dx=3.0, dy=-4.0),
        Similarity2D(rotation_deg=37.0),
        Similarity2D(scale=2.4),
        Similarity2D(scale=0.65, rotation_deg=-22.0, dx=-4.0, dy=11.0),
    ],
)
def test_fit_similarity_transform_maps_sample_points_to_reference(transform):
    reference = np.array([[0.0, 0.0], [2.0, 1.0], [-1.0, 4.0], [5.0, -2.0]])
    sample = transform.apply(reference)

    fitted = fit_similarity_transform(reference, sample)

    np.testing.assert_allclose(fitted.apply(sample), reference, atol=1e-12)


def test_fit_similarity_transform_rejects_degenerate_and_nonfinite_points():
    with pytest.raises(ValueError, match="share shape"):
        fit_similarity_transform(np.zeros((2, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="finite"):
        fit_similarity_transform([[0.0, 0.0], [np.inf, 1.0]], [[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="nonzero spatial extent"):
        fit_similarity_transform(np.zeros((2, 2)), [[0.0, 0.0], [1.0, 1.0]])


def test_fit_similarity_transform_does_not_reflect():
    reference = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 3.0], [4.0, 1.0]])
    reflected = reference.copy()
    reflected[:, 1] *= -1.0

    fitted = fit_similarity_transform(reference, reflected)

    assert fitted.scale > 0.0
    assert np.mean(np.sum((reference - fitted.apply(reflected)) ** 2, axis=1)) > 1.0


def test_fit_similarity_transform_weighted_and_noisy_recovery():
    source = np.array([[0.0, 0.0], [2.0, 1.0], [-1.0, 4.0], [5.0, -2.0]])
    source[-1] = [1e12, -2e12]
    expected = Similarity2D(scale=1.7, rotation_deg=28.0, dx=-3.0, dy=5.0)
    target = expected.apply(source)
    target[-1] += [100.0, -80.0]

    fitted = fit_similarity_transform(target, source, weights=np.array([1.0, 1.0, 1.0, 0.0]))

    np.testing.assert_allclose(fitted.apply(source[:3]), expected.apply(source[:3]), atol=1e-12)
    assert (
        fit_similarity_transform(target=target, source=source, weights=[1.0, 1.0, 1.0, 0.0])
        == fitted
    )
    noisy_source = source[:3]

    noisy_target = expected.apply(noisy_source) + np.random.default_rng(17).normal(
        scale=0.01, size=noisy_source.shape
    )
    noisy_fit = fit_similarity_transform(noisy_target, noisy_source)
    residual = noisy_fit.apply(noisy_source) - noisy_target
    assert np.sqrt(np.mean(np.sum(residual**2, axis=1))) < 0.02


def test_fit_similarity_transform_rejects_invalid_weights_and_degenerate_points():
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    target = source.copy()
    with pytest.raises(ValueError, match="N >= 2"):
        fit_similarity_transform(target[:1], source[:1])
    with pytest.raises(ValueError, match="source points must have nonzero spatial extent"):
        fit_similarity_transform(target, np.zeros_like(source))
    for weights, message in (
        ([1.0, 1.0], "shape"),
        ([0.0, 0.0, 0.0], "positive total"),
        ([1.0, -1.0, 1.0], "nonnegative"),
        ([1.0, np.inf, 1.0], "finite"),
    ):
        with pytest.raises(ValueError, match=message):
            fit_similarity_transform(target, source, weights=weights)

    with pytest.raises(ValueError, match="nonzero spatial extent"):
        fit_similarity_transform(np.zeros((3, 2)), source)


def test_fit_similarity_transform_rejects_numerically_singular_reflection():
    target = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    source = target.copy()
    source[:, 1] *= -1.0

    with pytest.raises(ValueError, match="numerically singular"):
        fit_similarity_transform(target, source)


def test_fit_similarity_transform_does_not_special_case_reversed_correspondence_order():
    target = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
    source = target[::-1]

    fitted = fit_similarity_transform(target, source)

    assert np.mean(np.sum((target - fitted.apply(source)) ** 2, axis=1)) > 0.1


def test_similarity_affine_conversion_is_consistent():
    similarity = Similarity2D(scale=1.75, rotation_deg=37.0, dx=5.0, dy=-8.0)
    points = np.array([[0.0, 1.0], [2.0, -3.0]])
    np.testing.assert_allclose(similarity.as_affine().apply(points), similarity.apply(points))


def _sample_cubic(bezier, parameters: np.ndarray) -> np.ndarray:
    points = np.asarray([bezier.p0, bezier.p1, bezier.p2, bezier.p3])
    t = parameters[:, None]
    one_minus_t = 1.0 - t
    return (
        one_minus_t**3 * points[0]
        + 3.0 * one_minus_t**2 * t * points[1]
        + 3.0 * one_minus_t * t**2 * points[2]
        + t**3 * points[3]
    )


def _sample_path(path: CubicPath, parameters: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(_sample_cubic(bezier, parameters) for bezier in path.to_beziers())


def test_transform_cubic_path_identity_and_translation():
    path = CubicPath(
        start=(1.0, 2.0),
        segments=(
            CubicSegment((2.0, 3.0), (4.0, 3.0), (5.0, 1.0)),
            CubicSegment((6.0, 0.0), (7.0, 3.0), (8.0, 2.0)),
        ),
    )

    assert transform_cubic_path(path, Affine2D.identity()) == path
    translated = transform_cubic_path(path, Affine2D.translation_only(3.0, -4.0))
    np.testing.assert_allclose(translated.start, [4.0, -2.0])
    np.testing.assert_allclose(translated.segments[0].c1, [5.0, -1.0])
    np.testing.assert_allclose(translated.segments[0].c2, [7.0, -1.0])
    np.testing.assert_allclose(translated.segments[0].end, [8.0, -3.0])
    np.testing.assert_allclose(translated.segments[1].end, [11.0, -2.0])
    assert len(translated.segments) == len(path.segments)


def test_transform_cubic_path_similarity_matches_affine_and_sampled_points():
    path = CubicPath(
        start=(1.0, 2.0),
        segments=(
            CubicSegment((2.0, 3.0), (4.0, 3.0), (5.0, 1.0)),
            CubicSegment((6.0, 0.0), (7.0, 3.0), (8.0, 2.0)),
        ),
    )
    transform = Similarity2D(scale=1.7, rotation_deg=37.0, dx=5.0, dy=-8.0)
    transformed = transform_cubic_path(path, transform)
    affine_transformed = transform_cubic_path(path, transform.as_affine())

    assert transformed == affine_transformed
    parameters = np.linspace(0.0, 1.0, 17)
    for original_samples, transformed_samples in zip(
        _sample_path(path, parameters),
        _sample_path(transformed, parameters),
        strict=True,
    ):
        np.testing.assert_allclose(
            transformed_samples,
            transform.apply(original_samples),
            atol=1e-12,
        )
