from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, radians, sin

import numpy as np
from numpy.typing import ArrayLike

from .types import CubicPath, CubicSegment, FloatArray


@dataclass(frozen=True)
class Affine2D:
    """2D affine map with row-major linear coefficients and translation.

    Points are mapped as ``[x', y'] = [[m00, m01], [m10, m11]] @ [x, y] + [dx, dy]``.
    ``apply_vectors()`` uses only the linear matrix. ``compose(other)`` means
    ``self ∘ other``: apply ``other`` first, then ``self``.
    """

    m00: float = 1.0
    m01: float = 0.0
    m10: float = 0.0
    m11: float = 1.0
    dx: float = 0.0
    dy: float = 0.0

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in self.as_tuple()):
            raise ValueError("affine transform values must be finite")

    @classmethod
    def identity(cls) -> Affine2D:
        """Return the identity affine transform."""
        return cls()

    @classmethod
    def translation_only(cls, dx: float = 0.0, dy: float = 0.0) -> Affine2D:
        """Return a translation-only affine transform."""
        return cls(dx=dx, dy=dy)

    @classmethod
    def scale(cls, sx: float, sy: float | None = None) -> Affine2D:
        """Return a scale transform, using a uniform scale when sy is omitted."""
        return cls(m00=sx, m11=sx if sy is None else sy)

    @classmethod
    def rotation(cls, angle_deg: float) -> Affine2D:
        """Return a rotation transform for an angle in degrees."""
        angle = radians(angle_deg)
        c, s = cos(angle), sin(angle)
        return cls(c, -s, s, c)

    @classmethod
    def from_basis(cls, origin: ArrayLike, x_axis: ArrayLike, y_axis: ArrayLike) -> Affine2D:
        """Build an affine transform from an origin and two basis vectors."""
        return cls(x_axis[0], y_axis[0], x_axis[1], y_axis[1], origin[0], origin[1])

    @classmethod
    def from_scale_shear_translation(
        cls,
        *,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        shear_x: float = 0.0,
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> Affine2D:
        """Build x'=scale_x*(x+shear_x*y)+dx, y'=scale_y*y+dy."""
        return cls(scale_x, scale_x * shear_x, 0.0, scale_y, dx, dy)

    @property
    def linear(self) -> FloatArray:
        """Return the 2x2 linear component as a float array."""
        return np.array([[self.m00, self.m01], [self.m10, self.m11]], dtype=float)

    @property
    def translation(self) -> FloatArray:
        """Return the translation as a float vector."""
        return np.array([self.dx, self.dy], dtype=float)

    @property
    def determinant(self) -> float:
        """Return the determinant of the linear component."""
        return self.m00 * self.m11 - self.m01 * self.m10

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        """Return the six affine parameters in row-major order."""
        return self.m00, self.m01, self.m10, self.m11, self.dx, self.dy

    def apply(self, points: ArrayLike) -> FloatArray:
        """Apply the affine transform to one point or an array of points."""
        values = np.asarray(points, dtype=float)
        if values.shape[-1] != 2 or not np.isfinite(values).all():
            raise ValueError("points must contain finite 2-D coordinates")
        return values @ self.linear.T + self.translation

    def apply_vectors(self, vectors: ArrayLike) -> FloatArray:
        """Apply the linear component to vectors without translation."""
        values = np.asarray(vectors, dtype=float)
        if values.shape[-1] != 2 or not np.isfinite(values).all():
            raise ValueError("vectors must contain finite 2-D coordinates")
        return values @ self.linear.T

    def inverse(self) -> Affine2D:
        """Return the inverse transform, raising for singular maps."""
        determinant = self.determinant
        if determinant == 0.0:
            raise ValueError("singular affine transform has no inverse")
        inverse_linear = (
            np.array([[self.m11, -self.m01], [-self.m10, self.m00]], dtype=float) / determinant
        )
        inverse_translation = -(inverse_linear @ self.translation)
        return Affine2D(
            inverse_linear[0, 0],
            inverse_linear[0, 1],
            inverse_linear[1, 0],
            inverse_linear[1, 1],
            inverse_translation[0],
            inverse_translation[1],
        )

    def compose(self, other: Affine2D) -> Affine2D:
        """Return ``self ∘ other``: apply ``other`` first, then ``self``."""
        linear = self.linear @ other.linear
        translation = self.linear @ other.translation + self.translation
        return Affine2D(
            linear[0, 0],
            linear[0, 1],
            linear[1, 0],
            linear[1, 1],
            translation[0],
            translation[1],
        )


@dataclass(frozen=True)
class Similarity2D:
    """Positive uniform scale, rotation in degrees, and translation in 2D.

    Applying the transform scales and rotates a point, then adds ``(dx, dy)``.
    ``as_affine()`` returns the equivalent affine map. This representation
    preserves shape because it does not include nonuniform scale or shear.
    """

    scale: float = 1.0
    rotation_deg: float = 0.0
    dx: float = 0.0
    dy: float = 0.0

    def __post_init__(self) -> None:
        values = (self.scale, self.rotation_deg, self.dx, self.dy)
        if not all(isfinite(value) for value in values):
            raise ValueError("similarity transform values must be finite")
        if self.scale <= 0.0:
            raise ValueError("scale must be positive")

    def as_affine(self) -> Affine2D:
        """Return the equivalent affine transform."""
        angle = radians(self.rotation_deg)
        c = self.scale * cos(angle)
        s = self.scale * sin(angle)
        return Affine2D(c, -s, s, c, self.dx, self.dy)

    def apply(self, points: ArrayLike) -> FloatArray:
        """Apply the similarity transform to one or more points."""
        return self.as_affine().apply(points)


def transform_cubic_path(path: CubicPath, transform: Affine2D | Similarity2D) -> CubicPath:
    """Return a cubic path with an affine or similarity transform applied."""
    affine = transform.as_affine() if isinstance(transform, Similarity2D) else transform
    return CubicPath(
        start=affine.apply(path.start),
        segments=tuple(
            CubicSegment(
                c1=affine.apply(segment.c1),
                c2=affine.apply(segment.c2),
                end=affine.apply(segment.end),
            )
            for segment in path.segments
        ),
    )


def fit_similarity_transform(
    target: ArrayLike,
    source: ArrayLike,
    *,
    weights: ArrayLike | None = None,
) -> Similarity2D:
    """Fit an orientation-preserving similarity mapping ``source`` onto ``target``.

    The fitted transform has positive uniform scale, rotation, and translation.
    Reflections are not representable by :class:`Similarity2D` and are never
    fitted. Optional nonnegative weights assign each point pair's relative
    contribution to the least-squares fit.
    """
    target_values = np.asarray(target, dtype=float)
    source_values = np.asarray(source, dtype=float)
    if (
        target_values.ndim != 2
        or target_values.shape[1] != 2
        or source_values.shape != target_values.shape
        or len(target_values) < 2
    ):
        raise ValueError("target and source points must share shape (N, 2), with N >= 2")
    if not np.isfinite(target_values).all() or not np.isfinite(source_values).all():
        raise ValueError("similarity-fit points must contain finite coordinates")

    if weights is None:
        fit_weights = np.ones(len(target_values), dtype=float)
    else:
        fit_weights = np.asarray(weights, dtype=float)
        if fit_weights.shape != (len(target_values),):
            raise ValueError("weights must have shape (N,)")
        if not np.isfinite(fit_weights).all() or np.any(fit_weights < 0.0):
            raise ValueError("weights must be finite and nonnegative")
    maximum_weight = float(np.max(fit_weights))
    if maximum_weight == 0.0:
        raise ValueError("weights must have positive total weight")
    fit_weights = fit_weights / maximum_weight
    fit_weights /= fit_weights.sum()
    active = fit_weights > 0.0
    fit_weights = fit_weights[active]
    source_values = source_values[active]
    target_values = target_values[active]

    first = int(np.flatnonzero(fit_weights)[0])
    source_origin = source_values[first]
    target_origin = target_values[first]
    with np.errstate(over="ignore", invalid="ignore"):
        source_offsets = source_values - source_origin
        target_offsets = target_values - target_origin
    if not np.isfinite(source_offsets).all() or not np.isfinite(target_offsets).all():
        raise ValueError("similarity fit is numerically singular")

    source_extent = float(np.max(np.abs(source_offsets)))
    if source_extent == 0.0:
        raise ValueError("source points must have nonzero spatial extent")
    target_extent = float(np.max(np.abs(target_offsets)))
    if target_extent == 0.0:
        raise ValueError("target points must have nonzero spatial extent")

    source_scaled = source_offsets / source_extent
    target_scaled = target_offsets / target_extent
    source_mean_scaled = fit_weights @ source_scaled
    target_mean_scaled = fit_weights @ target_scaled
    source_centered = source_scaled - source_mean_scaled
    target_centered = target_scaled - target_mean_scaled
    source_variance = float(np.sum(fit_weights * np.sum(source_centered**2, axis=1)))
    target_variance = float(np.sum(fit_weights * np.sum(target_centered**2, axis=1)))
    if not np.isfinite(source_variance) or source_variance <= np.finfo(float).eps:
        raise ValueError("source points must have nonzero spatial extent")
    if not np.isfinite(target_variance) or target_variance <= np.finfo(float).eps:
        raise ValueError("similarity fit is numerically singular")

    covariance = source_centered.T @ (fit_weights[:, None] * target_centered)
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.eye(2)
    correction[-1, -1] = 1.0 if np.linalg.det(left @ right_t) >= 0.0 else -1.0
    row_rotation = left @ correction @ right_t
    score = float(np.sum(fit_weights[:, None] * (source_centered @ row_rotation) * target_centered))
    singularity_threshold = np.finfo(float).eps * np.sqrt(source_variance * target_variance)
    if not np.isfinite(score) or score <= singularity_threshold:
        raise ValueError("similarity fit is numerically singular")

    normalized_scale = score / source_variance
    scale = (target_extent / source_extent) * normalized_scale
    source_mean = source_origin + source_extent * source_mean_scaled
    target_mean = target_origin + target_extent * target_mean_scaled
    translation = target_mean - scale * (source_mean @ row_rotation)
    if not np.isfinite(scale) or scale <= 0.0 or not np.isfinite(translation).all():
        raise ValueError("similarity fit is numerically singular")
    rotation_deg = float(np.degrees(np.arctan2(row_rotation[0, 1], row_rotation[0, 0])))
    return Similarity2D(
        scale=float(scale),
        rotation_deg=rotation_deg,
        dx=float(translation[0]),
        dy=float(translation[1]),
    )
