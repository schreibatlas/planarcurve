from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, radians, sin

import numpy as np
from numpy.typing import ArrayLike

from .types import FloatArray


@dataclass(frozen=True)
class Affine2D:
    """2D affine map with row-major linear coefficients and translation."""

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
    """Uniform scale, rotation, and translation in two dimensions."""

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
