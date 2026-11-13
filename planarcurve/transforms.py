from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, radians, sin

import numpy as np


@dataclass(frozen=True)
class Affine2D:
    """Full 2x2 affine linear map plus translation."""

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
        return cls()

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
        # Matches InkGrammar TopologyTransform semantics:
        # x' = scale_x * (x + shear_x*y) + dx; y' = scale_y*y + dy.
        return cls(scale_x, scale_x * shear_x, 0.0, scale_y, dx, dy)

    @property
    def linear(self) -> np.ndarray:
        return np.array([[self.m00, self.m01], [self.m10, self.m11]], dtype=float)

    @property
    def translation(self) -> np.ndarray:
        return np.array([self.dx, self.dy], dtype=float)

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        return self.m00, self.m01, self.m10, self.m11, self.dx, self.dy

    def apply(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        if values.shape[-1] != 2 or not np.isfinite(values).all():
            raise ValueError("points must contain finite 2-D coordinates")
        return values @ self.linear.T + self.translation


@dataclass(frozen=True)
class Similarity2D:
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
        angle = radians(self.rotation_deg)
        c = self.scale * cos(angle)
        s = self.scale * sin(angle)
        return Affine2D(c, -s, s, c, self.dx, self.dy)

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.as_affine().apply(points)
