from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SplineEndpointPolicy:
    """Endpoint constraints used by spline fitting, not by spline geometry."""

    start: Literal["locked", "free", "anchored"] = "locked"
    end: Literal["locked", "free", "anchored"] = "locked"

    def __post_init__(self) -> None:
        allowed = {"locked", "free", "anchored"}
        if self.start not in allowed or self.end not in allowed:
            raise ValueError("endpoint policy must be locked, free, or anchored")
