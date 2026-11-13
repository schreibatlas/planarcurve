"""Optional local parity check against an InkGrammar checkout.

Usage:
    PYTHONPATH=/path/to/inkgrammar:. python tools/parity_with_inkgrammar.py
"""

from __future__ import annotations

import numpy as np
from inkgrammar.fitting.curvature import evaluate_curvature_stroke as ink_evaluate
from inkgrammar.fitting.models import CurvatureStrokeGeometry as InkGeometry

from planarcurve.curvature import (
    CurvatureChange,
    CurvatureProfile,
    CurvatureTrajectory,
    evaluate_curvature_stroke,
)

payload = {
    "length": 1.7,
    "start_angle_deg": 25.0,
    "curvature": {
        "interpolation": "cubic-hermite",
        "start_rate_deg": 40.0,
        "changes": [
            {"at": 0.25, "rate_deg": 260.0},
            {"at": 0.55, "rate_deg": 200.0},
            {"at": 0.8, "rate_deg": -80.0},
        ],
        "end_rate_deg": 10.0,
    },
}
ink = ink_evaluate(InkGeometry.model_validate(payload), count=257)
ours = evaluate_curvature_stroke(
    CurvatureTrajectory(
        length=payload["length"],
        start_angle_deg=payload["start_angle_deg"],
        profile=CurvatureProfile(
            interpolation="cubic-hermite",
            start_rate_deg=40.0,
            changes=(
                CurvatureChange(0.25, 260.0),
                CurvatureChange(0.55, 200.0),
                CurvatureChange(0.8, -80.0),
            ),
            end_rate_deg=10.0,
        ),
    ),
    count=257,
)
np.testing.assert_array_equal(ours.u, ink.u)
np.testing.assert_allclose(ours.xy, ink.xy, rtol=0, atol=0)
np.testing.assert_allclose(ours.heading_deg, ink.heading_deg, rtol=0, atol=0)
np.testing.assert_allclose(ours.heading_rate_deg, ink.heading_rate_deg, rtol=0, atol=0)
print("curvature parity: exact")
