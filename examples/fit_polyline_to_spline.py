from pathlib import Path

import numpy as np

from planarcurve.fitting.spline import (
    SplineFitOptions,
    SplineObservation,
    fit_spline_control_points,
)
from planarcurve.spline import BSplineCurve

KNOTS = (0.0,) * 4 + (0.25, 0.5, 0.75) + (1.0,) * 4
TRUE_CONTROLS = np.array(
    [[0, 50], [15, 10], [30, 95], [50, 5], [70, 90], [85, 15], [100, 50]],
    dtype=float,
)
SEED_CONTROLS = np.array(
    [[0, 50], [15, 30], [30, 65], [50, 40], [70, 60], [85, 30], [100, 50]],
    dtype=float,
)

true_curve = BSplineCurve(KNOTS, tuple(map(tuple, TRUE_CONTROLS)))
seed_curve = BSplineCurve(KNOTS, tuple(map(tuple, SEED_CONTROLS)))
u = np.linspace(0.0, 1.0, 81)
rng = np.random.default_rng(7)
observed = true_curve.evaluate(u, derivatives=False).xy
observed = observed + rng.normal(0.0, 1.2, observed.shape)
observed[[20, 57]] += np.array([[0.0, 24.0], [0.0, -20.0]])

fit = fit_spline_control_points(
    seed_curve,
    [SplineObservation(u, observed)],
    options=SplineFitOptions(
        smoothness=0.01,
        seed_weight=0.0001,
        robust_scale=2.0,
        irls_iterations=5,
    ),
)
seed_points = seed_curve.evaluate(u, derivatives=False).xy
fitted_points = fit.curve.evaluate(u, derivatives=False).xy
seed_rms = float(np.sqrt(np.mean(np.sum((seed_points - observed) ** 2, axis=1))))


def polyline(points: np.ndarray, *, color: str, width: float) -> str:
    coordinates = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="{width}" '
        f'points="{coordinates}"/>'
    )


circles = "".join(
    f'<circle cx="{x:.3f}" cy="{y:.3f}" r="1.3" fill="#777"/>'
    for x, y in observed
)
svg = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="640" '
    'viewBox="-5 -5 110 110">'
    + circles
    + polyline(seed_points, color="#d97900", width=0.8)
    + polyline(fitted_points, color="#1769aa", width=1.4)
    + "</svg>"
)
Path("fit_polyline_to_spline.svg").write_text(svg, encoding="utf-8")
print(f"seed RMS: {seed_rms:.3f}")
print(f"fit RMS: {fit.residual_rms:.3f}")
print("wrote fit_polyline_to_spline.svg")
