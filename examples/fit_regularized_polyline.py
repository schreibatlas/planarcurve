"""Fit noisy samples of a cubic curve and inspect analytic spline geometry."""

from __future__ import annotations

import numpy as np

from planarcurve.fitting import fit_polyline_spline


def cubic_bezier(parameters: np.ndarray) -> np.ndarray:
    controls = np.array([[0.0, 0.0], [30.0, 60.0], [70.0, -60.0], [100.0, 0.0]])
    t = parameters[:, None]
    return (
        (1.0 - t) ** 3 * controls[0]
        + 3.0 * (1.0 - t) ** 2 * t * controls[1]
        + 3.0 * (1.0 - t) * t**2 * controls[2]
        + t**3 * controls[3]
    )


def main() -> None:
    rng = np.random.default_rng(19)
    parameters = np.linspace(0.0, 1.0, 121)
    observations = cubic_bezier(parameters)
    observations += rng.normal(0.0, 0.25, observations.shape)
    weights = np.ones(len(observations))
    weights[60] = 0.1

    fit = fit_polyline_spline(observations, weights=weights)
    samples = fit.curve.sample_by_arclength(101)
    assert samples.tangent is not None
    assert samples.curvature is not None
    heading = np.arctan2(samples.tangent[:, 1], samples.tangent[:, 0])
    curvature = samples.curvature
    heading_rate_per_progress = curvature * fit.diagnostics.normalized_scale

    assert np.isfinite(heading).all()
    assert np.isfinite(heading_rate_per_progress).all()
    print(f"controls: {fit.diagnostics.control_count}")
    print(f"normalized RMS residual: {fit.diagnostics.residual_rms:.6f}")
    print(f"heading at midpoint: {np.rad2deg(heading[50]):.2f} degrees")
    print(f"curvature at midpoint: {curvature[50]:.6f} per coordinate unit")
    print(f"d(theta)/dq at midpoint: {heading_rate_per_progress[50]:.6f} radians")


if __name__ == "__main__":
    main()
