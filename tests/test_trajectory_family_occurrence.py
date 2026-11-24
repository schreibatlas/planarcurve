from __future__ import annotations

import numpy as np

from planarcurve.fitting import (
    TrajectoryFitOptions,
    TrajectoryVariationPolicy,
    fit_trajectory_family,
    fit_trajectory_family_occurrence,
)
from planarcurve.transforms import Similarity2D, SimilarityFitBounds


def test_fixed_family_occurrence_is_directed_and_does_not_mutate_model():
    options = TrajectoryFitOptions(change_count=0, samples=32, restarts=0, max_nfev=80)
    points = np.column_stack((np.linspace(0.0, 1.0, 40), np.sin(np.linspace(0.0, 2.0, 40))))
    learned = fit_trajectory_family([points], fit_options=options).model
    before = learned

    result = fit_trajectory_family_occurrence(
        learned, points, fit_options=options, variation_policy=TrajectoryVariationPolicy()
    )
    reverse = fit_trajectory_family_occurrence(learned, points[::-1], fit_options=options)

    assert learned == before
    assert result.residuals
    assert not result.placement_constrained
    assert reverse.position_rmse > result.position_rmse


def test_fixed_family_fit_reports_bounded_placement_residual():
    options = TrajectoryFitOptions(change_count=0, samples=32, restarts=0, max_nfev=80)
    points = np.column_stack((np.linspace(0.0, 1.0, 40), np.sin(np.linspace(0.0, 2.0, 40))))
    model = fit_trajectory_family([points], fit_options=options).model
    rotated = Similarity2D(rotation_deg=30.0).apply(points)

    result = fit_trajectory_family_occurrence(
        model,
        rotated,
        fit_options=options,
        placement_bounds=SimilarityFitBounds.symmetric_rotation(10.0),
    )

    assert result.placement_constrained
    assert result.position_rmse > 0.01
