from __future__ import annotations

import numpy as np
import pytest

from planarcurve.fitting.parameters import (
    HeadingRateParameterization,
    decode_interval,
    decode_positive,
    encode_interval,
    encode_positive,
    ordered_positions_from_logits,
    ordered_positions_to_logits,
)
from planarcurve.trajectory import HeadingRateChange, HeadingRateProfile


def test_ordered_positions_round_trip_and_configurable_spacing():
    logits = np.array([0.3, -1.2, 2.0, 0.7])
    positions = ordered_positions_from_logits(logits, min_spacing=0.02)
    assert positions.size == 3
    assert np.all(np.diff(positions) >= 0.02 - 1e-12)
    np.testing.assert_allclose(
        ordered_positions_from_logits(
            ordered_positions_to_logits(positions, min_spacing=0.02), min_spacing=0.02
        ),
        positions,
        atol=1e-12,
    )


def test_position_parameterization_is_generic_without_spacing_policy():
    positions = ordered_positions_from_logits(np.array([-2.0, 3.0, 0.5]))
    assert np.all(np.diff(positions) > 0.0)
    assert positions[0] > 0.0 and positions[-1] < 1.0


def test_extreme_position_logits_stay_strictly_inside_the_interval():
    positions = ordered_positions_from_logits(np.array([-1000.0, 1000.0, -1000.0]))
    assert 0.0 < positions[0] < positions[1] < 1.0


def test_interval_and_positive_encoders_round_trip():
    for value in (-12.0, 0.0, 4.0):
        encoded = encode_interval(value, low=-20.0, high=10.0)
        decoded = decode_interval(encoded, low=-20.0, high=10.0)
        assert decoded == pytest.approx(value)
    assert decode_positive(encode_positive(2.5)) == pytest.approx(2.5)
    assert decode_positive(-1000.0) > 0.0


def test_parameterization_rejects_invalid_domain_inputs():
    with pytest.raises(ValueError, match="spacing"):
        ordered_positions_from_logits(np.zeros(4), min_spacing=0.5)
    with pytest.raises(ValueError, match="strictly ordered"):
        ordered_positions_to_logits(np.array([0.6, 0.5]))
    with pytest.raises(ValueError, match="positive"):
        encode_positive(0.0)



def test_heading_rate_parameterization_rejects_mismatched_profiles_and_values():
    codec = HeadingRateParameterization(
        change_count=1,
        min_change_spacing=0.1,
        max_heading_rate_deg_per_u=10.0,
    )
    profile = HeadingRateProfile(
        start_rate_deg_per_u=1.0,
        changes=(HeadingRateChange(0.5, 2.0),),
        end_rate_deg_per_u=3.0,
    )
    assert codec.decode(codec.encode(profile)) == profile

    with pytest.raises(ValueError, match="change count"):
        codec.encode(HeadingRateProfile(1.0, 2.0))
    with pytest.raises(ValueError, match="interpolation"):
        codec.encode(HeadingRateProfile(1.0, 2.0, changes=profile.changes, interpolation="linear"))
    with pytest.raises(ValueError, match="max_heading_rate"):
        codec.encode(HeadingRateProfile(20.0, 2.0, changes=profile.changes))
    with pytest.raises(ValueError, match="length"):
        codec.decode(np.zeros(codec.size + 1))
    with pytest.raises(ValueError, match="finite"):
        codec.decode(np.full(codec.size, np.nan))
    with pytest.raises(ValueError, match="spacing"):
        HeadingRateParameterization(3, 0.5, 10.0)
