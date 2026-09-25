import numpy as np
from uav_gap.reference import feasibility_reference


def test_reference_is_continuous_at_ballistic_segment_boundaries():
    for time in (2.2, 2.6, 4.6):
        left = feasibility_reference(time - 1e-6)
        right = feasibility_reference(time + 1e-6)
        for a, b in zip(left, right):
            assert np.allclose(a, b, atol=1e-4)


def test_reference_crosses_gate_center_with_force_direction_matching_roll():
    p, v, a = feasibility_reference(2.4)
    assert np.allclose(p, [0, 0, 1.5])
    assert np.isclose(np.arctan2(-a[1], 9.81 + a[2]), np.pi/4)
    assert v[0] == 3
