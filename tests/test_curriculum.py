import numpy as np
import pytest
from uav_gap.curriculum import curriculum_geometry

TABLE=((1.4,1.4,0.),(1.,.7,20.),(.8,.36,35.),(.66,.24,45.))

def test_default_preserves_all_original_geometries_exactly():
    levels=np.array([0,1,2,3,2,0])
    expected=np.array([TABLE[i] for i in levels],dtype=np.float32)
    assert np.array_equal(curriculum_geometry(levels,TABLE,3),expected)

def test_bridge_changes_current_level_only():
    actual=curriculum_geometry(np.array([0,2,1,2]),TABLE,2,.5)
    np.testing.assert_array_equal(actual[[0,2]],np.array([TABLE[0],TABLE[1]],dtype=np.float32))
    np.testing.assert_allclose(actual[[1,3]],[[.9,.53,27.5],[.9,.53,27.5]])

@pytest.mark.parametrize('fraction',[0,-.1,1.1,float('nan'),float('inf')])
def test_invalid_fraction_rejected(fraction):
    with pytest.raises(ValueError):curriculum_geometry([2],TABLE,2,fraction)

def test_level_zero_cannot_interpolate():
    with pytest.raises(ValueError):curriculum_geometry([0],TABLE,0,.5)
    np.testing.assert_array_equal(curriculum_geometry([0],TABLE,0),np.array([TABLE[0]],dtype=np.float32))
