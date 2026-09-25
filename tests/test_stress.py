import xml.etree.ElementTree as ET
import numpy as np
import pytest
import torch
from uav_gap.assets import scale_inertial_properties
from uav_gap.reset_state import quaternion_xyzw, sample_attitude_rates
from uav_gap.sensors import ObservationStream
from uav_gap.stress import CONDITIONS, stress_protocol, apply_scheduled_impulse


def test_mass_stress_scales_physics_and_preserves_collision():
    xml = '<robot><link><inertial><mass value=".25"/><inertia ixx=".01" iyy=".02" izz=".03"/></inertial><collision><geometry><box size=".46 .46 .12"/></geometry></collision></link></robot>'
    root = ET.fromstring(scale_inertial_properties(xml, 1.2))
    assert float(root.find('./link/inertial/mass').get('value')) == pytest.approx(.3)
    assert float(root.find('./link/inertial/inertia').get('izz')) == pytest.approx(.036)
    assert root.find('./link/collision/geometry/box').get('size') == '.46 .46 .12'


def test_reset_protocol_is_seeded_bounded_and_zero_when_disabled():
    rpy, rates = sample_attitude_rates(np.random.default_rng(11), 100, True)
    rpy2, rates2 = sample_attitude_rates(np.random.default_rng(11), 100, True)
    assert np.array_equal(rpy, rpy2) and np.array_equal(rates, rates2)
    assert (abs(rpy) <= np.deg2rad([8,8,10])).all() and (abs(rates) <= .25).all()
    assert abs(rpy).sum() > 0 and abs(rates).sum() > 0
    assert np.allclose(np.linalg.norm(quaternion_xyzw(rpy), axis=-1), 1.)
    assert np.allclose(quaternion_xyzw([[np.pi/2,0,0]])[0], [2**-.5,0,0,2**-.5])
    rpy, rates = sample_attitude_rates(np.random.default_rng(11), 2, False)
    assert not rpy.any() and not rates.any()


def test_three_missing_acquisitions_hold_depth_but_not_proprioception():
    stream = ObservationStream(delay_steps=1, drop_start=2, drop_frames=3)
    actual = []
    for step in range(7):
        result = stream(dict(depth=torch.full((1,1,64,64), float(step)),
                             proprio=torch.full((1,16), float(step))))
        actual.append(float(result['depth'][0,0,0,0]))
        assert result['proprio'][0,0] == max(0,step-1)
        assert result['proprio'][0,-1] == step
    assert actual == [0,0,1,1,1,1,5]


def test_stress_conditions_are_separate_and_impulse_is_one_shot():
    for condition in CONDITIONS:
        p = stress_protocol(condition)
        assert p['mass_scale'] == (1.2 if condition=='mass_plus20' else 1.)
        assert p['observation_delay_steps'] == (3 if condition=='delay_40ms' else 1)
    class Task:
        num_envs = 2
        calls = []
        def apply_world_impulse(self, value):
            self.calls.append(value)
    task = Task()
    for step in range(100):
        apply_scheduled_impulse(task, stress_protocol('lateral_impulse'), step)
    assert task.calls == [[[0.,.1,0.], [0.,.1,0.]]]
