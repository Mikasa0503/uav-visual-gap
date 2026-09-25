import pytest
import torch
from uav_gap.teacher_sensors import TeacherObservationDelay


def test_two_frame_state_delay_keeps_commands_current_and_has_no_future_leak():
    stream = TeacherObservationDelay(2)
    for step in range(6):
        observation = torch.full((3,27),float(step))
        delayed = stream(observation)
        expected = max(0,step-2)
        assert torch.equal(delayed[:,:12],torch.full((3,12),float(expected)))
        assert torch.equal(delayed[:,16:],torch.full((3,11),float(expected)))
        assert torch.equal(delayed[:,12:16],torch.full((3,4),float(step)))
        # Input buffers may be reused by the simulator without corrupting history.
        observation.fill_(99)


def test_zero_delay_is_identity_and_invalid_inputs_are_rejected():
    stream = TeacherObservationDelay(0)
    observation = torch.randn(2,27)
    torch.testing.assert_close(stream(observation),observation)
    with pytest.raises(ValueError):
        TeacherObservationDelay(-1)
    with pytest.raises(ValueError):
        stream(torch.zeros(2,16))
