import torch
from uav_gap.sensors import ObservationStream


def observation(value):
    return {'depth': torch.full((2,1,64,64), float(value)),
            'proprio': torch.full((2,16), float(value))}


def test_sensor_delay_is_causal_and_previous_command_is_current():
    stream = ObservationStream(2)
    for step, expected in enumerate([0,0,0,1,2]):
        actual = stream(observation(step))
        assert actual['depth'].eq(expected).all()
        assert actual['proprio'][:, :12].eq(expected).all()
        assert actual['proprio'][:, -4:].eq(step).all()
    stream.reset()
    assert stream(observation(9))['depth'].eq(9).all()


def test_stream_owns_history_and_returned_observations():
    stream = ObservationStream(1)
    initial = observation(1)
    result = stream(initial)
    result['depth'].zero_()
    initial['proprio'].zero_()
    actual = stream(observation(2))
    assert actual['depth'].eq(1).all()
    assert actual['proprio'][:, :12].eq(1).all()
