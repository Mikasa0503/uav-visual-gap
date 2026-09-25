import torch
from uav_gap.geometry import roll_matrix, swept_gate_collision


def sweep(y, roll):
    p0 = torch.tensor([[-1., y, 0.]])
    p1 = torch.tensor([[1., y, 0.]])
    r = roll_matrix(torch.tensor([roll]))
    return swept_gate_collision(p0, r, p1, r, torch.tensor([[.23, .23, .06]]),
        torch.zeros(1, 3), roll_matrix(torch.tensor([torch.pi/4])), torch.tensor([[.33, .12]]), .025)


def test_fast_motion_through_wall_is_detected_without_endpoint_contact():
    assert sweep(1., 0.).item()


def test_fast_aligned_motion_through_opening_is_clear():
    assert not sweep(0., torch.pi/4).item()


def test_wrong_attitude_fast_crossing_is_collision():
    assert sweep(0., 0.).item()
