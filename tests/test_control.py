import torch
from uav_gap.control import rate_wrench


def test_batched_hover_thrust_is_preserved_without_gravity_broadcast():
    actions = torch.tensor([[2.45, 0., 0., 0.], [4.9, 0., 0., 0.]])
    wrench = rate_wrench(actions, torch.zeros(2, 3), torch.eye(3).repeat(2, 1, 1))
    assert wrench.shape == (2, 6)
    assert torch.equal(wrench[:, 2], actions[:, 0])
    assert torch.equal(wrench[:, 3:], torch.zeros(2, 3))


def test_rate_feedback_tracks_body_rates_and_does_not_mutate_actions():
    actions = torch.tensor([[2., 1., 0., 0.]])
    saved = actions.clone()
    inertia = torch.diag(torch.tensor([.001, .001, .002])).unsqueeze(0)
    wrench = rate_wrench(actions, torch.zeros(1, 3), inertia)
    assert torch.allclose(wrench[:, 3:], torch.tensor([[.02, 0., 0.]]))
    assert torch.equal(actions, saved)


def test_negative_collective_is_clamped_to_physical_zero():
    wrench = rate_wrench(torch.tensor([[-1., 0., 0., 0.]]), torch.zeros(1, 3), torch.eye(3)[None])
    assert wrench[0, 2].item() == 0.
