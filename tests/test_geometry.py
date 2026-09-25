import torch
from uav_gap.geometry import gate_status, roll_matrix


def status(x=0., roll=0., y=0.):
    return gate_status(torch.tensor([[x, y, 0.]]), roll_matrix(torch.tensor([roll])),
                       torch.tensor([[.23, .23, .06]]), torch.zeros(1, 3),
                       roll_matrix(torch.tensor([torch.pi / 4])),
                       torch.tensor([[.33, .09]]), .025)


def test_horizontal_vehicle_cannot_pass_tilted_slit():
    collision, _, _, margin = status()
    assert collision.item() and margin.item() < 0


def test_matching_roll_fits_with_full_rotor_envelope():
    collision, _, _, margin = status(roll=torch.pi / 4)
    assert not collision.item() and margin.item() > 0


def test_center_crossing_is_not_full_body_crossing():
    _, _, after, _ = status(x=.1, roll=torch.pi / 4)
    assert not after.item()
    _, _, after, _ = status(x=.3, roll=torch.pi / 4)
    assert after.item()


def test_wall_contact_depends_on_offset_and_slab_overlap():
    assert status(roll=torch.pi / 4, y=.4)[0].item()
    assert not status(x=-1., y=.4)[0].item()


def test_thick_wall_overlap_is_detected_before_center_plane():
    assert status(x=-.24)[0].item()
