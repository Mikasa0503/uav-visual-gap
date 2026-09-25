"""Conservative full-vehicle box geometry, including rotor envelope.

All operations are batched torch operations. Call at EVERY physics substep;
endpoint checks alone do not provide continuous collision detection.
Frames: gate x is normal, y is aperture long axis, z is short axis.
Rotations map body/local coordinates to world coordinates.
"""

import torch


def roll_matrix(angle):
    c, s = torch.cos(angle), torch.sin(angle)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((o, z, z, z, c, -s, z, s, c), dim=-1).reshape(*angle.shape, 3, 3)


def gate_envelope(position, rotation, half_size, gate_center, gate_rotation):
    inverse = gate_rotation.transpose(-1, -2)
    center = torch.matmul(inverse, (position - gate_center).unsqueeze(-1)).squeeze(-1)
    relative_rotation = torch.matmul(inverse, rotation)
    extent = torch.matmul(relative_rotation.abs(), half_size.unsqueeze(-1)).squeeze(-1)
    return center, extent


def gate_status(position, rotation, half_size, gate_center, gate_rotation,
                aperture_half_size, wall_half_thickness):
    """Return collision, fully_before, fully_after and minimum aperture margin.

    This deliberately uses an enclosing box. The physical collision asset must
    use the same dimensions, preventing a mismatch with rotor visualization.
    """
    center, extent = gate_envelope(position, rotation, half_size, gate_center, gate_rotation)
    margin = aperture_half_size - (center[..., 1:].abs() + extent[..., 1:])
    overlap = center[..., 0].abs() <= extent[..., 0] + wall_half_thickness
    collision = overlap & (margin.min(dim=-1).values <= 0)
    before = center[..., 0] + extent[..., 0] < -wall_half_thickness
    after = center[..., 0] - extent[..., 0] > wall_half_thickness
    return collision, before, after, margin.min(dim=-1).values


def swept_gate_collision(p0, r0, p1, r1, half_size, gate_center, gate_rotation,
                         aperture_half_size, wall_half_thickness):
    """Conservative swept enclosure over one physics interval.

    Translation uses endpoint interval hulls; rotation adds a radius×angle
    bound to cover the shortest interpolated rotation. This can overestimate
    collision, never make a high-speed teleport through the wall look valid.
    """
    c0, e0 = gate_envelope(p0, r0, half_size, gate_center, gate_rotation)
    c1, e1 = gate_envelope(p1, r1, half_size, gate_center, gate_rotation)
    relative = r0.transpose(-1, -2) @ r1
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    padding = torch.linalg.vector_norm(half_size, dim=-1) * torch.acos(cosine)
    lo = torch.minimum(c0 - e0, c1 - e1) - padding[..., None]
    hi = torch.maximum(c0 + e0, c1 + e1) + padding[..., None]
    overlap = (lo[..., 0] <= wall_half_thickness) & (hi[..., 0] >= -wall_half_thickness)
    outside = ((lo[..., 1:] <= -aperture_half_size) | (hi[..., 1:] >= aperture_half_size)).any(-1)
    return overlap & outside
