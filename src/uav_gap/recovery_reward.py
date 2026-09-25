"""Optional dense post-crossing feedback; success and physics are unchanged."""
import torch


def recovery_feedback(crossed, distance, speed, upright_cosine):
    # Each component remains informative when another criterion is far away.
    # Bounded penalty <=0.11/step; no penalty before full-body crossing.
    cost = .05 * torch.tanh(distance) + .04 * torch.tanh(speed)
    cost += .01 * (1. - upright_cosine.clamp(-1., 1.))
    return -crossed.to(distance.dtype) * cost
