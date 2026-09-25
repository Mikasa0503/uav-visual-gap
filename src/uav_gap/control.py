"""SI-unit collective thrust/body-rate controller for project-owned use."""
import torch


def rate_wrench(actions, body_rates, inertia, gain=20.):
    """Actions: thrust in N, rates in rad/s; output: body wrench in N/Nm.

    Uses inertia-scaled rate feedback and gyroscopic compensation. It never
    reads a target position, gap pose, or privileged navigation state.
    """
    desired = actions[:, 1:].clone()
    desired[:, :2].clamp_(-8., 8.)
    desired[:, 2].clamp_(-4., 4.)
    angular_momentum = torch.bmm(inertia, body_rates.unsqueeze(-1)).squeeze(-1)
    feedback = torch.bmm(inertia, (gain * (desired - body_rates)).unsqueeze(-1)).squeeze(-1)
    wrench = torch.zeros((actions.shape[0], 6), device=actions.device, dtype=actions.dtype)
    wrench[:, 2] = actions[:, 0].clamp(0., 8.)
    wrench[:, 3:] = feedback + torch.cross(body_rates, angular_momentum, dim=-1)
    return wrench


def register_rate_controller(physical_inertia_scale=1.):
    # Delayed import preserves Isaac Gym's import-before-torch requirement at
    # application entrypoints; unit tests of rate_wrench need no simulator.
    from aerial_gym.control.controllers.base_lee_controller import BaseLeeController
    from aerial_gym.config.controller_config.lee_controller_config import control
    from aerial_gym.registry.controller_registry import controller_registry

    class CollectiveRateController(BaseLeeController):
        def update(self, command_actions):
            # The fixed controller retains nominal calibration under mass stress.
            return rate_wrench(command_actions, self.robot_body_angvel,
                               self.robot_inertia / physical_inertia_scale)

    controller_registry.register_controller('gap_collective_rates', CollectiveRateController, control)
