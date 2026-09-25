"""Episode accounting separate from rewards, with complete-body crossing."""

import torch


class EpisodeTracker:
    def __init__(self, num_envs, device, dt=.005, hold_seconds=.5, timeout_seconds=8.):
        self.dt, self.hold_seconds, self.timeout_seconds = dt, hold_seconds, timeout_seconds
        self.seen_before = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.crossed = self.seen_before.clone()
        self.done = self.seen_before.clone()
        self.elapsed = torch.zeros(num_envs, device=device)
        self.stable = torch.zeros_like(self.elapsed)

    def reset(self, indices):
        for array in (self.seen_before, self.crossed, self.done, self.elapsed, self.stable):
            array[indices] = 0

    def update(self, before, after, collision, out_of_bounds, goal_distance, speed, tilt):
        active = ~self.done
        self.elapsed += active * self.dt
        self.seen_before |= before & active
        failed = active & (collision | out_of_bounds)
        newly_crossed = active & ~failed & ~self.crossed & self.seen_before & after
        self.crossed |= newly_crossed
        stable = (active & ~failed & self.crossed & (goal_distance < .3)
                  & (speed < .3) & (tilt < torch.pi / 18))
        self.stable = torch.where(stable, self.stable + self.dt, torch.zeros_like(self.stable))
        success = active & ~failed & (self.stable >= self.hold_seconds - 1e-6)
        timeout = active & ~failed & ~success & (self.elapsed >= self.timeout_seconds - 1e-6)
        self.done |= failed | success | timeout
        return dict(crossing=newly_crossed, success=success, failure=failed, timeout=timeout)
