"""Frozen independent stress conditions, specified before result inspection."""
CONDITIONS = ('show', 'heldout', 'delay_40ms', 'depth_loss_3', 'mass_plus20', 'lateral_impulse')


def stress_protocol(condition, baseline_delay_steps=1):
    if condition not in CONDITIONS:
        raise ValueError('Unknown evaluation condition')
    return dict(version='stress_v1', condition=condition, policy_dt=.02,
        baseline_delay_steps=baseline_delay_steps,
        observation_delay_steps=baseline_delay_steps+(2 if condition=='delay_40ms' else 0),
        drop_start=50 if condition=='depth_loss_3' else None,
        drop_frames=3 if condition=='depth_loss_3' else 0,
        depth_loss_behavior='hold last successfully acquired depth; proprioception continues',
        mass_scale=1.2 if condition=='mass_plus20' else 1.,
        controller_calibration='nominal mass and inertia; no oracle compensation',
        impulse_step=50 if condition=='lateral_impulse' else None,
        world_impulse_ns=[0., .1, 0.] if condition=='lateral_impulse' else [0.,0.,0.])


def apply_scheduled_impulse(task, protocol, step):
    if step == protocol['impulse_step']:
        task.apply_world_impulse([protocol['world_impulse_ns']]*task.num_envs)
