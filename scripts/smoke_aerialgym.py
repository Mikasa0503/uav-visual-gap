"""Exercise real PhysX, body-rate control and Warp depth; no policy claims."""

import argparse
import json
import sys
import time
from uav_gap.runtime import import_simulator, require_idle_gpu, project_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=2)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    require_idle_gpu(args.gpu)
    # Upstream's argument parser must not consume our application's flags.
    sys.argv = [sys.argv[0]]
    import_simulator()
    import numpy as np
    import torch
    import warp as wp
    from aerial_gym.config.robot_config.base_quad_config import BaseQuadWithCameraCfg
    from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import BaseDepthCameraConfig
    from aerial_gym.config.env_config.env_with_obstacles import EnvWithObstaclesCfg
    from aerial_gym.config.sim_config.base_sim_config import BaseSimConfig
    from aerial_gym.registry.robot_registry import robot_registry
    from aerial_gym.registry.env_registry import env_config_registry
    from aerial_gym.registry.sim_registry import sim_config_registry
    from aerial_gym.robots.base_multirotor import BaseMultirotor
    from aerial_gym.sim.sim_builder import SimBuilder
    from uav_gap.control import register_rate_controller

    output = project_output('runs/runtime_smoke')
    output.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(project_output('.cache/warp'))
    torch.manual_seed(11)
    np.random.seed(11)

    class Camera(BaseDepthCameraConfig):
        width = 64
        height = 64
        randomize_placement = False
        segmentation_camera = False

    class Robot(BaseQuadWithCameraCfg):
        class sensor_config(BaseQuadWithCameraCfg.sensor_config):
            camera_config = Camera

        class init_config(BaseQuadWithCameraCfg.init_config):
            min_init_state = [0.5, 0.5, 0.5, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]
            max_init_state = min_init_state.copy()

        class control_allocator_config(BaseQuadWithCameraCfg.control_allocator_config):
            # Apply the bounded, delayed four-motor resultant to the COM.
            # This avoids relying on URDF rigid-body enumeration for motors.
            force_application_level = 'root_link'
            application_mask = [0]

    class Environment(EnvWithObstaclesCfg):
        class env(EnvWithObstaclesCfg.env):
            num_physics_steps_per_env_step_mean = 4
            num_physics_steps_per_env_step_std = 0
            sample_timestep_for_latency = False
            perturb_observations = False
            reset_on_collision = False
            lower_bound_min = [-3., -3., -3.]
            lower_bound_max = [-3., -3., -3.]
            upper_bound_min = [3., 3., 3.]
            upper_bound_max = [3., 3., 3.]

        class env_config(EnvWithObstaclesCfg.env_config):
            include_asset_type = {key: key.endswith('_wall') for key in
                                  EnvWithObstaclesCfg.env_config.include_asset_type}

    class Simulation(BaseSimConfig):
        class sim(BaseSimConfig.sim):
            dt = 0.005

    robot_registry.register('gap_smoke_quad', BaseMultirotor, Robot)
    env_config_registry.register('gap_smoke_room', Environment)
    sim_config_registry.register('gap_smoke_sim', Simulation)
    register_rate_controller()
    env = SimBuilder().build_env(sim_name='gap_smoke_sim', env_name='gap_smoke_room',
                                robot_name='gap_smoke_quad', controller_name='gap_collective_rates',
                                device='cuda:0', num_envs=args.num_envs, headless=True, use_warp=True)
    try:
        env.reset()
        env.render_sensors()
        tensors = env.get_obs()
        positions = tensors['robot_position']
        start = positions.clone()
        actions = torch.zeros((args.num_envs, 4), device='cuda:0')
        # Public action interface is SI: collective thrust [N], body rates [rad/s].
        actions[:, 0] = tensors['robot_mass'].reshape(-1) * 9.81
        # Start this hover-response test at actuator equilibrium, not the
        # upstream reset's random independent motor thrusts.
        env.robot_manager.robot.control_allocator.motor_model.current_motor_thrust[:] = actions[:, 0:1] / 4
        history = []
        elapsed_start = time.monotonic()
        for step in range(args.steps):
            actions[:, 1] = 0.15 if 25 <= step < 40 else 0.0
            env.step(actions)
            env.render_sensors()
            depth = tensors['depth_range_pixels']
            assert torch.isfinite(positions).all(), 'non-finite physical state'
            assert torch.isfinite(depth).all(), 'non-finite camera pixels'
            history.append(positions.detach().cpu().numpy().copy())
        depth_np = depth.detach().cpu().numpy()
        assert depth_np.shape == (args.num_envs, 1, 64, 64), depth_np.shape
        assert float(depth_np.std()) > 1e-4, 'camera has no scene geometry'
        displacement = float(torch.linalg.vector_norm(positions - start, dim=-1).max())
        assert displacement > 1e-5, 'physics did not advance'
        assert displacement < 1., 'small body-rate pulse caused excessive drift'
        final_rate = float(torch.linalg.vector_norm(tensors['robot_body_angvel'], dim=-1).max())
        assert final_rate < .15, 'body-rate controller did not settle'
        report = dict(kind='infrastructure_only', backend='AerialGym/PhysX/Warp',
                      num_envs=args.num_envs, steps=args.steps, simulated_seconds=args.steps * .02,
                      elapsed_seconds=time.monotonic() - elapsed_start,
                      depth_shape=list(depth_np.shape), depth_std=float(depth_np.std()),
                      max_displacement=displacement, gpu=torch.cuda.get_device_name(0),
                      final_body_rate=final_rate, force_application='root_link_motor_resultant',
                      controller='gap_collective_rates', physics_hz=200, sensor_hz=50)
        np.savez_compressed(output / 'trace.npz', position=np.stack(history), final_depth=depth_np)
        (output / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        env.IGE_env.gym.destroy_sim(env.IGE_env.sim)


if __name__ == '__main__':
    main()
