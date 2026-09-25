"""Project-owned Aerial Gym task. Import only after Isaac Gym initialization."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from aerial_gym.utils.math import quat_to_rotation_matrix
from uav_gap.assets import VEHICLE_HALF_SIZE
from uav_gap.geometry import gate_status, roll_matrix, swept_gate_collision
from uav_gap.termination import EpisodeTracker
from uav_gap.control import register_rate_controller
from uav_gap.runtime import ROOT
from uav_gap.reset_state import sample_attitude_rates, quaternion_xyzw
from uav_gap.curriculum import curriculum_geometry


@dataclass
class TaskConfig:
    num_envs: int = 64
    seed: int = 11
    cameras: bool = False
    level: int = 0
    randomize: bool = True
    auto_reset: bool = True
    dt: float = .005
    substeps: int = 4
    max_seconds: float = 8.
    perturb_reset: bool = False
    mass_scale: float = 1.
    recovery_reward: str = 'original'
    curriculum_fraction: float = 1.


class GapTask:
    # Nominal geometry is provisional until feasibility test succeeds.
    LEVELS = ((1.4, 1.4, 0.), (1., .7, 20.), (.8, .36, 35.), (.66, .24, 45.))
    teacher_dim = 27
    student_dim = 16
    device = 'cuda:0'

    def __init__(self, config):
        from aerial_gym.config.robot_config.base_quad_config import BaseQuadWithCameraCfg
        from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import BaseDepthCameraConfig
        from aerial_gym.config.env_config.env_with_obstacles import EnvWithObstaclesCfg
        from aerial_gym.config.asset_config.env_object_config import panel_asset_params
        from aerial_gym.config.sim_config.base_sim_config import BaseSimConfig
        from aerial_gym.registry.robot_registry import robot_registry
        from aerial_gym.registry.env_registry import env_config_registry
        from aerial_gym.registry.sim_registry import sim_config_registry
        from aerial_gym.robots.base_multirotor import BaseMultirotor
        from aerial_gym.sim.sim_builder import SimBuilder
        self.cfg = config
        if config.recovery_reward not in ('original', 'dense_v2'):
            raise ValueError('Unknown recovery reward version')
        if config.mass_scale not in (1., 1.2):
            raise ValueError('Supported physical mass scales: 1.0, 1.2')
        self.num_envs = config.num_envs
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)

        class Camera(BaseDepthCameraConfig):
            width = height = 64
            randomize_placement = False
            segmentation_camera = False

        class Robot(BaseQuadWithCameraCfg):
            class sensor_config(BaseQuadWithCameraCfg.sensor_config):
                enable_camera = config.cameras
                camera_config = Camera

            class robot_asset(BaseQuadWithCameraCfg.robot_asset):
                asset_folder = str(ROOT / 'assets/gap')
                file = 'quad_mass120.urdf' if config.mass_scale == 1.2 else 'quad.urdf'
                flip_visual_attachments = False

            class init_config(BaseQuadWithCameraCfg.init_config):
                min_init_state = [.125, .5, .3, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]
                max_init_state = min_init_state.copy()

            class control_allocator_config(BaseQuadWithCameraCfg.control_allocator_config):
                force_application_level = 'root_link'
                application_mask = [0]

        class Panel(panel_asset_params):
            num_assets = 4
            asset_folder = str(ROOT / 'assets/gap')
            file = 'panel.urdf'
            keep_in_env = True
            fix_base_link = True
            flip_visual_attachments = False
            collision_mask = 1
            min_state_ratio = [.5, .5, .3, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]
            max_state_ratio = min_state_ratio.copy()

        class Environment(EnvWithObstaclesCfg):
            class env(EnvWithObstaclesCfg.env):
                num_envs = config.num_envs
                num_physics_steps_per_env_step_mean = 1
                num_physics_steps_per_env_step_std = 0
                reset_on_collision = False
                sample_timestep_for_latency = False
                perturb_observations = False
                create_ground_plane = True
                lower_bound_min = lower_bound_max = [-4., -4., 0.]
                upper_bound_min = upper_bound_max = [4., 4., 5.]

            class env_config:
                include_asset_type = {'gap_panels': True}
                asset_type_to_dict_map = {'gap_panels': Panel}

        class Simulation(BaseSimConfig):
            class sim(BaseSimConfig.sim):
                dt = config.dt
                class physx(BaseSimConfig.sim.physx):
                    contact_collection = 2

        robot_registry.register('visual_gap_quad', BaseMultirotor, Robot)
        env_config_registry.register('visual_gap_world', Environment)
        sim_config_registry.register('visual_gap_physics', Simulation)
        register_rate_controller(physical_inertia_scale=config.mass_scale)
        self.sim = SimBuilder().build_env(sim_name='visual_gap_physics', env_name='visual_gap_world',
            robot_name='visual_gap_quad', controller_name='gap_collective_rates', device=self.device,
            num_envs=config.num_envs, headless=True, use_warp=config.cameras)
        self.t = self.sim.get_obs()
        self.nominal_mass = self.t['robot_mass'].clone() / config.mass_scale
        n, d = self.num_envs, self.device
        self.ids = torch.arange(n, device=d)
        self.gate_center = torch.tensor([0., 0., 1.5], device=d).repeat(n, 1)
        self.goal = torch.tensor([2., 0., 1.5], device=d).repeat(n, 1)
        self.gate_angle = torch.zeros(n, device=d)
        self.gate_rotation = torch.eye(3, device=d).repeat(n, 1, 1)
        self.aperture = torch.ones(n, 2, device=d)
        self.half_size = torch.tensor(VEHICLE_HALF_SIZE, device=d).repeat(n, 1)
        self.prev_action = torch.zeros(n, 4, device=d)
        self.tracker = EpisodeTracker(n, d, dt=config.dt, timeout_seconds=config.max_seconds)
        self.episode_return = torch.zeros(n, device=d)
        self.episode_id = np.zeros(n, dtype=np.int64)
        self.physics_steps = 0
        self.reset()

    @property
    def position(self):
        return self.t['robot_position']

    @property
    def rotation(self):
        return quat_to_rotation_matrix(self.t['robot_orientation'])

    def reset(self, indices=None, scenarios=None):
        ids = self.ids if indices is None else indices
        if len(ids) == 0:
            return self.teacher_obs()
        self.sim.reset_idx(ids)
        indices_cpu = ids.cpu().numpy()
        k = len(indices_cpu)
        levels = np.full(k, self.cfg.level)
        if self.cfg.randomize and self.cfg.level > 0:
            old = self.rng.random(k) < .2
            levels[old] = self.rng.integers(0, self.cfg.level, old.sum())
        geometry = curriculum_geometry(levels, self.LEVELS, self.cfg.level, self.cfg.curriculum_fraction)
        center = np.tile([0., 0., 1.5], (k, 1))
        start = np.tile([-3., 0., 1.5], (k, 1))
        velocity = np.zeros((k, 3))
        rpy, world_rates = sample_attitude_rates(self.rng, k, self.cfg.randomize and self.cfg.perturb_reset)
        if self.cfg.randomize:
            center[:, 1:] += self.rng.uniform([-.15, -.1], [.15, .1], (k, 2))
            start += self.rng.uniform([-.2, -.15, -.1], [.2, .15, .1], (k, 3))
            velocity += self.rng.uniform(-.08, .08, (k, 3))
            geometry[:, 2] += self.rng.uniform(-5., 5., k) * (levels > 0)
        if scenarios is not None:
            if len(scenarios) != k:
                raise ValueError('One explicit scenario per reset environment required')
            for j, scenario in enumerate(scenarios):
                center[j] = scenario.get('gate_center', center[j])
                start[j] = scenario.get('start', start[j])
                velocity[j] = scenario.get('velocity', velocity[j])
                geometry[j] = scenario.get('geometry', geometry[j])
                rpy[j] = scenario.get('rpy', rpy[j])
                world_rates[j] = scenario.get('world_rates', world_rates[j])
        self.gate_center[ids] = torch.as_tensor(center, device=self.device, dtype=torch.float32)
        self.gate_angle[ids] = torch.as_tensor(np.deg2rad(geometry[:, 2]), device=self.device)
        self.gate_rotation[ids] = roll_matrix(self.gate_angle[ids])
        self.aperture[ids] = torch.as_tensor(geometry[:, :2] / 2, device=self.device)
        # Four panels leave the requested rectangular opening. Static within each
        # episode; reset both PhysX actor poses and Warp acceleration structures.
        offset = torch.zeros((k, 4, 3), device=self.device)
        offset[:, 0, 1] = self.aperture[ids, 0] + 4
        offset[:, 1, 1] = -self.aperture[ids, 0] - 4
        offset[:, 2, 2] = self.aperture[ids, 1] + 4
        offset[:, 3, 2] = -self.aperture[ids, 1] - 4
        panel = self.t['env_asset_state_tensor']
        panel[ids, :, :3] = self.gate_center[ids, None] + torch.einsum('nij,nkj->nki', self.gate_rotation[ids], offset)
        panel[ids, :, 3:7] = 0
        panel[ids, :, 3] = torch.sin(self.gate_angle[ids, None] / 2)
        panel[ids, :, 6] = torch.cos(self.gate_angle[ids, None] / 2)
        panel[ids, :, 7:] = 0
        state = self.t['robot_state_tensor']
        state[ids] = 0
        state[ids, :3] = torch.as_tensor(start, device=self.device, dtype=torch.float32)
        state[ids, 3:7] = torch.as_tensor(quaternion_xyzw(rpy), device=self.device, dtype=torch.float32)
        state[ids, 7:10] = torch.as_tensor(velocity, device=self.device, dtype=torch.float32)
        state[ids, 10:13] = torch.as_tensor(world_rates, device=self.device, dtype=torch.float32)
        self.sim.robot_manager.robot.update_states()
        self.sim.robot_manager.robot.control_allocator.motor_model.current_motor_thrust[ids] = self.nominal_mass[ids, None] * 9.81 / 4
        self.sim.IGE_env.write_to_sim()
        if self.cfg.cameras:
            self.sim.warp_env.reset_idx(ids)
            self.sim.render_sensors()
        self.tracker.reset(ids)
        self.prev_action[ids] = 0
        self.episode_return[ids] = 0
        self.episode_id[indices_cpu] += 1
        return self.teacher_obs()

    def _body(self, vector):
        return (self.rotation.transpose(-1, -2) @ vector.unsqueeze(-1)).squeeze(-1)

    def student_proprio(self, noisy=False):
        # Explicit allowlist. No gate fields or terminal-state flags are used.
        goal = self._body(self.goal - self.position)
        velocity = self.t['robot_body_linvel'].clone()
        rates = self.t['robot_body_angvel'].clone()
        gravity = -self.rotation[:, 2, :]
        if noisy:
            goal += torch.randn_like(goal) * .02
            velocity += torch.randn_like(velocity) * .03
            rates += torch.randn_like(rates) * .01
            gravity += torch.randn_like(gravity) * .005
        return torch.cat((goal, velocity, rates, gravity, self.prev_action), dim=-1)

    def teacher_obs(self):
        relative_gate = self._body(self.gate_center - self.position)
        rotation = self.rotation.transpose(-1, -2) @ self.gate_rotation
        axes = rotation[:, :, 1:].reshape(self.num_envs, 6)
        return torch.cat((self.student_proprio(), relative_gate, axes, self.aperture), dim=-1)

    def student_obs(self):
        if not self.cfg.cameras:
            raise RuntimeError('Student observations require real rendered depth')
        depth = self.t['depth_range_pixels'].clamp(0, 1).clone()
        return {'depth': depth, 'proprio': self.student_proprio(noisy=True)}

    def _distance(self):
        through = torch.linalg.vector_norm(self.position - self.gate_center, dim=-1)
        through += torch.linalg.vector_norm(self.gate_center - self.goal, dim=-1)
        direct = torch.linalg.vector_norm(self.position - self.goal, dim=-1)
        return torch.where(self.tracker.crossed, direct, through)

    def step(self, actions):
        actions = actions.clamp(-1, 1)
        si = actions.clone()
        si[:, 0] = (actions[:, 0] + 1) * self.nominal_mass * 9.81
        si[:, 1:3] *= 6.
        si[:, 3] *= 3.
        old_distance = self._distance()
        active_start = ~self.tracker.done
        flags = {name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                 for name in ('success', 'failure', 'timeout', 'crossing')}
        for _ in range(self.cfg.substeps):
            p0, r0 = self.position.clone(), self.rotation.clone()
            self.sim.step(si)
            r1 = self.rotation
            contact, before, after, _ = gate_status(self.position, r1, self.half_size,
                self.gate_center, self.gate_rotation, self.aperture, .025)
            swept = swept_gate_collision(p0, r0, self.position, r1, self.half_size,
                self.gate_center, self.gate_rotation, self.aperture, .025)
            physical = self.t['crashes'].bool()
            outside = ((self.position[:, :2].abs() > 3.8).any(-1) |
                       (self.position[:, 2] < .08) | (self.position[:, 2] > 4.8))
            finite = torch.isfinite(self.t['robot_state_tensor']).all(-1)
            tilt = torch.acos(r1[:, 2, 2].clamp(-1, 1))
            events = self.tracker.update(before, after, contact | swept | physical,
                outside | ~finite, torch.linalg.vector_norm(self.goal - self.position, dim=-1),
                torch.linalg.vector_norm(self.t['robot_linvel'], dim=-1), tilt)
            for name in flags:
                flags[name] |= events[name]
            self.physics_steps += self.num_envs
        # Distance-progress shaping and one-time events. This is not claimed to be
        # policy-invariant potential shaping; traversal tilt is not penalized.
        reward = 4. * (old_distance - self._distance()) - .01
        reward += flags['crossing'] * 10 + flags['success'] * 30 - flags['failure'] * 15
        reward -= .002 * (actions - self.prev_action).square().sum(-1)
        goal_distance = torch.linalg.vector_norm(self.goal - self.position, dim=-1)
        speed = torch.linalg.vector_norm(self.t['robot_linvel'], dim=-1)
        reward += self.tracker.crossed * .1 * torch.exp(-2 * goal_distance - speed)
        if self.cfg.recovery_reward == 'dense_v2':
            from uav_gap.recovery_reward import recovery_feedback
            reward += recovery_feedback(self.tracker.crossed, goal_distance, speed, r1[:, 2, 2])
        reward = torch.nan_to_num(reward) * active_start
        self.prev_action[:] = actions
        self.episode_return += reward
        done = flags['success'] | flags['failure'] | flags['timeout']
        info = {**flags, 'time_outs': flags['timeout'], 'episode_return': self.episode_return.clone(),
                'terminal_position': self.position.clone(), 'episode_seconds': self.tracker.elapsed.clone()}
        if self.cfg.cameras:
            self.sim.render_sensors()
        if self.cfg.auto_reset and done.any():
            self.reset(done.nonzero(as_tuple=False).flatten())
        return self.teacher_obs(), reward, done, info

    def close(self):
        self.sim.IGE_env.gym.destroy_sim(self.sim.IGE_env.sim)

    def apply_world_impulse(self, impulse):
        """Apply an instantaneous external translational impulse, in N·s."""
        impulse = torch.as_tensor(impulse, dtype=torch.float32, device=self.device)
        if impulse.shape != (self.num_envs, 3):
            raise ValueError('Expected one world-frame impulse per environment')
        impulse = impulse * (~self.tracker.done)[:, None]
        self.t['robot_state_tensor'][:, 7:10] += impulse / self.t['robot_mass'][:, None]
        self.sim.robot_manager.robot.update_states()
        self.sim.IGE_env.write_to_sim()
