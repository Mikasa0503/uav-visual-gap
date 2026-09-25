"""rl-games integration; explicit lifecycle and initial checkpoint capture."""
import json
import time
from collections import deque
import numpy as np
import torch
from gym import spaces
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from uav_gap.task import GapTask, TaskConfig


class GapVectorEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.task = env_configurations.configurations[config_name]['env_creator'](**kwargs)

    def step(self, actions):
        return self.task.step(actions)

    def reset(self):
        return self.task.reset()

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        return dict(action_space=spaces.Box(-1., 1., (4,), dtype=np.float32),
                    observation_space=spaces.Box(-np.inf, np.inf, (27,), dtype=np.float32),
                    agents=1, value_size=1)

    def get_env_state(self):
        # Preserve curriculum and RNG; episode state deliberately reset on resume.
        return {'level': self.task.cfg.level, 'rng': self.task.rng.bit_generator.state}

    def set_env_state(self, state):
        if state:
            self.task.cfg.level = state['level']
            self.task.rng.bit_generator.state = state['rng']


def register_task(config):
    env_configurations.register('visual_gap', {'vecenv_type': 'VISUAL_GAP',
        'env_creator': lambda **kw: GapTask(config)})
    vecenv.register('VISUAL_GAP', lambda config_name, num_actors, **kw: GapVectorEnv(config_name, num_actors, **kw))


class ProgressObserver(AlgoObserver):
    def __init__(self, directory):
        self.directory = directory
        self.recent = deque(maxlen=1000)
        self.crossings = 0
        self.frames = 0
        self.started = time.monotonic()

    def after_init(self, algo):
        self.algo = algo

    def process_infos(self, infos, done_indices):
        self.frames += self.algo.num_actors
        self.crossings += int(infos['crossing'].sum())
        for i in done_indices.flatten().cpu().tolist():
            self.recent.append({name: float(infos[name][i]) for name in
                                ('success', 'failure', 'timeout', 'episode_return', 'episode_seconds')})

    def after_clear_stats(self):
        pass

    def after_print_stats(self, frame, epoch_num, total_time):
        entry = {'frames': int(self.frames), 'epoch': int(epoch_num),
                 'elapsed_seconds': time.monotonic()-self.started,
                 'level': self.algo.vec_env.task.cfg.level, 'recent_episodes': len(self.recent),
                 'crossing_events': self.crossings}
        if self.recent:
            entry.update({k: float(np.mean([row[k] for row in self.recent])) for k in self.recent[0]})
        with (self.directory/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(entry)+'\n')
        if epoch_num <= 10 or epoch_num % 10 == 0:
            stem = self.directory / ('snapshot_%06d' % epoch_num)
            self.algo.save(str(stem))
            stem.with_suffix('.json').write_text(json.dumps(entry, indent=2))
        print('GAP_PROGRESS ' + json.dumps(entry), flush=True)
