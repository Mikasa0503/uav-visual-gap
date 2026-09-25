"""Teacher inference with the same rl-games normalizers and action contract."""
import torch
from rl_games.algos_torch.model_builder import ModelBuilder


def load_teacher(config, checkpoint, num_envs=1, device='cuda:0'):
    definition = ModelBuilder().load(config['params'])
    model = definition.build({'actions_num': 4, 'input_shape': (27,), 'num_seqs': num_envs,
                              'value_size': 1, 'normalize_input': True, 'normalize_value': True})
    weights = torch.load(checkpoint, map_location=device)
    model.load_state_dict(weights['model'], strict=True)
    model = model.to(device).eval()
    return model


@torch.no_grad()
def teacher_action(model, observations):
    result = model({'is_train': False, 'prev_actions': None, 'obs': observations, 'rnn_states': None})
    return result['mus'].clamp(-1, 1)
