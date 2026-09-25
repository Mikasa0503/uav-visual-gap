from pathlib import Path
import torch
import yaml
from rl_games.algos_torch.model_builder import ModelBuilder
from uav_gap.inference import load_teacher, teacher_action


def test_checkpoint_preserves_deterministic_policy_and_normalizers(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root/'configs/teacher_ppo.yaml').read_text())
    model = ModelBuilder().load(config['params']).build({'actions_num': 4, 'input_shape': (27,),
        'num_seqs': 2, 'value_size': 1, 'normalize_input': True, 'normalize_value': True})
    obs = torch.randn(2, 27)
    model.eval()
    before = teacher_action(model, obs)
    path = tmp_path/'teacher.pth'
    torch.save({'model': model.state_dict()}, path)
    restored = load_teacher(config, path, 2, 'cpu')
    assert torch.equal(before, teacher_action(restored, obs))
    assert torch.equal(before, teacher_action(restored, obs))
