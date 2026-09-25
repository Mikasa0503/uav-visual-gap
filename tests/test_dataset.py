import json
import numpy as np
import pytest
import torch
from uav_gap.dataset import EpisodeWindows, collate_windows, save_episode, supervised_loss
from uav_gap.student import make_student, load_student


def example(length):
    rng = np.random.default_rng(length)
    return dict(depth=rng.random((length, 1, 64, 64)).astype(np.float32),
        proprio=rng.random((length, 16)).astype(np.float32),
        teacher_action=rng.uniform(-1, 1, (length, 4)).astype(np.float32),
        applied_action=np.zeros((length, 4), dtype=np.float32))


def test_dataset_masks_tail_and_preserves_teacher_vs_executed(tmp_path):
    save_episode(tmp_path/'episode.npz', example(70), {'success': False})
    (tmp_path/'manifest.json').write_text(json.dumps(dict(split='train', labels=70, complete=True,
        episodes=[dict(file='episode.npz', length=70)])))
    dataset = EpisodeWindows([tmp_path])
    assert len(dataset) == 2
    batch = collate_windows([dataset[0], dataset[1]])
    assert batch['prefix'] == 64
    assert batch['mask'].sum() == 70
    assert batch['reset'][0, 64] and batch['reset'][1, 0]
    assert batch['applied_action'].eq(0).all()
    assert batch['teacher_action'].abs().sum() > 0
    with pytest.raises(ValueError, match='Duplicate'):
        EpisodeWindows([tmp_path, tmp_path])


@pytest.mark.parametrize('kind', ['gru', 'stack4'])
def test_causal_prefix_padding_equals_complete_episode_and_masks_loss(kind):
    torch.manual_seed(5)
    model = make_student(kind)
    first, second = example(7), example(9)
    rows = [dict(first, start=0), dict(second, start=5)]
    batch = collate_windows(rows)
    # Exact whole-episode oracle includes no artificial history resets.
    expected = []
    for row in rows:
        predicted, _ = model(torch.from_numpy(row['depth'])[None], torch.from_numpy(row['proprio'])[None])
        expected.append((predicted[0, row['start']:] - torch.from_numpy(row['teacher_action'][row['start']:])).square().mean(-1))
    target = torch.cat(expected).mean()
    actual = supervised_loss(model, batch)
    assert torch.allclose(actual, target, atol=1e-6)
    changed = {k:v.clone() if torch.is_tensor(v) else v for k,v in batch.items()}
    changed['teacher_action'][:, batch['prefix']:][~batch['mask']] = 1e5
    assert torch.allclose(supervised_loss(model, changed), actual)
    actual.backward()
    assert model.depth_encoder.layers[0].weight.grad.abs().sum() > 0


def test_student_checkpoint_has_no_teacher_dependency(tmp_path):
    model = make_student('gru').eval()
    path = tmp_path/'student.pth'
    torch.save(dict(kind='gru', model=model.state_dict(), seed=11, updates=3, label_presentations=40), path)
    restored, metadata = load_student(path, device='cpu')
    depth, proprio = torch.rand(1,2,1,64,64), torch.rand(1,2,16)
    with torch.no_grad():
        assert torch.equal(model(depth, proprio)[0], restored(depth, proprio)[0])
    assert metadata['updates'] == 3
