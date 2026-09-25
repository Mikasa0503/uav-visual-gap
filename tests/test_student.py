import pytest
import torch
from uav_gap.student import make_student


def test_memory_ablation_has_matching_parameter_budget():
    gru = sum(p.numel() for p in make_student('gru').parameters())
    stack = sum(p.numel() for p in make_student('stack4').parameters())
    assert abs(stack-gru)/gru < .01


@pytest.mark.parametrize('kind', ['gru', 'stack4'])
def test_student_is_causal_and_reset_discards_previous_episode(kind):
    torch.manual_seed(11)
    model = make_student(kind).eval()
    depth = torch.rand(2, 5, 1, 64, 64)
    proprio = torch.rand(2, 5, 16)
    with torch.no_grad():
        original, _ = model(depth, proprio)
        changed = depth.clone()
        changed[:, 3:] = 0
        modified, _ = model(changed, proprio)
        assert torch.allclose(original[:, :3], modified[:, :3], atol=1e-6)
        reset = torch.zeros(2, 5, dtype=torch.bool)
        reset[:, 3] = True
        combined, _ = model(depth, proprio, reset=reset)
        fresh, _ = model(depth[:, 3:], proprio[:, 3:])
        assert torch.allclose(combined[:, 3:], fresh, atol=1e-6)
        assert original.shape == (2, 5, 4)
        assert (original.abs() <= 1).all()


@pytest.mark.parametrize('kind', ['gru', 'stack4'])
def test_streaming_inference_matches_sequence_training(kind):
    model = make_student(kind).eval()
    depth, proprio = torch.rand(2, 4, 1, 64, 64), torch.rand(2, 4, 16)
    with torch.no_grad():
        batch, _ = model(depth, proprio)
        state, outputs = None, []
        for t in range(4):
            output, state = model(depth[:, t:t+1], proprio[:, t:t+1], state)
            outputs.append(output)
        assert torch.allclose(batch, torch.cat(outputs, dim=1), atol=1e-6)
