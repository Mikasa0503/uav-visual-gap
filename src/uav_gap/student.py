"""Deployable students accept only depth, proprioception and their own memory."""
import torch
from torch import nn


class DepthEncoder(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, 32, 5, 2), nn.ELU(),
            nn.Conv2d(32, 64, 3, 2), nn.ELU(), nn.Conv2d(64, 128, 3, 2), nn.ELU(),
            nn.Flatten(), nn.Linear(128*6*6, 128), nn.ELU())

    def forward(self, depth):
        return self.layers(depth)


class RecurrentStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_encoder = DepthEncoder()
        self.proprio_encoder = nn.Sequential(nn.Linear(16, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU())
        self.memory = nn.GRU(192, 128, batch_first=True)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 4), nn.Tanh())

    def forward(self, depth, proprio, hidden=None, reset=None):
        batch, length = depth.shape[:2]
        vision = self.depth_encoder(depth.reshape(batch*length, 1, 64, 64)).reshape(batch, length, 128)
        feature = torch.cat((vision, self.proprio_encoder(proprio)), dim=-1)
        if hidden is None:
            hidden = feature.new_zeros(1, batch, 128)
        if reset is None:
            output, hidden = self.memory(feature, hidden)
        elif not reset[:,1:].any():
            # Dataset windows/burn-in chunks reset only at their first frame.
            # Preserve exactly that boundary, then use one recurrent invocation
            # rather than 64 separate launches. Mid-chunk resets keep the general path.
            hidden = hidden * (~reset[:,0]).to(hidden.dtype)[None,:,None]
            output,hidden = self.memory(feature,hidden)
        else:
            frames = []
            for t in range(length):
                hidden = hidden * (~reset[:, t]).to(hidden.dtype)[None, :, None]
                value, hidden = self.memory(feature[:, t:t+1], hidden)
                frames.append(value)
            output = torch.cat(frames, dim=1)
        return self.head(output), hidden


class StackedStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_encoder = DepthEncoder(channels=4)
        self.proprio_encoder = nn.Sequential(nn.Linear(16, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU())
        # Match GRU policy parameter count within 1%, so memory comparison is not
        # dominated by a large difference in model capacity.
        self.head = nn.Sequential(nn.Linear(192, 384), nn.ELU(), nn.Linear(384, 128), nn.ELU(),
                                  nn.Linear(128, 64), nn.ELU(),
                                  nn.Linear(64, 4), nn.Tanh())

    def forward(self, depth, proprio, hidden=None, reset=None):
        batch, length = depth.shape[:2]
        if hidden is None:
            hidden = depth[:, 0].repeat(1, 3, 1, 1)
        stacks = []
        for t in range(length):
            if reset is not None:
                hidden = torch.where(reset[:, t, None, None, None], depth[:, t].repeat(1, 3, 1, 1), hidden)
            stacked = torch.cat((hidden, depth[:, t]), dim=1)
            stacks.append(stacked)
            hidden = stacked[:, 1:]
        vision = self.depth_encoder(torch.stack(stacks, dim=1).reshape(batch*length, 4, 64, 64))
        feature = torch.cat((vision.reshape(batch, length, 128), self.proprio_encoder(proprio)), dim=-1)
        return self.head(feature), hidden


def make_student(kind):
    if kind == 'gru':
        return RecurrentStudent()
    if kind == 'stack4':
        return StackedStudent()
    raise ValueError('Unknown student kind: ' + str(kind))


def load_student(path, device='cuda:0'):
    checkpoint = torch.load(path, map_location=device)
    model = make_student(checkpoint['kind']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    metadata = {k: checkpoint[k] for k in ('kind', 'seed', 'updates', 'label_presentations')}
    metadata['observation_delay_steps'] = checkpoint.get('configuration', {}).get('observation_delay_steps', 0)
    metadata['parameters'] = sum(p.numel() for p in model.parameters())
    return model, metadata


@torch.no_grad()
def benchmark_single_student(model, visual, iterations=100):
    """Batch-one GPU policy latency on a real sensor sample; no simulator timing."""
    import numpy as np
    depth, proprio = visual['depth'][:1, None], visual['proprio'][:1, None]
    hidden = None
    for _ in range(10):
        _, hidden = model(depth, proprio, hidden)
    events = []
    for _ in range(iterations):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        _, hidden = model(depth, proprio, hidden)
        stop.record()
        events.append((start,stop))
    torch.cuda.synchronize()
    latency = [start.elapsed_time(stop) for start,stop in events]
    return dict(batch_size=1, warmup=10, iterations=iterations,
                gpu_policy_ms_median=float(np.median(latency)), gpu_policy_ms_p95=float(np.percentile(latency,95)),
                includes_sensor_rendering=False, includes_cpu_transfer=False)
