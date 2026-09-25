"""Episode datasets with exact causal prefixes and masked 64-step supervision."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


FIELDS = ('depth', 'proprio', 'teacher_action', 'applied_action')


def save_episode(path, arrays, metadata):
    path = Path(path)
    length = len(arrays['depth'])
    expected = {'depth': (length, 1, 64, 64), 'proprio': (length, 16),
                'teacher_action': (length, 4), 'applied_action': (length, 4)}
    if length == 0 or any(arrays[k].shape != v for k, v in expected.items()):
        raise ValueError('Invalid episode dimensions')
    if any(not np.isfinite(arrays[k]).all() for k in FIELDS):
        raise ValueError('Nonfinite demonstration data')
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **{k: arrays[k].astype(np.float16 if k == 'depth' else np.float32)
                                for k in FIELDS})
    path.with_suffix('.json').write_text(json.dumps(dict(metadata, length=length), indent=2))


class EpisodeWindows(Dataset):
    """Keep all outcomes; aggregate datasets without relabeling executed actions.

    Each target window has its entire episode prefix. Training consumes that
    prefix without gradient, then applies TBPTT to the target window only.
    Thus a random window does not falsely reset a recurrent deployment policy.
    """
    def __init__(self, directories, length=64):
        self.length = length
        self.episodes, self.windows = [], []
        self.manifests = []
        seen = set()
        for directory in directories:
            directory = Path(directory).resolve()
            manifest = json.loads((directory/'manifest.json').read_text())
            if manifest['split'] != 'train':
                raise ValueError('Only training rollouts can enter supervised training')
            if not manifest.get('complete', False):
                raise ValueError('Collection must complete before supervised training')
            self.manifests.append(manifest)
            for entry in manifest['episodes']:
                path = (directory/entry['file']).resolve()
                path.relative_to(directory)
                if path in seen:
                    raise ValueError('Duplicate episode in aggregate dataset')
                seen.add(path)
                self.episodes.append(path)
                self.windows.extend((len(self.episodes)-1, start, min(start+length, entry['length']))
                                    for start in range(0, entry['length'], length))
        if not self.windows:
            raise ValueError('Empty dataset')

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode, start, stop = self.windows[index]
        with np.load(self.episodes[episode], allow_pickle=False) as data:
            return dict({k: np.array(data[k][:stop], dtype=np.float32) for k in FIELDS}, start=start)


def collate_windows(rows):
    """Left-pad prefixes, right-pad supervision; mask padding and reset at start."""
    batch, prefix = len(rows), max(row['start'] for row in rows)
    target = max(len(row['depth'])-row['start'] for row in rows)
    total = prefix + target
    output = {k: torch.zeros(batch, total, *rows[0][k].shape[1:]) for k in FIELDS}
    reset = torch.zeros(batch, total, dtype=torch.bool)
    mask = torch.zeros(batch, target, dtype=torch.bool)
    for i, row in enumerate(rows):
        offset = prefix - row['start']
        length = len(row['depth'])
        for k in FIELDS:
            output[k][i, offset:offset+length] = torch.from_numpy(row[k])
        reset[i, offset] = True
        mask[i, :length-row['start']] = True
    return dict(output, reset=reset, mask=mask, prefix=prefix)


def supervised_loss(model, batch):
    prefix = batch['prefix']
    hidden = None
    if prefix:
        # Chunk burn-in to avoid a large CNN workspace for long episode prefixes.
        with torch.no_grad():
            for start in range(0, prefix, 64):
                stop = min(prefix, start+64)
                _, hidden = model(batch['depth'][:, start:stop], batch['proprio'][:, start:stop],
                                  hidden, batch['reset'][:, start:stop])
    predicted, _ = model(batch['depth'][:, prefix:], batch['proprio'][:, prefix:],
                         hidden, batch['reset'][:, prefix:])
    error = (predicted - batch['teacher_action'][:, prefix:]).square().mean(-1)
    return (error * batch['mask']).sum() / batch['mask'].sum().clamp_min(1)
