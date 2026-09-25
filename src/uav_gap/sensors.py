"""Causal observation transport; the known previous command is never delayed."""
from collections import deque


class ObservationStream:
    def __init__(self, delay_steps=1, drop_start=None, drop_frames=0):
        if delay_steps < 0:
            raise ValueError('Observation delay cannot be negative')
        self.delay_steps = delay_steps
        self.history = deque(maxlen=delay_steps+1)
        self.drop_start, self.drop_frames = drop_start, drop_frames
        self.step, self.last_depth = 0, None

    def __call__(self, observation):
        current = {k:v.clone() for k,v in observation.items()}
        if (self.drop_start is not None and self.drop_start <= self.step < self.drop_start+self.drop_frames
                and self.last_depth is not None):
            current['depth'] = self.last_depth.clone()
        else:
            self.last_depth = current['depth'].clone()
        self.step += 1
        if not self.history:
            # Reset has no prior episode history; use the first current sample.
            for _ in range(self.delay_steps):
                self.history.append({k:v.clone() for k,v in current.items()})
        self.history.append(current)
        result = {k:v.clone() for k,v in self.history[0].items()}
        result['proprio'][:, -4:] = current['proprio'][:, -4:]
        return result

    def reset(self):
        self.history.clear()
        self.step, self.last_depth = 0, None
