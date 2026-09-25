"""Opt-in geometric bridge for current-level training samples only."""
import numpy as np


def curriculum_geometry(levels, table, current_level, fraction=1.):
    if not np.isfinite(fraction) or not 0. < fraction <= 1.:
        raise ValueError('curriculum_fraction must be in (0, 1]')
    if not 0 <= current_level < len(table):
        raise ValueError('Invalid curriculum level')
    if current_level == 0 and fraction != 1.:
        raise ValueError('Level zero has no preceding geometry')
    # Preserve the exact old conversion path for the default recipe.
    geometry = np.array([table[level] for level in levels], dtype=np.float32)
    if fraction != 1.:
        previous = np.asarray(table[current_level-1], dtype=np.float32)
        target = np.asarray(table[current_level], dtype=np.float32)
        geometry[np.asarray(levels) == current_level] = previous + fraction*(target-previous)
    return geometry
