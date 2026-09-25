"""Versioned training reset perturbations, using world-frame angular velocity."""
import numpy as np


def quaternion_xyzw(rpy):
    rpy = np.asarray(rpy)
    sr, sp, sy = np.sin(rpy/2).T
    cr, cp, cy = np.cos(rpy/2).T
    return np.stack((sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                     cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy), axis=-1)


def sample_attitude_rates(rng, count, enabled):
    if not enabled:
        return np.zeros((count,3)), np.zeros((count,3))
    # Initial recovery challenge is bounded and independent of the aperture pose.
    rpy = rng.uniform(-1, 1, (count,3)) * np.deg2rad([8.,8.,10.])
    world_rates = rng.uniform(-.25, .25, (count,3))
    return rpy, world_rates
