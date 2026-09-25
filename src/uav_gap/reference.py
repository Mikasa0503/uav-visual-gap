"""Non-learning feasibility reference; NEVER label these rollouts as a policy."""
import numpy as np


def quintic(p0, v0, a0, p1, v1, a1, duration, time):
    p0, v0, a0, p1, v1, a1 = [np.asarray(x, dtype=float) for x in (p0, v0, a0, p1, v1, a1)]
    t = np.clip(time / duration, 0, 1)
    c = np.zeros((6, 3))
    c[:3] = p0, v0 * duration, .5 * a0 * duration**2
    residual = np.stack((p1 - c[:3].sum(0), v1 * duration - c[1] - 2*c[2],
                         a1 * duration**2 - 2*c[2]))
    c[3:] = np.linalg.solve(np.array([[1, 1, 1], [3, 4, 5], [6, 12, 20]]), residual)
    p = sum(c[i] * t**i for i in range(6))
    v = sum(i*c[i] * t**(i-1) for i in range(1, 6)) / duration
    a = sum(i*(i-1)*c[i] * t**(i-2) for i in range(2, 6)) / duration**2
    return p, v, a


def feasibility_reference(time, angle_deg=45., speed=3., approach=2.2, recovery=2.0):
    angle = np.deg2rad(angle_deg)
    center = np.array([0., 0., 1.5])
    acceleration = np.array([0., -9.81*np.sin(angle)*np.cos(angle), -9.81*np.sin(angle)**2])
    half_duration = .6 / speed
    velocity = np.array([speed, 0., 0.])
    enter_p = center - velocity*half_duration + .5*acceleration*half_duration**2
    exit_p = center + velocity*half_duration + .5*acceleration*half_duration**2
    enter_v, exit_v = velocity - acceleration*half_duration, velocity + acceleration*half_duration
    if time < approach:
        return quintic([-3., 0., 1.5], [0., 0., 0.], [0., 0., 0.],
                        enter_p, enter_v, acceleration, approach, time)
    if time < approach + 2*half_duration:
        t = time - approach - half_duration
        return center + velocity*t + .5*acceleration*t*t, velocity + acceleration*t, acceleration
    end_time = approach + 2*half_duration + recovery
    if time < end_time:
        return quintic(exit_p, exit_v, acceleration, [2., 0., 1.5], [0., 0., 0.], [0., 0., 0.],
                        recovery, time - approach - 2*half_duration)
    return np.array([2., 0., 1.5]), np.zeros(3), np.zeros(3)
