import torch
from uav_gap.termination import EpisodeTracker


def step(tracker, before=False, after=False, collision=False, speed=0., goal=0.):
    b = lambda value: torch.tensor([value])
    return tracker.update(b(before), b(after), b(collision), b(False),
                          b(float(goal)), b(float(speed)), b(0.))


def test_success_requires_crossing_then_continuous_stable_hold():
    tracker = EpisodeTracker(1, 'cpu', dt=.1)
    step(tracker, before=True)
    assert step(tracker, after=True)['crossing'].item()
    for _ in range(3):
        assert not step(tracker, after=True)['success'].item()
    assert step(tracker, after=True)['success'].item()


def test_collision_has_priority_over_crossing_and_success():
    tracker = EpisodeTracker(1, 'cpu', dt=.5)
    step(tracker, before=True)
    result = step(tracker, after=True, collision=True)
    assert result['failure'].item()
    assert not result['success'].item() and not result['crossing'].item()


def test_no_success_when_spawned_behind_wall():
    tracker = EpisodeTracker(1, 'cpu', dt=.5)
    assert not step(tracker, after=True)['success'].item()


def test_crossing_reward_cannot_repeat_and_instability_resets_hold():
    tracker = EpisodeTracker(1, 'cpu', dt=.1)
    step(tracker, before=True)
    assert step(tracker, after=True)['crossing'].item()
    assert not step(tracker, before=True)['crossing'].item()
    assert not step(tracker, after=True, speed=1.)['crossing'].item()
    assert tracker.stable.item() == 0


def test_timeout_is_distinct_and_terminal_emits_once():
    tracker = EpisodeTracker(1, 'cpu', dt=.1, timeout_seconds=.2)
    step(tracker, before=True, goal=5.)
    assert step(tracker, before=True, goal=5.)['timeout'].item()
    assert not step(tracker, before=True, goal=5.)['timeout'].item()
    tracker.reset(torch.tensor([0]))
    assert not tracker.done.item() and tracker.elapsed.item() == 0
