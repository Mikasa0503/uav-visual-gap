"""Real-physics checks of fixed actuation calibration and external impulse."""
import argparse
import json
import sys
from uav_gap.runtime import require_idle_gpu, import_simulator, project_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mass-scale', type=float, default=1.)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    require_idle_gpu()
    sys.argv = [sys.argv[0]]
    import_simulator()
    import torch
    from uav_gap.task import GapTask, TaskConfig
    directory = project_output('runs/'+args.name)
    directory.mkdir(parents=True, exist_ok=False)
    task = GapTask(TaskConfig(num_envs=2, level=0, randomize=True, perturb_reset=True,
                             auto_reset=False, mass_scale=args.mass_scale))
    try:
        randomized = task.t['robot_state_tensor'].clone()
        assert randomized[:,3:6].abs().sum() > 0 and randomized[:,10:13].abs().sum() > 0
        task.reset(scenarios=[dict(start=[-3,0,1.5], velocity=[0,0,0], rpy=[0,0,0], world_rates=[0,0,0])]*2)
        torch.testing.assert_close(task.nominal_mass, torch.full_like(task.nominal_mass, .25), rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(task.t['robot_mass'], task.nominal_mass*args.mass_scale)
        initial_z = task.position[:,2].clone()
        for _ in range(20):
            task.step(torch.zeros(2,4,device='cuda:0'))
        height_change = task.position[:,2]-initial_z
        if args.mass_scale == 1.:
            assert height_change.abs().max() < .02, height_change
        else:
            assert height_change.max() < -.05, height_change
        before = task.t['robot_linvel'].clone()
        task.apply_world_impulse([[0,.1,0]]*2)
        actual = task.t['robot_linvel'] - before
        expected = torch.tensor([[0,.1,0]]*2,device='cuda:0') / task.t['robot_mass'][:,None]
        torch.testing.assert_close(actual, expected)
        result = dict(mass_scale=args.mass_scale, physical_mass=task.t['robot_mass'].cpu().tolist(),
            nominal_mass=task.nominal_mass.cpu().tolist(), height_change_after_400ms=height_change.cpu().tolist(),
            impulse_velocity_change=actual.cpu().tolist(), randomized_initial_state=randomized.cpu().tolist())
        (directory/'report.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        task.close()


if __name__ == '__main__':
    main()
