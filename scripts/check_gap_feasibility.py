"""Fly physical, actuated reference rollouts to validate task feasibility."""
import argparse
import json
import sys
from uav_gap.runtime import import_simulator, require_idle_gpu, project_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=3)
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--cameras', action='store_true')
    parser.add_argument('--name', default='gap_feasibility')
    parser.add_argument('--parameter-index', type=int)
    args = parser.parse_args()
    require_idle_gpu()
    sys.argv = [sys.argv[0]]
    import_simulator()
    import numpy as np
    import torch
    from uav_gap.task import GapTask, TaskConfig
    from uav_gap.reference import feasibility_reference
    task = GapTask(TaskConfig(num_envs=args.num_envs, level=args.level, cameras=args.cameras,
                              randomize=False, auto_reset=False))
    out = project_output('runs/' + args.name)
    out.mkdir(parents=True, exist_ok=True)
    try:
        n, d = task.num_envs, task.device
        parameters = [(kp, kd, ka, lead) for kp in (4., 6.) for kd in (4., 6.)
                      for ka in (4., 8.) for lead in (.02, .03, .04, .05)]
        if args.parameter_index is not None:
            parameters = [parameters[args.parameter_index]]
        parameters = [parameters[i % len(parameters)] for i in range(n)]
        gains = torch.tensor(parameters, device=d)
        results = [None] * n
        history, quats, cmds, frames = [], [], [], []

        def orientation(acc):
            thrust = acc + torch.tensor([0., 0., 9.81], device=d)
            z = torch.nn.functional.normalize(thrust, dim=-1)
            heading = torch.tensor([1., 0., 0.], device=d).expand_as(z)
            y = torch.nn.functional.normalize(torch.cross(z, heading, dim=-1), dim=-1)
            x = torch.cross(y, z, dim=-1)
            return torch.stack((x, y, z), dim=-1)

        dt = task.cfg.dt * task.cfg.substeps
        angle = task.LEVELS[args.level][2]
        for step in range(round(8 / dt)):
            time = step * dt
            p, v, acc = feasibility_reference(time, angle_deg=angle)
            p = torch.as_tensor(p, device=d, dtype=torch.float32)
            v = torch.as_tensor(v, device=d, dtype=torch.float32)
            acc = torch.as_tensor(acc, device=d, dtype=torch.float32)
            future = np.stack([feasibility_reference(time + lead, angle_deg=angle)[2]
                               for _, _, _, lead in parameters])
            future_next = np.stack([feasibility_reference(time + lead + .005, angle_deg=angle)[2]
                                    for _, _, _, lead in parameters])
            acc_future = torch.as_tensor(future, device=d, dtype=torch.float32)
            desired_acc = acc_future + gains[:, :1]*(p-task.position) + gains[:, 1:2]*(v-task.t['robot_linvel'])
            rd = orientation(desired_acc)
            current = task.rotation
            delta = current.transpose(-1, -2) @ rd
            error = .5 * torch.stack((delta[:, 2, 1]-delta[:, 1, 2], delta[:, 0, 2]-delta[:, 2, 0],
                                      delta[:, 1, 0]-delta[:, 0, 1]), dim=-1)
            actions = torch.zeros(n, 4, device=d)
            desired_force = desired_acc + torch.tensor([0., 0., 9.81], device=d)
            collective_acc = (desired_force * current[:, :, 2]).sum(-1).clamp(0, 19.62)
            actions[:, 0] = collective_acc / 9.81 - 1
            r_ref = orientation(acc_future)
            r_next = orientation(torch.as_tensor(future_next, device=d, dtype=torch.float32))
            angular_delta = r_ref.transpose(-1, -2) @ r_next
            omega_ref = .5 / .005 * torch.stack((angular_delta[:, 2, 1]-angular_delta[:, 1, 2],
                angular_delta[:, 0, 2]-angular_delta[:, 2, 0], angular_delta[:, 1, 0]-angular_delta[:, 0, 1]), dim=-1)
            omega_body = ((current.transpose(-1, -2) @ r_ref) @ omega_ref[..., None]).squeeze(-1)
            rates = gains[:, 2:3] * error + omega_body
            actions[:, 1:3] = rates[:, :2] / 6
            actions[:, 3] = rates[:, 2] / 3
            obs, _, done, info = task.step(actions)
            assert obs.shape == (n, 27) and torch.isfinite(obs).all()
            history.append(task.position.cpu().numpy().copy())
            quats.append(task.t['robot_orientation'].cpu().numpy().copy())
            cmds.append(actions.cpu().numpy().copy())
            if args.cameras and step % 5 == 0:
                frames.append(task.student_obs()['depth'].cpu().numpy().copy())
            for i in done.nonzero(as_tuple=False).flatten().cpu().tolist():
                results[i] = dict(index=i, parameters=parameters[i], success=bool(info['success'][i]),
                    failure=bool(info['failure'][i]), timeout=bool(info['timeout'][i]),
                    crossed=bool(task.tracker.crossed[i]), seconds=float(info['episode_seconds'][i]),
                    position=info['terminal_position'][i].cpu().tolist())
            if task.tracker.done.all():
                break
        report = dict(kind='scripted_physical_feasibility_not_learning', level=args.level,
                      geometry=task.LEVELS[args.level], results=results,
                      success_count=sum(bool(r and r['success']) for r in results), num_envs=n)
        (out/'report.json').write_text(json.dumps(report, indent=2))
        np.savez_compressed(out/'trace.npz', position=np.stack(history), quaternion=np.stack(quats),
                            action=np.stack(cmds), depth=np.array(frames), dt=dt)
        print(json.dumps(report, indent=2))
    finally:
        task.close()


if __name__ == '__main__':
    main()
