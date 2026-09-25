"""Teacher BC or student-controlled DAgger rollouts with real depth observations."""
import argparse
import hashlib
import json
import sys
from uav_gap.runtime import require_idle_gpu, import_simulator, ROOT, project_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--student', help='Omit for teacher-controlled BC; provide for pure student DAgger')
    parser.add_argument('--level', type=int, default=3)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--label-budget', type=int, help='Exact teacher-query budget; overrides episode count')
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--seed', type=int, default=41011)
    parser.add_argument('--observation-delay-steps', type=int, default=1)
    parser.add_argument('--perturb-reset', action='store_true')
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    if args.num_envs <= 0 or (args.label_budget is None and (args.episodes <= 0 or args.episodes % args.num_envs)):
        raise ValueError('episodes must be a positive multiple of num-envs')
    if args.label_budget is not None and args.label_budget <= 0:
        raise ValueError('Label budget must be positive')
    require_idle_gpu()
    sys.argv = [sys.argv[0]]
    import_simulator()
    import numpy as np
    import torch
    import yaml
    from uav_gap.task import GapTask, TaskConfig
    from uav_gap.inference import load_teacher, teacher_action
    from uav_gap.student import load_student
    from uav_gap.dataset import FIELDS, save_episode
    from uav_gap.sensors import ObservationStream

    teacher_path = (ROOT/args.teacher).resolve()
    teacher_path.relative_to(ROOT)
    configuration = yaml.safe_load((teacher_path.parent/'configuration.yaml').read_text())
    teacher = load_teacher(configuration, teacher_path, args.num_envs)
    student, student_metadata = None, None
    if args.student:
        student_path = (ROOT/args.student).resolve()
        student_path.relative_to(ROOT)
        student, student_metadata = load_student(student_path)
        if student_metadata['observation_delay_steps'] != args.observation_delay_steps:
            raise ValueError('Student rollout must match the training sensor delay')
    directory = project_output('datasets/'+args.name)
    directory.mkdir(parents=True, exist_ok=False)
    cfg = TaskConfig(num_envs=args.num_envs, seed=args.seed, level=args.level,
                     cameras=True, randomize=True, auto_reset=False, perturb_reset=args.perturb_reset)
    task = GapTask(cfg)
    manifest = dict(schema_version=1, split='train', seed=args.seed, level=args.level,
        controller='student_dagger' if student else 'teacher_bc', beta=0. if student else 1.,
        teacher=str(teacher_path.relative_to(ROOT)),
        teacher_sha256=hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
        student=args.student, student_metadata=student_metadata,
        student_sha256=hashlib.sha256(student_path.read_bytes()).hexdigest() if student else None,
        task=vars(cfg),
        policy_hz=50, observation_delay_steps=args.observation_delay_steps,
        observation='Warp depth and noisy simulated proprioception; no VIO; latest known previous command',
        requested_label_budget=args.label_budget,
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (ROOT/'src/uav_gap').glob('*.py')}, episodes=[], labels=0, teacher_queries=0)
    try:
        wave = 0
        while True:
            obs = task.reset()
            buffers = [{k: [] for k in FIELDS} for _ in range(args.num_envs)]
            hidden = None
            stream = ObservationStream(args.observation_delay_steps)
            initial = dict(gate_center=task.gate_center.cpu().tolist(),
                           aperture=task.aperture.cpu().tolist(),
                           gate_angle=task.gate_angle.cpu().tolist(), start=task.position.cpu().tolist(),
                           root_state=task.t['robot_state_tensor'].cpu().tolist())
            saved = set()

            def save(i, info=None):
                if i in saved or not buffers[i]['depth']:
                    return
                episode = wave*args.num_envs+i
                filename = 'episode_%06d.npz' % episode
                data = {k: np.stack(buffers[i][k]) for k in FIELDS}
                metadata = dict(episode=episode, budget_truncated=info is None,
                    success=bool(info['success'][i]) if info is not None else False,
                    failure=bool(info['failure'][i]) if info is not None else False,
                    timeout=bool(info['timeout'][i]) if info is not None else False,
                    crossed=bool(task.tracker.crossed[i]),
                    seconds=float(info['episode_seconds'][i]) if info is not None else len(data['depth'])*.02,
                    initial={k:v[i] for k,v in initial.items()})
                save_episode(directory/filename, data, metadata)
                manifest['episodes'].append(dict(file=filename, length=len(data['depth']), **metadata))
                manifest['labels'] += len(data['depth'])
                saved.add(i)

            for step in range(400):
                active = (~task.tracker.done).cpu().numpy()
                chosen = np.flatnonzero(active)
                if args.label_budget is not None:
                    chosen = chosen[:args.label_budget-manifest['teacher_queries']]
                visual = stream(task.student_obs())
                # Only retained live states are queried; no hidden discarded labels.
                ids = torch.as_tensor(chosen, device=task.device, dtype=torch.long)
                labels = torch.zeros(args.num_envs, 4, device=task.device)
                labels[ids] = teacher_action(teacher, obs[ids])
                manifest['teacher_queries'] += len(chosen)
                with torch.no_grad():
                    if student:
                        predicted, hidden = student(visual['depth'][:, None], visual['proprio'][:, None], hidden)
                        applied = predicted[:, 0]
                    else:
                        applied = labels
                arrays = {k: v.detach().cpu().numpy() for k, v in dict(depth=visual['depth'],
                    proprio=visual['proprio'], teacher_action=labels, applied_action=applied).items()}
                for i in chosen:
                    for k in FIELDS:
                        buffers[i][k].append(arrays[k][i].copy())
                obs, _, done, info = task.step(applied)
                for i in done.nonzero(as_tuple=False).flatten().cpu().tolist():
                    if i in chosen:
                        save(i, info)
                if args.label_budget is not None and manifest['teacher_queries'] == args.label_budget:
                    for i in range(args.num_envs):
                        save(i)
                    break
                if task.tracker.done.all():
                    break
            budget_complete = args.label_budget is not None and manifest['teacher_queries'] == args.label_budget
            if not task.tracker.done.all() and not budget_complete:
                raise RuntimeError('Unfinished collection episodes')
            # Atomic progress metadata; incomplete collections remain explicitly marked.
            wave += 1
            manifest['complete'] = budget_complete if args.label_budget is not None else wave*args.num_envs == args.episodes
            assert manifest['teacher_queries'] == manifest['labels']
            temporary = directory/'manifest.tmp'
            temporary.write_text(json.dumps(manifest, indent=2))
            temporary.replace(directory/'manifest.json')
            print(json.dumps(dict(episodes=len(manifest['episodes']), labels=manifest['labels'],
                                  success=sum(x['success'] for x in manifest['episodes']))), flush=True)
            if manifest['complete']:
                break
    finally:
        task.close()


if __name__ == '__main__':
    main()
