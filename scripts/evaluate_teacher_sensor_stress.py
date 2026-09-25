"""Separate teacher sensor-condition evaluator; active formal base sources stay frozen."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from uav_gap.runtime import require_idle_gpu, import_simulator, ROOT, project_output
from uav_gap.stress import stress_protocol, apply_scheduled_impulse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--level', type=int, choices=[3], default=3)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--condition', choices=['delay_40ms','depth_loss_3'], required=True)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--name', required=True)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--declaration',default='configs/teacher_sensor_test_v1.json')
    args = parser.parse_args()
    require_idle_gpu()
    sys.argv = [sys.argv[0]]
    import_simulator()
    import numpy as np
    import torch
    import yaml
    from uav_gap.task import GapTask, TaskConfig
    from uav_gap.inference import load_teacher, teacher_action
    from uav_gap.teacher_sensors import TeacherObservationDelay
    checkpoint = (ROOT/args.checkpoint).resolve()
    checkpoint.relative_to(ROOT)
    configuration = yaml.safe_load((checkpoint.parent/'configuration.yaml').read_text())
    path = ROOT/'configs/scenes'/(args.split+'.json')
    manifest = json.loads(path.read_text())
    scenes = [x.copy() for x in manifest['scenarios'] if x['condition'] == args.condition][:args.count]
    if len(scenes) != args.count:
        raise ValueError('Not enough predeclared scenarios')
    # Early curriculum stages use the same predeclared initial conditions, but
    # stage geometry is explicitly different from the final-scene evaluation.
    if args.level != 3:
        for scene in scenes:
            scene['geometry'] = list(GapTask.LEVELS[args.level])
    declaration_path = (ROOT/args.declaration).resolve()
    declaration_path.relative_to(ROOT)
    declaration = json.loads(declaration_path.read_text())
    if declaration['baseline_observation_delay_steps']!=0 or declaration['current_known_command_indices']!=[12,13,14,15]:
        raise ValueError('Teacher sensor declaration does not match the observation schema')
    protocol = stress_protocol(args.condition, baseline_delay_steps=0)
    delay = declaration['extra_observation_delay_steps'] if args.condition=='delay_40ms' else 0
    if protocol['observation_delay_steps']!=delay:
        raise ValueError('Teacher delay differs from the declared sensor protocol')
    protocol.update(depth_input_available=False,depth_drop_applied=False,
        applicability='privileged_state_delay' if delay else 'not_applicable_no_depth_input',
        known_command_indices=declaration['current_known_command_indices'],
        interpretation='Delayed measured state with latest known command' if delay else declaration['depth_loss_interpretation'])
    stream = TeacherObservationDelay(delay)
    cfg = TaskConfig(num_envs=len(scenes), seed=manifest['seed'], level=args.level,
                     randomize=False, auto_reset=False, cameras=False, mass_scale=protocol['mass_scale'])
    task = GapTask(cfg)
    directory = project_output('runs/evaluation/' + args.name)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        model = load_teacher(configuration, checkpoint, len(scenes))
        obs = task.reset(scenarios=scenes)
        with torch.random.fork_rng(devices=[0]):
            for _ in range(10):
                teacher_action(model,obs[:1])
            events = []
            for _ in range(100):
                begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                teacher_action(model,obs[:1])
                end.record()
                events.append((begin,end))
            torch.cuda.synchronize()
            latency = [a.elapsed_time(b) for a,b in events]
        rows = [None]*len(scenes)
        crossing_times = [None]*len(scenes)
        positions, quaternions, actions_record, observations_record, raw_observations = [], [], [], [], []
        for step in range(400):
            apply_scheduled_impulse(task, protocol, step)
            raw_obs = task.teacher_obs()
            obs = stream(raw_obs)
            actions = teacher_action(model, obs)
            if args.record:
                positions.append(task.position.cpu().numpy().copy())
                quaternions.append(task.t['robot_orientation'].cpu().numpy().copy())
                actions_record.append(actions.cpu().numpy().copy())
                observations_record.append(obs.cpu().numpy().copy())
                raw_observations.append(raw_obs.cpu().numpy().copy())
            obs, _, done, info = task.step(actions)
            for i in info['crossing'].nonzero(as_tuple=False).flatten().cpu().tolist():
                crossing_times[i] = float(task.tracker.elapsed[i])
            for i in done.nonzero(as_tuple=False).flatten().cpu().tolist():
                success,seconds = bool(info['success'][i]),float(info['episode_seconds'][i])
                rows[i] = dict(scene_id=scenes[i]['id'], success=bool(info['success'][i]),
                               failure=bool(info['failure'][i]), timeout=bool(info['timeout'][i]),
                               crossed=bool(task.tracker.crossed[i]),
                               seconds=seconds,steps=step+1,crossing_seconds=crossing_times[i],
                               recovery_seconds=seconds-crossing_times[i] if success and crossing_times[i] is not None else None,
                               final_position=info['terminal_position'][i].cpu().tolist())
            if task.tracker.done.all():
                break
        if any(row is None for row in rows):
            raise RuntimeError('Evaluation ended with unfinished episodes')
        successes = [row for row in rows if row['success']]
        summary = dict(kind='state_teacher_evaluation', split=args.split, condition=args.condition,
                       level=args.level, checkpoint=str(checkpoint.relative_to(ROOT)),
                       sensor_declaration_sha256=hashlib.sha256(declaration_path.read_bytes()).hexdigest(),
                       evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       teacher_sensor_source_sha256=hashlib.sha256((ROOT/'src/uav_gap/teacher_sensors.py').read_bytes()).hexdigest(),
                       checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       num_episodes=len(rows), success_rate=sum(x['success'] for x in rows)/len(rows),
                       crossing_rate=sum(x['crossed'] for x in rows)/len(rows),
                       failures=sum(x['failure'] for x in rows), timeouts=sum(x['timeout'] for x in rows),
                       success_seconds_mean=float(np.mean([x['seconds'] for x in successes])) if successes else None,
                       recovery_seconds_mean=float(np.mean([x['recovery_seconds'] for x in successes])) if successes else None,
                       single_robot_inference=dict(batch_size=1,warmup=10,iterations=100,
                           gpu_policy_ms_median=float(np.median(latency)),gpu_policy_ms_p95=float(np.percentile(latency,95)),
                           includes_sensor_rendering=False,includes_cpu_transfer=False),
                       stress_protocol=protocol, physical_mass_kg=task.t['robot_mass'].cpu().tolist(),
                       nominal_actuation_mass_kg=task.nominal_mass.cpu().tolist())
        (directory/'summary.json').write_text(json.dumps(summary, indent=2))
        (directory/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        (directory/'scenarios.json').write_text(json.dumps(scenes, indent=2))
        if args.record:
            np.savez_compressed(directory/'trace.npz', position=np.stack(positions),
                                quaternion=np.stack(quaternions), actions=np.stack(actions_record), dt=.02,
                                teacher_observation=np.stack(observations_record),
                                raw_teacher_observation=np.stack(raw_observations),
                                pose_timing='pre_action',terminal_steps=np.array([x['steps'] for x in rows]))
        print(json.dumps(summary, indent=2))
    finally:
        task.close()


if __name__ == '__main__':
    main()
