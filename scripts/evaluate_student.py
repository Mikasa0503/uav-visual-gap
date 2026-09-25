"""Teacher-free visual student evaluation with real closed-loop trajectories."""
import argparse
import hashlib
import json
import sys
from uav_gap.runtime import require_idle_gpu, import_simulator, ROOT, project_output
from uav_gap.stress import CONDITIONS, stress_protocol, apply_scheduled_impulse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--level', type=int, default=3)
    parser.add_argument('--split', choices=['validation','test','showcase'], default='validation')
    parser.add_argument('--condition', choices=CONDITIONS, default='show')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--name', required=True)
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    require_idle_gpu()
    sys.argv = [sys.argv[0]]
    import_simulator()
    import numpy as np
    import torch
    from uav_gap.student import load_student, benchmark_single_student
    from uav_gap.task import GapTask, TaskConfig
    from uav_gap.sensors import ObservationStream
    path = ROOT/'configs/scenes'/(args.split+'.json')
    manifest = json.loads(path.read_text())
    scenes = [x.copy() for x in manifest['scenarios'] if x['condition']==args.condition][:args.count]
    if len(scenes) != args.count:
        raise ValueError('Not enough declared scenarios')
    if args.level != 3:
        for scene in scenes:
            scene['geometry'] = list(GapTask.LEVELS[args.level])
    checkpoint = (ROOT/args.checkpoint).resolve()
    checkpoint.relative_to(ROOT)
    model, metadata = load_student(checkpoint)
    protocol = stress_protocol(args.condition, metadata['observation_delay_steps'])
    task = GapTask(TaskConfig(num_envs=len(scenes), seed=manifest['seed'], cameras=True,
                            level=args.level, randomize=False, auto_reset=False,
                            mass_scale=protocol['mass_scale']))
    directory = project_output('runs/evaluation/'+args.name)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        task.reset(scenarios=scenes)
        # Benchmarking must not consume the evaluation sensor-noise random stream.
        with torch.random.fork_rng(devices=[0]):
            single_latency = benchmark_single_student(model, task.student_obs())
        rows, crossing_times = [None]*len(scenes), [None]*len(scenes)
        hidden, timings = None, []
        stream = ObservationStream(protocol['observation_delay_steps'], protocol['drop_start'], protocol['drop_frames'])
        records = {k: [] for k in ('position','quaternion','actions','depth','proprio')}
        for step in range(400):
            apply_scheduled_impulse(task, protocol, step)
            visual = stream(task.student_obs())
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.no_grad():
                start.record()
                actions, hidden = model(visual['depth'][:, None], visual['proprio'][:, None], hidden)
                actions = actions[:, 0]
                end.record()
            timings.append((start, end))
            if args.record:
                # Observation, pose and action are recorded at the SAME pre-step time.
                for name, value in dict(position=task.position, quaternion=task.t['robot_orientation'],
                                       actions=actions, **visual).items():
                    records[name].append(value.cpu().numpy().copy().astype(np.float16 if name=='depth' else np.float32))
            _, _, done, info = task.step(actions)
            for i in info['crossing'].nonzero(as_tuple=False).flatten().cpu().tolist():
                crossing_times[i] = float(task.tracker.elapsed[i])
            for i in done.nonzero(as_tuple=False).flatten().cpu().tolist():
                seconds = float(info['episode_seconds'][i])
                success = bool(info['success'][i])
                rows[i] = dict(scene_id=scenes[i]['id'], success=success, failure=bool(info['failure'][i]),
                    timeout=bool(info['timeout'][i]), crossed=bool(task.tracker.crossed[i]),
                    seconds=seconds, steps=step+1, crossing_seconds=crossing_times[i],
                    recovery_seconds=seconds-crossing_times[i] if success and crossing_times[i] is not None else None,
                    final_position=info['terminal_position'][i].cpu().tolist())
            if task.tracker.done.all():
                break
        if any(row is None for row in rows):
            raise RuntimeError('Unfinished evaluation episodes')
        torch.cuda.synchronize()
        latency = np.array([a.elapsed_time(b) for a,b in timings])
        successes = [x for x in rows if x['success']]
        summary = dict(kind='visual_student_evaluation', teacher_loaded=False,
            student=metadata, split=args.split, condition=args.condition, level=args.level,
            checkpoint=str(checkpoint.relative_to(ROOT)),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), num_episodes=len(rows),
            success_rate=len(successes)/len(rows), crossing_rate=sum(x['crossed'] for x in rows)/len(rows),
            failures=sum(x['failure'] for x in rows), timeouts=sum(x['timeout'] for x in rows),
            success_seconds_mean=float(np.mean([x['seconds'] for x in successes])) if successes else None,
            recovery_seconds_mean=float(np.mean([x['recovery_seconds'] for x in successes])) if successes else None,
            inference=dict(batch_size=len(scenes), includes_first_call=True,
                           gpu_batch_ms_median=float(np.median(latency)), gpu_batch_ms_p95=float(np.percentile(latency,95))),
            single_robot_inference=single_latency, stress_protocol=protocol,
            physical_mass_kg=task.t['robot_mass'].cpu().tolist(),
            nominal_actuation_mass_kg=task.nominal_mass.cpu().tolist(),
            baseline_observation_delay_ms=20*metadata['observation_delay_steps'],
            proprioception='noisy simulated state estimate, not VIO; latest known previous command')
        (directory/'summary.json').write_text(json.dumps(summary, indent=2))
        (directory/'episodes.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
        (directory/'scenarios.json').write_text(json.dumps(scenes, indent=2))
        if args.record:
            np.savez_compressed(directory/'trace.npz', **{k:np.stack(v) for k,v in records.items()},
                                dt=.02, pose_timing='pre_action', terminal_steps=np.array([x['steps'] for x in rows]))
        print(json.dumps(summary, indent=2))
    finally:
        task.close()


if __name__ == '__main__':
    main()
