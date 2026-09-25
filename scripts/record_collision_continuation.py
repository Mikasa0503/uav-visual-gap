"""Observe genuine learned-policy dynamics after scored failure without resetting.

The first terminal result remains final. Post-terminal simulation is film-only,
uses the SAME policy/controller/physics, and does not change training/evaluation.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from uav_gap.runtime import ROOT,project_output,require_idle_gpu,import_simulator


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--record',required=True,help='Existing pilot film checkpoint record')
    parser.add_argument('--name',required=True)
    parser.add_argument('--tail-seconds',type=float,default=2.)
    args=parser.parse_args()
    if not 0<args.tail_seconds<=3: raise ValueError('Bound the diagnostic tail to 0-3 seconds')
    source=(ROOT/args.record).resolve();source.relative_to(ROOT)
    reference=json.loads(source.read_text())
    checkpoint=ROOT/reference['checkpoint']
    if sha(checkpoint)!=reference['checkpoint_sha256']: raise ValueError('Checkpoint changed')
    require_idle_gpu();sys.argv=[sys.argv[0]];import_simulator()
    import numpy as np
    import torch
    from uav_gap.student import load_student,benchmark_single_student
    from uav_gap.sensors import ObservationStream
    from uav_gap.task import GapTask,TaskConfig
    manifest_path=ROOT/'configs/scenes/showcase.json'
    manifest=json.loads(manifest_path.read_text());scenes=manifest['scenarios']
    if len(scenes)!=1: raise ValueError('Continuation uses the single fixed nominal showcase')
    model,metadata=load_student(checkpoint)
    task=GapTask(TaskConfig(num_envs=1,seed=manifest['seed'],cameras=True,level=3,
                            randomize=False,auto_reset=False))
    output=project_output('runs/evaluation/'+args.name);output.mkdir(parents=True,exist_ok=False)
    records={k:[] for k in ('position','quaternion','actions','depth','proprio')}
    physics={k:[] for k in ('position','quaternion','velocity','angular_velocity','physical_contact','contact_force','collision_counter')}
    try:
        task.reset(scenarios=scenes)
        with torch.random.fork_rng(devices=[0]):
            latency=benchmark_single_student(model,task.student_obs())
        original_step=task.sim.step
        def observed_physics_step(si):
            result=original_step(si)
            for name,value in dict(position=task.position,quaternion=task.t['robot_orientation'],
                velocity=task.t['robot_linvel'],angular_velocity=task.t['robot_body_angvel'],
                physical_contact=(torch.linalg.vector_norm(task.t['robot_contact_force_tensor'],dim=-1)>task.sim.cfg.env.collision_force_threshold),
                contact_force=task.t['robot_contact_force_tensor'],collision_counter=task.t['crashes']).items():
                physics[name].append(value.cpu().numpy().copy())
            return result
        task.sim.step=observed_physics_step
        hidden=None;stream=ObservationStream(metadata['observation_delay_steps'])
        terminal=None;crossing=None;stop_after=None
        tail_steps=round(args.tail_seconds/.02)
        for step in range(400+tail_steps):
            visual=stream(task.student_obs())
            with torch.no_grad():
                action,hidden=model(visual['depth'][:,None],visual['proprio'][:,None],hidden)
                action=action[:,0]
            for name,value in dict(position=task.position,quaternion=task.t['robot_orientation'],actions=action,**visual).items():
                records[name].append(value.cpu().numpy().copy().astype(np.float16 if name=='depth' else np.float32))
            _,_,done,info=task.step(action)
            if bool(info['crossing'][0]): crossing=float(task.tracker.elapsed[0])
            if terminal is None and bool(done[0]):
                success=bool(info['success'][0]);seconds=float(info['episode_seconds'][0])
                terminal=dict(scene_id=scenes[0]['id'],success=success,failure=bool(info['failure'][0]),
                    timeout=bool(info['timeout'][0]),crossed=bool(task.tracker.crossed[0]),
                    seconds=seconds,steps=step+1,crossing_seconds=crossing,
                    recovery_seconds=seconds-crossing if success and crossing is not None else None,
                    final_position=info['terminal_position'][0].cpu().tolist())
                # Successful clips finish normally; failures/timeouts show the aftermath.
                stop_after=step+1+(0 if success else tail_steps)
            if stop_after is not None and step+1>=stop_after: break
        if terminal is None: raise RuntimeError('No scored terminal result')
        if not all(np.isfinite(np.asarray(physics[k])).all() for k in ('position','quaternion','velocity')):
            raise RuntimeError('Non-finite continued physics; do not render as valid motion')
        terminal['recorded_steps']=len(records['position'])
        terminal['post_terminal_seconds']=(terminal['recorded_steps']-terminal['steps'])*.02
        # Compare the common pre-terminal portion to the existing original-speed recording.
        original=ROOT/reference['evaluations']['showcase']['directory']
        prefix={}
        with np.load(original/'trace.npz',allow_pickle=False) as previous:
            n=min(terminal['steps'],len(previous['position']))
            for key in ('position','quaternion','actions'):
                prefix[key+'_max_abs_difference']=float(np.max(np.abs(np.asarray(records[key])[:n]-previous[key][:n])))
        summary=dict(kind='visual_student_evaluation',teacher_loaded=False,student=metadata,
            split='showcase',condition='show',level=3,checkpoint=str(checkpoint.relative_to(ROOT)),
            checkpoint_sha256=sha(checkpoint),manifest_sha256=sha(manifest_path),num_episodes=1,
            success_rate=float(terminal['success']),crossing_rate=float(terminal['crossed']),
            failures=int(terminal['failure']),timeouts=int(terminal['timeout']),
            success_seconds_mean=terminal['seconds'] if terminal['success'] else None,
            recovery_seconds_mean=terminal['recovery_seconds'],single_robot_inference=latency,
            recording_purpose='Film-only continuous physics after the unchanged scored terminal result',
            continuation=dict(seconds=terminal['post_terminal_seconds'],control='Same closed-loop visual policy; no reset or external impulse',
                physics='Original task materials and collision model, unchanged',score_frozen_at_seconds=terminal['seconds']),
            original_film_record=str(source.relative_to(ROOT)),original_film_record_sha256=sha(source),
            original_prefix_comparison=prefix,recorder_sha256=sha(Path(__file__)))
        (output/'summary.json').write_text(json.dumps(summary,indent=2))
        (output/'episodes.jsonl').write_text(json.dumps(terminal)+'\n')
        (output/'scenarios.json').write_text(json.dumps(scenes,indent=2))
        np.savez_compressed(output/'trace.npz',**{k:np.stack(v) for k,v in records.items()},
            dt=.02,pose_timing='pre_action',terminal_steps=np.array([terminal['steps']]),
            recorded_steps=np.array([terminal['recorded_steps']]))
        np.savez_compressed(output/'physics_trace.npz',**{k:np.stack(v) for k,v in physics.items()},
            dt=.005,pose_timing='post_action')
        contact=np.asarray(physics['physical_contact']).reshape(len(physics['physical_contact']),-1).any(axis=1)
        position=np.asarray(physics['position'])[:,0];velocity=np.asarray(physics['velocity'])[:,0]
        near_wall=np.abs(position[:,0])<.65
        reversed_near_wall=near_wall & (velocity[:,0]<-.05)
        audit=dict(recorded_seconds=len(physics['position'])*.005,scored_terminal_seconds=terminal['seconds'],
            terminal=terminal,physical_contact_sample_count=int(contact.sum()),
            contact_force_threshold_n=float(task.sim.cfg.env.collision_force_threshold),
            first_physical_contact_seconds=float((np.flatnonzero(contact)[0]+1)*.005) if contact.any() else None,
            negative_x_velocity_near_wall_sample_count=int(reversed_near_wall.sum()),
            min_vx_near_wall=float(velocity[near_wall,0].min()) if near_wall.any() else None,
            max_vx_near_wall=float(velocity[near_wall,0].max()) if near_wall.any() else None,
            prefix_comparison=prefix,
            note='Contact and velocity are measured physics evidence, not scripted recoil. Visual inspection still required.')
        (output/'continuation_audit.json').write_text(json.dumps(audit,indent=2))
        print(json.dumps(audit,indent=2))
    finally: task.close()


if __name__=='__main__': main()
