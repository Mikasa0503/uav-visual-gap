"""Offline recovery diagnosis from existing validation traces (no test tuning)."""
import hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
from uav_gap.runtime import ROOT,project_output

rows=[]
for seed in (11,22):
    for epoch in (280,290,300):
        directory=ROOT/('runs/evaluation/teacher_formal_v2_seed%d_l0_epoch%d_show'%(seed,epoch))
        summary=json.loads((directory/'summary.json').read_text())
        episodes=[json.loads(line) for line in (directory/'episodes.jsonl').read_text().splitlines()]
        cases=[]
        with np.load(directory/'trace.npz') as trace:
            dt=float(trace['dt'])
            for i,episode in enumerate(episodes):
                n=min(episode['steps'],len(trace['position']))
                p=trace['position'][:n,i]
                distance=np.linalg.norm(p-np.array([2.,0.,1.5]),axis=1)
                speed=np.linalg.norm(np.diff(p,axis=0)/dt,axis=1)
                rotation=Rotation.from_quat(trace['quaternion'][:n,i]).as_matrix()
                tilt=np.degrees(np.arccos(np.clip(rotation[:,2,2],-1,1)))
                crossed=np.arange(n)*dt>=episode['crossing_seconds'] if episode['crossing_seconds'] is not None else np.zeros(n,dtype=bool)
                stable=crossed[1:]&(distance[1:]<.3)&(speed<.3)&(tilt[1:]<10)
                longest=current=0
                for value in stable:
                    current=current+1 if value else 0
                    longest=max(longest,current)
                cases.append(dict(scene_id=episode['scene_id'],final_distance=float(distance[-1]),min_distance=float(distance.min()),
                    last_second_speed=float(speed[-50:].mean()),last_second_tilt_deg=float(tilt[-50:].mean()),
                    sampled_longest_stable_seconds=longest*dt,action_saturation_fraction=float((np.abs(trace['actions'][:n,i])>.98).mean())))
        keys=[k for k in cases[0] if k!='scene_id']
        rows.append(dict(seed=seed,epoch=epoch,success_rate=summary['success_rate'],crossing_rate=summary['crossing_rate'],
            failures=summary['failures'],timeouts=summary['timeouts'],medians={k:float(np.median([r[k] for r in cases])) for k in keys},
            source=str(directory.relative_to(ROOT)),trace_sha256=hashlib.sha256((directory/'trace.npz').read_bytes()).hexdigest(),cases=cases))
out=project_output('runs/diagnostics/formal_v2_recovery');out.mkdir(parents=True,exist_ok=False)
result=dict(rows=rows,note='Pose-derived speed is a finite-difference estimate at 50Hz; sampled stability is diagnostic only, never replaces authoritative 200Hz termination. Each seed uses identical validation cases. No final-test scenes inspected.')
(out/'report.json').write_text(json.dumps(result,indent=2))
lines=['# Formal v2 wide-gate recovery diagnosis','',
    'Seed22 crossed every validation aperture but timed out without recovery at all three promotion snapshots. Seed11 completed the same gate.','',
    '| Seed | Epoch | Success | Crossing | Final distance (m) | Last-second speed (m/s) | Last-second tilt (deg) |',
    '|---|---|---|---|---|---|---|']
for row in rows:
    m=row['medians'];lines.append('| %d | %d | %.0f%% | %.0f%% | %.3f | %.3f | %.2f |'%(row['seed'],row['epoch'],100*row['success_rate'],100*row['crossing_rate'],m['final_distance'],m['last_second_speed'],m['last_second_tilt_deg']))
lines+=['','Last-second quantities and distances are medians over 100 episodes; successful episodes end earlier. Speed is estimated from saved positions, not direct velocities.',
    '', 'Development hypothesis: the original multiplicative recovery bonus becomes weak when distance and speed are simultaneously large, and lacks direct post-crossing upright feedback. Test bounded independent post-crossing costs for distance, speed and tilt. This is a hypothesis, not a proven causal conclusion.',
    '', 'Keep all existing failed runs. Do not lower success thresholds, extend a frozen formal budget in place, choose only favorable seeds, or report diagnostic runs as formal results.']
(out/'report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:12]))
