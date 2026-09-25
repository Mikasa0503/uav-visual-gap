"""Compare saved 50 Hz poses near the gate; never infer physical contacts."""
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation
from uav_gap.runtime import ROOT, project_output

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows = []
for version, epochs in ((2, (1130,1140,1150)), (3, (1430,1440,1450))):
    for epoch in epochs:
        directory = ROOT/('runs/evaluation/teacher_formal_v%d_seed11_l2_epoch%d_show'%(version,epoch))
        summary = json.loads((directory/'summary.json').read_text())
        episodes = [json.loads(x) for x in (directory/'episodes.jsonl').read_text().splitlines()]
        scenes = json.loads((directory/'scenarios.json').read_text())
        cases = []
        with np.load(directory/'trace.npz') as trace:
            for i,(ep,scene) in enumerate(zip(episodes,scenes)):
                n=ep['steps']; p=trace['position'][:n,i]; q=trace['quaternion'][:n,i]
                rotation=Rotation.from_quat(q).as_matrix()
                gate=Rotation.from_euler('x',scene['geometry'][2],degrees=True).as_matrix()
                center=(p-np.array(scene['gate_center']))@gate
                relative=np.einsum('ij,njk->nik',gate.T,rotation)
                extent=np.abs(relative)@np.array([.23,.23,.06])
                margin=np.array(scene['geometry'][:2])/2-np.abs(center[:,1:])-extent[:,1:]
                # Last pre-action pose nearest gate; success cases sample the actual crossing.
                j=int(np.argmin(np.abs(center[:,0])))
                euler=Rotation.from_matrix(rotation[j]).as_euler('xyz',degrees=True)
                speed=np.linalg.norm(np.diff(p,axis=0)/float(trace['dt']),axis=1)
                cases.append(dict(scene_id=ep['scene_id'],success=ep['success'],crossed=ep['crossed'],
                    seconds=ep['seconds'],terminal_position=ep['final_position'],
                    sample_seconds=j*float(trace['dt']),gate_x=float(center[j,0]),
                    gate_y=float(center[j,1]),gate_z=float(center[j,2]),
                    extent_y=float(extent[j,1]),extent_z=float(extent[j,2]),
                    margin_y=float(margin[j,0]),margin_z=float(margin[j,1]),
                    roll_deg=float(euler[0]),pitch_deg=float(euler[1]),yaw_deg=float(euler[2]),
                    approach_speed=float(speed[max(0,j-1)]),
                    action_saturation_fraction=float((np.abs(trace['actions'][:n,i])>.98).mean())))
        keys=['seconds','gate_x','gate_y','gate_z','extent_y','extent_z','margin_y','margin_z',
              'roll_deg','pitch_deg','yaw_deg','approach_speed','action_saturation_fraction']
        rows.append(dict(version=version,epoch=epoch,success_rate=summary['success_rate'],
            crossing_rate=summary['crossing_rate'],failures=summary['failures'],timeouts=summary['timeouts'],
            medians={k:float(np.median([c[k] for c in cases])) for k in keys},cases=cases,
            sources={str((directory/f).relative_to(ROOT)):sha(directory/f) for f in
                     ['summary.json','episodes.jsonl','scenarios.json','trace.npz']}))
out=project_output('runs/diagnostics/formal_v3_aperture');out.mkdir(parents=True,exist_ok=False)
result=dict(rows=rows,note='50Hz pre-action poses, not terminal200Hz state. Nearest-gate samples occur at different x for successes/failures. Negative margin identifies projected envelope obstruction, not an actual PhysX contact event or authoritative terminal cause. No test scenes used.')
(out/'report.json').write_text(json.dumps(result,indent=2))
for row in rows:
    print(json.dumps({k:v for k,v in row.items() if k not in ('cases','sources')}))
