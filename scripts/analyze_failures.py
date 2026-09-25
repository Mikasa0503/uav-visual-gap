"""Descriptive failure localization from saved traces; no policy/model updates."""
import argparse
import json
import math
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
from uav_gap.runtime import ROOT
from uav_gap.assets import VEHICLE_HALF_SIZE

parser = argparse.ArgumentParser()
parser.add_argument('--evaluation', required=True)
args = parser.parse_args()
directory = (ROOT/args.evaluation).resolve()
directory.relative_to(ROOT)
episodes = [json.loads(x) for x in (directory/'episodes.jsonl').read_text().splitlines()]
scenes = json.loads((directory/'scenarios.json').read_text())
with np.load(directory/'trace.npz',allow_pickle=False) as data:
    position, orientation, dt = data['position'], data['quaternion'], float(data['dt'])
    failures = []
    for i,row in enumerate(episodes):
        if not row['failure']:
            continue
        step = min(len(position)-1,row.get('steps',int(math.ceil(row['seconds']/dt)))-1)
        p = np.array(row['final_position'])
        near = 'near_aperture' if abs(p[0])<.5 else 'floor' if p[2]<.15 else 'bounds' if (abs(p[:2])>3.7).any() else 'other'
        width,height,angle = scenes[i]['geometry']
        gate = Rotation.from_euler('x',angle,degrees=True).as_matrix()
        body = Rotation.from_quat(orientation[step,i]).as_matrix()
        center = gate.T @ (position[step,i]-np.array(scenes[i]['gate_center']))
        envelope = abs(gate.T @ body) @ np.array(VEHICLE_HALF_SIZE)
        margins = np.array([width,height])/2-abs(center[1:])-envelope[1:]
        failures.append(dict(scene_id=row['scene_id'], region=near, seconds=row['seconds'],
            crossed=row['crossed'], gate_angle_deg=angle, gate_center=scenes[i]['gate_center'],
            sampled_source_index=step, sampled_local_center=center.tolist(),
            sampled_clearance_margins_yz=margins.tolist()))
result = dict(num_episodes=len(episodes), failures=len(failures), regions=dict(Counter(x['region'] for x in failures)),
    timeouts=sum(x['timeout'] for x in episodes), details=failures,
    note='Descriptive terminal localization and nearest recorded 50Hz pose; not exact 200Hz collision attribution.')
(directory/'failure_diagnostics.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2))
