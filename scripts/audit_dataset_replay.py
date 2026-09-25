"""Replay serialized student rollouts to detect observation/action alignment bugs."""
import argparse
import hashlib
import json
import numpy as np
import torch
from uav_gap.runtime import ROOT
from uav_gap.student import load_student

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--count', type=int, default=16)
args = parser.parse_args()
directory = (ROOT/args.dataset).resolve()
directory.relative_to(ROOT)
manifest = json.loads((directory/'manifest.json').read_text())
if manifest['controller'] != 'student_dagger' or not manifest['complete']:
    raise ValueError('A completed student-controlled dataset is required')
checkpoint = ROOT/manifest['student']
model, metadata = load_student(checkpoint,device='cpu')
indices = np.linspace(0,len(manifest['episodes'])-1,min(args.count,len(manifest['episodes'])),dtype=int)
rows = []
with torch.no_grad():
    for index in indices:
        entry = manifest['episodes'][int(index)]
        with np.load(directory/entry['file'],allow_pickle=False) as episode:
            depth = torch.from_numpy(episode['depth'].astype(np.float32))[None]
            proprio = torch.from_numpy(episode['proprio'])[None]
            applied = torch.from_numpy(episode['applied_action'])
        predicted, _ = model(depth,proprio)
        streamed, hidden = [],None
        for t in range(depth.shape[1]):
            output, hidden = model(depth[:,t:t+1],proprio[:,t:t+1],hidden)
            streamed.append(output[0,0])
        streamed = torch.stack(streamed)
        rows.append(dict(file=entry['file'],frames=len(applied),
            serialized_vs_applied_max_abs=float((predicted[0]-applied).abs().max()),
            serialized_vs_applied_rmse=float((predicted[0]-applied).square().mean().sqrt()),
            batch_vs_stream_max_abs=float((predicted[0]-streamed).abs().max())))
report = dict(dataset=args.dataset,checkpoint=manifest['student'],
    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),student=metadata,
    episodes=len(rows),frames=sum(x['frames'] for x in rows),
    max_serialized_action_error=max(x['serialized_vs_applied_max_abs'] for x in rows),
    max_batch_stream_error=max(x['batch_vs_stream_max_abs'] for x in rows),details=rows,
    note='CPU replay of float16-stored depth vs original GPU float32 inputs; no physics/model updates.')
(directory/'replay_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='details'},indent=2))
