"""Offline diagnostics of fitting, previous-action shortcuts and visual sensitivity."""
import argparse
import json
import numpy as np
import torch
from uav_gap.runtime import ROOT, project_output
from uav_gap.student import load_student

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',required=True)
parser.add_argument('--datasets',nargs='+',required=True)
parser.add_argument('--count',type=int,default=16)
parser.add_argument('--name',required=True)
args = parser.parse_args()
checkpoint = (ROOT/args.checkpoint).resolve()
checkpoint.relative_to(ROOT)
model, metadata = load_student(checkpoint,device='cpu')
results = []
with torch.no_grad():
    for name in args.datasets:
        directory = (ROOT/name).resolve()
        directory.relative_to(ROOT)
        manifest = json.loads((directory/'manifest.json').read_text())
        indices = np.linspace(0,len(manifest['episodes'])-1,min(args.count,len(manifest['episodes'])),dtype=int)
        sums = np.zeros(5)
        per_action = np.zeros(4)
        frames = 0
        for index in indices:
            path = directory/manifest['episodes'][int(index)]['file']
            with np.load(path,allow_pickle=False) as data:
                depth = torch.from_numpy(data['depth'].astype(np.float32))[None]
                proprio = torch.from_numpy(data['proprio'])[None]
                target = torch.from_numpy(data['teacher_action'])[None]
            predicted,_ = model(depth,proprio)
            # Circular time shift preserves marginal images but breaks their
            # alignment with the current physical state. It is a diagnostic,
            # not a closed-loop ablation or proof of task-relevant perception.
            shifted,_ = model(depth.roll(depth.shape[1]//2,dims=1),proprio)
            errors = (predicted-target).square()
            sums[:4] += [float(errors.sum()),float((shifted-target).square().sum()),
                         float((proprio[:,:,-4:]-target).square().sum()),
                         float((predicted-shifted).square().sum())]
            per_action += errors.sum((0,1)).numpy()
            sums[4] += float((target.abs()>.95).sum())
            frames += target.shape[1]
        results.append(dict(dataset=name,episodes=len(indices),frames=frames,
            mse=float(sums[0]/(frames*4)),time_shifted_depth_mse=float(sums[1]/(frames*4)),
            previous_action_baseline_mse=float(sums[2]/(frames*4)),
            visual_action_change_rms=float(np.sqrt(sums[3]/(frames*4))),
            per_action_mse=(per_action/frames).tolist(),saturated_label_fraction=float(sums[4]/(frames*4))))
report = dict(checkpoint=args.checkpoint,student=metadata,results=results,
              note='Offline sampled training episodes; not closed-loop validation.')
path = project_output('runs/diagnostics/'+args.name+'.json')
path.parent.mkdir(parents=True,exist_ok=True)
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
