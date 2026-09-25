"""Descriptive held-out geometry diagnostics; no policy selection or testing."""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from uav_gap.runtime import ROOT, project_output

parser = argparse.ArgumentParser()
parser.add_argument('--evaluations',nargs='+',required=True)
parser.add_argument('--name',required=True)
args = parser.parse_args()
fig,axes = plt.subplots(2,len(args.evaluations),figsize=(5*len(args.evaluations),6),squeeze=False,
                      constrained_layout=True)
reports = []
for col,name in enumerate(args.evaluations):
    folder = (ROOT/name).resolve()
    folder.relative_to(ROOT)
    summary = json.loads((folder/'summary.json').read_text())
    if summary['split']!='validation' or summary['condition']!='heldout':
        raise ValueError('This diagnostic is for held-out validation only')
    scenes = json.loads((folder/'scenarios.json').read_text())
    episodes = [json.loads(x) for x in (folder/'episodes.jsonl').read_text().splitlines()]
    assert [x['id'] for x in scenes]==[x['scene_id'] for x in episodes]
    angle = np.array([x['geometry'][2] for x in scenes])
    success = np.array([x['success'] for x in episodes])
    bins = []
    for low in range(40,50,2):
        selected = (angle>=low)&(angle<low+2)
        bins.append(dict(angle_interval=[low,low+2],episodes=int(selected.sum()),
                         successes=int(success[selected].sum())))
    height = np.array([x['gate_center'][2] for x in scenes])-1.5
    height_groups = []
    for label,selected in [('below_-5cm',height<-.05),('within_5cm',abs(height)<=.05),('above_5cm',height>.05)]:
        height_groups.append(dict(group=label,episodes=int(selected.sum()),successes=int(success[selected].sum())))
    reports.append(dict(evaluation=name,angle_bins=bins,height_groups=height_groups))
    for row,component in enumerate((1,2)):
        offset = np.array([x['gate_center'][component] for x in scenes])-(1.5 if component==2 else 0)
        ax = axes[row,col]
        ax.scatter(angle[success],offset[success],s=22,c='#239b56',label='success')
        ax.scatter(angle[~success],offset[~success],s=26,c='#c0392b',marker='x',label='failure/timeout')
        ax.set(xlabel='Gate roll (degrees)',ylabel='Gate '+('Y' if component==1 else 'Z')+' offset (m)')
        ax.grid(alpha=.15)
        if row==0:
            ax.set_title('%s updates | %d/%d' % (summary['student']['updates'],success.sum(),len(success)))
            ax.legend(fontsize=8)
fig.suptitle('Fixed held-out validation cases: descriptive geometry failure map')
output = project_output('artifacts/learning/'+args.name)
output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(output.with_suffix('.png'),dpi=150)
output.with_suffix('.json').write_text(json.dumps(dict(results=reports,
    note='Small fixed validation bins; descriptive associations, not causal attribution or new independent trials.'),indent=2))
print(json.dumps(reports,indent=2))
