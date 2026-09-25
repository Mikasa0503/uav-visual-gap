"""Plot recorded imitation loss separately from held-out closed-loop performance."""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from uav_gap.runtime import ROOT, project_output
from uav_gap.learning_history import lineage

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
args = parser.parse_args()
directory = (ROOT/'runs/experiments'/args.run).resolve()
directory.relative_to(ROOT)
request = json.loads((directory/'request.json').read_text())

fig, axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
rounds = []
schedule = [(folder, item, row) for folder,item in lineage(ROOT,directory) for row in item['rounds']]
for folder, item, row in schedule:
    progress_path = ROOT/'checkpoints'/row['stem']/'progress.jsonl'
    if progress_path.exists():
        progress = [json.loads(x) for x in progress_path.read_text().splitlines()]
        if progress:
            x = np.array([p['update'] for p in progress])
            y = np.array([p['mse'] for p in progress])
            axes[0].plot(x,y,alpha=.25,color='#2874a6',linewidth=.5)
            window = min(25,len(y))
            axes[0].plot(x[window-1:],np.convolve(y,np.ones(window)/window,mode='valid'),color='#2874a6')
            phase = 'BC' if row['index']==0 or request['group']=='bc_gru' else 'DAgger %d' % row['index']
            axes[0].axvline(x[0],color='gray',linewidth=.6,linestyle=':')
            axes[0].text(x[0],.97,phase,transform=axes[0].get_xaxis_transform(),va='top',fontsize=8)
    result = folder/('round%d.json' % row['index'])
    if result.exists():
        rounds.append(json.loads(result.read_text()))
axes[0].set(yscale='log',xlabel='Optimizer updates',ylabel='Normalized-action MSE',title='Supervised fit (not task success)')
for condition,color in [('show','#2874a6'),('heldout','#d35400')]:
    x,y = [],[]
    for row in rounds:
        match = next(item for item in row['evaluations'] if item['condition']==condition)
        x.append(row['labels'])
        y.append(100*match['success_rate'])
    if x:
        axes[1].plot(x,y,'o-',label=condition,color=color)
axes[1].axhline(90,color='#2874a6',linestyle=':',linewidth=.8,label='show target')
axes[1].axhline(80,color='#d35400',linestyle=':',linewidth=.8,label='heldout target')
axes[1].set(ylim=(-3,103),xlabel='Cumulative unique teacher labels',ylabel='Success (%)',
            title='Closed-loop validation: 100 cases / point')
axes[1].legend(fontsize=8,loc='center left')
for ax in axes:
    ax.grid(alpha=.15)
complete = (directory/'complete.json').exists()
fig.suptitle('%s, seed %d | %s | %s' % (request['group'],request['seed'],
    request['config']['stage'],'completed' if complete else 'in progress'),fontsize=12)
path = project_output('artifacts/learning/'+args.run+'_visual.png')
path.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(path,dpi=160)
print(path)
