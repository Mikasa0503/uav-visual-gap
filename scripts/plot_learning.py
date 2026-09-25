"""Plot authentic recorded training metrics, clearly separate from evaluation."""
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from uav_gap.runtime import project_output

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
args = parser.parse_args()
rows = [json.loads(line) for line in project_output('checkpoints/'+args.run+'/progress.jsonl').read_text().splitlines()]
rows = [row for row in rows if 'success' in row]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
x = [row['frames']/1e6 for row in rows]
axes[0].plot(x, [100*row['success'] for row in rows], color='#007c91', lw=2)
axes[0].set(ylabel='Training success rate (%)', ylim=(-2, 102))
axes[1].plot(x, [row['episode_return'] for row in rows], color='#a7462d', lw=2)
axes[1].set(ylabel='Training episode return')
for ax in axes:
    ax.set_xlabel('Environment transitions (millions)')
    ax.grid(alpha=.2)
fig.suptitle('PPO state teacher — wide-gate curriculum\nRolling training metrics, not held-out evaluation')
fig.tight_layout()
out = project_output('artifacts/learning')
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out/(args.run+'.png'), dpi=160)
print(out/(args.run+'.png'))
