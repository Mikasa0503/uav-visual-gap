"""Static scientific checks of stored trajectories and actual sensor frames."""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from uav_gap.runtime import project_output

parser = argparse.ArgumentParser()
parser.add_argument('--run', default='feasibility_camera')
args = parser.parse_args()
data = np.load(project_output('runs/'+args.run+'/trace.npz'))
p = data['position'][:, 0]
out = project_output('artifacts/diagnostics')
out.mkdir(parents=True, exist_ok=True)
fig = plt.figure(figsize=(13, 7))
ax = fig.add_subplot(2, 3, (1, 4), projection='3d')
ax.plot(p[:, 0], p[:, 1], p[:, 2], color='#007c91', lw=2)
ax.scatter(*p[0], color='#a7462d', label='Start')
ax.scatter(2, 0, 1.5, color='#2c8b47', label='Goal')
angle = np.pi/4
local = np.array([[-.33,-.12],[.33,-.12],[.33,.12],[-.33,.12],[-.33,-.12]])
y = local[:,0]*np.cos(angle)-local[:,1]*np.sin(angle)
z = 1.5+local[:,0]*np.sin(angle)+local[:,1]*np.cos(angle)
ax.plot(np.zeros(5), y, z, color='#a7462d', lw=3)
ax.set(xlabel='x (m)', ylabel='y (m)', zlabel='z (m)', title='Actuated physical trajectory')
ax.view_init(20, -70)
ax.legend()
pos = fig.add_subplot(2, 3, (2, 3))
t = np.arange(len(p))*.02
for i, name in enumerate('xyz'):
    pos.plot(t, p[:,i], label=name)
pos.set(xlabel='Simulated time (s)', ylabel='Position (m)', title='Position history')
pos.legend()
depth = data['depth']
for panel, target in [(5,0), (6,min(21,len(depth)-1))]:
    sensor = fig.add_subplot(2, 3, panel)
    image = sensor.imshow(depth[target,0,0], vmin=0, vmax=1, cmap='viridis')
    sensor.set_title('Actual depth at %.1fs' % (target*.1))
    sensor.axis('off')
fig.suptitle('Infrastructure validation — scripted feasibility, NOT a learned policy', fontsize=14)
fig.tight_layout()
path = out/(args.run+'.png')
fig.savefig(path, dpi=150)
print(path)
