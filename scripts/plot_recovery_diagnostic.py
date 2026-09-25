"""Compare the same predeclared validation case, without smoothing its poses."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.spatial.transform import Rotation
from uav_gap.runtime import ROOT,project_output
from video_font import simplified_chinese_font

font=FontProperties(fname=str(simplified_chinese_font()),size=11)
fig,axes=plt.subplots(2,2,figsize=(12,8),dpi=150)
curves=[]
for family,title,color in [('teacher_formal_v2_seed11_l0','原奖励 · 种子11','#159895'),
                          ('teacher_formal_v2_seed22_l0','原奖励 · 种子22','#ca7b15'),
                          ('teacher_recovery_dense_v2_diagnostic_seed22_l0','附加惩罚 · 种子22','#c94848')]:
    directory=ROOT/'runs/evaluation'/(family+'_epoch300_show')
    episode=json.loads((directory/'episodes.jsonl').read_text().splitlines()[0])
    with np.load(directory/'trace.npz') as trace:
        count=episode['steps'];p=trace['position'][:count,0].copy();q=trace['quaternion'][:count,0].copy();dt=float(trace['dt'])
    t=np.arange(count)*dt;distance=np.linalg.norm(p-[2,0,1.5],axis=1)
    speed=np.linalg.norm(np.diff(p,axis=0)/dt,axis=1)
    tilt=np.degrees(np.arccos(np.clip(Rotation.from_quat(q).as_matrix()[:,2,2],-1,1)))
    axes[0,0].plot(p[:,0],p[:,2],label=title,color=color)
    axes[0,0].plot(p[-1,0],p[-1,2],'x',color=color)
    axes[0,1].plot(t,distance,color=color)
    axes[1,0].plot(t[1:],speed,color=color)
    axes[1,1].plot(t,tilt,color=color)
    curves.append(dict(source=str(directory.relative_to(ROOT)),trace_sha256=hashlib.sha256((directory/'trace.npz').read_bytes()).hexdigest(),episode=episode))
axes[0,0].plot(2,1.5,'*',ms=13,color='black',label='目标')
axes[0,0].axvline(0,c='gray',lw=1,ls='--');axes[0,0].axhline(0,c='gray',lw=1)
axes[0,0].set_xlim(-3.2,3);axes[0,0].set_ylim(0,2.4)
axes[0,0].set_xlabel('前进位置 x（m）',fontproperties=font);axes[0,0].set_ylabel('高度 z（m）',fontproperties=font)
axes[0,0].legend(prop=font,loc='lower left')
for ax,ylabel,threshold in [(axes[0,1],'到目标的距离（m）',.3),(axes[1,0],'位置差分估计速度（m/s）',.3),(axes[1,1],'倾角（度）',10)]:
    ax.set_xlabel('仿真时间（秒）',fontproperties=font);ax.set_ylabel(ylabel,fontproperties=font)
    ax.axhline(threshold,color='black',lw=1,ls='--')
for ax in axes.flat:ax.grid(alpha=.2)
fig.suptitle('恢复失败诊断：同一个验证起点，原始轨迹',fontproperties=FontProperties(fname=font.get_file(),size=18))
fig.text(.5,.015,'第0回合，仅作行为诊断；速度为50Hz位置差分估计。正式成功判定仍使用200Hz真实状态。',ha='center',fontproperties=font)
fig.tight_layout(rect=[0,.035,1,.95])
out=project_output('artifacts/diagnostics/recovery_dense_v2_seed22');out.mkdir(parents=True,exist_ok=False)
fig.savefig(out/'comparison.png');plt.close(fig)
(out/'sources.json').write_text(json.dumps(dict(selection='episode0 of all three fixed validation batches; no smoothing',curves=curves,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
print(out/'comparison.png')
