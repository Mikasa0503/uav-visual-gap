"""Compact full pilot validation curve for the short film; no smoothing."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from uav_gap.runtime import ROOT,project_output

from video_font import simplified_chinese_font, validate_text
font=FontProperties(fname=str(simplified_chinese_font()))
rows=[]
for family,rounds in [('visual_pilot_v1',range(4)),('visual_pilot_extension',range(4,6)),
                     ('visual_pilot_lowrate',range(6,8)),('visual_pilot_moredata',range(8,10))]:
    for index in rounds:
        row=dict(round=index,sources={})
        for condition in ('show','heldout'):
            path=ROOT/('runs/evaluation/%s_seed11_dagger_gru_r%d_%s/summary.json'%(family,index,condition))
            summary=json.loads(path.read_text())
            assert summary['split']=='validation' and summary['num_episodes']==100 and summary['student']['seed']==11
            row[condition]=100*summary['success_rate'];row['updates']=summary['student']['updates']
            row['sources'][condition]=dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        rows.append(row)
assert len(rows)==10 and rows[-1]['updates']==12000
out=project_output('artifacts/learning/concise_curve_sc');out.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(12.8,7.2),dpi=100,facecolor='#102330')
ax.set_facecolor('#102330')
for key,label,color in [('show','固定场景','#8ee7e6'),('heldout','留出几何','#ffbd79')]:
    ax.plot([r['updates'] for r in rows],[r[key] for r in rows],'-o',color=color,lw=3,ms=8,label=label)
    ax.annotate('%d/100'%rows[-1][key],(12000,rows[-1][key]),xytext=(-7,9 if key=='show' else -24),
                textcoords='offset points',ha='right',color=color,fontsize=18)
ax.set_xlim(0,12500);ax.set_ylim(-3,108);ax.set_xticks([0,4000,8000,12000]);ax.set_yticks([0,20,40,60,80,100])
ax.tick_params(colors='#dbe6ed',labelsize=16)
for spine in ax.spines.values():spine.set_color('#425869')
ax.grid(color='#425869',alpha=.5)
ax.set_xlabel('优化更新次数',fontproperties=font,fontsize=20,color='#dbe6ed')
ax.set_ylabel('成功率（%）',fontproperties=font,fontsize=20,color='#dbe6ed')
legend=ax.legend(loc='lower right',facecolor='#102330',edgecolor='#425869',prop=FontProperties(fname=font.get_file(),size=18))
for label in legend.get_texts():label.set_color('#dbe6ed')
fig.suptitle('训练会波动：完整验证曲线保留',fontproperties=font,fontsize=28,color='#8ee7e6',y=.96)
fig.text(.5,.035,'seed 11 先导实验 · 每个点 100 回合 · 折线连接实测点，没有平滑',ha='center',fontproperties=font,fontsize=16,color='#dbe6ed')
fig.subplots_adjust(left=.1,right=.96,bottom=.16,top=.87)
texts=[t.get_text() for t in fig.findobj(match=matplotlib.text.Text)]
for text in texts:
    validate_text(text,font.get_file())
fig.savefig(out/'learning_curve.png',dpi=100);plt.close(fig)
(out/'learning_curve.json').write_text(json.dumps(dict(rows=rows,texts=texts,font=dict(family='Noto Sans CJK SC',sha256=hashlib.sha256(Path(font.get_file()).read_bytes()).hexdigest()),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
print(out/'learning_curve.png')
