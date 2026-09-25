"""Generate the final one-page brief only from complete, source-bound results."""
import argparse,json,os,subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg');matplotlib.rcParams['pdf.fonttype']=42
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from uav_gap.runtime import ROOT,project_output
from uav_gap.formal_names import protocol_version,test_prefix
from uav_gap.handoff import digest
from uav_gap.results import seed_statistics,verify_summary
from video_font import simplified_chinese_font,validate_text
from pdf_font import build_pdf_font

GROUPS=('teacher','bc_gru','dagger_gru','dagger_stack4')
CONDITIONS=('show','heldout','delay_40ms','depth_loss_3','mass_plus20','lateral_impulse')
LABELS=('固定狭缝','未见几何','额外40ms延迟','连续丢3帧深度','质量增加20%','侧向冲量')


def read(path):return json.loads(path.read_text())


def load_results(version):
    path=ROOT/('artifacts/results/formal_'+version+'/results.json');data=read(path)
    if data['version']!=version or data['evaluation_count']!=72 or data['episode_count']!=7200:
        raise ValueError('Complete formal results required')
    for family,key in [('formal_matrix_','matrix_request_sha256'),('teacher_sensor_test_','supplement_request_sha256')]:
        directory=ROOT/'runs/experiments'/(family+version)
        actual=digest(directory/'request.json')
        if read(directory/'complete.json')['request_sha256']!=actual or data[key]!=actual:
            raise ValueError('Formal completion/provenance mismatch')
    evidence={r['name']:r for r in data['evidence']}
    expected={(g,s,c) for g in GROUPS for s in (11,22,33) for c in CONDITIONS}
    if len(data['rows'])!=72 or {(r['group'],r['seed'],r['condition']) for r in data['rows']}!=expected:
        raise ValueError('Incomplete seed/group/condition coverage')
    for row in data['rows']:
        name='%s_seed%d_%s_%s'%(test_prefix(version),row['seed'],row['group'],row['condition'])
        directory=ROOT/'runs/evaluation'/name
        for file,value in evidence[name]['files'].items():
            if digest(directory/file)!=value:raise ValueError('Underlying result changed: '+name)
        verify_summary(read(directory/'summary.json'),row)
    for aggregate in data['aggregates']:
        subset=[r for r in data['rows'] if r['group']==aggregate['group'] and r['condition']==aggregate['condition']]
        for key,value in aggregate['metrics'].items():
            if seed_statistics([r[key] for r in subset])!=value:raise ValueError('Aggregate does not match all seeds')
    return path,data


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',type=protocol_version,default='v3')
    parser.add_argument('--layout-proof',action='store_true');args=parser.parse_args()
    if args.layout_proof:path,data=None,None
    else:path,data=load_results(args.version)
    out=project_output('artifacts/reports/'+('formal_brief_layout_proof' if args.layout_proof else 'interview_brief_final_'+args.version))
    out.mkdir(parents=True,exist_ok=False)
    aggregates={} if data is None else {(r['group'],r['condition']):r for r in data['aggregates']}
    def metric(group,condition,key='success_rate',scale=100):
        if data is None:return '待完成'
        stats=aggregates[group,condition]['metrics'][key]
        if stats['mean'] is None:return '无成功样本'
        digits=1 if key=='success_rate' else 2
        value=format(stats['mean']*scale,'.%df'%digits)
        value+=' ± '+format(stats['sample_sd']*scale,'.%df'%digits) if stats['sample_sd'] is not None else '（SD未定义）'
        if stats['defined_seeds']!=3:value+=' [%d/3]'%stats['defined_seeds']
        return value
    font_path=str(simplified_chinese_font());texts=[]
    fig=plt.figure(figsize=(8.27,11.69),dpi=180,facecolor='#f6f8fb')
    canvas=fig.add_axes([0,0,1,1]);canvas.set_axis_off()
    def text(x,y,value,size=9,color='#183447',**kwargs):
        validate_text(value,font_path);texts.append(value)
        fig.text(x,y,value,fontproperties=FontProperties(fname=font_path,size=size),color=color,va='top',**kwargs)
    def section(y,value):
        canvas.add_patch(Rectangle((.064,y-.017),.005,.018,color='#087f85'))
        text(.08,y,value,12,'#087f85')
    text(.065,.96,'视觉无人机：倾斜穿缝与稳定恢复',22)
    text(.067,.916,'PPO 教师 → CNN+GRU 行为克隆 → DAgger 闭环纠正',11,'#536b79')
    text(.067,.89,'版式校验稿：尚无正式数据，不作为结果使用' if data is None else '正式 '+args.version+' | 3 个训练种子 | 6 个测试条件 | 7,200 回合',9.5,'#b36b00' if data is None else '#087f85')
    section(.85,'任务与具身算法')
    text(.08,.824,'46×46×12 cm 旋翼包络，穿越 66×24 cm、倾斜 45° 狭缝后稳定悬停。',9.4)
    text(.08,.802,'成功：完整穿过，距目标<0.3m、速度<0.3m/s、倾角<10°，保持0.5s。',9)
    text(.08,.78,'物理/控制200Hz，视觉策略50Hz；策略输出推力与机体系角速度。',9)
    text(.08,.752,'教师：3×256 MLP，特权状态与分级课程；学生：64×64深度＋16维辅助信息。',8.8)
    text(.08,.731,'CNN＋128维GRU，因果序列训练；DAgger使用学生实际闭环轨迹进行教师标注。',8.8)
    text(.08,.71,'每组每种子196,608个标签、12,000次更新；四帧模型与GRU参数差约0.26%。',8.8)
    section(.671,'完整测试：成功率 %，三种子均值 ± 样本标准差')
    headers=('测试条件','状态教师','BC-GRU','DAgger-GRU','DAgger-4帧')
    xs=(.08,.32,.485,.65,.815)
    for x,title in zip(xs,headers):text(x,.64,title,8.6,'#087f85')
    for index,(condition,label) in enumerate(zip(CONDITIONS,LABELS)):
        y=.611-index*.034
        if index%2==0:canvas.add_patch(Rectangle((.065,y-.027),.87,.032,color='#e8eff3'))
        text(xs[0],y,label,8.5)
        for x,group in zip(xs[1:],GROUPS):
            text(x,y,'N/A（对照）' if group=='teacher' and condition=='depth_loss_3' else metric(group,condition),8.3)
    text(.08,.393,'教师没有深度输入；丢帧项仅为输入不变对照，不代表视觉鲁棒性。',7.9,'#536b79')
    section(.362,'主方法细节与失败证据')
    if data is None:
        detail='逐种子固定/留出成功率、完成时间、推理延迟和失败计数：待完整结果核验。'
        lines=[detail,'真实结果生成时将绑定评测轨迹、最终模型与汇总文件的哈希。']
    else:
        model=[r for r in data['rows'] if r['group']=='dagger_gru']
        rates=[]
        for condition in ('show','heldout'):
            values=[next(r['success_rate'] for r in model if r['condition']==condition and r['seed']==s) for s in (11,22,33)]
            rates.append('/'.join('%.0f'%(100*v) for v in values))
        lines=['种子11/22/33：固定狭缝 '+rates[0]+'%；未见几何 '+rates[1]+'%。',
            '固定狭缝：完成时间 '+metric('dagger_gru','show','success_seconds_mean',1)+' s；策略p50 '+metric('dagger_gru','show','policy_latency_median_ms',1)+' ms。',
            '主方法全部1,800回合：失败%d，超时%d；穿过后失败%d、穿过后超时%d。'%(
                sum(r['failures'] for r in model),sum(r['timeouts'] for r in model),sum(r['failure_after_crossing'] for r in model),sum(r['timeout_after_crossing'] for r in model))]
    for i,line in enumerate(lines):text(.08,.333-i*.023,line,8.7)
    text(.08,.255,'时间仅统计成功回合；缺失种子不补零。策略延迟不含传感器、传输或控制。',8,'#536b79')
    section(.222,'证据边界与面试讨论')
    text(.08,.194,'每种子复用同一批100个场景；种子标准差不是置信区间，也不是300个独立场景。',8.4)
    text(.08,.173,'不同扰动条件有各自固定起点；参数/标签/更新预算匹配不等于FLOPs匹配。',8.4)
    text(.08,.152,'自身信息来自加噪、延迟的仿真状态，未实现VIO；不作真实飞行迁移声明。',8.4)
    text(.08,.131,'可展开：感知与特权信息差、DAgger分布纠偏、时序记忆、课程稳定性与失败诊断。',8.4)
    text(.065,.084,'21.68秒先导Demo展示四个学习阶段与留出飞行；正式评测录像待完成。',8.5,'#087f85')
    text(.065,.061,'报告状态：版式校验，所有指标待完成' if data is None else '完整结果 SHA256：'+digest(path)[:24]+'…',7.8,'#536b79')
    text(.065,.041,'原始轨迹、每种子指标、固定模型预算和失败实验均保留，未选取最佳种子代替总体。',7.8,'#536b79')
    ttf=out/'brief_sc_subset.ttf';build_pdf_font(font_path,''.join(texts),ttf)
    for artist in fig.texts:artist.set_fontproperties(FontProperties(fname=str(ttf),size=artist.get_fontsize()))
    pdf=out/'interview_brief.pdf';fig.savefig(pdf);plt.close(fig)
    (out/'brief_text.txt').write_text('\n\n'.join(texts)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(status='layout_proof_not_results' if data is None else 'formal_report_pending_visual_review',version=args.version,
        formal_results_sha256=None if path is None else digest(path),pdf_sha256=digest(pdf),script_sha256=digest(Path(__file__)),font_sha256=digest(ttf)),indent=2))
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None)
    subprocess.run(['/usr/bin/pdftoppm','-scale-to','1800','-png','-singlefile',str(pdf),str(out/'rendered')],env=env,check=True)
    subprocess.run(['/usr/bin/pdftotext',str(pdf),str(out/'extracted.txt')],env=env,check=True)
    from collections import Counter
    if Counter(''.join(''.join(texts).split()))!=Counter(''.join((out/'extracted.txt').read_text().split())):
        raise ValueError('PDF text extraction differs from source; inspect before delivery')
    print(pdf)


if __name__=='__main__':main()
