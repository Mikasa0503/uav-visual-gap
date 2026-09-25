"""One-page pilot interview brief; final formal metrics are never invented."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype']=42
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch
from matplotlib.font_manager import FontProperties
from PIL import Image
from uav_gap.runtime import ROOT,project_output
from video_font import simplified_chinese_font,validate_text


def read(path):return json.loads(path.read_text())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=project_output('artifacts/reports/interview_brief_pilot_v2');out.mkdir(parents=True,exist_ok=False)
    sources={}
    for condition in ('show','heldout'):
        path=ROOT/('runs/evaluation/visual_pilot_moredata_seed11_dagger_gru_r9_'+condition+'/summary.json')
        summary=read(path)
        if not (summary['split']=='validation' and summary['num_episodes']==100 and summary['teacher_loaded'] is False
                and summary['student']['updates']==12000 and summary['student']['seed']==11):
            raise ValueError('Pilot brief requires the exact final-budget pilot evidence')
        sources[condition]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),summary=summary)
    assert sources['show']['summary']['checkpoint_sha256']==sources['heldout']['summary']['checkpoint_sha256']
    film=ROOT/'artifacts/film/concise_learning_film_v10_spinning_rotors/manifest.json';film_info=read(film)
    font_path=str(simplified_chinese_font())
    fig=plt.figure(figsize=(8.27,11.69),dpi=180,facecolor='#f6f8fb')
    canvas=fig.add_axes([0,0,1,1]);canvas.set_axis_off();texts=[]
    ink='#163043';muted='#526876';teal='#087f85';gold='#b66b00'

    def text(x,y,value,size=10,color=ink,weight='normal',**kwargs):
        validate_text(value,font_path);texts.append(value)
        fig.text(x,y,value,fontproperties=FontProperties(fname=font_path,size=size,weight=weight),color=color,va='top',**kwargs)

    def box(x,y,w,h,color='#ffffff',edge='none'):
        canvas.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.008',facecolor=color,edgecolor=edge,transform=canvas.transAxes))

    def heading(x,y,value):
        canvas.add_patch(Rectangle((x,y-.016),.005,.017,color=teal,transform=canvas.transAxes))
        text(x+.014,y,value,11.5,teal)

    text(.065,.96,'视觉无人机：倾斜穿缝与稳定恢复',22)
    text(.067,.918,'PPO 特权教师 → CNN+GRU 视觉策略 → DAgger 闭环纠正',10.3,muted)
    text(.067,.891,'先导验证稿  |  正式多种子对照尚未完成',9.5,gold)
    image_path=ROOT/'artifacts/film/concise_learning_film_v10_spinning_rotors/qa_03.png'
    ax=fig.add_axes([.065,.726,.575,.142]);ax.imshow(Image.open(image_path).crop((0,400,1280,720)));ax.set_axis_off()
    text(.068,.722,'真实仿真回放：转体穿缝后，在目标圆环处稳住',8,muted)
    for y,key,label in [(.806,'show','固定狭缝验证'),(.735,'heldout','留出几何验证')]:
        box(.687,y,.244,.057)
        count=round(sources[key]['summary']['success_rate']*100)
        text(.704,y+.05,'%d/100'%count,21,teal)
        text(.706,y+.02,label,8.5,muted)
    heading(.065,.676,'任务与成功标准')
    text(.081,.648,'46×46×12 cm 旋翼包络，穿过 66×24 cm、倾斜 45° 的狭缝。',10)
    text(.081,.626,'完整穿过后：距目标 <0.3 m、速度 <0.3 m/s、倾角 <10°，持续 0.5 s。',9)
    text(.081,.607,'物理与控制 200 Hz，策略与深度 50 Hz；接触失败，超时单独记录。',9,muted)
    heading(.065,.574,'具身算法链路')
    cards=[(.065,'01  PPO 教师','特权位姿与狭缝信息\n3×256 MLP；分级课程\n学习推力与机体系角速度'),
           (.365,'02  视觉行为克隆','64×64 深度＋16维自身信息\nCNN 编码＋128维 GRU\n因果序列训练与隐状态重置'),
           (.665,'03  DAgger 纠正','学生闭环飞行，教师标注\n聚合偏离示范分布的状态\n196,608 标签／12,000 更新')]
    for x,title,body in cards:
        box(x,.447,.269,.098)
        text(x+.009,.535,title,11,teal)
        text(x+.009,.511,body,8.4,linespacing=1.65)
    heading(.065,.414,'能力与验证证据')
    text(.081,.386,'感知到控制',10,teal)
    text(.081,.368,'视觉策略不读取狭缝真值或穿越标志；\n评测不加载教师，输出直接进入固定角速度控制器。',8.7,linespacing=1.7)
    text(.081,.327,'学习与闭环纠错',10,teal)
    text(.081,.309,'保留真实检查点及完整波动曲线；\n失败后继续 2 秒真实物理，记录接触力与速度反向。',8.7,linespacing=1.7)
    text(.081,.268,'可复核工程',10,teal)
    text(.081,.25,'固定场景、数据与模型哈希、统一回放镜头；\n源代码与配置可审阅；量化结论仍需对应的本地评测证据。',8.7,linespacing=1.7)
    # Separate future scope from measured pilot numbers.
    box(.58,.223,.35,.184,'#e8eff3')
    text(.595,.394,'正式对照：待完成',11,gold)
    text(.595,.366,'3 个种子：11 / 22 / 33\nBC-GRU / DAgger-GRU / 四帧堆叠\n匹配标签与优化预算，报告种子差异\n未见几何、额外延迟、深度丢帧\n质量变化、侧向扰动\n4 组×3 种子×6 条件，共 7,200 回合',8.5,linespacing=1.75)
    heading(.065,.195,'边界与面试讨论')
    text(.081,.168,'当前 98/100 与 88/100 来自 seed 11 的验证集，不是最终测试或三种子均值。',8.6)
    text(.081,.147,'自身状态由仿真真值加噪声与延迟构造；未实现 VIO，不作真实飞行迁移声明。',8.6)
    text(.081,.126,'可展开：教师到学生的信息差、分布偏移、记忆与四帧堆叠、恢复奖励与课程稳定性。',8.6)
    text(.065,.077,'Demo：%.2f 秒，4 个能力阶段＋预先指定的留出飞行。'%film_info['duration_seconds'],8.8,teal)
    text(.065,.053,'最终先导模型 SHA256：'+sources['show']['summary']['checkpoint_sha256'][:20]+'…',7.6,muted)
    text(.065,.035,'轨迹来自真实仿真，文字按标准字体排版；完整证据见项目内 JSON、检查点和原始录像。',7.6,muted)
    from pdf_font import build_pdf_font
    pdf_font=out/'brief_sc_subset.ttf'
    build_pdf_font(font_path,''.join(texts),pdf_font)
    for artist in fig.texts:
        artist.set_fontproperties(FontProperties(fname=str(pdf_font),size=artist.get_fontsize()))
    pdf=out/'interview_brief_pilot.pdf';fig.savefig(pdf,metadata={'Title':'UAV Visual Gap - Pilot validation interview brief','Author':'UAV Visual Gap project'})
    plt.close(fig)
    (out/'brief_text.txt').write_text('\n\n'.join(texts)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(status='pilot_draft_not_final_formal_report',sources={k:{a:b for a,b in v.items() if a!='summary'} for k,v in sources.items()},
        image_path=str(image_path.relative_to(ROOT)),image_sha256=sha(image_path),film_manifest_sha256=sha(film),
        font_family='Noto Sans CJK SC',font_sha256=sha(Path(font_path)),pdf_font_sha256=sha(pdf_font),
        font_conversion='Standard SC CFF glyphs converted to a TrueType subset (quadratic approximation error <=1 font unit); PDF text remains Unicode and copyable.',pdf_sha256=sha(pdf),script_sha256=sha(Path(__file__))),indent=2))
    print(pdf)


if __name__=='__main__':main()
