"""Four capability milestones with real collision aftermath; no padded flight clips."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
import assemble_learning_film as video
from uav_gap.runtime import ROOT,project_output

from video_font import simplified_chinese_font, validate_text

# Extract the SC face explicitly; TTC face 0 is Japanese.
video.FONT=str(simplified_chinese_font())
TITLES={250:'01  初期：能靠近，仍会撞门',3000:'02  能穿过去，还不会稳定恢复',
        4000:'03  首次完整完成：穿缝并稳住',12000:'04  更成熟：恢复更快，成功更稳定'}


def portable_manifest_value(value):
    if isinstance(value, str):
        return value.replace(str(ROOT), '.')
    if isinstance(value, list):
        return [portable_manifest_value(item) for item in value]
    if isinstance(value, dict):
        return {key: portable_manifest_value(item) for key, item in value.items()}
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',default='concise_learning_film_reproduction')
    parser.add_argument('--selection',default='artifacts/render/concise_selected_v4/complete.json')
    parser.add_argument('--heldout-video',default='artifacts/video/pilot_final_heldout_000.mp4')
    parser.add_argument('--flight-only',action='store_true',default=True,
                        help='Use the current flight-only edit with one lower-left data line (default).')
    parser.add_argument('--with-overlays-and-curve',dest='flight_only',action='store_false',
                        help='Rebuild the older explanatory-overlay and learning-curve edit.')
    args=parser.parse_args()
    output=project_output('artifacts/film/'+args.name)
    if output.exists(): raise FileExistsError('Preserve previous film versions')
    batch=video.read(ROOT/args.selection)
    if [r['updates'] for r in batch['clips']]!=list(TITLES):
        raise ValueError('The concise edit must retain all four distinct capability milestones')
    raws=[video.verify_video(ROOT/r['video']) for r in batch['clips']]
    if any(r['summary']['teacher_loaded'] or r['summary']['student']['updates']!=u
           for r,u in zip(raws,TITLES)): raise ValueError('Wrong visual policy identity')
    for raw in raws:
        if not raw['episode']['success'] and raw['episode']['post_terminal_seconds']<1.98:
            raise ValueError('Failure clips need genuine post-terminal physics, not a hold')
    # The approach milestone may only be called a collision if actual physics recorded contact.
    collision=video.read(ROOT/raws[0]['render']['source_evaluation']/'contact_response_audit.json')
    if not collision['wall_contact'] or not collision['wall_velocity_reversal']:
        raise ValueError('Collision caption needs measured wall contact AND velocity reversal')
    heldout=video.verify_video(ROOT/args.heldout_video)
    if heldout['render']['episode']!=0 or heldout['summary']['checkpoint_sha256']!=raws[-1]['summary']['checkpoint_sha256']:
        raise ValueError('Held-out case must be episode 0 of the same final model')
    output.mkdir(parents=True)
    editor=video.Editor(output)
    for raw,updates in zip(raws,TITLES):
        record=video.read(ROOT/('runs/film/pilot_learning_v1/update_%06d.json'%updates))
        validation=video.read(ROOT/record['evaluations']['validation']['directory']/'summary.json')
        if validation['checkpoint_sha256']!=raw['summary']['checkpoint_sha256']:
            raise ValueError('Caption validation is not from the displayed checkpoint')
        if args.flight_only:
            if raw['render'].get('flight_only') is not True:
                raise ValueError('Flight-only assembly requires annotation-free source renders')
            minimal='%02d   ·   %s 次更新   ·   验证 %d/100'%(
                list(TITLES).index(updates)+1,format(updates,','),round(100*validation['success_rate']))
            filters=[editor.text(minimal,24,666,20,'0x8ee7e6')]
        else:
            details='%s   ·   %s 次更新   ·   %s 个教师标签'%(record['phase'],format(updates,','),format(record['labels'],','))
            score='固定场景验证 %d/100   ·   同一场景、同一镜头、原速回放'%round(100*validation['success_rate'])
            filters=['drawbox=x=0:y=0:w=1280:h=207:color=0x102330:t=fill',
                     editor.text(TITLES[updates],40,24,32,'0x8ee7e6'),editor.text(details,40,76,22),
                     editor.text(score,40,116,22)]
            if not raw['episode']['success']:
                filters.append(editor.text('撞上门框后弹回' if updates==250 else '未能稳住，回撞墙面后弹回',40,163,22,'0xffd28a',
                                           'gte(t,%.6f)'%raw['episode']['seconds']))
            else:
                filters.append(editor.text('完成耗时 %.2f 秒；穿缝＋稳定恢复，才算成功'%raw['episode']['seconds'],40,163,22,'0xa1ecaa'))
        path=editor.encode('stage_%06d'%updates,['-i',str(ROOT/raw['path'])],','.join(filters),raw['frames'])
        editor.add(path,TITLES[updates],updates=updates,source_video=raw['path'],source_sha256=raw['sha256'],
            full_episode=True,end_hold_frames=0,retiming=False,scored_outcome=raw['outcome'],
            post_terminal_seconds=raw['episode'].get('post_terminal_seconds',0),
            validation_successes=round(100*validation['success_rate']))
    if args.flight_only:
        if heldout['render'].get('flight_only') is not True:
            raise ValueError('Flight-only assembly requires annotation-free held-out render')
        filters=[editor.text('留出场景   ·   最终策略   ·   验证 88/100',24,666,20,'0x8ee7e6')]
    else:
        filters=['drawbox=x=0:y=0:w=1280:h=207:color=0x102330:t=fill',
            editor.text('换一组狭缝位置与角度',40,24,32,'0x8ee7e6'),
            editor.text('同一最终策略   ·   预先指定的留出场景第 0 回合',40,76,22),
            editor.text('留出几何验证 88/100   ·   无教师介入',40,116,22)]
    path=editor.encode('heldout',['-i',str(ROOT/heldout['path'])],','.join(filters),heldout['frames'])
    editor.add(path,'预先指定的留出场景',source_video=heldout['path'],source_sha256=heldout['sha256'],full_episode=True,end_hold_frames=0)
    if not args.flight_only:
        curve=ROOT/'artifacts/learning/concise_curve_sc/learning_curve.png'
        path=editor.encode('learning_curve',['-loop','1','-framerate','25','-i',str(curve)],'format=yuv420p',75)
        editor.add(path,'完整学习曲线保留退步',source_figure=str(curve.relative_to(ROOT)),
                   source_sha256=video.sha(curve),source_data_sha256=video.sha(curve.with_suffix('.json')))
    listing=output/'concat.txt'
    listing.write_text(''.join("file '%s'\n"%(ROOT/p['path']) for p in editor.parts))
    film=output/'uav_visual_learning_concise.mp4'
    command=['/usr/bin/ffmpeg','-y','-hide_banner','-loglevel','error','-f','concat','-safe','0',
             '-i',str(listing),'-c','copy','-movflags','+faststart',str(film)]
    subprocess.run(command,env=video.ENV,check=True)
    count,duration=video.probe(film)
    if count!=sum(p['frames'] for p in editor.parts) or not 20<=duration<=30:
        raise ValueError('Unexpected concise-film duration or frame count')
    manifest=dict(video_sha256=video.sha(film),duration_seconds=duration,frames=count,fps=25,
        assembler_sha256=video.sha(Path(__file__)),editor_sha256=video.sha(ROOT/'scripts/assemble_learning_film.py'),
        font=dict(family='Noto Sans CJK SC',
                  path=Path(video.FONT).resolve().relative_to(ROOT).as_posix(),
                  sha256=video.sha(Path(video.FONT))),
        user_revision=('Show only gap-flight footage with one minimal lower-left data line and no ending card.'
                       if args.flight_only else
                       'Use the platform-free HCSP Iris visual with the same Plastic_ABS and plastic_red colors as iris_batVisualOnly.usd. Preserve collision physics, full flights, full learning curve, standard Simplified Chinese text, and no opening or closing explanation cards.'),
        all_selected_stages_retained=True,full_flight_clips=True,retiming=False,end_frame_holds=False,
        flight_only=args.flight_only,ending_removed=args.flight_only,
        evidence='Seed-11 pilot validation, not formal final-test results',parts=editor.parts,
        commands=portable_manifest_value(editor.commands+[command]))
    captions=[dict(file=p.name,text=p.read_text()) for p in sorted(output.glob('caption_*.txt'))]
    for caption in captions:
        validate_text(caption['text'],video.FONT)
    (output/'captions.json').write_text(json.dumps(captions,indent=2,ensure_ascii=False))
    manifest['captions_sha256']=video.sha(output/'captions.json')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print(json.dumps(dict(video=str(film.relative_to(ROOT)),duration_seconds=duration,frames=count),indent=2))


if __name__=='__main__': main()
