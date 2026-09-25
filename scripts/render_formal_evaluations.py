"""Render the first frozen case for every formal group/seed/condition, CPU only."""
import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from uav_gap.runtime import ROOT,project_output
from uav_gap.formal_names import test_prefix


def read(path): return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def selection(config):
    prefix=test_prefix(config['version'])
    return [dict(name='%s_seed%d_%s_%s'%(prefix,seed,group,condition),
                 seed=seed,group=group,condition=condition,episode=config['episode_index'])
            for group in config['groups'] for seed in config['seeds'] for condition in config['conditions']]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/formal_video_v2.json')
    parser.add_argument('--name',default='formal_uncut_v2')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    config_path=(ROOT/args.config).resolve();config_path.relative_to(ROOT)
    config=read(config_path);clips=selection(config)
    if len(clips)!=config['expected_clips'] or len({r['name'] for r in clips})!=len(clips):
        raise ValueError('Video declaration must contain each case exactly once')
    if config['episode_index']!=0 or not 1<=config['workers']<=4 or not 1<=config['threads_per_worker']<=16:
        raise ValueError('Invalid case selection/CPU budget')
    if args.dry_run:
        print(json.dumps(clips,indent=2));return
    for name in (config['matrix'],config['supplement']):
        experiment=ROOT/'runs/experiments'/name
        if read(experiment/'complete.json')['request_sha256']!=sha(experiment/'request.json'):
            raise ValueError('Formal evidence must finish before final video batch')
    sources={s:sha(ROOT/s) for s in ('scripts/render_trace.py','scripts/encode_replay.py',
                                    'scripts/render_formal_evaluations.py','src/uav_gap/formal_names.py',str(config_path.relative_to(ROOT)))}
    request=dict(clips=clips,config=config,source_hashes=sources)
    base=project_output('artifacts/render/'+args.name);base.mkdir(parents=True,exist_ok=True)
    if (base/'request.json').exists() and read(base/'request.json')!=request:
        raise ValueError('Resume requires unchanged render selection/sources/settings')
    (base/'request.json').write_text(json.dumps(request,indent=2))
    (base/'pid').write_text(str(os.getpid()))
    videos=project_output('artifacts/video/'+args.name);videos.mkdir(parents=True,exist_ok=True)
    test_hash=sha(ROOT/'configs/scenes/test.json')
    main_request=read(ROOT/'runs/experiments'/config['matrix']/'request.json')
    if test_hash!=main_request['source_hashes']['configs/scenes/test.json']:
        raise ValueError('Formal test manifest changed')
    completed={}

    def render(clip):
        if any(sha(ROOT/s)!=h for s,h in sources.items()): raise ValueError('Renderer source changed mid-batch')
        evaluation=ROOT/'runs/evaluation'/clip['name'];summary=read(evaluation/'summary.json')
        if not (summary['split']=='test' and summary['num_episodes']==100 and summary['level']==3
                and summary['manifest_sha256']==test_hash and summary['condition']==clip['condition']):
            raise ValueError('Unexpected evaluation identity')
        frame_dir=base/clip['name'];frame_manifest=frame_dir/'render_manifest.json'
        if not frame_manifest.exists():
            command=[str(ROOT/'third_party/blender-4.2.21-linux-x64/blender'),'--background','--python',
                str(ROOT/'scripts/render_trace.py'),'--','--evaluation',str(evaluation.relative_to(ROOT)),
                '--output',str(frame_dir.relative_to(ROOT)),'--episode',str(clip['episode']),'--all',
                '--width',str(config['width']),'--samples',str(config['samples']),
                '--threads',str(config['threads_per_worker'])]
            with (base/(clip['name']+'.log')).open('a') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        manifest=read(frame_manifest)
        if not (manifest['episode']==clip['episode'] and manifest['trace_sha256']==sha(evaluation/'trace.npz')
                and manifest['renderer_sha256']==sources['scripts/render_trace.py']
                and manifest['evaluation_summary_sha256']==sha(evaluation/'summary.json')
                and manifest['resolution']==[config['width'],config['width']*9//16]
                and manifest['cycles_samples']==config['samples']):
            raise ValueError('Rendered episode differs from declaration/source')
        video=videos/(clip['name']+'.mp4')
        if not video.exists():
            subprocess.run([str(ROOT/'scripts/run.sh'),'scripts/encode_replay.py','--frames',
                str(frame_dir.relative_to(ROOT)),'--output',str(video.relative_to(ROOT))],cwd=ROOT,
                check=True,capture_output=True,text=True)
        encoded=read(video.with_suffix('.json'))
        if encoded['render']!=manifest or encoded['video_sha256']!=sha(video):
            raise ValueError('Encoded replay provenance mismatch')
        episode=[json.loads(line) for line in (evaluation/'episodes.jsonl').read_text().splitlines()][clip['episode']]
        result=next(k for k in ('success','failure','timeout') if episode[k])
        return dict(**clip,episode_outcome=result,batch_success_rate=summary['success_rate'],
            path=str(video.relative_to(ROOT)),sha256=sha(video),frames=len(manifest['source_indices']),
            duration_seconds=len(manifest['source_indices'])/manifest['output_fps'],
            visual_robustness_applicable=not(clip['group']=='teacher' and clip['condition']=='depth_loss_3'))

    with ThreadPoolExecutor(max_workers=config['workers']) as pool:
        futures={pool.submit(render,c):c for c in clips}
        for future in as_completed(futures):
            row=future.result();completed[row['name']]=row
            print(json.dumps(dict(completed=row['name'],count=len(completed),total=len(clips))),flush=True)
    rows=[completed[c['name']] for c in clips]
    (base/'complete.json').write_text(json.dumps(dict(request_sha256=sha(base/'request.json'),clips=rows),indent=2))
    lines=['# Uncut formal evaluation replays','',
           'Episode 0 from every frozen group/seed/condition, selected before final tests and retained regardless of outcome.',
           'Each clip plays the complete regularly sampled physics trace at original speed. No hold, excerpt or retiming.',
           'The displayed batch score uses all 100 cases; this one episode is not an estimate of overall performance.',
           'Teacher depth-loss cases have no camera input and are N/A for visual robustness.','',
           '| Group | Seed | Condition | Episode outcome | Batch success /100 | Full replay |',
           '| --- | --- | --- | --- | --- | --- |']
    for row in rows:
        condition=row['condition']+(' (N/A visual; control)' if not row['visual_robustness_applicable'] else '')
        lines.append('| %s | %d | %s | %s | %d | [%0.2fs](%s.mp4) |'%(row['group'],row['seed'],condition,
            row['episode_outcome'],round(100*row['batch_success_rate']),row['duration_seconds'],row['name']))
    (videos/'index.md').write_text('\n'.join(lines)+'\n')
    (videos/'manifest.json').write_text(json.dumps(dict(request=request,clips=rows),indent=2))


if __name__=='__main__': main()
