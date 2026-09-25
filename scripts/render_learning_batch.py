"""Bounded CPU rendering alongside the live, serial GPU recording job."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uav_gap.runtime import ROOT,project_output


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recording',required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--threads',type=int,default=16)
    parser.add_argument('--width',type=int,default=1280)
    parser.add_argument('--samples',type=int,default=12)
    args = parser.parse_args()
    if any(not re.fullmatch(r'[a-zA-Z0-9_-]+',s) for s in (args.name,args.recording)):
        raise ValueError('Invalid job names')
    if not 1<=args.workers<=4 or not 1<=args.threads<=16:
        raise ValueError('CPU rendering is bounded to four workers with at most sixteen threads each')
    recording = project_output('runs/film/'+args.recording)
    request = read(recording/'request.json')
    output = project_output('artifacts/render/'+args.name)
    output.mkdir(parents=True,exist_ok=True)
    renderer,encoder = ROOT/'scripts/render_trace.py',ROOT/'scripts/encode_replay.py'
    settings = dict(recording_request_sha256=digest(recording/'request.json'),
                    renderer_sha256=digest(renderer),encoder_sha256=digest(encoder),**vars(args))
    settings_path = output/'request.json'
    if settings_path.exists() and read(settings_path)!=settings:
        raise ValueError('Rendering resume requires identical sources and settings')
    settings_path.write_text(json.dumps(settings,indent=2))
    (output/'pid').write_text(str(os.getpid()))
    pending = {row['updates']:row for row in request['checkpoints']}
    done = []

    def render(row,record):
        if any(record.get(k)!=v for k,v in row.items()):
            raise ValueError('Completed recording differs from the frozen selection')
        if digest(renderer)!=settings['renderer_sha256'] or digest(encoder)!=settings['encoder_sha256']:
            raise ValueError('Renderer changed during the batch')
        showcase = ROOT/record['evaluations']['showcase']['directory']
        validation = ROOT/record['evaluations']['validation']['directory']/'summary.json'
        for path in (showcase,validation):
            path.resolve().relative_to(ROOT)
        frames = output/('update_%06d' % row['updates'])
        manifest_path = frames/'render_manifest.json'
        if not manifest_path.exists():
            command = [str(ROOT/'third_party/blender-4.2.21-linux-x64/blender'),'--background','--python',str(renderer),'--',
                '--evaluation',str(showcase.relative_to(ROOT)),'--output',str(frames.relative_to(ROOT)),
                '--all','--width',str(args.width),'--samples',str(args.samples),'--threads',str(args.threads),
                '--phase',row['phase'],'--validation-summary',str(validation.relative_to(ROOT))]
            with (output/('update_%06d.log' % row['updates'])).open('a') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        manifest = read(manifest_path)
        if not (manifest['trace_sha256']==digest(showcase/'trace.npz')
                and manifest['renderer_sha256']==settings['renderer_sha256']
                and manifest['validation_summary_sha256']==digest(validation)
                and manifest['resolution']==[args.width,int(args.width*9/16)]
                and manifest['cycles_samples']==args.samples and manifest['phase']==row['phase']):
            raise ValueError('Rendered replay provenance/settings mismatch')
        video = project_output('artifacts/video/%s/update_%06d.mp4' % (args.name,row['updates']))
        if not video.exists():
            subprocess.run([str(ROOT/'scripts/run.sh'),str(encoder),'--frames',str(frames.relative_to(ROOT)),
                '--output',str(video.relative_to(ROOT))],cwd=ROOT,check=True,capture_output=True,text=True)
        encoded = read(video.with_suffix('.json'))
        if encoded['render']!=manifest or encoded['video_sha256']!=digest(video):
            raise ValueError('Encoded video differs from its verified replay')
        return dict(updates=row['updates'],video=str(video.relative_to(ROOT)),frames=len(manifest['source_indices']),
                    video_sha256=encoded['video_sha256'])

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        while pending or futures:
            for future in list(futures):
                if future.done():
                    result = future.result()
                    done.append(result)
                    del futures[future]
                    print(json.dumps(result),flush=True)
            for update,row in list(pending.items()):
                if len(futures)>=args.workers:
                    break
                path = recording/('update_%06d.json' % update)
                if path.exists():
                    futures[pool.submit(render,row,read(path))] = update
                    del pending[update]
            if pending and (recording/'complete.json').exists():
                if any(not (recording/('update_%06d.json' % update)).exists() for update in pending):
                    raise RuntimeError('Completed recording is missing selected episode metadata')
            elif pending:
                pid = int((recording/'pid').read_text())
                cmdline = Path('/proc/%d/cmdline' % pid)
                try:
                    command = cmdline.read_bytes()
                except FileNotFoundError:
                    command = b''
                alive = b'record_learning_showcase.py' in command and args.recording.encode() in command
                if not alive and not (recording/'complete.json').exists():
                    raise RuntimeError('Recording job stopped before all selected episodes were available')
            if pending or futures:
                time.sleep(2)
    (output/'complete.json').write_text(json.dumps(dict(settings=settings,clips=sorted(done,key=lambda x:x['updates'])),indent=2))


if __name__=='__main__':
    main()
