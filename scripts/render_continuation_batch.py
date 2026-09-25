"""CPU replay of concise learning milestones, including real post-failure dynamics."""
import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from uav_gap.runtime import ROOT,project_output


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',default='concise_learning_v3')
    parser.add_argument('--name',default='concise_learning_v3')
    parser.add_argument('--flight-only',action='store_true')
    parser.add_argument('--updates',help='Optional comma-separated subset of captured update counts')
    args=parser.parse_args()
    recording=ROOT/'runs/film'/args.capture
    capture=read(recording/'complete.json')
    if not 1<=len(capture['records'])<=6: raise ValueError('Expected a bounded completed milestone selection')
    selected_updates=([int(value) for value in args.updates.split(',')] if args.updates else
                      [row['updates'] for row in capture['records']])
    records=[row for row in capture['records'] if row['updates'] in selected_updates]
    if len(records)!=len(selected_updates) or [row['updates'] for row in records]!=selected_updates:
        raise ValueError('Requested updates must appear once in capture order')
    sources={name:sha(ROOT/name) for name in ('scripts/render_trace.py','scripts/encode_replay.py',
                                             'scripts/render_continuation_batch.py')}
    settings=dict(capture_sha256=sha(recording/'complete.json'),sources=sources,samples=8,width=1280,
                  workers=4,threads=16,flight_only=args.flight_only,updates=selected_updates)
    base=project_output('artifacts/render/'+args.name);base.mkdir(parents=True,exist_ok=True)
    if (base/'request.json').exists() and read(base/'request.json')!=settings:
        raise ValueError('Resume requires identical render sources and settings')
    (base/'request.json').write_text(json.dumps(settings,indent=2))
    def render(row):
        if any(sha(ROOT/s)!=h for s,h in sources.items()): raise ValueError('Renderer changed mid-batch')
        updates=row['updates'];evaluation=ROOT/row['directory']
        old=read(ROOT/('runs/film/pilot_learning_v1/update_%06d.json'%updates))
        validation=ROOT/old['evaluations']['validation']['directory']/'summary.json'
        frames=base/('update_%06d'%updates);manifest_path=frames/'render_manifest.json'
        if not manifest_path.exists():
            command=[str(ROOT/'third_party/blender-4.2.21-linux-x64/blender'),'--background','--python',
                str(ROOT/'scripts/render_trace.py'),'--','--evaluation',str(evaluation.relative_to(ROOT)),
                '--output',str(frames.relative_to(ROOT)),'--all','--width','1280','--samples','8','--threads','16',
                '--phase',old['phase'],'--validation-summary',str(validation.relative_to(ROOT))]
            if args.flight_only: command.append('--flight-only')
            with (base/('update_%06d.log'%updates)).open('a') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        manifest=read(manifest_path)
        if manifest['trace_sha256']!=sha(evaluation/'trace.npz') or manifest['renderer_sha256']!=sources['scripts/render_trace.py']:
            raise ValueError('Continuation replay provenance mismatch')
        if manifest.get('flight_only',False)!=args.flight_only:
            raise ValueError('Replay annotation mode mismatch')
        video=project_output('artifacts/video/%s/update_%06d.mp4'%(args.name,updates))
        if not video.exists():
            subprocess.run([str(ROOT/'scripts/run.sh'),'scripts/encode_replay.py','--frames',str(frames.relative_to(ROOT)),
                '--output',str(video.relative_to(ROOT))],cwd=ROOT,check=True,capture_output=True,text=True)
        encoded=read(video.with_suffix('.json'))
        if encoded['render']!=manifest or encoded['video_sha256']!=sha(video):
            raise ValueError('Encoded continuation differs from physical replay')
        return dict(updates=updates,video=str(video.relative_to(ROOT)),sha256=sha(video),
                    frames=len(manifest['source_indices']),continuation=manifest['continuation'])
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(render,row) for row in records]):
            row=future.result();results.append(row);print(json.dumps(row),flush=True)
    (base/'complete.json').write_text(json.dumps(dict(settings=settings,clips=sorted(results,key=lambda r:r['updates'])),indent=2))


if __name__=='__main__': main()
