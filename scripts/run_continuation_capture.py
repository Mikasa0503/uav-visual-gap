"""Borrow an idle GPU between owned curriculum stages, then always resume it.

Only the known orchestration process is temporarily stopped. Its current child
finishes normally. No trainer or other GPU user's process is signaled.
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from uav_gap.runtime import ROOT,project_output


def identity(pid):
    path=Path('/proc')/str(pid)
    stat=(path/'stat').read_text().split(') ',1)[1].split()
    return (path.stat().st_uid,stat[19],(path/'cmdline').read_bytes())


def children(pid):
    found=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        try:
            fields=(path/'stat').read_text().split(') ',1)[1].split()
            if int(fields[1])==pid and fields[0]!='Z': found.append(int(path.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError): pass
    return found


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--curriculum-pid',required=True,type=int)
    parser.add_argument('--name',default='concise_learning_v3')
    parser.add_argument('--updates',default='0,1000,3000,4000,5500,12000')
    args=parser.parse_args()
    updates=[int(value) for value in args.updates.split(',')]
    if not updates or updates!=sorted(set(updates)): raise ValueError('Expected unique chronological checkpoint updates')
    original=identity(args.curriculum_pid)
    if original[0]!=os.getuid() or b'scripts/run_teacher_curriculum.py\0' not in original[2] or b'teacher_formal_v2\0' not in original[2]:
        raise ValueError('Only the owned, known formal-v2 curriculum coordinator may be paused')
    directory=project_output('runs/film/'+args.name);directory.mkdir(parents=True,exist_ok=False)
    request=dict(updates=updates,tail_seconds=2.,curriculum_pid=args.curriculum_pid,
        curriculum_start_ticks=original[1],selection='Explicit existing-checkpoint selection for physically verified concise demonstration',
        user_revision='Shorter film with clear progression and actual physics after collision failure',
        sources={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in
            ('scripts/record_collision_continuation.py','scripts/run_continuation_capture.py')})
    (directory/'request.json').write_text(json.dumps(request,indent=2))
    state=dict(queue_pid=os.getpid(),curriculum_pid=args.curriculum_pid,status='starting',records=[])
    def save(): (directory/'queue_state.json').write_text(json.dumps(state,indent=2))
    def terminate(signum,frame): raise SystemExit('Capture interrupted by signal %d'%signum)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    save();paused=False
    try:
        if identity(args.curriculum_pid)!=original: raise RuntimeError('Coordinator identity changed')
        os.kill(args.curriculum_pid,signal.SIGSTOP);paused=True
        state.update(status='waiting_for_current_child_to_finish',paused_unix=time.time());save()
        deadline=time.monotonic()+1200
        while True:
            gpu=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            active_children=children(args.curriculum_pid)
            if not active_children and not gpu: break
            if time.monotonic()>deadline: raise TimeoutError('No idle stage boundary within twenty minutes')
            state.update(active_children=active_children,gpu_pids=gpu);save();time.sleep(5)
        state['status']='recording';save()
        for updates in request['updates']:
            for source,expected in request['sources'].items():
                if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=expected:
                    raise ValueError('Capture source changed mid-queue')
            name='%s_u%06d'%(args.name,updates)
            command=[str(ROOT/'scripts/run.sh'),'scripts/record_collision_continuation.py','--record',
                'runs/film/pilot_learning_v1/update_%06d.json'%updates,'--name',name,'--tail-seconds','2']
            with (directory/(name+'.log')).open('w') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            audit=json.loads((ROOT/'runs/evaluation'/name/'continuation_audit.json').read_text())
            state['records'].append(dict(updates=updates,directory='runs/evaluation/'+name,audit=audit));save()
            print(json.dumps(dict(completed=name,audit=audit)),flush=True)
        state['status']='capture_complete';save()
        (directory/'complete.json').write_text(json.dumps(state,indent=2))
    finally:
        if paused:
            try:
                if identity(args.curriculum_pid)==original:
                    os.kill(args.curriculum_pid,signal.SIGCONT)
                    state['curriculum_resumed']=True
                else: state['curriculum_resumed']=False
            except FileNotFoundError: state['curriculum_resumed']=False
            state['finished_unix']=time.time();save()


if __name__=='__main__': main()
