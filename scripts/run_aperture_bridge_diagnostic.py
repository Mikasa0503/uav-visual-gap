"""Fixed-budget development-only test of gradual L1-to-L2 transfer."""
import hashlib
import json
import os
import subprocess
import time
from uav_gap.runtime import ROOT, project_output, check_gpu_available


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    check_gpu_available()
    name='teacher_aperture_bridge_diagnostic_seed11'
    directory=project_output('runs/experiments/'+name)
    parent=ROOT/'checkpoints/teacher_formal_v3_seed11_l1/final.pth'
    if sha(parent)!='7ea4dfa6c3d029517cb4af313466d6ce52e86787ede2fa11f6b510bce723176e':
        raise ValueError('Unexpected diagnostic parent')
    source_paths=list((ROOT/'src/uav_gap').glob('*.py'))+[ROOT/'scripts'/f for f in
        ['train_teacher.py','evaluate_teacher.py','run_aperture_bridge_diagnostic.py','run.sh']]
    source_paths += [ROOT/'configs/teacher_ppo.yaml',ROOT/'configs/scenes/validation.json']
    source_paths += list((ROOT/'assets/gap').glob('*.urdf'))
    phases=[dict(tag='bridgeone',fraction=1/3,epochs=200,total=1250),
            dict(tag='bridgetwo',fraction=2/3,epochs=200,total=1450),
            dict(tag='full',fraction=1.,epochs=400,total=1850)]
    request=dict(role='development_validation_only',seed=11,level=2,num_envs=512,
        parent_checkpoint=str(parent.relative_to(ROOT)),parent_sha256=sha(parent),
        recovery_reward='original',perturb_reset=False,phases=phases,
        validation_epochs=[1830,1840,1850],count=100,threshold=.85,
        rationale='Test gradual geometry transfer from the passed V3 L1 checkpoint after V3 L2 failed0/100 at all three snapshots; no test scenes or automatic budget extensions.',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in source_paths})
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'request.json').write_text(json.dumps(request,indent=2))
    (directory/'pid').write_text(str(os.getpid()))

    def execute(args,label):
        for path,value in request['source_hashes'].items():
            if sha(ROOT/path)!=value:raise ValueError('Diagnostic source changed: '+path)
        (directory/'state.json').write_text(json.dumps(dict(status='running',stage=label)))
        start=time.monotonic()
        with (directory/(label+'.log')).open('w') as stream:
            result=subprocess.run([str(ROOT/'scripts/run.sh')]+args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        with (directory/'stage_timings.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(label=label,elapsed_seconds=time.monotonic()-start,
                returncode=result.returncode,command=args))+'\n')
        result.check_returncode()

    previous=request['parent_checkpoint']
    for phase in phases:
        stage=name+'_'+phase['tag']
        execute(['scripts/train_teacher.py','--name',stage,'--seed','11','--num-envs','512',
            '--level','2','--epochs',str(phase['epochs']),'--curriculum-fraction',str(phase['fraction']),
            '--checkpoint',previous,'--recovery-reward','original'],stage)
        checkpoint=ROOT/'checkpoints'/stage
        task=json.loads((checkpoint/'task.json').read_text())
        progress=json.loads((checkpoint/'progress.jsonl').read_text().splitlines()[-1])
        if progress['epoch']!=phase['total'] or task['curriculum_fraction']!=phase['fraction']:
            raise ValueError('Diagnostic phase mismatch')
        previous=str((checkpoint/'final.pth').relative_to(ROOT))
    reports=[]
    for epoch in request['validation_epochs']:
        label=name+'_epoch%d_show'%epoch
        checkpoint='checkpoints/%s_full/snapshot_%06d.pth'%(name,epoch)
        execute(['scripts/evaluate_teacher.py','--checkpoint',checkpoint,'--level','2',
            '--split','validation','--condition','show','--count','100','--name',label,'--record'],label)
        summary_path=ROOT/'runs/evaluation'/label/'summary.json'
        summary=json.loads(summary_path.read_text())
        if summary['checkpoint_sha256']!=sha(ROOT/checkpoint) or summary['num_episodes']!=100:
            raise ValueError('Evaluation identity mismatch')
        reports.append(dict(epoch=epoch,summary=str(summary_path.relative_to(ROOT)),
            summary_sha256=sha(summary_path),**{k:summary[k] for k in
            ['success_rate','crossing_rate','failures','timeouts']}))
    result=dict(development_only=True,request_sha256=sha(directory/'request.json'),
        reports=reports,passed=all(r['success_rate']>=.85 for r in reports),
        final_checkpoint=previous,final_checkpoint_sha256=sha(ROOT/previous))
    (directory/'complete.json').write_text(json.dumps(result,indent=2))
    (directory/'state.json').write_text(json.dumps(dict(status='complete')))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
