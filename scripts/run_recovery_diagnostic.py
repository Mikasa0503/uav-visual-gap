"""Fixed-budget development runs; never masquerade as formal matrix results."""
import hashlib,json,os,subprocess,time
from pathlib import Path
from uav_gap.runtime import ROOT,project_output

name='teacher_recovery_dense_v2_diagnostic'
out=project_output('runs/experiments/'+name)
out.mkdir(parents=True,exist_ok=False)
request=dict(role='development_validation_only',seeds=[22,33,11],epochs=300,level=0,num_envs=512,
    recovery_reward='dense_v2',perturb_reset=False,validation_count=100,
    evaluation_epochs=[280,290,300],promotion_threshold=.85,
    change='Post-crossing bounded independent distance, speed and tilt costs. Same physics, termination, PPO and reset protocol.',
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT/'src/uav_gap/task.py',ROOT/'src/uav_gap/recovery_reward.py',ROOT/'scripts/train_teacher.py',ROOT/'configs/teacher_ppo.yaml']})
(out/'request.json').write_text(json.dumps(request,indent=2));(out/'pid').write_text(str(os.getpid()))
rows=[]

def run(args,label):
    started=time.time()
    with (out/(label+'.log')).open('w') as log:
        result=subprocess.run([str(ROOT/'scripts/run.sh')]+args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    with (out/'stage_timings.jsonl').open('a') as log:
        log.write(json.dumps(dict(label=label,elapsed_seconds=time.time()-started,returncode=result.returncode,command=args))+'\n')
    result.check_returncode()

for seed in request['seeds']:
    family=name+'_seed%d_l0'%seed
    run(['scripts/train_teacher.py','--name',family,'--seed',str(seed),'--num-envs','512',
        '--level','0','--epochs','300','--recovery-reward','dense_v2'],family+'_train')
    reports=[]
    for epoch in request['evaluation_epochs']:
        label=family+'_epoch%d_show'%epoch
        run(['scripts/evaluate_teacher.py','--checkpoint','checkpoints/%s/snapshot_%06d.pth'%(family,epoch),
            '--level','0','--split','validation','--condition','show','--count','100','--name',label,'--record'],label)
        summary=json.loads((ROOT/'runs/evaluation'/label/'summary.json').read_text())
        reports.append(dict(epoch=epoch,success_rate=summary['success_rate'],crossing_rate=summary['crossing_rate'],failures=summary['failures'],timeouts=summary['timeouts']))
    row=dict(seed=seed,reports=reports,passed=all(r['success_rate']>=.85 for r in reports))
    rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(row),flush=True)
# Complete the whole predeclared seed set even if one fails; no seed filtering.
(out/'complete.json').write_text(json.dumps(dict(results=rows,all_passed=all(r['passed'] for r in rows)),indent=2))
