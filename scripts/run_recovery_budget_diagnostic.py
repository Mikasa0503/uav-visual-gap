"""One bounded development continuation after the full dense-reward diagnostic."""
import hashlib,json,os,subprocess,time
from pathlib import Path
from uav_gap.runtime import ROOT,project_output

name='teacher_recovery_budget_diagnostic_seed22_l0'
upstream=ROOT/'runs/experiments/teacher_recovery_dense_v2_diagnostic'
directory=project_output('runs/experiments/'+name)
checkpoint=ROOT/'checkpoints/teacher_formal_v2_seed22_l0/final.pth'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
# No PID signaling. Require the exact upstream coordinator if it is still live.
pid=int((upstream/'pid').read_text())
proc=Path('/proc')/str(pid)
identity=(proc/'stat').read_text().split()[21] if proc.exists() else None
if proc.exists() and b'scripts/run_recovery_diagnostic.py' not in (proc/'cmdline').read_bytes():
    raise RuntimeError('Upstream PID belongs to another command')
request=dict(role='development_validation_only',upstream=str(upstream.relative_to(ROOT)),upstream_pid=pid,
    seed=22,level=0,num_envs=512,additional_epochs=300,total_epochs=600,
    parent_checkpoint=str(checkpoint.relative_to(ROOT)),parent_sha256=sha(checkpoint),
    recovery_reward='original',evaluation_epochs=[580,590,600],count=100,
    rationale='Separate longer-training hypothesis from negative recovery-cost shaping. Continue the original failed teacher in a new diagnostic directory; never overwrite or extend the frozen formal run in place.',
    sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/uav_gap/task.py',ROOT/'src/uav_gap/recovery_reward.py',
        ROOT/'scripts/train_teacher.py',ROOT/'scripts/evaluate_teacher.py',ROOT/'configs/teacher_ppo.yaml']})
directory.mkdir(parents=True,exist_ok=False)
(directory/'request.json').write_text(json.dumps(request,indent=2));(directory/'pid').write_text(str(os.getpid()))
(directory/'state.json').write_text(json.dumps(dict(status='waiting_for_upstream',upstream_pid=pid)))
while proc.exists() and (proc/'stat').read_text().split()[21]==identity:
    # Completion marker is written only after every upstream subprocess has exited.
    if (upstream/'complete.json').exists():break
    time.sleep(10)
if not (upstream/'complete.json').exists():
    raise RuntimeError('Upstream terminated without all three diagnostic seed results')
upstream_result=json.loads((upstream/'complete.json').read_text())
if {r['seed'] for r in upstream_result['results']}!={11,22,33}:
    raise ValueError('Incomplete diagnostic seed coverage')
if upstream_result['all_passed']:
    raise RuntimeError('All dense-reward gates passed; reassess before a budget continuation')
if sha(checkpoint)!=request['parent_sha256']:
    raise ValueError('Original parent checkpoint changed')
for path,value in request['sources'].items():
    if sha(ROOT/path)!=value:raise ValueError('Diagnostic source changed while queued')

def run(args,label):
    start=time.time()
    (directory/'state.json').write_text(json.dumps(dict(status='running',stage=label)))
    with (directory/(label+'.log')).open('w') as log:
        result=subprocess.run([str(ROOT/'scripts/run.sh')]+args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    with (directory/'stage_timings.jsonl').open('a') as log:
        log.write(json.dumps(dict(label=label,elapsed_seconds=time.time()-start,returncode=result.returncode,command=args))+'\n')
    result.check_returncode()

run(['scripts/train_teacher.py','--name',name,'--seed','22','--num-envs','512','--level','0','--epochs','300',
     '--checkpoint',request['parent_checkpoint'],'--recovery-reward','original'],'train')
reports=[]
for epoch in request['evaluation_epochs']:
    label=name+'_epoch%d_show'%epoch
    run(['scripts/evaluate_teacher.py','--checkpoint','checkpoints/%s/snapshot_%06d.pth'%(name,epoch),
         '--level','0','--split','validation','--condition','show','--count','100','--name',label,'--record'],label)
    summary=json.loads((ROOT/'runs/evaluation'/label/'summary.json').read_text())
    reports.append(dict(epoch=epoch,success_rate=summary['success_rate'],crossing_rate=summary['crossing_rate'],failures=summary['failures'],timeouts=summary['timeouts']))
result=dict(seed=22,development_only=True,reports=reports,passed=all(r['success_rate']>=.85 for r in reports),
    upstream_complete_sha256=sha(upstream/'complete.json'),parent_checkpoint_sha256=request['parent_sha256'])
(directory/'complete.json').write_text(json.dumps(result,indent=2))
(directory/'state.json').write_text(json.dumps(dict(status='complete')))
print(json.dumps(result),flush=True)
