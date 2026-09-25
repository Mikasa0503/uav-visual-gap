"""Fixed-budget three-seed teacher curriculum with explicit validation gates."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from uav_gap.runtime import ROOT, project_output


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    names = ['task','assets','geometry','termination','control','runtime','reset_state','rl_adapter','inference','stress','recovery_reward','curriculum']
    paths = [ROOT/'src/uav_gap'/(name+'.py') for name in names]
    paths += [ROOT/'scripts'/name for name in ('train_teacher.py','evaluate_teacher.py','run_teacher_curriculum.py','run.sh')]
    paths += [ROOT/'assets/gap'/name for name in ('quad.urdf','quad_mass120.urdf','panel.urdf')]
    paths += [ROOT/'configs/teacher_ppo.yaml',ROOT/'configs/scenes/validation.json']
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def schedule(config, family):
    import re
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', family):
        raise ValueError('Invalid experiment family')
    phases = config.get('phases')
    if phases is None:
        epochs = config['epochs_per_level']
        if len(epochs)!=4:
            raise ValueError('Four curriculum levels required')
        phases = [dict(tag='l%d'%level,level=level,epochs=budget,perturb_reset=config['perturb_reset'],
                       promotion_check=True,exact_final=level==3) for level,budget in enumerate(epochs)]
    if len({p['tag'] for p in phases})!=len(phases):
        raise ValueError('Curriculum phase tags must be unique')
    for index,phase in enumerate(phases):
        if (not re.fullmatch(r'l[0-3](?:_[a-z]+)?',phase['tag'])
                or phase['level'] not in range(4) or type(phase['epochs']) is not int
                or phase['epochs']<30 or phase['epochs']%10):
            raise ValueError('Invalid curriculum phase')
        if index and (phase['level']-phases[index-1]['level'] not in (0,1)
                      or phase['level']>phases[index-1]['level'] and not phases[index-1]['promotion_check']):
            raise ValueError('A harder level requires the previous promotion gate')
        if phase['exact_final']!=(index==len(phases)-1):
            raise ValueError('Only the final phase receives exact-final checks')
    if phases[0]['level']!=0 or phases[-1]['level']!=3 or not phases[-1]['perturb_reset']:
        raise ValueError('Curriculum must start at level zero and finish at level three with reset v2')
    if config['seeds'] != [11,22,33] or config['perturb_reset'] is not True:
        raise ValueError('Formal protocol requires seeds 11/22/33 and reset v2')
    reward_version=config.get('recovery_reward','original')
    if reward_version not in ('original','dense_v2'):
        raise ValueError('Unknown declared teacher recovery reward')
    rows = []
    for seed in config['seeds']:
        previous, cumulative = None, 0
        for phase in phases:
            level,budget = phase['level'],phase['epochs']
            cumulative += budget
            name = '%s_seed%d_%s' % (family,seed,phase['tag'])
            checkpoint = 'checkpoints/'+name+'/final.pth'
            train = ['scripts/train_teacher.py','--name',name,'--seed',str(seed),
                     '--num-envs',str(config['num_envs']),'--level',str(level),
                     '--epochs',str(budget)]
            if 'recovery_reward' in config:
                train += ['--recovery-reward',reward_version]
            if phase['perturb_reset']:
                train.append('--perturb-reset')
            if previous:
                train += ['--checkpoint',previous]
            evaluations = []
            for epoch in (cumulative-20,cumulative-10,cumulative) if phase['promotion_check'] else ():
                evaluations.append(dict(checkpoint='checkpoints/%s/snapshot_%06d.pth' % (name,epoch),
                    name='%s_epoch%d_show' % (name,epoch),condition='show',role='promotion'))
            if phase['exact_final']:
                for condition in ('show','heldout'):
                    evaluations.append(dict(checkpoint=checkpoint,name=name+'_final_'+condition,
                                            condition=condition,role='exact_final'))
            rows.append(dict(seed=seed,level=level,name=name,previous=previous,train=train,
                             perturb_reset=phase['perturb_reset'],exact_final=phase['exact_final'],
                             checkpoint=checkpoint,epochs=cumulative,evaluations=evaluations))
            previous = checkpoint
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/teacher_formal_v1.json')
    parser.add_argument('--name',default='teacher_formal_v1')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args = parser.parse_args()
    path = (ROOT/args.config).resolve()
    path.relative_to(ROOT)
    config = read(path)
    rows = schedule(config,args.name)
    if args.dry_run:
        print(json.dumps(rows,indent=2))
        return
    directory = project_output('runs/experiments/'+args.name)
    request = dict(config=config,config_sha256=digest(path),rows=rows,
                   source_hashes=source_hashes(),
                   validation_sha256=digest(ROOT/'configs/scenes/validation.json'),
                   ppo_sha256=digest(ROOT/'configs/teacher_ppo.yaml'))
    if directory.exists():
        if not args.resume or read(directory/'request.json')!=request:
            raise ValueError('Resume requires the identical formal request')
    else:
        directory.mkdir(parents=True)
        (directory/'request.json').write_text(json.dumps(request,indent=2))
    (directory/'pid').write_text(str(os.getpid()))

    def execute(command,label):
        if source_hashes()!=request['source_hashes']:
            raise RuntimeError('Teacher protocol source changed during execution')
        start = time.monotonic()
        started = datetime.now(timezone.utc).isoformat()
        with (directory/(label+'.log')).open('a') as log:
            log.write(json.dumps(dict(command=command))+'\n')
            log.flush()
            result = subprocess.run([str(ROOT/'scripts/run.sh')]+command,cwd=ROOT,stdout=log,
                                    stderr=subprocess.STDOUT)
        with (directory/'stage_timings.jsonl').open('a') as ledger:
            ledger.write(json.dumps(dict(stage=label,started_utc=started,
                elapsed_seconds=time.monotonic()-start,returncode=result.returncode,
                accounting='GPU-stage wall time including initialization',command=command))+'\n')
        result.check_returncode()

    for row in rows:
        checkpoint = ROOT/row['checkpoint']
        if not checkpoint.exists():
            execute(row['train'],row['name']+'_train')
        task = read(checkpoint.parent/'task.json')
        initial = read(checkpoint.parent/'initial.json')
        if (task['seed']!=row['seed'] or task['level']!=row['level'] or task['perturb_reset']!=row['perturb_reset']
                or task['num_envs']!=config['num_envs'] or initial['resumed_from']!=row['previous']
                or task.get('recovery_reward','original')!=config.get('recovery_reward','original')):
            raise ValueError('Existing teacher training protocol mismatch')
        last = json.loads((checkpoint.parent/'progress.jsonl').read_text().splitlines()[-1])
        if last['epoch'] != row['epochs']:
            raise ValueError('Teacher did not complete the fixed epoch budget')
        reports = []
        for evaluation in row['evaluations']:
            summary_path = ROOT/'runs/evaluation'/evaluation['name']/'summary.json'
            if not summary_path.exists():
                command = ['scripts/evaluate_teacher.py','--checkpoint',evaluation['checkpoint'],
                    '--level',str(row['level']),'--split','validation','--condition',evaluation['condition'],
                    '--count',str(config['validation_count']),'--name',evaluation['name'],'--record']
                execute(command,evaluation['name'])
            report = read(summary_path)
            if not (report['kind']=='state_teacher_evaluation' and report['split']=='validation'
                    and report['condition']==evaluation['condition'] and report['level']==row['level']
                    and report['num_episodes']==config['validation_count']
                    and report['checkpoint_sha256']==digest(ROOT/evaluation['checkpoint'])
                    and report['manifest_sha256']==request['validation_sha256']):
                raise ValueError('Teacher evaluation provenance mismatch')
            reports.append(dict(role=evaluation['role'],summary=str(summary_path.relative_to(ROOT)),
                                condition=evaluation['condition'],success_rate=report['success_rate']))
        passed = all(item['success_rate']>=config['promotion_threshold'] for item in reports
                     if item['role']=='promotion')
        if row['exact_final']:
            passed = passed and all(item['success_rate']>=config['final_threshold'] for item in reports
                                   if item['role']=='exact_final' and item['condition']=='show')
        result = dict(seed=row['seed'],level=row['level'],epochs=row['epochs'],passed=passed,
                      checkpoint=row['checkpoint'],checkpoint_sha256=digest(checkpoint),evaluations=reports)
        (directory/(row['name']+'.json')).write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
        if not passed:
            raise RuntimeError('Curriculum gate failed; preserve the result and reassess the protocol')
    (directory/'complete.json').write_text(json.dumps(dict(seeds=config['seeds'],levels=4),indent=2))


if __name__=='__main__':
    main()
