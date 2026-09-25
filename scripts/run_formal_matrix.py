"""Frozen teacher/student matrix, followed only then by all final test conditions."""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime,timezone
from uav_gap.runtime import ROOT,project_output
from uav_gap.formal_names import protocol_version,test_prefix as formal_test_prefix
from run_visual_rounds import protocol_sources,digest,read
from run_teacher_curriculum import source_hashes as teacher_sources

SEEDS = (11,22,33)
GROUPS = ('dagger_gru','bc_gru','dagger_stack4')
CONDITIONS = ('show','heldout','delay_40ms','depth_loss_3','mass_plus20','lateral_impulse')


def schedule(version='v1'):
    protocol_version(version)
    teacher,visual = 'teacher_formal_'+version,'visual_formal_'+version
    test_prefix = formal_test_prefix(version)
    stages = [dict(name='teachers',kind='teacher_training',directory='runs/experiments/'+teacher,
        command=['scripts/run_teacher_curriculum.py','--name',teacher,
                 '--config','configs/'+teacher+'.json'])]
    for group in GROUPS:
        for seed in SEEDS:
            name = '%s_seed%d_%s' % (visual,seed,group)
            stages.append(dict(name=name,kind='student_training',directory='runs/experiments/'+name,
                command=['scripts/run_visual_rounds.py','--config','configs/'+visual+'.json',
                         '--name',visual,'--group',group,'--seed',str(seed),
                         '--teacher','checkpoints/%s_seed%d_l3/final.pth' % (teacher,seed),
                         '--teacher-validation','runs/evaluation/%s_seed%d_l3_final_show/summary.json' % (teacher,seed)]))
    # No final-test episode runs before ALL training budgets have completed.
    for group in GROUPS:
        for seed in SEEDS:
            for condition in CONDITIONS:
                name = '%s_seed%d_%s_%s' % (test_prefix,seed,group,condition)
                checkpoint = 'checkpoints/%s_seed%d_%s_r9/final.pth' % (visual,seed,group)
                stages.append(dict(name=name,kind='student_test',checkpoint=checkpoint,condition=condition,
                    directory='runs/evaluation/'+name,command=['scripts/evaluate_student.py','--checkpoint',checkpoint,
                    '--level','3','--split','test','--condition',condition,'--count','100','--name',name,'--record']))
    for seed in SEEDS:
        for condition in ('show','heldout','mass_plus20','lateral_impulse'):
            name = '%s_seed%d_teacher_%s' % (test_prefix,seed,condition)
            checkpoint = 'checkpoints/%s_seed%d_l3/final.pth' % (teacher,seed)
            stages.append(dict(name=name,kind='teacher_test',checkpoint=checkpoint,condition=condition,
                directory='runs/evaluation/'+name,command=['scripts/evaluate_teacher.py','--checkpoint',checkpoint,
                '--level','3','--split','test','--condition',condition,'--count','100','--name',name,'--record']))
    return stages


def sources(version='v1'):
    result = dict(teacher_sources(),**protocol_sources())
    result['src/uav_gap/formal_names.py'] = digest(ROOT/'src/uav_gap/formal_names.py')
    result['scripts/run_formal_matrix.py'] = digest(ROOT/'scripts/run_formal_matrix.py')
    for name in ('teacher_formal_'+version+'.json','visual_formal_'+version+'.json','scenes/test.json'):
        result['configs/'+name] = digest(ROOT/'configs'/name)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--version',type=protocol_version,default='v1')
    args = parser.parse_args()
    stages = schedule(args.version)
    if args.dry_run:
        print(json.dumps(stages,indent=2))
        return
    request = dict(version=args.version,seeds=SEEDS,groups=GROUPS,stages=stages,source_hashes=sources(args.version))
    # Normalize JSON tuples so identical-request resume compares equal.
    request = json.loads(json.dumps(request))
    directory = project_output('runs/experiments/formal_matrix_'+args.version)
    if directory.exists():
        if not args.resume or read(directory/'request.json')!=request:
            raise ValueError('Formal matrix resume requires identical sources and frozen protocol')
    else:
        directory.mkdir(parents=True)
        (directory/'request.json').write_text(json.dumps(request,indent=2))
    (directory/'pid').write_text(str(os.getpid()))
    completed = []
    for stage in stages:
        if sources(args.version)!=request['source_hashes']:
            raise RuntimeError('Formal protocol changed during execution')
        target = ROOT/stage['directory']
        training = stage['kind'].endswith('_training')
        command = list(stage['command'])
        if training and target.exists():
            command.append('--resume')
        if training or not (target/'summary.json').exists():
            start,started = time.monotonic(),datetime.now(timezone.utc).isoformat()
            (directory/'active_stage.json').write_text(json.dumps(stage,indent=2))
            with (directory/(stage['name']+'.log')).open('a') as log:
                log.write(json.dumps(dict(command=command))+'\n')
                log.flush()
                result = subprocess.run([str(ROOT/'scripts/run.sh')]+command,cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT)
            with (directory/'stage_timings.jsonl').open('a') as ledger:
                ledger.write(json.dumps(dict(stage=stage['name'],started_utc=started,
                    elapsed_seconds=time.monotonic()-start,returncode=result.returncode,
                    accounting='Serial stage wall time including initialization',command=command))+'\n')
            result.check_returncode()
        if training:
            if not (target/'complete.json').exists():
                raise RuntimeError('Training subprocess did not complete its whole declared budget')
        else:
            summary = read(target/'summary.json')
            kind = 'visual_student_evaluation' if stage['kind']=='student_test' else 'state_teacher_evaluation'
            if not (summary['kind']==kind and summary['split']=='test' and summary['level']==3
                    and summary['condition']==stage['condition'] and summary['num_episodes']==100
                    and summary['checkpoint_sha256']==digest(ROOT/stage['checkpoint'])
                    and summary['manifest_sha256']==request['source_hashes']['configs/scenes/test.json']):
                raise ValueError('Final test summary provenance mismatch')
            if kind=='visual_student_evaluation' and summary['teacher_loaded'] is not False:
                raise ValueError('Visual testing must not load a teacher')
            if not (target/'trace.npz').exists():
                raise ValueError('Formal test needs the full recorded physics trace')
        completed.append(stage['name'])
        print(json.dumps(dict(completed=stage['name'],stages_done=len(completed),total_stages=len(stages))),flush=True)
    (directory/'complete.json').write_text(json.dumps(dict(request_sha256=digest(directory/'request.json'),
                                                         completed=completed),indent=2))


if __name__=='__main__':
    main()
