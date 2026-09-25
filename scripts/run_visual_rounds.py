"""Serial BC/DAgger rounds with budget checks and completed-stage resume."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime,timezone
from uav_gap.runtime import ROOT, project_output
from uav_gap.experiment import GROUPS, build_rounds


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def protocol_sources():
    modules = ('assets','control','dataset','experiment','geometry','inference','reset_state',
               'runtime','sensors','stress','student','task','termination')
    paths = [ROOT/'src/uav_gap'/(name+'.py') for name in modules]
    paths += [ROOT/'scripts'/name for name in ('collect_demonstrations.py','train_student.py',
                                              'evaluate_student.py','run_visual_rounds.py','run.sh')]
    paths += [ROOT/'assets/gap'/name for name in ('quad.urdf','quad_mass120.urdf','panel.urdf')]
    return {str(path.relative_to(ROOT)):digest(path) for path in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/visual_pilot.json')
    parser.add_argument('--name', required=True, help='Shared experiment family, not the group suffix')
    parser.add_argument('--group', choices=GROUPS, required=True)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--teacher-validation', required=True, help='Exact-checkpoint fixed-scene summary.json')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config_path, teacher, validation = [(ROOT/value).resolve() for value in
                                       (args.config,args.teacher,args.teacher_validation)]
    for path in (config_path,teacher,validation):
        path.relative_to(ROOT)
    config = read(config_path)
    rounds = build_rounds(config,args.name,args.group,args.seed,args.teacher)
    if args.dry_run:
        print(json.dumps(rounds,indent=2))
        return
    teacher_sha = digest(teacher)
    report = read(validation)
    if not (report['kind']=='state_teacher_evaluation' and report['level']==3
            and report['split']=='validation' and report['condition']=='show'
            and report['checkpoint_sha256']==teacher_sha and report['num_episodes']>=100
            and report['success_rate']>=.95):
        raise ValueError('Teacher must pass exact-checkpoint final-gap validation before collection')
    teacher_task = read(teacher.parent/'task.json')
    if bool(teacher_task.get('perturb_reset',False)) != config['perturb_reset']:
        raise ValueError('Teacher and collection must use the same reset protocol')
    if config.get('initial_checkpoint'):
        initial_path = (ROOT/config['initial_checkpoint']).resolve()
        initial_path.relative_to(ROOT)
        initial = read(initial_path.with_suffix('.json'))
        initial_config = read(initial_path.parent/'configuration.json')
        expected_kind = 'stack4' if args.group=='dagger_stack4' else 'gru'
        if (initial['updates'] != config['initial_updates'] or initial['unique_labels'] != config['initial_unique_labels']
                or initial_config['seed'] != args.seed or initial_config['kind'] != expected_kind):
            raise ValueError('Continuation checkpoint budget/seed/architecture mismatch')
        if set(initial_config['datasets_sha256']) != set(config['initial_datasets']):
            raise ValueError('Continuation must preserve the full previous aggregate')
        for data_path in config['initial_datasets']:
            manifest_path = (ROOT/data_path/'manifest.json').resolve()
            manifest_path.relative_to(ROOT)
            data = read(manifest_path)
            if (digest(manifest_path) != initial_config['datasets_sha256'][data_path]
                    or data['teacher_sha256'] != teacher_sha or not data['complete']):
                raise ValueError('Continuation dataset provenance mismatch')
            if args.group=='bc_gru' and data['controller'] != 'teacher_bc':
                raise ValueError('BC cannot inherit student-controlled DAgger datasets')
    directory = project_output('runs/experiments/%s_seed%d_%s' % (args.name,args.seed,args.group))
    request = dict(config=config, config_sha256=digest(config_path), name=args.name,group=args.group,
                   seed=args.seed,teacher=args.teacher,teacher_sha256=teacher_sha,
                   teacher_validation=str(validation.relative_to(ROOT)), rounds=rounds)
    if config.get('freeze_sources'):
        request['source_hashes'] = protocol_sources()
    if directory.exists():
        if not args.resume or read(directory/'request.json') != request:
            raise ValueError('Existing experiment requires --resume with the identical request')
    else:
        directory.mkdir(parents=True)
        (directory/'request.json').write_text(json.dumps(request,indent=2))
    (directory/'pid').write_text(str(os.getpid()))

    def execute(command,label):
        if 'source_hashes' in request and protocol_sources()!=request['source_hashes']:
            raise RuntimeError('Formal visual protocol source changed during execution')
        start,started = time.monotonic(),datetime.now(timezone.utc).isoformat()
        with (directory/(label+'.log')).open('a') as log:
            log.write(json.dumps(dict(command=command))+'\n')
            log.flush()
            result = subprocess.run([str(ROOT/'scripts/run.sh')]+command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        with (directory/'stage_timings.jsonl').open('a') as ledger:
            ledger.write(json.dumps(dict(stage=label,started_utc=started,elapsed_seconds=time.monotonic()-start,
                returncode=result.returncode,accounting='GPU-stage wall time including initialization',command=command))+'\n')
        result.check_returncode()

    for row in rounds:
        index = row['index']
        manifest_path = ROOT/'datasets'/row['data_name']/'manifest.json'
        if not manifest_path.exists():
            execute(row['collect'],'round%d_collect' % index)
        data = read(manifest_path)
        wanted_student = row['collect'][row['collect'].index('--student')+1] if '--student' in row['collect'] else None
        wanted_seed = int(row['collect'][row['collect'].index('--seed')+1])
        wanted_labels = int(row['collect'][row['collect'].index('--label-budget')+1])
        if not (data['complete'] and data['labels']==data['teacher_queries']==wanted_labels
                and data['teacher_sha256']==teacher_sha and data['student']==wanted_student
                and data['seed']==wanted_seed and data['level']==3 and data['split']=='train'
                and data['task']['perturb_reset']==config['perturb_reset']
                and data['observation_delay_steps']==config['observation_delay_steps']):
            raise ValueError('Existing data does not match this round protocol')
        if wanted_student and data.get('student_sha256') and data['student_sha256'] != digest(ROOT/wanted_student):
            raise ValueError('Collection student checkpoint hash mismatch')
        checkpoint = ROOT/row['checkpoint']
        if not checkpoint.exists():
            execute(row['train'],'round%d_train' % index)
        final = read(checkpoint.with_suffix('.json'))
        if final['updates'] != row['cumulative_updates'] or final['unique_labels'] != row['cumulative_unique_labels']:
            raise ValueError('Checkpoint violates the matched experiment budget')
        trained = read(checkpoint.parent/'configuration.json')
        if trained['seed'] != args.seed or trained['kind'] != ('stack4' if args.group=='dagger_stack4' else 'gru'):
            raise ValueError('Checkpoint group/seed mismatch')
        if '--learning-rate' in row['train']:
            expected_rate = float(row['train'][row['train'].index('--learning-rate')+1])
            if trained.get('effective_learning_rate') != expected_rate:
                raise ValueError('Checkpoint learning-rate schedule mismatch')
        for path in row['datasets']:
            if trained['datasets_sha256'][path] != digest(ROOT/path/'manifest.json'):
                raise ValueError('Checkpoint dataset provenance mismatch')
        summaries = []
        for command in row['evaluate']:
            evaluation_name = command[command.index('--name')+1]
            summary_path = ROOT/'runs/evaluation'/evaluation_name/'summary.json'
            if not summary_path.exists():
                execute(command,evaluation_name)
            summary = read(summary_path)
            if not (summary['kind']=='visual_student_evaluation' and summary['teacher_loaded'] is False
                    and summary['checkpoint_sha256']==digest(checkpoint)
                    and summary['num_episodes']==config['validation_count']):
                raise ValueError('Evaluation provenance mismatch')
            summaries.append(summary)
        result = dict(round=index,labels=row['cumulative_unique_labels'],updates=row['cumulative_updates'],
                      evaluations=summaries)
        (directory/('round%d.json' % index)).write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(round=index,labels=result['labels'],updates=result['updates'],
                              success={x['condition']:x['success_rate'] for x in summaries})),flush=True)
    (directory/'complete.json').write_text(json.dumps(dict(rounds=len(rounds),teacher_sha256=teacher_sha),indent=2))


if __name__=='__main__':
    main()
