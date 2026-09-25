"""Complete teacher sensor-condition coverage after the frozen main matrix."""
import argparse
import hashlib
import json
import subprocess
from uav_gap.runtime import ROOT,project_output
from uav_gap.formal_names import protocol_version,test_prefix


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--version',type=protocol_version,default='v1')
    args = parser.parse_args()
    declaration_path = ROOT/('configs/teacher_sensor_test_'+args.version+'.json')
    config = read(declaration_path)
    if config.get('test_prefix','formal_test')!=test_prefix(args.version):
        raise ValueError('Sensor declaration uses a different protocol namespace')
    stages = []
    for seed in config['seeds']:
        checkpoint = config['checkpoint_pattern'].format(seed=seed)
        for condition in config['conditions']:
            name = '%s_seed%d_teacher_%s' % (config.get('test_prefix','formal_test'),seed,condition)
            stages.append(dict(name=name,checkpoint=checkpoint,condition=condition,seed=seed,
                command=['scripts/evaluate_teacher_sensor_stress.py','--checkpoint',checkpoint,'--level','3',
                    '--split',config['split'],'--condition',condition,'--count',str(config['episodes_per_condition']),
                    '--name',name,'--record','--declaration',str(declaration_path.relative_to(ROOT))]))
    if args.dry_run:
        print(json.dumps(stages,indent=2))
        return
    base = ROOT/'runs/experiments'/config.get('matrix','formal_matrix_v1')
    completion = read(base/'complete.json')
    if completion['request_sha256']!=digest(base/'request.json'):
        raise ValueError('The full main matrix must finish before supplementary final tests')
    test_sha = digest(ROOT/'configs/scenes/test.json')
    if read(base/'request.json')['source_hashes']['configs/scenes/test.json']!=test_sha:
        raise ValueError('Frozen test scenarios changed')
    request = dict(stages=stages,declaration_sha256=digest(declaration_path),manifest_sha256=test_sha,
        sources={name:digest(ROOT/name) for name in ('scripts/evaluate_teacher_sensor_stress.py',
            'scripts/run_teacher_sensor_supplement.py','scripts/audit_teacher_sensor_trace.py',
            'src/uav_gap/teacher_sensors.py','src/uav_gap/formal_names.py')})
    directory = project_output('runs/experiments/teacher_sensor_test_'+args.version)
    if directory.exists():
        if not args.resume or read(directory/'request.json')!=request:
            raise ValueError('Supplement resume requires the identical declared protocol')
    else:
        directory.mkdir(parents=True)
        (directory/'request.json').write_text(json.dumps(request,indent=2))
    results = []
    for stage in stages:
        for name,value in request['sources'].items():
            if digest(ROOT/name)!=value:
                raise ValueError('Supplement source changed during execution')
        # Bind to the same final teacher evaluated by the main matrix.
        checkpoint_sha = digest(ROOT/stage['checkpoint'])
        base_summary = read(ROOT/'runs/evaluation'/('%s_seed%d_teacher_show' % (
            config.get('test_prefix','formal_test'),stage['seed']))/'summary.json')
        if base_summary['checkpoint_sha256']!=checkpoint_sha:
            raise ValueError('Teacher changed after the main formal test')
        target = ROOT/'runs/evaluation'/stage['name']
        if not (target/'summary.json').exists():
            with (directory/(stage['name']+'.log')).open('a') as log:
                subprocess.run([str(ROOT/'scripts/run.sh')]+stage['command'],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        summary = read(target/'summary.json')
        if not (summary['split']=='test' and summary['condition']==stage['condition']
                and summary['num_episodes']==config['episodes_per_condition'] and summary['level']==3
                and summary['checkpoint_sha256']==checkpoint_sha and summary['manifest_sha256']==test_sha
                and summary['sensor_declaration_sha256']==request['declaration_sha256']
                and (target/'trace.npz').exists()):
            raise ValueError('Teacher sensor-condition evidence does not match the declaration')
        expected = 'privileged_state_delay' if stage['condition']=='delay_40ms' else 'not_applicable_no_depth_input'
        if summary['stress_protocol']['applicability']!=expected:
            raise ValueError('Teacher sensor applicability was misreported')
        # A declared delay is insufficient: verify the input arrays actually used.
        subprocess.run([str(ROOT/'scripts/run.sh'),'scripts/audit_teacher_sensor_trace.py',
            '--evaluation',str(target.relative_to(ROOT))],cwd=ROOT,check=True)
        audit = read(target/'sensor_trace_audit.json')
        if not audit['exact'] or audit['episodes']!=config['episodes_per_condition']:
            raise ValueError('Executed teacher sensor trace did not pass its audit')
        results.append(dict(name=stage['name'],success_rate=summary['success_rate'],applicability=expected,
                            sensor_trace_audit_sha256=digest(target/'sensor_trace_audit.json')))
        print(json.dumps(results[-1]),flush=True)
    (directory/'complete.json').write_text(json.dumps(dict(results=results,
        request_sha256=digest(directory/'request.json')),indent=2))


if __name__=='__main__':
    main()
