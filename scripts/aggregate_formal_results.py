"""Verify and summarize all 72 frozen final evaluations; CPU only, no tuning.

--inventory is safe during training and reports availability without metrics.
Full reports require both the main matrix and teacher supplement to be complete.
"""
import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from uav_gap.runtime import ROOT,project_output
from uav_gap.formal_names import protocol_version,test_prefix
from uav_gap.results import episode_metrics,verify_summary,seed_statistics,paired_outcomes

SEEDS=(11,22,33)
GROUPS=('teacher','bc_gru','dagger_gru','dagger_stack4')
CONDITIONS=('show','heldout','delay_40ms','depth_loss_3','mass_plus20','lateral_impulse')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def require_complete(directory):
    request=directory/'request.json'
    complete=read(directory/'complete.json')
    if complete['request_sha256']!=sha(request):
        raise ValueError('Completion does not match request: '+str(directory))
    return read(request),complete


def stage_name(version,seed,group,condition):
    prefix=test_prefix(version)
    return '%s_seed%d_%s_%s'%(prefix,seed,group,condition)


def inventory(version):
    rows=[]
    for group in GROUPS:
        for seed in SEEDS:
            for condition in CONDITIONS:
                name=stage_name(version,seed,group,condition)
                directory=ROOT/'runs/evaluation'/name
                rows.append(dict(name=name,available=all((directory/p).exists() for p in
                    ('summary.json','episodes.jsonl','scenarios.json','trace.npz'))))
    return dict(expected_evaluations=72,available_evaluations=sum(r['available'] for r in rows),
                main_complete=(ROOT/('runs/experiments/formal_matrix_'+version+'/complete.json')).exists(),
                supplement_complete=(ROOT/('runs/experiments/teacher_sensor_test_'+version+'/complete.json')).exists(),
                metric_inspection=False,evaluations=rows)


def mean_sd(stats,scale=1):
    if stats['mean'] is None: return 'undefined'
    value='%.2f'%(scale*stats['mean'])
    value+=' +/- %.2f'%(scale*stats['sample_sd']) if stats['sample_sd'] is not None else ' (SD undefined)'
    if stats['defined_seeds']!=stats['total_seeds']:
        value+=' [%d/%d defined seeds]'%(stats['defined_seeds'],stats['total_seeds'])
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--version',type=protocol_version,default='v2')
    parser.add_argument('--inventory',action='store_true')
    parser.add_argument('--output',default=None)
    args=parser.parse_args()
    coverage=inventory(args.version)
    if args.inventory:
        print(json.dumps(coverage,indent=2))
        return
    base=ROOT/('runs/experiments/formal_matrix_'+args.version)
    supplement=ROOT/('runs/experiments/teacher_sensor_test_'+args.version)
    request,complete=require_complete(base)
    sensor_request,sensor_complete=require_complete(supplement)
    if complete['completed']!=[s['name'] for s in request['stages']]:
        raise ValueError('Main completion omits or reorders declared stages')
    if [s['name'] for s in sensor_complete['results']]!=[s['name'] for s in sensor_request['stages']]:
        raise ValueError('Supplement completion omits declared stages')
    if coverage['available_evaluations']!=72:
        raise ValueError('All 72 evaluations with full evidence are required')
    manifest=ROOT/'configs/scenes/test.json'
    test_hash=sha(manifest)
    if test_hash!=request['source_hashes']['configs/scenes/test.json'] or test_hash!=sensor_request['manifest_sha256']:
        raise ValueError('Test manifest changed')
    frozen_scenes=read(manifest)['scenarios']
    output=project_output(args.output or ('artifacts/results/formal_'+args.version))
    if output.exists(): raise FileExistsError('Preserve old report; use a new output directory')
    rows=[]; episodes_by_key={}; evidence=[]
    for group in GROUPS:
        for seed in SEEDS:
            checkpoint=ROOT/('checkpoints/teacher_formal_%s_seed%d_l3/final.pth'%(args.version,seed)
                if group=='teacher' else 'checkpoints/visual_formal_%s_seed%d_%s_r9/final.pth'%(args.version,seed,group))
            checkpoint_hash=sha(checkpoint)
            budget=None if group=='teacher' else read(checkpoint.with_suffix('.json'))
            if budget and (budget['updates']!=12000 or budget['unique_labels']!=196608):
                raise ValueError('Student fixed-budget checkpoint mismatch')
            for condition in CONDITIONS:
                name=stage_name(args.version,seed,group,condition)
                directory=ROOT/'runs/evaluation'/name
                summary=read(directory/'summary.json')
                if not (summary['split']=='test' and summary['level']==3 and summary['condition']==condition
                        and summary['checkpoint_sha256']==checkpoint_hash and summary['manifest_sha256']==test_hash
                        and summary['num_episodes']==100):
                    raise ValueError('Evaluation identity/provenance mismatch: '+name)
                scenes=read(directory/'scenarios.json')
                expected=[s for s in frozen_scenes if s['condition']==condition]
                if scenes!=expected: raise ValueError('Executed scenarios differ from frozen test cases')
                episodes=[json.loads(line) for line in (directory/'episodes.jsonl').read_text().splitlines()]
                metrics=episode_metrics(episodes,[s['id'] for s in expected])
                verify_summary(summary,metrics)
                if group!='teacher':
                    if summary['kind']!='visual_student_evaluation' or summary['teacher_loaded'] is not False:
                        raise ValueError('Student inference must be teacher-free')
                    if summary['student']['seed']!=seed or summary['student']['updates']!=12000:
                        raise ValueError('Wrong student seed/budget')
                    expected_model='stack4' if group=='dagger_stack4' else 'gru'
                    if summary['student']['kind']!=expected_model:
                        raise ValueError('Wrong student architecture')
                elif summary['kind']!='state_teacher_evaluation':
                    raise ValueError('Wrong teacher evaluation kind')
                not_applicable=group=='teacher' and condition=='depth_loss_3'
                if group=='teacher' and condition in ('delay_40ms','depth_loss_3'):
                    audit=read(directory/'sensor_trace_audit.json')
                    if (not audit['exact'] or audit['episodes']!=100 or
                            audit['trace_sha256']!=sha(directory/'trace.npz')):
                        raise ValueError('Teacher sensor trace lacks an exact execution audit')
                    if not_applicable and audit['applicability']!='not_applicable_no_depth_input':
                        raise ValueError('Teacher depth-loss control must be labeled N/A')
                latency=summary['single_robot_inference']
                if latency['batch_size']!=1 or latency['includes_sensor_rendering'] or latency['includes_cpu_transfer']:
                    raise ValueError('Unexpected policy latency scope')
                subprocess.run([str(ROOT/'scripts/run.sh'),'scripts/analyze_failures.py','--evaluation',
                    str(directory.relative_to(ROOT))],cwd=ROOT,check=True,capture_output=True,text=True)
                diagnostics=read(directory/'failure_diagnostics.json')
                if diagnostics['failures']!=metrics['failures'] or diagnostics['timeouts']!=metrics['timeouts']:
                    raise ValueError('Failure diagnostic counts disagree')
                row=dict(group=group,seed=seed,condition=condition,**metrics,
                    policy_latency_median_ms=latency['gpu_policy_ms_median'],
                    policy_latency_p95_ms=latency['gpu_policy_ms_p95'],
                    unique_labels=budget['unique_labels'] if budget else None,
                    updates=budget['updates'] if budget else None,
                    label_presentations=budget['label_presentations'] if budget else None,
                    visual_robustness_applicable=not not_applicable,
                    failure_regions=diagnostics['regions'])
                rows.append(row);episodes_by_key[group,seed,condition]=episodes
                evidence.append(dict(name=name,checkpoint_sha256=checkpoint_hash,
                    files={p:sha(directory/p) for p in ('summary.json','episodes.jsonl','scenarios.json',
                                                       'trace.npz','failure_diagnostics.json')}))
                print(json.dumps(dict(verified=name,completed=len(rows),total=72)),flush=True)
    aggregates=[]
    for group in GROUPS:
        for condition in CONDITIONS:
            subset=[r for r in rows if r['group']==group and r['condition']==condition]
            metrics={key:seed_statistics([r[key] for r in subset]) for key in
                ('success_rate','crossing_rate','failures','timeouts','success_seconds_mean','recovery_seconds_mean',
                 'policy_latency_median_ms','policy_latency_p95_ms')}
            regions=Counter()
            for r in subset: regions.update(r['failure_regions'])
            aggregates.append(dict(group=group,condition=condition,metrics=metrics,seed_order=list(SEEDS),
                success_by_seed=[r['success_rate'] for r in subset],failure_regions=dict(regions),
                visual_robustness_applicable=not(group=='teacher' and condition=='depth_loss_3')))
    comparisons=[]
    for first,second in [('dagger_gru','bc_gru'),('dagger_gru','dagger_stack4')]:
        for condition in CONDITIONS:
            pairs=[dict(seed=seed,**paired_outcomes(episodes_by_key[first,seed,condition],
                episodes_by_key[second,seed,condition])) for seed in SEEDS]
            comparisons.append(dict(first=first,second=second,condition=condition,per_seed=pairs,
                difference_percentage_points=seed_statistics([p['success_difference_percentage_points'] for p in pairs])))
    targets=[]
    for condition,target in [('show',.90),('heldout',.80)]:
        row=next(r for r in aggregates if r['group']=='dagger_gru' and r['condition']==condition)
        targets.append(dict(condition=condition,target=target,mean=row['metrics']['success_rate']['mean'],
            mean_meets_target=row['metrics']['success_rate']['mean']>=target,
            every_seed_meets_target=all(x>=target for x in row['success_by_seed'])))
    output.mkdir(parents=True)
    result=dict(version=args.version,evaluation_count=72,episode_count=7200,seeds=list(SEEDS),rows=rows,
                aggregates=aggregates,paired_comparisons=comparisons,student_targets=targets,
                test_manifest_sha256=test_hash,evidence=evidence,
                matrix_request_sha256=sha(base/'request.json'),supplement_request_sha256=sha(supplement/'request.json'),
                aggregator_sha256=sha(Path(__file__)),statistics_sha256=sha(ROOT/'src/uav_gap/results.py'),
                note='Three training seeds reuse the same 100 cases per condition; not 300 independent scenes.')
    (output/'results.json').write_text(json.dumps(result,indent=2))
    fields=[key for key in rows[0] if key!='failure_regions']
    with (output/'per_seed.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows)
    lines=['# Formal final-test results','',
        'Fixed final-budget checkpoints; 3 training seeds x 100 identical cases per condition.',
        'Values are equal-weight seed means +/- sample SD, not confidence intervals.',
        'Timing averages include successful episodes only. Undefined seeds are reported, never zero-filled.',
        'Policy latency is warm batch-one GPU inference; it excludes sensing, CPU transfer and control.',
        'Own-state input is noisy/delayed simulated state, not VIO. No real-flight claim.','',
        '| Group | Condition | Success % (11 / 22 / 33) | Mean +/- SD % | Failure count | Timeout count | Completion s | Recovery s | Policy p50 ms | Policy p95 ms |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in aggregates:
        m=row['metrics'];condition=row['condition']+(' (N/A visual; input-unaffected control)' if not row['visual_robustness_applicable'] else '')
        cells=[row['group'],condition,' / '.join('%.0f'%(100*x) for x in row['success_by_seed']),
            mean_sd(m['success_rate'],100),mean_sd(m['failures']),mean_sd(m['timeouts']),
            mean_sd(m['success_seconds_mean']),mean_sd(m['recovery_seconds_mean']),
            mean_sd(m['policy_latency_median_ms']),mean_sd(m['policy_latency_p95_ms'])]
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['','## Matched-budget comparisons','',
        'DAgger-GRU minus the comparator, in percentage points, paired by training seed and case.',
        'The stack has four depth frames plus current proprioception; GRU retains recurrent features.',
        'Matching parameters, unique labels and updates does not imply matching FLOPs or valid label presentations.','',
        '| Contrast | Condition | Mean difference +/- seed SD (pp) |','| --- | --- | --- |']
    for row in comparisons:
        lines.append('| %s - %s | %s | %s |'%(row['first'],row['second'],row['condition'],mean_sd(row['difference_percentage_points'])))
    lines+=['','## Failure interpretation','',
        'Per-seed CSV separates failures/timeouts before and after full-body crossing.',
        'results.json includes terminal-position region counts. These are descriptive locations,',
        'not exact causal collision labels: detailed diagnostics use the nearest recorded 50Hz pose,',
        'while physical termination runs at 200Hz. Do not infer the precise cause from region alone.','',
        'Student stress uses baseline 20ms delay (60ms under extra-40ms), a three-acquisition depth hold',
        'at step 50, nominal-controller +20% physical mass/inertia, or a 0.1 N s lateral impulse',
        'at step 50 for surviving episodes. Teacher state delay changes 0ms to 40ms.',
        'Stress scenarios use their own predeclared randomized starts; differences from show combine',
        'condition and finite-case variation, so they are not paired clean-vs-stress causal estimates.','']
    (output/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(output),evaluations=72,episodes=7200,targets=targets),indent=2))


if __name__=='__main__':
    main()
