"""Summarize every fixed-budget dense-recovery validation, including failures."""
import json,hashlib
from pathlib import Path
from uav_gap.runtime import ROOT,project_output


def read(path):return json.loads(path.read_text())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    experiment=ROOT/'runs/experiments/teacher_recovery_dense_v2_diagnostic'
    request=read(experiment/'request.json');complete=read(experiment/'complete.json')
    if sorted(row['seed'] for row in complete['results'])!=[11,22,33]:raise ValueError('All diagnostic seeds are required')
    rows=[]
    for seed in (11,22,33):
        for variant,prefix in [('original','teacher_formal_v2'),('dense_v2','teacher_recovery_dense_v2_diagnostic')]:
            family=prefix+'_seed%d_l0'%seed
            training=ROOT/'checkpoints'/family
            if not (training/'final.pth').exists():
                if variant=='dense_v2':raise ValueError('Missing diagnostic teacher')
                rows.append(dict(seed=seed,variant=variant,available=False,reason='Formal v2 stopped at seed22; seed33 baseline was never trained.'))
                continue
            task=read(training/'task.json')
            if task.get('recovery_reward','original')!=variant or task['seed']!=seed or task['level']!=0 or task['perturb_reset']:
                raise ValueError('Wrong training identity')
            reports=[]
            for epoch in (280,290,300):
                evaluation=ROOT/'runs/evaluation'/(family+'_epoch%d_show'%epoch)
                report=read(evaluation/'summary.json')
                checkpoint=training/('snapshot_%06d.pth'%epoch)
                if not (report['split']=='validation' and report['condition']=='show' and report['num_episodes']==100
                        and report['level']==0 and report['checkpoint_sha256']==sha(checkpoint)):
                    raise ValueError('Wrong validation/checkpoint identity')
                episodes=[json.loads(s) for s in (evaluation/'episodes.jsonl').read_text().splitlines()]
                if len(episodes)!=100 or sum(r['success'] for r in episodes)!=round(100*report['success_rate']):
                    raise ValueError('Episode/summary mismatch')
                reports.append(dict(epoch=epoch,successes=sum(r['success'] for r in episodes),crossings=sum(r['crossed'] for r in episodes),
                    failures=sum(r['failure'] for r in episodes),timeouts=sum(r['timeout'] for r in episodes),
                    source=str(evaluation.relative_to(ROOT)),summary_sha256=sha(evaluation/'summary.json')))
            rows.append(dict(seed=seed,variant=variant,available=True,passed=all(r['successes']>=85 for r in reports),reports=reports))
    dense=[r for r in rows if r['variant']=='dense_v2']
    if any(r['passed']!=next(x['passed'] for x in complete['results'] if x['seed']==r['seed']) for r in dense):
        raise ValueError('Coordinator gate reports disagree')
    result=dict(role='development_validation_not_formal_test',rows=rows,request_sha256=sha(experiment/'request.json'),
        complete_sha256=sha(experiment/'complete.json'),all_dense_seeds_passed=all(r['passed'] for r in dense),
        limitations='Baseline seed33 does not exist. This is not a complete three-seed paired reward ablation. Repeated snapshots use the same100starts, not300independentcases. No final-test data used.')
    out=project_output('artifacts/diagnostics/recovery_dense_v2_summary');out.mkdir(parents=True,exist_ok=False)
    (out/'results.json').write_text(json.dumps(result,indent=2))
    lines=['# Recovery reward diagnostic','',
        'Development validation only. Every run uses300epochs,512environments and the wide l0 aperture. No final-test results were inspected.','',
        '| Seed | Reward | Successes at280 /290 /300 (each out of100) | Final crossing /failure /timeout | Three-snapshot gate |',
        '|---|---|---|---|---|']
    for row in rows:
        if not row['available']:
            lines.append('| %d | %s | not run | not run | unavailable |'%(row['seed'],row['variant']));continue
        final=row['reports'][-1]
        lines.append('| %d | %s | %s | %d /%d /%d | %s |'%(row['seed'],row['variant'],
            ' /'.join(str(r['successes']) for r in row['reports']),final['crossings'],final['failures'],final['timeouts'],'pass' if row['passed'] else 'fail'))
    lines+=['',result['limitations'],'',
        'The dense variant adds bounded independent penalties only after full-body crossing. Physics, sensors, controller, reward events and success thresholds are unchanged.',
        'Promote no recipe on one favorable snapshot. Preserve negative results; an independent original-reward continuation to600epochs tests optimization budget in a new development directory. It never extends the frozen formal run in place.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
