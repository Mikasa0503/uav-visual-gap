"""Record outcome-independent learning snapshots and exact-checkpoint batch scores."""
import argparse
import hashlib
import json
import os
import re
import subprocess
from uav_gap.runtime import ROOT, project_output
from uav_gap.learning_history import make_film_request,read


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment',required=True,help='Completed runs/experiments directory relative to project')
    parser.add_argument('--name',required=True)
    parser.add_argument('--count',type=int,default=30)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',args.name):
        raise ValueError('Invalid recording name')
    experiment = (ROOT/args.experiment).resolve()
    experiment.relative_to(ROOT)
    request = make_film_request(ROOT,experiment,args.count)
    if args.dry_run:
        print(json.dumps(request,indent=2))
        return
    directory = project_output('runs/film/'+args.name)
    if directory.exists():
        if not args.resume or read(directory/'request.json')!=request:
            raise ValueError('Resume requires the identical frozen film request')
    else:
        directory.mkdir(parents=True)
        (directory/'request.json').write_text(json.dumps(request,indent=2))
    (directory/'pid').write_text(str(os.getpid()))
    records = []
    for row in request['checkpoints']:
        if digest(ROOT/row['checkpoint'])!=row['checkpoint_sha256']:
            raise RuntimeError('Selected checkpoint changed after the recording request')
        summaries = {}
        for split,count in [('validation',100),('showcase',1)]:
            name = '%s_u%06d_%s' % (args.name,row['updates'],split)
            if split=='validation' and row['existing_validation']:
                name = row['existing_validation']
            folder = ROOT/'runs/evaluation'/name
            path = folder/'summary.json'
            if not path.exists():
                command = ['scripts/evaluate_student.py','--checkpoint',row['checkpoint'],
                    '--split',split,'--condition','show','--count',str(count),'--name',name]
                if split=='showcase':
                    command.append('--record')
                with (directory/(name+'.log')).open('a') as log:
                    log.write(json.dumps(dict(command=command))+'\n')
                    log.flush()
                    subprocess.run([str(ROOT/'scripts/run.sh')]+command,cwd=ROOT,stdout=log,
                                   stderr=subprocess.STDOUT,check=True)
            summary = read(path)
            if not (summary['kind']=='visual_student_evaluation' and summary['teacher_loaded'] is False
                    and summary['split']==split and summary['condition']=='show' and summary['level']==3
                    and summary['num_episodes']==count and summary['checkpoint_sha256']==row['checkpoint_sha256']
                    and summary['manifest_sha256']==request[split+'_sha256']
                    and summary['student']['updates']==row['updates']):
                raise ValueError('Recorded evaluation provenance mismatch')
            if split=='showcase' and not (folder/'trace.npz').is_file():
                raise ValueError('Showcase requires the actual recorded physical trace')
            summaries[split] = dict(directory=str(folder.relative_to(ROOT)),success_rate=summary['success_rate'],
                                   summary_sha256=digest(path))
        record = dict(row,evaluations=summaries)
        records.append(record)
        (directory/('update_%06d.json' % row['updates'])).write_text(json.dumps(record,indent=2))
        print(json.dumps(dict(updates=row['updates'],showcase_success=summaries['showcase']['success_rate'],
                              batch_success=summaries['validation']['success_rate'])),flush=True)
    (directory/'complete.json').write_text(json.dumps(dict(request_sha256=digest(directory/'request.json'),
                                                          checkpoints=records),indent=2))


if __name__=='__main__':
    main()
