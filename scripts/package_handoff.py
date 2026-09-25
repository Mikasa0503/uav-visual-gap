"""Package only a fully verified formal report/video set; refuse partial handoffs."""
import argparse,json,re,subprocess,tarfile
from pathlib import Path
from uav_gap.runtime import ROOT,project_output
from uav_gap.handoff import Inventory,digest,verify_coverage,verify_brief


def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--version',required=True)
    parser.add_argument('--brief',required=True,help='Reviewed final interview report, project-relative')
    parser.add_argument('--film',default='artifacts/film/concise_learning_film_v10_spinning_rotors/uav_visual_learning_concise.mp4')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch(r'v[1-9][0-9]*',args.version):raise ValueError('Invalid protocol version')
    result_path=ROOT/('artifacts/results/formal_'+args.version+'/results.json')
    result=read(result_path)
    if result['version']!=args.version:raise ValueError('Report version mismatch')
    render=ROOT/('artifacts/render/formal_uncut_'+args.version)
    rendered=read(render/'complete.json')
    if rendered['request_sha256']!=digest(render/'request.json'):raise ValueError('Render completion mismatch')
    verify_coverage(result,rendered['clips'])
    matrix=ROOT/('runs/experiments/formal_matrix_'+args.version)
    supplement=ROOT/('runs/experiments/teacher_sensor_test_'+args.version)
    for directory,key in ((matrix,'matrix_request_sha256'),(supplement,'supplement_request_sha256')):
        if read(directory/'complete.json')['request_sha256']!=digest(directory/'request.json') or result[key]!=digest(directory/'request.json'):
            raise ValueError('Report and experiment requests disagree')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('Commit source changes before producing a reproducible handoff')
    inventory=Inventory(ROOT)
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    for name in filter(None,tracked):
        if name.startswith('third_party/'):raise ValueError('Do not redistribute licensed third-party sources')
        inventory.add(name)
    for path,value in read(matrix/'request.json')['source_hashes'].items():inventory.add(path,value)
    for directory in (matrix,supplement,render):
        for name in ('request.json','complete.json'):inventory.add(str((directory/name).relative_to(ROOT)))
    for name in ('results.json','per_seed.csv','report.md'):
        inventory.add(str((result_path.parent/name).relative_to(ROOT)))
    checkpoints=set()
    evidence_by_name={r['name']:r for r in result['evidence']}
    declared={r['name'] for r in read(matrix/'request.json')['stages'] if r['kind'].endswith('_test')}
    declared.update(r['name'] for r in read(supplement/'request.json')['stages'])
    if set(evidence_by_name)!=declared or {c['name'] for c in rendered['clips']}!=declared:
        raise ValueError('Report or video names differ from frozen evaluation declarations')
    if len(result['evidence'])!=72 or len({r['name'] for r in result['evidence']})!=72:
        raise ValueError('Incomplete report provenance')
    for row in result['evidence']:
        evaluation=ROOT/'runs/evaluation'/row['name']
        if set(row['files'])!={'summary.json','episodes.jsonl','scenarios.json','trace.npz','failure_diagnostics.json'}:
            raise ValueError('Report lacks full episode evidence')
        for name,value in row['files'].items():inventory.add(str((evaluation/name).relative_to(ROOT)),value)
        summary=read(evaluation/'summary.json')
        inventory.add(summary['checkpoint'],row['checkpoint_sha256']);checkpoints.add(summary['checkpoint'])
        audit=evaluation/'sensor_trace_audit.json'
        if audit.exists():inventory.add(str(audit.relative_to(ROOT)))
    if len(checkpoints)!=12:raise ValueError('Expected 3 final teachers and 9 final students')
    for name in checkpoints:
        checkpoint=ROOT/name
        for sibling in checkpoint.parent.iterdir():
            if sibling.suffix in ('.json','.yaml','.jsonl'):
                inventory.add(str(sibling.relative_to(ROOT)))
        configuration=checkpoint.parent/'configuration.json'
        if configuration.exists():
            for dataset,value in read(configuration)['datasets_sha256'].items():
                inventory.add(dataset+'/manifest.json',value)
    for clip in rendered['clips']:
        inventory.add(clip['path'],clip['sha256'])
        metadata_path=Path(clip['path']).with_suffix('.json')
        inventory.add(str(metadata_path))
        metadata=read(ROOT/metadata_path)
        evidence=evidence_by_name[clip['name']]['files']
        if (metadata['video_sha256']!=clip['sha256'] or metadata['render']['episode']!=0
                or metadata['render']['trace_sha256']!=evidence['trace.npz']
                or metadata['render']['evaluation_summary_sha256']!=evidence['summary.json']):
            raise ValueError('Video and reported episode evidence do not match')
    video_dir=ROOT/('artifacts/video/formal_uncut_'+args.version)
    for name in ('index.md','manifest.json'):inventory.add(str((video_dir/name).relative_to(ROOT)))
    inventory.add(args.brief);inventory.add(args.film)
    brief=ROOT/args.brief
    if brief.suffix.lower()!='.pdf':raise ValueError('Final interview brief must be a one-page PDF')
    brief_provenance=brief.parent/'provenance.json'
    verify_brief(read(brief_provenance),args.version,digest(result_path),digest(brief))
    inventory.add(str(brief_provenance.relative_to(ROOT)))
    import os
    pdf_env=dict(os.environ);pdf_env.pop('LD_LIBRARY_PATH',None)
    pdf_info=subprocess.check_output(['/usr/bin/pdfinfo',str(brief)],env=pdf_env,text=True)
    if not re.search(r'^Pages:\s+1\s*$',pdf_info,re.MULTILINE):raise ValueError('Interview brief must have exactly one page')
    film_manifest=read((ROOT/args.film).parent/'manifest.json')
    if digest(ROOT/args.film)!=film_manifest['video_sha256']:raise ValueError('Demo hash mismatch')
    for name in ('manifest.json','qa.json','captions.json','demo_captions.txt'):
        inventory.add(str(((ROOT/args.film).parent/name).relative_to(ROOT)))
    inventory.add('runs/runtime_inventory.json')
    inventory.add('assets/gap/manifest.json')
    output=project_output('artifacts/handoff/'+args.version)
    info=dict(version=args.version,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        files=inventory.files,file_count=len(inventory.files),total_bytes=sum(r['bytes'] for r in inventory.files.values()),
        scope='Code/configs, environment inventory, 12 final policies, all 7200 episode traces/metrics, 72 predeclared full replays, reviewed brief and labeled pilot demo.',
        exclusions='Licensed third-party binaries and full training datasets are not redistributed. Dataset manifests and exact regeneration commands remain in training configuration; deterministic bitwise re-training is not promised.')
    if args.check_only:
        print(json.dumps({k:v for k,v in info.items() if k!='files'},indent=2));return
    output.mkdir(parents=True,exist_ok=False)
    (output/'inventory.json').write_text(json.dumps(info,indent=2))
    inventory.verify()
    archive=output/('uav_visual_gap_'+args.version+'.tar.gz')
    with tarfile.open(archive,'w:gz') as bundle:
        for name in sorted(inventory.files):bundle.add(ROOT/name,arcname=name,recursive=False)
        bundle.add(output/'inventory.json',arcname='HANDOFF_INVENTORY.json',recursive=False)
    # Verify archived payload hashes, not only source paths.
    import hashlib
    with tarfile.open(archive,'r:gz') as bundle:
        for name,row in inventory.files.items():
            stream=bundle.extractfile(name);h=hashlib.sha256()
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            if h.hexdigest()!=row['sha256']:raise ValueError('Archived payload mismatch: '+name)
    (output/'archive_sha256.txt').write_text(digest(archive)+'  '+archive.name+'\n')
    print(json.dumps(dict(archive=str(archive),files=len(inventory.files),sha256=digest(archive)),indent=2))


if __name__=='__main__':main()
