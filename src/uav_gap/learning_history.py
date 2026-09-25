"""Outcome-independent selection of genuine chronological learning snapshots."""
import hashlib
import json


def read(path):
    return json.loads(path.read_text())


def lineage(root, folder, seen=None):
    seen = set() if seen is None else seen
    folder = folder.resolve()
    folder.relative_to(root)
    if folder in seen:
        raise ValueError('Experiment lineage contains a cycle')
    seen.add(folder)
    item = read(folder/'request.json')
    parent = item['config'].get('parent_experiment')
    history = lineage(root,root/parent,seen) if parent else []
    if history:
        previous = history[-1][1]
        if any(previous[key]!=item[key] for key in ('group','seed','teacher_sha256')):
            raise ValueError('Experiment lineage changes group, seed or teacher')
        if item['config'].get('initial_checkpoint')!=previous['rounds'][-1]['checkpoint']:
            raise ValueError('Continuation does not start from the parent final checkpoint')
    return history+[(folder,item)]


def balanced_selection(candidates,count):
    """Keep zero and every round boundary, then fill largest gaps in update space.

    Ties prefer earlier updates. Success rates and trajectories are never inputs.
    """
    by_update = {row['updates']:row for row in candidates}
    if len(by_update)!=len(candidates) or 0 not in by_update:
        raise ValueError('Unique update counts including the untrained policy are required')
    selected = {row['updates'] for row in candidates if row['mandatory']}
    if 0 not in selected or max(by_update) not in selected or not len(selected)<=count<=len(candidates):
        raise ValueError('Count must include all round boundaries and fit available checkpoints')
    while len(selected)<count:
        chosen = max((u for u in by_update if u not in selected),
                     key=lambda u:(min(abs(u-s) for s in selected),-u))
        selected.add(chosen)
    return [by_update[u] for u in sorted(selected)]


def candidates(root,folder):
    result = []
    previous_updates = 0
    for directory,request in lineage(root,folder):
        if not (directory/'complete.json').exists():
            raise ValueError('Freeze a film selection only after the experiment is complete')
        for row in request['rounds']:
            stage = (root/'checkpoints'/row['stem']).resolve()
            stage.relative_to(root)
            paths = sorted(stage.glob('snapshot_*.pth'))+[stage/'final.pth']
            if not result:
                paths.insert(0,stage/'initial.pth')
            round_candidates = {}
            for path in paths:
                meta = read(path.with_suffix('.json'))
                updates = meta['updates']
                if (meta['unique_labels']!=row['cumulative_unique_labels']
                        or not previous_updates<=updates<=row['cumulative_updates']):
                    raise ValueError('Checkpoint budgets disagree with the recorded round')
                round_candidates[updates] = dict(checkpoint=str(path.relative_to(root)),
                    updates=updates,labels=meta['unique_labels'],label_presentations=meta['label_presentations'],
                    round=row['index'],phase='BC' if row['index']==0 or request['group']=='bc_gru' else 'DAgger',
                    mandatory=path.name in ('initial.pth','final.pth'),
                    existing_validation=row['stem']+'_show' if path.name=='final.pth' else None)
            result.extend(round_candidates.values())
            previous_updates = row['cumulative_updates']
    return result


def make_film_request(root,folder,count):
    chosen = balanced_selection(candidates(root,folder),count)
    for row in chosen:
        row['checkpoint_sha256'] = hashlib.sha256((root/row['checkpoint']).read_bytes()).hexdigest()
    return dict(experiment=str(folder.relative_to(root)),count=count,
        selection='Keep initial and all round finals; greedily maximize distance in optimizer-update space; ties earlier. No outcomes used.',
        showcase_sha256=hashlib.sha256((root/'configs/scenes/showcase.json').read_bytes()).hexdigest(),
        validation_sha256=hashlib.sha256((root/'configs/scenes/validation.json').read_bytes()).hexdigest(),
        checkpoints=chosen)
