"""Strict, relocatable evidence inventories for completed experiment handoffs."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class Inventory:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.files={}

    def add(self,relative,expected=None):
        relative=Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Inventory paths must be relative and contained')
        path=self.root/relative
        path.resolve().relative_to(self.root)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Only existing regular files can be packaged: '+str(relative))
        sha=digest(path)
        if expected is not None and sha!=expected:
            raise ValueError('Evidence hash mismatch: '+str(relative))
        self.files[relative.as_posix()]=dict(sha256=sha,bytes=path.stat().st_size)

    def verify(self):
        for name,row in self.files.items():
            path=self.root/name
            if not path.is_file() or digest(path)!=row['sha256']:
                raise ValueError('Evidence changed after inventory: '+name)


def verify_coverage(result,clips):
    expected={(g,s,c) for g in ('teacher','bc_gru','dagger_gru','dagger_stack4')
              for s in (11,22,33) for c in ('show','heldout','delay_40ms','depth_loss_3','mass_plus20','lateral_impulse')}
    metric_keys=[(r['group'],r['seed'],r['condition']) for r in result['rows']]
    clip_keys=[(r['group'],r['seed'],r['condition']) for r in clips]
    if len(metric_keys)!=72 or set(metric_keys)!=expected or len(clip_keys)!=72 or set(clip_keys)!=expected:
        raise ValueError('All 72 unique metric groups and uncut videos are required')
    if result['evaluation_count']!=72 or result['episode_count']!=7200:
        raise ValueError('Incomplete formal episode coverage')
    if any(c['episode']!=0 for c in clips):
        raise ValueError('Video selection differs from predeclared episode 0')


def verify_brief(provenance,version,results_sha256,pdf_sha256):
    if not (provenance.get('status')=='formal_final_report' and provenance.get('version')==version
            and provenance.get('formal_results_sha256')==results_sha256
            and provenance.get('pdf_sha256')==pdf_sha256):
        raise ValueError('Final handoff requires a reviewed brief bound to the completed formal results, not a pilot draft')
