import importlib.util
import json
from pathlib import Path


def test_formal_video_selection_is_complete_fixed_and_outcome_independent():
    root=Path(__file__).parents[1]
    spec=importlib.util.spec_from_file_location('formal_video',root/'scripts/render_formal_evaluations.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=json.loads((root/'configs/formal_video_v2.json').read_text())
    selected=module.selection(config)
    assert len(selected)==len({row['name'] for row in selected})==72
    assert {row['episode'] for row in selected}=={0}
    assert len({(row['group'],row['seed']) for row in selected})==12
    assert all(len([row for row in selected if row['condition']==condition])==12 for condition in config['conditions'])
    assert selected[0]['name']=='formal_v2_test_seed11_teacher_show'
    assert selected[-1]['name']=='formal_v2_test_seed33_dagger_stack4_lateral_impulse'
    assert all(set(row)=={'name','seed','group','condition','episode'} for row in selected)
