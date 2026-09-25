import importlib.util
import sys
from pathlib import Path
import pytest
from uav_gap.formal_names import protocol_version,test_prefix as prefix


def test_version_names_preserve_history_and_reject_path_injection():
    assert prefix('v1')=='formal_test'
    assert prefix('v2')=='formal_v2_test'
    assert prefix('v3')=='formal_v3_test'
    for value in ('v0','v01','v3/other','../v3','v3; echo x',None):
        with pytest.raises(ValueError):protocol_version(value)


def test_new_protocol_training_reporting_and_video_share_namespace():
    root=Path(__file__).parents[1]
    sys.path.insert(0,str(root/'scripts'))
    try:
        import run_formal_matrix,aggregate_formal_results,render_formal_evaluations
        stages=run_formal_matrix.schedule('v3')
        tests=[s for s in stages if s['kind'].endswith('_test')]
        assert len(tests)==66 and len(stages)==76
        import json
        config=json.loads((root/'configs/formal_video_v2.json').read_text());config['version']='v3'
        clips=render_formal_evaluations.selection(config)
        for clip in clips:
            assert clip['name']==aggregate_formal_results.stage_name('v3',clip['seed'],clip['group'],clip['condition'])
        assert {s['name'] for s in tests}<={c['name'] for c in clips}
        assert not ({s['name'] for s in tests}&{s['name'] for s in run_formal_matrix.schedule('v2') if s['kind'].endswith('_test')})
    finally:sys.path.pop(0)
