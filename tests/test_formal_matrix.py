import importlib.util
import sys
from pathlib import Path


def test_all_nine_student_trainings_precede_the_54_fixed_final_tests():
    scripts = Path(__file__).parents[1]/'scripts'
    sys.path.insert(0,str(scripts))
    try:
        spec = importlib.util.spec_from_file_location('formal_matrix',scripts/'run_formal_matrix.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        stages = module.schedule()
    finally:
        sys.path.pop(0)
    assert stages[0]['kind']=='teacher_training'
    students = [s for s in stages if s['kind']=='student_training']
    tests = [s for s in stages if s['kind']=='student_test']
    assert len(students)==9 and len(tests)==54
    assert all(s['kind'].endswith('_training') for s in stages[:10])
    assert all(s['kind'].endswith('_test') for s in stages[10:])
    assert len({s['name'] for s in stages})==len(stages)==76
    assert all(s['checkpoint'].endswith('_r9/final.pth') for s in tests)
    assert all('--record' in s['command'] for s in stages[10:])
    assert len([s for s in stages if s['kind']=='teacher_test'])==12
    revised = module.schedule('v2')
    assert len(revised)==76
    assert 'configs/teacher_formal_v2.json' in revised[0]['command']
    assert revised[0]['directory']=='runs/experiments/teacher_formal_v2'
    assert not ({s['name'] for s in stages[1:]} & {s['name'] for s in revised[1:]})
    assert all('formal_v2' in s['checkpoint'] for s in revised[10:])
