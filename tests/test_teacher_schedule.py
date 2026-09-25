import importlib.util
import json
from pathlib import Path


def test_formal_teachers_have_identical_budgets_and_three_validation_gates():
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location('teacher_schedule',root/'scripts/run_teacher_curriculum.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads((root/'configs/teacher_formal_v1.json').read_text())
    rows = module.schedule(config,'formal_test')
    assert len(rows)==12
    for seed in (11,22,33):
        stages = [r for r in rows if r['seed']==seed]
        assert [r['epochs'] for r in stages]==[300,750,1150,2250]
        assert stages[0]['previous'] is None
        for i,row in enumerate(stages):
            assert '--perturb-reset' in row['train']
            if i:
                assert row['previous']==stages[i-1]['checkpoint']
            gates = [e for e in row['evaluations'] if e['role']=='promotion']
            assert len(gates)==3
            assert len({e['checkpoint'] for e in gates})==3
        assert {e['condition'] for e in stages[-1]['evaluations'] if e['role']=='exact_final'}=={'show','heldout'}


def test_v2_preserves_final_difficulty_and_total_budget_with_a_staged_reset_curriculum():
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location('teacher_schedule_v2',root/'scripts/run_teacher_curriculum.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads((root/'configs/teacher_formal_v2.json').read_text())
    rows = module.schedule(config,'teacher_formal_v2')
    assert len(rows)==18
    for seed in (11,22,33):
        stages = [r for r in rows if r['seed']==seed]
        assert stages[0]['previous'] is None
        assert [r['epochs'] for r in stages]==[300,450,750,1150,1750,2250]
        assert [r['level'] for r in stages]==[0,1,1,2,3,3]
        assert [r['perturb_reset'] for r in stages]==[False]*5+[True]
        assert stages[1]['evaluations']==[]  # No promotion after the partial warmup.
        assert stages[-1]['checkpoint']=='checkpoints/teacher_formal_v2_seed%d_l3/final.pth' % seed
        assert all(s['previous']==stages[i-1]['checkpoint'] for i,s in enumerate(stages) if i)
        assert len([e for e in stages[-1]['evaluations'] if e['role']=='promotion'])==3
        assert {e['condition'] for e in stages[-1]['evaluations'] if e['role']=='exact_final'}=={'show','heldout'}


def test_declared_teacher_reward_is_forwarded_not_silently_ignored():
    root=Path(__file__).parents[1]
    spec=importlib.util.spec_from_file_location('teacher_reward_schedule',root/'scripts/run_teacher_curriculum.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=json.loads((root/'configs/teacher_formal_v2.json').read_text())
    config['recovery_reward']='original'
    rows=module.schedule(config,'reward_declaration_test')
    assert all(row['train'][row['train'].index('--recovery-reward')+1]=='original' for row in rows)
    import pytest
    config['recovery_reward']='unrecorded_variant'
    with pytest.raises(ValueError):module.schedule(config,'reward_declaration_test')


def test_v3_adds_same_wide_gate_budget_to_all_seeds_from_scratch():
    root=Path(__file__).parents[1]
    spec=importlib.util.spec_from_file_location('teacher_schedule_v3',root/'scripts/run_teacher_curriculum.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=json.loads((root/'configs/teacher_formal_v3.json').read_text())
    rows=module.schedule(config,'teacher_formal_v3')
    assert len(rows)==21
    for seed in (11,22,33):
        stages=[r for r in rows if r['seed']==seed]
        assert stages[0]['previous'] is None and stages[0]['evaluations']==[]
        assert [r['epochs'] for r in stages]==[300,600,750,1050,1450,2050,2550]
        assert [r['level'] for r in stages]==[0,0,1,1,2,3,3]
        assert [e['checkpoint'] for e in stages[1]['evaluations']]==[
            'checkpoints/teacher_formal_v3_seed%d_l0/snapshot_%06d.pth'%(seed,e) for e in (580,590,600)]
        assert stages[-1]['checkpoint']=='checkpoints/teacher_formal_v3_seed%d_l3/final.pth'%seed
        assert stages[-1]['perturb_reset'] and not any(r['perturb_reset'] for r in stages[:-1])
        assert all(r['train'][r['train'].index('--recovery-reward')+1]=='original' for r in stages)
