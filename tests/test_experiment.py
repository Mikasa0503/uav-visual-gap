import json
from pathlib import Path
from uav_gap.experiment import GROUPS, build_rounds


def test_three_groups_share_initial_data_and_exact_budgets():
    config = json.loads((Path(__file__).parents[1]/'configs/visual_pilot.json').read_text())
    schedules = {group:build_rounds(config,'pilot',group,11,'checkpoints/teacher/final.pth') for group in GROUPS}
    assert len({rows[0]['data_name'] for rows in schedules.values()}) == 1
    for group, rows in schedules.items():
        assert rows[-1]['cumulative_unique_labels'] == 65536
        assert rows[-1]['cumulative_updates'] == 4000
        for index,row in enumerate(rows):
            assert len(row['datasets']) == index+1
            assert ('--student' in row['collect']) == (group.startswith('dagger') and index>0)
            if index:
                assert rows[index-1]['checkpoint'] in row['train']
            assert '--perturb-reset' in row['collect']
            assert all('test' not in command for command in row['evaluate'])


def test_continuation_preserves_history_and_adds_only_new_budget():
    config = json.loads((Path(__file__).parents[1]/'configs/visual_pilot_extension.json').read_text())
    rows = build_rounds(config,'extension','dagger_gru',11,'checkpoints/teacher/final.pth')
    assert [r['index'] for r in rows] == [4,5]
    assert rows[0]['datasets'][:4] == config['initial_datasets']
    assert config['initial_checkpoint'] in rows[0]['collect']
    assert config['initial_checkpoint'] in rows[0]['train']
    assert rows[-1]['cumulative_unique_labels'] == 98304
    assert rows[-1]['cumulative_updates'] == 6000
    assert 'shared' not in rows[0]['data_name']


def test_learning_rate_schedule_is_explicit_and_equal_across_groups():
    config = json.loads((Path(__file__).parents[1]/'configs/visual_pilot.json').read_text())
    config['learning_rates'] = [3e-4,3e-4,1e-4,1e-4]
    for group in GROUPS:
        rows = build_rounds(config,'rate_test',group,11,'checkpoints/teacher/final.pth')
        assert [float(row['train'][row['train'].index('--learning-rate')+1]) for row in rows] == config['learning_rates']


def test_variable_round_budgets_match_the_pilot_recipe_for_all_groups():
    config = json.loads((Path(__file__).parents[1]/'configs/visual_pilot.json').read_text())
    config.update(rounds=10,labels_per_round=[16384]*8+[32768]*2,updates_per_round=[1000]*8+[2000]*2)
    for group in GROUPS:
        rows = build_rounds(config,'formal',group,11,'checkpoints/teacher/final.pth')
        assert rows[7]['cumulative_unique_labels']==131072
        assert rows[8]['cumulative_unique_labels']==163840
        assert rows[-1]['cumulative_unique_labels']==196608
        assert rows[-1]['cumulative_updates']==12000
        assert [int(r['collect'][r['collect'].index('--label-budget')+1]) for r in rows]==config['labels_per_round']
        assert [int(r['train'][r['train'].index('--updates')+1]) for r in rows]==config['updates_per_round']
