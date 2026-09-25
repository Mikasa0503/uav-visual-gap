import pytest
from uav_gap.handoff import Inventory,verify_coverage


def test_inventory_rejects_changed_or_escaping_evidence(tmp_path):
    (tmp_path/'evidence.json').write_text('original')
    inventory=Inventory(tmp_path);inventory.add('evidence.json');inventory.verify()
    with pytest.raises(ValueError):inventory.add('../outside')
    with pytest.raises(ValueError):inventory.add('evidence.json','wrong hash')
    (tmp_path/'evidence.json').write_text('changed')
    with pytest.raises(ValueError):inventory.verify()


def test_coverage_rejects_duplicate_or_missing_group_even_when_count_is_72():
    rows=[dict(group=g,seed=s,condition=c) for g in ('teacher','bc_gru','dagger_gru','dagger_stack4')
          for s in (11,22,33) for c in ('show','heldout','delay_40ms','depth_loss_3','mass_plus20','lateral_impulse')]
    result=dict(rows=rows,evaluation_count=72,episode_count=7200)
    clips=[dict(r,episode=0) for r in rows];verify_coverage(result,clips)
    clips[-1]=clips[0]
    with pytest.raises(ValueError):verify_coverage(result,clips)


def test_pilot_brief_cannot_be_passed_off_as_formal_final_report():
    from uav_gap.handoff import verify_brief
    data=dict(status='pilot_draft_not_final_formal_report',version='v3',formal_results_sha256='metrics',pdf_sha256='pdf')
    with pytest.raises(ValueError):verify_brief(data,'v3','metrics','pdf')
    data['status']='formal_final_report'
    verify_brief(data,'v3','metrics','pdf')
    with pytest.raises(ValueError):verify_brief(data,'v3','changed metrics','pdf')
