import pytest
from uav_gap.learning_history import balanced_selection


def test_selection_keeps_initial_final_and_regression_round_boundaries():
    candidates = [dict(updates=u,mandatory=u in (0,1000,2000,3000),success_rate=float(u!=2000))
                  for u in range(0,3001,50)]
    chosen = balanced_selection(candidates,12)
    assert len(chosen)==12
    assert {0,1000,2000,3000}.issubset({x['updates'] for x in chosen})
    assert [x['updates'] for x in chosen]==sorted(x['updates'] for x in chosen)
    flipped = [dict(row,success_rate=1-row['success_rate']) for row in reversed(candidates)]
    assert [x['updates'] for x in chosen]==[x['updates'] for x in balanced_selection(flipped,12)]


def test_selection_refuses_to_hide_rounds_or_duplicate_policy_points():
    candidates = [dict(updates=u,mandatory=True) for u in (0,1000,2000)]
    with pytest.raises(ValueError):
        balanced_selection(candidates,2)
    with pytest.raises(ValueError):
        balanced_selection(candidates+[dict(updates=1000,mandatory=True)],3)
