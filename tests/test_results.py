import copy
import pytest
from uav_gap.results import episode_metrics, seed_statistics, verify_summary, paired_outcomes


def cases():
    return [dict(scene_id='a',success=True,failure=False,timeout=False,crossed=True,
                 seconds=3.,crossing_seconds=1.,recovery_seconds=2.),
            dict(scene_id='b',success=False,failure=True,timeout=False,crossed=False,
                 seconds=.5,crossing_seconds=None,recovery_seconds=None),
            dict(scene_id='c',success=False,failure=False,timeout=True,crossed=True,
                 seconds=8.,crossing_seconds=1.,recovery_seconds=None)]


def test_success_conditioned_timing_and_failure_partition():
    metrics=episode_metrics(cases(),['a','b','c'])
    assert metrics['success_rate']==1/3
    assert metrics['success_seconds_mean']==3.
    assert metrics['recovery_seconds_mean']==2.
    assert metrics['failure_before_crossing']==1
    assert metrics['timeout_after_crossing']==1
    verify_summary(metrics,metrics)
    changed=dict(metrics,success_seconds_mean=11.5/3)
    with pytest.raises(ValueError): verify_summary(changed,metrics)


def test_no_success_is_undefined_not_zero_or_timeout():
    metrics=episode_metrics(cases()[1:],['b','c'])
    assert metrics['success_seconds_mean'] is None
    assert metrics['recovery_seconds_mean'] is None
    stats=seed_statistics([None,2.,4.])
    assert stats['mean']==3. and stats['defined_seeds']==2 and stats['total_seeds']==3
    assert stats['sample_sd']==pytest.approx(2**.5)
    assert seed_statistics([None,None,None])['mean'] is None


def test_cases_and_outcomes_fail_closed():
    original=cases()
    with pytest.raises(ValueError): episode_metrics(original,['c','b','a'])
    invalid=copy.deepcopy(original);invalid[0]['failure']=True
    with pytest.raises(ValueError): episode_metrics(invalid,['a','b','c'])
    invalid=copy.deepcopy(original);invalid[0]['crossed']=False
    with pytest.raises(ValueError): episode_metrics(invalid,['a','b','c'])
    invalid=copy.deepcopy(original);invalid[0]['recovery_seconds']=99.
    with pytest.raises(ValueError): episode_metrics(invalid,['a','b','c'])
    with pytest.raises(ValueError): seed_statistics([float('nan')])


def test_paired_seed_comparison_does_not_treat_seeds_as_new_scenes():
    first=cases();second=copy.deepcopy(first)
    second[0]['success']=False;second[1]['success']=True
    result=paired_outcomes(first,second)
    assert result==dict(first_only_success=1,second_only_success=1,same_outcome=1,
                        success_difference_percentage_points=0.)
    with pytest.raises(ValueError): paired_outcomes(first,list(reversed(second)))
    stats=seed_statistics([90,95,100])
    assert stats['mean']==95 and stats['sample_sd']==5
