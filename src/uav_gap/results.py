"""CPU-only statistical checks for the frozen experiment; no model selection."""
import math
from statistics import mean, stdev


def seed_statistics(values):
    """Equal weight per training seed; undefined timing is never a zero."""
    valid = [float(x) for x in values if x is not None]
    if any(not math.isfinite(x) for x in valid):
        raise ValueError('Non-finite result')
    return dict(mean=mean(valid) if valid else None,
                sample_sd=stdev(valid) if len(valid)>1 else None,
                minimum=min(valid) if valid else None,
                maximum=max(valid) if valid else None,
                defined_seeds=len(valid), total_seeds=len(values))


def episode_metrics(episodes, expected_ids):
    if [row['scene_id'] for row in episodes] != expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError('Missing, duplicated, reordered or unexpected evaluation cases')
    if not episodes:
        raise ValueError('Empty evaluation')
    for row in episodes:
        if any(type(row[k]) is not bool for k in ('success','failure','timeout','crossed')):
            raise ValueError('Invalid outcome flags')
        if sum(row[k] for k in ('success','failure','timeout')) != 1:
            raise ValueError('Every episode needs exactly one terminal outcome')
        if not math.isfinite(row['seconds']) or not 0 < row['seconds'] <= 8.02:
            raise ValueError('Invalid terminal duration')
        if row['success']:
            if not row['crossed'] or row['crossing_seconds'] is None or row['recovery_seconds'] is None:
                raise ValueError('Success requires a crossing and measured recovery')
            if not 0 <= row['crossing_seconds'] < row['seconds']:
                raise ValueError('Invalid crossing time')
            if not math.isclose(row['recovery_seconds'],row['seconds']-row['crossing_seconds'],abs_tol=1e-6):
                raise ValueError('Recovery must equal completion minus crossing')
        elif row['recovery_seconds'] is not None:
            raise ValueError('Failed episodes must not enter success-conditioned recovery')
    success = [row for row in episodes if row['success']]
    counts = {key:sum(row[key] for row in episodes) for key in ('success','failure','timeout','crossed')}
    return dict(num_episodes=len(episodes),successes=counts['success'],failures=counts['failure'],
                timeouts=counts['timeout'],success_rate=counts['success']/len(episodes),
                crossing_rate=counts['crossed']/len(episodes),
                success_seconds_mean=mean(row['seconds'] for row in success) if success else None,
                recovery_seconds_mean=mean(row['recovery_seconds'] for row in success) if success else None,
                failure_before_crossing=sum(row['failure'] and not row['crossed'] for row in episodes),
                failure_after_crossing=sum(row['failure'] and row['crossed'] for row in episodes),
                timeout_before_crossing=sum(row['timeout'] and not row['crossed'] for row in episodes),
                timeout_after_crossing=sum(row['timeout'] and row['crossed'] for row in episodes))


def verify_summary(summary, metrics):
    for key in ('num_episodes','failures','timeouts','success_rate','crossing_rate',
                'success_seconds_mean','recovery_seconds_mean'):
        actual, expected = summary[key], metrics[key]
        if actual is None or expected is None:
            if actual is not expected:
                raise ValueError('Summary/episode disagreement: '+key)
        elif not math.isclose(actual,expected,rel_tol=1e-6,abs_tol=1e-6):
            raise ValueError('Summary/episode disagreement: '+key)


def paired_outcomes(first, second):
    if [row['scene_id'] for row in first] != [row['scene_id'] for row in second]:
        raise ValueError('Paired comparison requires identical ordered cases')
    wins = sum(a['success'] and not b['success'] for a,b in zip(first,second))
    losses = sum(b['success'] and not a['success'] for a,b in zip(first,second))
    return dict(first_only_success=wins,second_only_success=losses,
                same_outcome=len(first)-wins-losses,
                success_difference_percentage_points=100*(wins-losses)/len(first))
