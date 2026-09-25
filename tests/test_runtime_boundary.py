import fcntl
import json
import subprocess
from types import SimpleNamespace

import pytest

from uav_gap import runtime
from uav_gap.runtime import ROOT, project_output


def test_outputs_cannot_escape_project():
    with pytest.raises(ValueError):
        project_output('../outside-project')


def test_absolute_external_output_rejected():
    with pytest.raises(ValueError):
        project_output('/tmp/outside-project')


def test_known_output_remains_inside():
    assert project_output('runs/evaluation') == ROOT / 'runs/evaluation'


def test_gpu_admission_rejects_external_process_and_releases_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    monkeypatch.setattr(runtime, '_gpu_lease', None)
    answers = iter(['40960, 31890', '1234, other-trainer'])
    monkeypatch.setattr(runtime.subprocess, 'run',
                        lambda *args, **kwargs: SimpleNamespace(stdout=next(answers)))
    with pytest.raises(RuntimeError, match='GPU is unavailable'):
        runtime.require_idle_gpu()
    with (tmp_path / '.cache/gpu0.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_gpu_admission_holds_project_lock_until_job_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    monkeypatch.setattr(runtime, '_gpu_lease', None)
    monkeypatch.setattr(runtime, 'start_gpu_monitor', lambda *args: None)
    answers = iter(['40960, 40900', ''])
    monkeypatch.setattr(runtime.subprocess, 'run',
                        lambda *args, **kwargs: SimpleNamespace(stdout=next(answers)))
    runtime.require_idle_gpu()
    try:
        with (tmp_path / '.cache/gpu0.lock').open('a+') as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        runtime._gpu_lease[1].close()
        runtime._gpu_lease = None


def test_coordinator_gpu_check_does_not_hold_child_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    answers = iter(['40960, 40900', ''])
    monkeypatch.setattr(runtime.subprocess, 'run',
                        lambda *args, **kwargs: SimpleNamespace(stdout=next(answers)))
    assert runtime.check_gpu_available() == (40960, 40900)
    assert not (tmp_path / '.cache/gpu0.lock').exists()


def test_gpu_runtime_limit_keeps_eight_gib_headroom():
    assert not runtime.gpu_memory_limit_exceeded(40960, 8192)
    assert runtime.gpu_memory_limit_exceeded(40960, 8191)
    assert runtime.gpu_memory_limit_exceeded(16384, 8000)


def test_gpu_monitor_interrupts_only_its_own_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    answers = iter(['40960, 8191', '12345, 21000'])
    monkeypatch.setattr(runtime.subprocess, 'run',
                        lambda *args, **kwargs: SimpleNamespace(stdout=next(answers)))
    signalled = []
    monkeypatch.setattr(runtime.os, 'kill', lambda pid, sig: signalled.append((pid, sig)))
    runtime.start_gpu_monitor(0, 12345)
    runtime._gpu_monitor[1].join(timeout=2)
    assert not runtime._gpu_monitor[1].is_alive()
    assert signalled == [(12345, runtime.signal.SIGINT)]
    row = json.loads((tmp_path / 'logs/gpu_usage_12345.jsonl').read_text().strip())
    assert row['action'] == 'interrupt_memory_limit'
    assert row['foreign_pids'] == []


def test_gpu_monitor_yields_when_another_job_enters(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    answers = iter(['40960, 35000', '12345, 5000\n67890, 500'])
    monkeypatch.setattr(runtime.subprocess, 'run',
                        lambda *args, **kwargs: SimpleNamespace(stdout=next(answers)))
    signalled = []
    monkeypatch.setattr(runtime.os, 'kill', lambda pid, sig: signalled.append((pid, sig)))
    runtime.start_gpu_monitor(0, 12345)
    runtime._gpu_monitor[1].join(timeout=2)
    assert not runtime._gpu_monitor[1].is_alive()
    assert signalled == [(12345, runtime.signal.SIGINT)]
    row = json.loads((tmp_path / 'logs/gpu_usage_12345.jsonl').read_text().strip())
    assert row['action'] == 'yield_to_other_job'
    assert row['foreign_pids'] == [67890]


def test_gpu_monitor_stops_own_job_after_repeated_sampling_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)

    def fail_sample(*args, **kwargs):
        raise subprocess.CalledProcessError(1, 'nvidia-smi')

    monkeypatch.setattr(runtime.subprocess, 'run', fail_sample)
    signalled = []
    monkeypatch.setattr(runtime.os, 'kill', lambda pid, sig: signalled.append((pid, sig)))
    runtime.start_gpu_monitor(0, 12345, interval_seconds=0)
    runtime._gpu_monitor[1].join(timeout=2)
    assert not runtime._gpu_monitor[1].is_alive()
    assert signalled == [(12345, runtime.signal.SIGINT)]
    rows = [json.loads(line) for line in
            (tmp_path / 'logs/gpu_usage_12345.jsonl').read_text().splitlines()]
    assert [row['action'] for row in rows] == [
        'monitor_error', 'monitor_error', 'interrupt_monitor_failure']
