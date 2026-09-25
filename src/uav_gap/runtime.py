"""Explicit runtime boundary and read-only GPU availability checks."""

import os
import subprocess
import sys
import fcntl
import json
import signal
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_ENV = os.environ.get('UAV_GAP_ENV')
ENV = Path(_ENV).expanduser().resolve() if _ENV else None
_gpu_lease = None
_gpu_monitor = None


def assert_environment():
    active = Path(sys.prefix).resolve()
    if ENV is not None and active != ENV:
        raise RuntimeError('Use the interpreter selected by UAV_GAP_ENV via scripts/run.sh')
    if ENV is None and active.name != 'uav_gap':
        raise RuntimeError('Activate the dedicated uav_gap environment or set UAV_GAP_ENV')


def project_output(relative):
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT)
    return path


def check_gpu_available(gpu=0, minimum_free_mib=32768):
    """Read-only admission check for coordinator processes that spawn GPU jobs."""
    result = subprocess.run(
            ['nvidia-smi', '--id=%d' % gpu,
             '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True)
    fields = [item.strip() for item in result.stdout.strip().split(',')]
    if len(fields) != 2:
        raise RuntimeError('Cannot read GPU memory inventory')
    total_mib, free_mib = map(int, fields)
    minimum = min(minimum_free_mib, int(total_mib * .8))
    result = subprocess.run(
            ['nvidia-smi', '--id=%d' % gpu,
             '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
            capture_output=True, text=True, check=True)
    processes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if processes or free_mib < minimum:
        raise RuntimeError('GPU is unavailable: %d/%d MiB free, need %d MiB; processes: %s' %
                           (free_mib, total_mib, minimum, '; '.join(processes) or 'none'))
    return total_mib, free_mib


def gpu_memory_limit_exceeded(total_mib, free_mib):
    """Leave at least 8 GiB free and stop before 80% of the GPU is occupied."""
    return free_mib < max(8192, int(total_mib * .2))


def start_gpu_monitor(gpu, pid, interval_seconds=2):
    """Record total GPU memory and interrupt only this job near the safety limit."""
    global _gpu_monitor
    path = ROOT / 'logs' / ('gpu_usage_%d.jsonl' % pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    stopped = threading.Event()

    def sample():
        consecutive_errors = 0
        with path.open('a') as output:
            while not stopped.is_set():
                try:
                    result = subprocess.run(
                        ['nvidia-smi', '--id=%d' % gpu,
                         '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, check=True, timeout=10)
                    fields = [int(item.strip()) for item in result.stdout.strip().split(',')]
                    if len(fields) != 2:
                        raise ValueError('Invalid GPU memory sample')
                    total_mib, free_mib = fields
                    processes_result = subprocess.run(
                        ['nvidia-smi', '--id=%d' % gpu,
                         '--query-compute-apps=pid,used_gpu_memory',
                         '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, check=True, timeout=10)
                    process_rows = [line.strip().split(',', 1) for line in
                                    processes_result.stdout.splitlines() if line.strip()]
                    active_pids = [int(row[0].strip()) for row in process_rows]
                    foreign_pids = [active_pid for active_pid in active_pids if active_pid != pid]
                    memory_limit = gpu_memory_limit_exceeded(total_mib, free_mib)
                    action = ('interrupt_memory_limit' if memory_limit else
                              'yield_to_other_job' if foreign_pids else 'sample')
                    entry = dict(time_unix=time.time(), gpu=gpu, pid=pid,
                                 total_mib=total_mib, free_mib=free_mib,
                                 used_mib=total_mib-free_mib, foreign_pids=foreign_pids,
                                 action=action)
                    output.write(json.dumps(entry)+'\n')
                    output.flush()
                    consecutive_errors = 0
                    if action != 'sample':
                        os.kill(pid, signal.SIGINT)
                        return
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    consecutive_errors += 1
                    fatal = consecutive_errors >= 3
                    output.write(json.dumps(dict(time_unix=time.time(), gpu=gpu, pid=pid,
                                                 action=('interrupt_monitor_failure' if fatal else
                                                         'monitor_error'),
                                                 consecutive_errors=consecutive_errors,
                                                 error=str(exc)))+'\n')
                    output.flush()
                    if fatal:
                        os.kill(pid, signal.SIGINT)
                        return
                stopped.wait(interval_seconds)

    thread = threading.Thread(target=sample, name='uav-gap-gpu-monitor', daemon=True)
    thread.start()
    _gpu_monitor = (stopped, thread)


def require_idle_gpu(gpu=0, minimum_free_mib=32768):
    """Reserve the project GPU slot until this process exits."""
    global _gpu_lease
    if _gpu_lease is not None:
        if _gpu_lease[0] != gpu:
            raise RuntimeError('A different GPU is already reserved in this process')
        return
    lock_path = ROOT / '.cache' / ('gpu%d.lock' % gpu)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lease = lock_path.open('a+')
    try:
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another UAV Visual Gap GPU job is running') from exc
        check_gpu_available(gpu, minimum_free_mib)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        _gpu_lease = (gpu, lease)
        start_gpu_monitor(gpu, os.getpid())
    except BaseException:
        _gpu_lease = None
        lease.close()
        raise


def import_simulator():
    assert_environment()
    if 'torch' in sys.modules:
        raise RuntimeError('Isaac Gym must be imported before torch')
    import isaacgym
    expected = (ROOT / 'third_party/isaacgym').resolve()
    Path(isaacgym.__file__).resolve().relative_to(expected)
    import aerial_gym
    Path(aerial_gym.__file__).resolve().relative_to(ROOT / 'third_party/aerial_gym_simulator')
    return isaacgym, aerial_gym
