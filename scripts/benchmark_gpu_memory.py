"""Bounded real-simulator memory calibration; no policy or training claim."""
import argparse
import json
import os
import subprocess
import sys
import time

from uav_gap.runtime import ROOT, import_simulator, require_idle_gpu


def sampled_gpu_memory(pid, since_unix):
    """Summarize this run's whole-device monitor samples, not PyTorch memory."""
    path = ROOT/'logs'/('gpu_usage_%d.jsonl' % pid)
    if not path.exists():
        return dict(samples=0, peak_used_mib=None, minimum_free_mib=None)
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get('time_unix', 0) >= since_unix and row.get('action') == 'sample':
            rows.append(row)
    return dict(samples=len(rows),
                peak_used_mib=max((row['used_mib'] for row in rows), default=None),
                minimum_free_mib=min((row['free_mib'] for row in rows), default=None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, required=True)
    parser.add_argument('--steps', type=int, default=30)
    args = parser.parse_args()
    if args.num_envs < 1 or not 1 <= args.steps <= 100:
        raise ValueError('Calibration requires positive env count and 1-100 steps')
    run_start_unix = time.time()
    require_idle_gpu()
    sys.argv = [sys.argv[0]]
    import_simulator()
    import torch
    from uav_gap.task import GapTask, TaskConfig

    start = time.monotonic()
    task = GapTask(TaskConfig(num_envs=args.num_envs, level=3, cameras=True,
                              randomize=False, auto_reset=False))
    try:
        actions = torch.zeros(args.num_envs, 4, device='cuda:0')
        for _ in range(args.steps):
            task.step(actions)
            observation = task.student_obs()
            if not torch.isfinite(observation['depth']).all():
                raise RuntimeError('Nonfinite depth in calibration')
        torch.cuda.synchronize()
        result = subprocess.run(
            ['nvidia-smi', '--id=0', '--query-gpu=memory.total,memory.used,memory.free',
             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        total, used, free = [int(item.strip()) for item in result.stdout.strip().split(',')]
        sampled = sampled_gpu_memory(os.getpid(), run_start_unix)
        report = dict(kind='gpu_memory_calibration_not_training', num_envs=args.num_envs,
                      steps=args.steps, cameras=True, physics_hz=200, policy_hz=50,
                      gpu_total_mib=total, gpu_used_mib_at_end=used, gpu_free_mib_at_end=free,
                      gpu_sample_count=sampled['samples'],
                      gpu_sampled_peak_used_mib=max(used, sampled['peak_used_mib'] or 0),
                      gpu_sampled_minimum_free_mib=min(free, sampled['minimum_free_mib'] or free),
                      torch_peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20, 1),
                      torch_peak_reserved_mib=round(torch.cuda.max_memory_reserved()/2**20, 1),
                      elapsed_seconds=round(time.monotonic()-start, 3))
        output = ROOT/'runs/diagnostics'/('gpu_memory_%04d_envs.json' % args.num_envs)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        task.close()


if __name__ == '__main__':
    main()
