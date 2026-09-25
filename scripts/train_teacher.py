"""Train an unassisted PPO teacher; save its initial policy before learning."""
import argparse
import json
import sys
import hashlib
from uav_gap.runtime import import_simulator, require_idle_gpu, project_output, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--num-envs', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--level', type=int, default=0)
    parser.add_argument('--name', default='teacher_seed11_pilot')
    parser.add_argument('--checkpoint')
    parser.add_argument('--perturb-reset', action='store_true')
    parser.add_argument('--recovery-reward', choices=('original','dense_v2'), default='original')
    parser.add_argument('--curriculum-fraction', type=float, default=1.)
    args = parser.parse_args()
    require_idle_gpu()
    sys.argv = [sys.argv[0]]
    import_simulator()
    import torch
    import yaml
    from rl_games.torch_runner import Runner
    from uav_gap.rl_adapter import register_task, ProgressObserver
    from uav_gap.task import TaskConfig
    directory = project_output('checkpoints/' + args.name)
    directory.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((ROOT/'configs/teacher_ppo.yaml').read_text())
    cfg['params']['seed'] = args.seed
    training = cfg['params']['config']
    training.update(name=args.name, full_experiment_name=args.name, num_actors=args.num_envs,
                    max_epochs=args.epochs, train_dir=str(ROOT/'runs/rl_games'),
                    minibatch_size=min(4096, args.num_envs*32))
    task_cfg = TaskConfig(seed=args.seed, num_envs=args.num_envs, level=args.level,
                         perturb_reset=args.perturb_reset, recovery_reward=args.recovery_reward,
                         curriculum_fraction=args.curriculum_fraction)
    register_task(task_cfg)
    observer = ProgressObserver(directory)
    runner = Runner(observer)
    runner.load(cfg)
    agent = runner.algo_factory.create(runner.algo_name, base_name='run', params=runner.params)
    (directory/'configuration.yaml').write_text(yaml.safe_dump(cfg))
    (directory/'task.json').write_text(json.dumps(vars(task_cfg), indent=2))
    sources = list((ROOT/'src').rglob('*.py')) + list((ROOT/'scripts').glob('*.py'))
    (directory/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2))
    try:
        if args.checkpoint:
            agent.restore(args.checkpoint)
            # An explicit stage request overrides the restored curriculum level.
            agent.vec_env.task.cfg.level = args.level
            observer.frames = int(agent.frame)
            agent.max_epochs = agent.epoch_num + args.epochs
        agent.save(str(directory/'initial'))
        (directory/'initial.json').write_text(json.dumps({'frames': observer.frames, 'seed': args.seed,
            'level': args.level, 'resumed_from': args.checkpoint}, indent=2))
        agent.train()
        agent.save(str(directory/'final'))
    finally:
        agent.vec_env.task.close()


if __name__ == '__main__':
    main()
