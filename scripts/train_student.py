"""Fixed-update BC/DAgger supervised training over immutable rollout datasets."""
import argparse
import hashlib
import json
import time
from uav_gap.runtime import ROOT, project_output, require_idle_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--kind', choices=['gru', 'stack4'], default='gru')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--updates', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, help='Explicit round override; otherwise preserve resumed Adam rate')
    parser.add_argument('--checkpoint')
    parser.add_argument('--name', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if min(args.updates, args.batch_size, args.save_every) <= 0:
        raise ValueError('Training budgets must be positive')
    if args.learning_rate is not None and not 0 < args.learning_rate < 1:
        raise ValueError('Learning rate must be finite and between zero and one')
    if args.device.startswith('cuda'):
        require_idle_gpu()
    import torch
    from torch.utils.data import DataLoader
    from uav_gap.student import make_student
    from uav_gap.dataset import EpisodeWindows, collate_windows, supervised_loss

    torch.manual_seed(args.seed)
    paths = [(ROOT/path).resolve() for path in args.datasets]
    for path in paths:
        path.relative_to(ROOT)
    dataset = EpisodeWindows(paths, length=64)
    delays = {m['observation_delay_steps'] for m in dataset.manifests}
    if len(delays) != 1:
        raise ValueError('All aggregation rounds must have the same sensor delay')
    directory = project_output('checkpoints/'+args.name)
    directory.mkdir(parents=True, exist_ok=False)
    model = make_student(args.kind).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    updates, presentations = 0, 0
    if args.checkpoint:
        path = (ROOT/args.checkpoint).resolve()
        path.relative_to(ROOT)
        checkpoint = torch.load(path, map_location=args.device)
        if checkpoint['kind'] != args.kind or checkpoint['seed'] != args.seed:
            raise ValueError('Student kind/seed must match continuation checkpoint')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        updates, presentations = checkpoint['updates'], checkpoint['label_presentations']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if args.device.startswith('cuda'):
            torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint['cuda_rng']])
    if args.learning_rate is not None:
        for group in optimizer.param_groups:
            group['lr'] = args.learning_rate
    # New deterministic sampler per round; round boundaries are deliberate, not
    # advertised as bitwise resume of an interrupted minibatch iterator.
    generator = torch.Generator().manual_seed(args.seed+updates)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
                        num_workers=0, collate_fn=collate_windows)
    iterator = iter(loader)
    metadata = dict(vars(args), observation_delay_steps=delays.pop(), sequence_length=64,
        effective_learning_rate=optimizer.param_groups[0]['lr'],
        parameters=sum(p.numel() for p in model.parameters()),
        architecture='gru_v1' if args.kind=='gru' else 'stack4_capacitymatched_v2',
        burn_in='entire causal episode prefix, no gradient',
        input_schema=dict(depth=[1,64,64], proprio=16),
        datasets_sha256={str(p.relative_to(ROOT)): hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()
                         for p in paths}, unique_labels=sum(m['labels'] for m in dataset.manifests),
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (ROOT/'src/uav_gap').glob('*.py')})
    (directory/'configuration.json').write_text(json.dumps(metadata, indent=2))

    def save(name):
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), kind=args.kind,
            seed=args.seed, updates=updates, label_presentations=presentations,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()
                if args.device.startswith('cuda') else [], configuration=metadata), directory/(name+'.pth'))
        (directory/(name+'.json')).write_text(json.dumps(dict(updates=updates,
            label_presentations=presentations, unique_labels=metadata['unique_labels']), indent=2))

    save('initial')
    start_time = time.monotonic()
    for local_update in range(1, args.updates+1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {k:v.to(args.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
        model.train()
        loss = supervised_loss(model, batch)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite supervised loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite gradient norm')
        optimizer.step()
        updates += 1
        presentations += int(batch['mask'].sum())
        record = dict(update=updates, mse=float(loss.detach()), grad_norm=float(norm),
            label_presentations=presentations, elapsed_seconds=time.monotonic()-start_time)
        with (directory/'progress.jsonl').open('a') as output:
            output.write(json.dumps(record)+'\n')
        if local_update <= 10 or local_update % args.save_every == 0:
            save('snapshot_%06d' % updates)
            print(json.dumps(record), flush=True)
    save('final')


if __name__ == '__main__':
    main()
