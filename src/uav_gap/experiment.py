"""Explicit, equal-budget visual experiment schedules without simulator imports."""
import re

GROUPS = ('bc_gru', 'dagger_gru', 'dagger_stack4')


def build_rounds(config, name, group, seed, teacher):
    if group not in GROUPS or not re.fullmatch(r'[a-zA-Z0-9_-]+', name):
        raise ValueError('Invalid experiment group/name')
    for key in ('rounds','batch_size','save_every','num_envs','validation_count'):
        if not isinstance(config[key],int) or config[key] <= 0:
            raise ValueError('Positive integer required for '+key)
    budgets = {}
    for key in ('labels_per_round','updates_per_round'):
        values = config[key] if isinstance(config[key],list) else [config[key]]*config['rounds']
        if len(values)!=config['rounds'] or any(type(v) is not int or v<=0 for v in values):
            raise ValueError('One positive integer budget per round is required for '+key)
        budgets[key] = values
    kind = 'stack4' if group=='dagger_stack4' else 'gru'
    data = list(config.get('initial_datasets',[]))
    if len(data) != len(set(data)):
        raise ValueError('Duplicate initial datasets')
    result, previous = [], config.get('initial_checkpoint')
    start_round = config.get('start_round',0)
    base_labels, base_updates = config.get('initial_unique_labels',0), config.get('initial_updates',0)
    if (data or start_round or base_labels or base_updates) and previous is None:
        raise ValueError('Continuation requires an initial checkpoint')
    for local_index in range(config['rounds']):
        round_index = start_round+local_index
        stem = '%s_seed%d_%s_r%d' % (name,seed,group,round_index)
        # All groups share the identical teacher rollout set for round zero.
        data_name = '%s_seed%d_shared_r0' % (name,seed) if previous is None else stem
        data.append('datasets/'+data_name)
        collect = ['scripts/collect_demonstrations.py','--teacher',teacher,'--level','3',
            '--label-budget',str(budgets['labels_per_round'][local_index]),'--num-envs',str(config['num_envs']),
            '--seed',str(41000+seed+1000*round_index),'--name',data_name,
            '--observation-delay-steps',str(config['observation_delay_steps'])]
        if config['perturb_reset']:
            collect.append('--perturb-reset')
        if previous is not None and group.startswith('dagger'):
            collect += ['--student', previous]
        train = ['scripts/train_student.py','--datasets']+data+['--kind',kind,'--seed',str(seed),
            '--updates',str(budgets['updates_per_round'][local_index]),'--batch-size',str(config['batch_size']),
            '--save-every',str(config['save_every']),'--name',stem]
        if previous is not None:
            train += ['--checkpoint', previous]
        if 'learning_rates' in config:
            rates = config['learning_rates']
            if len(rates) != config['rounds'] or any(not 0 < rate < 1 for rate in rates):
                raise ValueError('One valid learning rate is required per round')
            train += ['--learning-rate', str(rates[local_index])]
        checkpoint = 'checkpoints/'+stem+'/final.pth'
        evaluate = [['scripts/evaluate_student.py','--checkpoint',checkpoint,'--level','3',
            '--split','validation','--condition',condition,'--count',str(config['validation_count']),
            '--name',stem+'_'+condition,'--record'] for condition in ('show','heldout')]
        result.append(dict(index=round_index, stem=stem, data_name=data_name, datasets=list(data),
            checkpoint=checkpoint, collect=collect, train=train, evaluate=evaluate,
            cumulative_unique_labels=base_labels+sum(budgets['labels_per_round'][:local_index+1]),
            cumulative_updates=base_updates+sum(budgets['updates_per_round'][:local_index+1])))
        previous = checkpoint
    return result
