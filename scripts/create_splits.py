"""Predeclare disjoint validation/test manifests before learned evaluation."""
import json
import hashlib
import numpy as np
import yaml
from uav_gap.runtime import ROOT, project_output

out = project_output('configs/scenes')
out.mkdir(exist_ok=True)
for split, seed in [('validation', 22092026), ('test', 23092026)]:
    rng = np.random.default_rng(seed)
    result = []
    for condition in ('show', 'heldout', 'delay_40ms', 'depth_loss_3', 'mass_plus20', 'lateral_impulse'):
        for i in range(100):
            start = [-3 + rng.uniform(-.2, .2), rng.uniform(-.15, .15), 1.5 + rng.uniform(-.1, .1)]
            center = [0., 0., 1.5]
            angle = 45.
            if condition == 'heldout':
                center = [0., rng.uniform(-.15, .15), 1.5+rng.uniform(-.1, .1)]
                angle += rng.uniform(-5., 5.)
            result.append(dict(id='%s_%s_%03d' % (split, condition, i), condition=condition,
                               gate_center=center, geometry=[.66, .24, angle], start=start,
                               velocity=rng.uniform(-.08, .08, 3).tolist()))
    path = out / (split + '.json')
    text = json.dumps({'seed': seed, 'scenarios': result}, indent=2)
    if path.exists() and path.read_text() != text:
        raise RuntimeError('Refusing to overwrite a frozen scene manifest: ' + str(path))
    path.write_text(text)
    print(path, len(result))

# The film uses the nominal show scene frozen before policy training, not a
# validation case selected because a particular checkpoint happened to succeed.
definition = ROOT/'configs/task.yaml'
canonical = yaml.safe_load(definition.read_text())['show_scene']
showcase = dict(seed=43022026, purpose='Nominal fixed film scenario, not a statistical evaluation',
    definition=str(definition.relative_to(ROOT)), definition_sha256=hashlib.sha256(definition.read_bytes()).hexdigest(),
    scenarios=[dict(id='showcase_nominal',condition='show',**canonical)])
path = out/'showcase.json'
text = json.dumps(showcase,indent=2)
if path.exists() and path.read_text()!=text:
    raise RuntimeError('Refusing to change the frozen film scene')
path.write_text(text)
print(path,1)
