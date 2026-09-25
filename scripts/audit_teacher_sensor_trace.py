"""Verify executed teacher sensor timing against recorded raw measurements."""
import argparse
import hashlib
import json
import numpy as np
from uav_gap.runtime import ROOT

parser = argparse.ArgumentParser()
parser.add_argument('--evaluation',required=True)
args = parser.parse_args()
directory = (ROOT/args.evaluation).resolve()
directory.relative_to(ROOT)
summary = json.loads((directory/'summary.json').read_text())
delay = summary['stress_protocol']['observation_delay_steps']
path = directory/'trace.npz'
with np.load(path,allow_pickle=False) as data:
    raw,actual = data['raw_teacher_observation'],data['teacher_observation']
    expected = raw[np.maximum(0,np.arange(len(raw))-delay)].copy()
    expected[:,:,12:16] = raw[:,:,12:16]
    np.testing.assert_array_equal(actual,expected)
    result = dict(exact=True,frames=len(raw),episodes=raw.shape[1],delay_steps=delay,
                  applicability=summary['stress_protocol']['applicability'],
                  trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(directory/'sensor_trace_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
