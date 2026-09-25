"""Audit actual 200Hz wall contact and velocity reversal in a continuation recording."""
import argparse,hashlib,json
import numpy as np
from uav_gap.runtime import ROOT
from uav_gap.contact_evidence import contact_response

parser=argparse.ArgumentParser();parser.add_argument('--evaluation',required=True);args=parser.parse_args()
directory=(ROOT/args.evaluation).resolve();directory.relative_to(ROOT)
path=directory/'physics_trace.npz'
with np.load(path,allow_pickle=False) as data:
    result=contact_response(data['position'][:,0],data['velocity'][:,0],data['contact_force'][:,0],float(data['dt']))
result['physics_trace_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
(directory/'contact_response_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
