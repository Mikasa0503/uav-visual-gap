import numpy as np
from uav_gap.contact_evidence import contact_response


def test_reversal_without_wall_contact_is_not_a_collision():
    p=np.tile([.2,0.,1.5],(50,1));v=np.zeros_like(p);v[:10,0]=3.;v[10:,0]=-1.
    assert contact_response(p,v,np.zeros_like(p))==dict(wall_contact=False,wall_velocity_reversal=False)


def test_measured_wall_contact_and_later_reversal():
    p=np.tile([.2,0.,1.5],(50,1));v=np.zeros_like(p);v[:,0]=-2.;v[15:,0]=.8
    f=np.zeros_like(p);f[10,0]=75.
    r=contact_response(p,v,f)
    assert r['wall_contact'] and r['wall_velocity_reversal']
    assert r['first_wall_contact_seconds']==.055
    assert r['first_opposite_vx_seconds']==.08
    assert r['vx_before_contact']==-2.


def test_floor_contact_is_not_misclassified_as_wall():
    p=np.tile([.2,0.,.1],(50,1));v=np.zeros_like(p);f=np.tile([2.,0.,100.],(50,1))
    assert not contact_response(p,v,f)['wall_contact']
