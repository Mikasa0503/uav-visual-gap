"""Measured near-wall contact and subsequent velocity reversal, without causal overclaim."""
import numpy as np


def contact_response(position,velocity,force,dt=.005,threshold=.05,window_seconds=.15):
    position,velocity,force=map(np.asarray,(position,velocity,force))
    if position.shape!=velocity.shape or force.shape!=position.shape or position.ndim!=2 or position.shape[1]!=3:
        raise ValueError('Expected matching T x 3 recorded physical arrays')
    if dt<=0 or not all(np.isfinite(a).all() for a in (position,velocity,force)):
        raise ValueError('Invalid physical samples')
    # Gate plane normal is world x; avoid labeling floor friction as wall contact.
    wall=(np.abs(position[:,0])<.65)&(position[:,2]>.4)&(np.abs(force[:,0])>threshold)&(np.abs(force[:,0])>np.abs(force[:,2]))
    indices=np.flatnonzero(wall)
    if not len(indices): return dict(wall_contact=False,wall_velocity_reversal=False)
    i=int(indices[0]);end=min(len(position),i+int(round(window_seconds/dt))+1)
    before=float(velocity[max(0,i-1),0]);direction=np.sign(before)
    opposite=np.flatnonzero(velocity[i:end,0]*direction < -threshold) if abs(before)>threshold else np.array([],dtype=int)
    reverse=int(i+opposite[0]) if len(opposite) else None
    return dict(wall_contact=True,first_wall_contact_seconds=(i+1)*dt,
        first_wall_contact_position=position[i].tolist(),first_wall_contact_force_n=force[i].tolist(),
        vx_before_contact=before,vx_first_contact=float(velocity[i,0]),response_window_seconds=window_seconds,
        vx_min_in_window=float(velocity[i:end,0].min()),vx_max_in_window=float(velocity[i:end,0].max()),
        wall_velocity_reversal=reverse is not None,
        first_opposite_vx_seconds=(reverse+1)*dt if reverse is not None else None,
        note='Measured contact and subsequent closed-loop motion, not an isolated restitution measurement.')
