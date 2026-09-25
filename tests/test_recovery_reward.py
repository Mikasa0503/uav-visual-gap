import torch
from uav_gap.recovery_reward import recovery_feedback


def test_before_crossing_has_no_attitude_or_goal_bias():
    x=torch.tensor([0.,1.,100.])
    assert torch.equal(recovery_feedback(torch.zeros(3,dtype=torch.bool),x,x,-torch.ones(3)),torch.zeros(3))


def test_independent_feedback_remains_informative_and_bounded():
    cross=torch.ones(4,dtype=torch.bool)
    distance=torch.tensor([1.,.5,1.,1.])
    speed=torch.tensor([1.,1.,.5,1.])
    upright=torch.tensor([.5,.5,.5,1.])
    reward=recovery_feedback(cross,distance,speed,upright)
    assert (reward[1:]>reward[0]).all()
    extremes=recovery_feedback(cross,torch.ones(4)*1e6,torch.ones(4)*1e6,torch.tensor([-1.,0.,1.,1.]))
    assert (extremes>=-.110001).all() and (extremes<=0).all()
    assert recovery_feedback(cross,torch.zeros(4),torch.zeros(4),torch.ones(4)).eq(0).all()
