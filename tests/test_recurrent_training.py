import copy
import torch
from uav_gap.student import RecurrentStudent


def stepwise_reference(model,depth,proprio,hidden,reset):
    batch,length = depth.shape[:2]
    vision = model.depth_encoder(depth.reshape(batch*length,1,64,64)).reshape(batch,length,128)
    feature = torch.cat((vision,model.proprio_encoder(proprio)),dim=-1)
    outputs = []
    for t in range(length):
        hidden = hidden*(~reset[:,t]).to(hidden.dtype)[None,:,None]
        value,hidden = model.memory(feature[:,t:t+1],hidden)
        outputs.append(value)
    return model.head(torch.cat(outputs,dim=1)),hidden


def test_chunked_gru_preserves_actions_hidden_state_and_training_gradients():
    torch.manual_seed(404)
    model = RecurrentStudent()
    reference = copy.deepcopy(model)
    depth,proprio = torch.rand(3,64,1,64,64),torch.randn(3,64,16)
    reset = torch.zeros(3,64,dtype=torch.bool)
    reset[:2,0] = True
    hidden = torch.randn(1,3,128,requires_grad=True)
    hidden_reference = hidden.detach().clone().requires_grad_()
    prediction,state = model(depth,proprio,hidden,reset)
    expected,expected_state = stepwise_reference(reference,depth,proprio,hidden_reference,reset)
    torch.testing.assert_close(prediction,expected,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(state,expected_state,atol=1e-6,rtol=1e-5)
    target = torch.randn_like(prediction)
    (prediction-target).square().mean().backward()
    (expected-target).square().mean().backward()
    torch.testing.assert_close(hidden.grad,hidden_reference.grad,atol=1e-7,rtol=1e-4)
    assert torch.equal(hidden.grad[:,:2],torch.zeros_like(hidden.grad[:,:2]))
    for p,q in zip(model.parameters(),reference.parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-4)
