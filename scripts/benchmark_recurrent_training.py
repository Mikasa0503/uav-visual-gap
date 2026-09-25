"""Check CUDA numerical parity and wall time of fused versus stepwise GRU training."""
import copy
import json
import time
from uav_gap.runtime import require_idle_gpu,ROOT,project_output

require_idle_gpu()
import torch
from uav_gap.student import RecurrentStudent

torch.manual_seed(404)
model = RecurrentStudent().cuda()
reference = copy.deepcopy(model)
model.memory.flatten_parameters()
reference.memory.flatten_parameters()
depth,proprio = torch.rand(8,64,1,64,64,device='cuda'),torch.randn(8,64,16,device='cuda')
reset = torch.zeros(8,64,dtype=torch.bool,device='cuda')
reset[:4,0] = True
hidden = torch.randn(1,8,128,device='cuda')
target = torch.randn(8,64,4,device='cuda')


def stepwise():
    vision = reference.depth_encoder(depth.reshape(512,1,64,64)).reshape(8,64,128)
    feature = torch.cat((vision,reference.proprio_encoder(proprio)),dim=-1)
    state,outputs = hidden,[]
    for t in range(64):
        state = state*(~reset[:,t]).to(state.dtype)[None,:,None]
        output,state = reference.memory(feature[:,t:t+1],state)
        outputs.append(output)
    return reference.head(torch.cat(outputs,dim=1)),state


actual,state = model(depth,proprio,hidden,reset)
expected,expected_state = stepwise()
torch.testing.assert_close(actual,expected,atol=2e-6,rtol=1e-4)
torch.testing.assert_close(state,expected_state,atol=2e-6,rtol=1e-4)
(actual-target).square().mean().backward()
(expected-target).square().mean().backward()
gradient_error = max(float((p.grad-q.grad).abs().max()) for p,q in zip(model.parameters(),reference.parameters()))
for p,q in zip(model.parameters(),reference.parameters()):
    torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=1e-3)
results = {}
for name,module,forward in [('stepwise',reference,stepwise),('chunked',model,lambda:model(depth,proprio,hidden,reset))]:
    timings = []
    for index in range(25):
        torch.cuda.synchronize()
        started = time.perf_counter()
        module.zero_grad(set_to_none=True)
        predicted,_ = forward()
        (predicted-target).square().mean().backward()
        torch.cuda.synchronize()
        if index>=5:
            timings.append(1000*(time.perf_counter()-started))
    results[name] = dict(median_ms=sorted(timings)[len(timings)//2],iterations=len(timings))
result = dict(batch=8,steps=64,device=torch.cuda.get_device_name(),timings=results,
    max_action_error=float((actual-expected).abs().max()),max_gradient_error=gradient_error,
    speedup=results['stepwise']['median_ms']/results['chunked']['median_ms'],
    note='Both GRU weight buffers explicitly flattened. Synthetic training-window forward/backward only; excludes loading, burn-in and Adam. Evaluation path is unchanged.')
path = project_output('runs/diagnostics/recurrent_training_benchmark_flattened.json')
path.parent.mkdir(parents=True,exist_ok=True)
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
