import torch
from time import time
from model import Net, InferenceGraph

model_name = 'b3c128nbt'
num_evals = 300
fp16 = False
cudnn_benchmark = False

device = torch.device('cuda')
net = Net()
dummy_input = lambda: torch.rand(1, 6, 12, 12, device=device)
if fp16:
    net = net.half()
    dummy_input = lambda: torch.rand(1, 6, 12, 12, device=device, dtype=torch.half)
net = net.to(device)

print()
print('Model:', model_name)
print('# params:', sum(p.numel() for p in net.parameters()))
print('Device:', device)
print('FP16:', fp16)
print('CUDNN:', torch.backends.cudnn.enabled)
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = cudnn_benchmark
    print('CUDNN Benchmark:', torch.backends.cudnn.benchmark)
print()

with torch.inference_mode():
    (policy, opp_policy), value = net(dummy_input())
    print(policy.shape, opp_policy.shape, value.shape)

start = time()
net.train()
for _ in range(num_evals):
    net.forward(dummy_input())
    torch.cuda.synchronize()
print('train nnEval/s:', num_evals / (time() - start))

start = time()
net.eval()
with torch.no_grad():
    for _ in range(num_evals):
        net.forward(dummy_input())
        torch.cuda.synchronize()
print('no_grad nnEval/s:', num_evals / (time() - start))

start = time()
net.eval()
with torch.inference_mode():
    for _ in range(num_evals):
        net.forward(dummy_input())
        torch.cuda.synchronize()
print('inference_mode nnEval/s:', num_evals / (time() - start))

print()

with torch.inference_mode():
    start = time()
    net_t = torch.jit.trace(net, dummy_input())
    net_t.forward(dummy_input())
    print(f'Tracing cost {time() - start:.3f}s')

    start = time()
    for _ in range(num_evals):
        net_t.forward(dummy_input())
        torch.cuda.synchronize()
print('inference_mode traced nnEval/s:', num_evals / (time() - start))

print()

with torch.inference_mode():
    start = time()
    net_s = torch.jit.script(net)
    net_s.forward(dummy_input())
    print(f'Scripting cost {time() - start:.3f}s')
    start = time()
    for _ in range(num_evals):
        net_s.forward(dummy_input())
        torch.cuda.synchronize()
print('inference_mode scripted nnEval/s:', num_evals / (time() - start))

print()

with torch.inference_mode():
    start = time()
    net_t = torch.jit.trace(net, dummy_input())
    net_t = torch.jit.optimize_for_inference(net_t)
    net_t.forward(dummy_input())
    print(f'Trace+optimize cost {time() - start:.3f}s')
    start = time()
    for _ in range(num_evals):
        net_t.forward(dummy_input())
        torch.cuda.synchronize()
print('inference_mode trace+optimize nnEval/s:', num_evals / (time() - start))

print()

with torch.inference_mode():
    start = time()
    net_s = torch.jit.script(net)
    net_s = torch.jit.optimize_for_inference(net_s)
    net_s.forward(dummy_input())
    print(f'Script+optimize cost {time() - start:.3f}s')
    start = time()
    for _ in range(num_evals):
        net_s.forward(dummy_input())
        torch.cuda.synchronize()
print('inference_mode script+optimize nnEval/s:', num_evals / (time() - start))

print()
start = time()
graph = torch.cuda.CUDAGraph()
inputs = dummy_input()

with torch.no_grad():
    # warmup
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            net.forward(inputs)
    torch.cuda.current_stream().wait_stream(stream)

    # capture
    with torch.cuda.graph(graph, stream=stream):
        outputs = net(inputs)

print('CUDA graph capture cost:', time() - start)

start = time()
for _ in range(num_evals):
    inputs.copy_(dummy_input())
    graph.replay()
    torch.cuda.synchronize()
print('CUDA graph nnEval/s:', num_evals / (time() - start))

graph.reset()
