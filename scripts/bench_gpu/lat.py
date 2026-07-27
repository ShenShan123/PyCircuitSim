"""V7.2.0 plan §3.4 — idle-GPU round-trip latency floor."""
import time, numpy as np, torch
torch.cuda.init()
def med(fn, n=200):
    for _ in range(20): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(n):
        t0=time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
    return np.median(ts)*1e6
a=torch.zeros(1,device='cuda')
print(f"empty sync (no work)      : {med(lambda: None):7.1f} us")
print(f"tiny kernel + sync        : {med(lambda: a.add_(1.0)):7.1f} us")
h=torch.zeros((6144,4),dtype=torch.float32,pin_memory=True)
d=torch.zeros((6144,17),dtype=torch.float32,device='cuda')
print(f"H2D 96KB pinned + sync    : {med(lambda: h.to('cuda',non_blocking=True)):7.1f} us")
print(f"D2H 408KB -> numpy + sync : {med(lambda: d.cpu().numpy()):7.1f} us")
def rt():
    x=h.to('cuda',non_blocking=True); y=(x.sum(1,keepdim=True)+d[:,:1]); y.cpu()
print(f"full round trip (6144)    : {med(rt):7.1f} us")
