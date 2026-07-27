"""GPU NN cost + memory at the user's real array sizes (movement-inclusive)."""
import os,sys,time
os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np, torch
torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False
sys.path.insert(0,"/data2/shenshan/PyCircuitSim")
from pycircuitsim.models.mosfet_directnet import NMOS_NN
CKPT=("/data2/shenshan/PyCircuitSim/external_compact_models/bsimar/"
      "checkpoints/tsmc5_dn_large_nmos_best.pt")
d=NMOS_NN(name="M",nodes=["d","g","s","b"],model_path=CKPT,L=16e-9,NFIN=10.0,tech_code=0)
d._nn_model.to("cuda"); d._device=torch.device("cuda")
for a in ("_geo_norm_t","_tech_code_tensor","_v_mean","_v_std_t","_v_min","_v_max","_clamp_beta"):
    setattr(d,a,getattr(d,a).to("cuda"))
print(f"{'array':>9} {'NMOS rows':>10} {'e2e/iter':>10} {'us/dev':>8} {'peak GB':>8} {'41 iters':>9}")
for r,c in [(32,16),(64,32),(128,64),(192,96),(256,128)]:
    N=4*r*c   # NMOS rows (4 NMOS per 6T cell)
    raw=np.random.default_rng(0).uniform(0,0.7,(N,4)).astype(np.float32)
    xg=d._geo_norm_t.unsqueeze(0).expand(N,-1).contiguous()
    tc=d._tech_code_tensor.expand(N).contiguous()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    def it():
        v=torch.from_numpy(raw).to("cuda")
        vn=d._clamp_norm_voltages(v).detach().requires_grad_(True)
        xf=torch.cat([vn,xg],dim=1)
        with torch.enable_grad():
            out=d._nn_model(xf,tech_codes=tc)
            gi=torch.autograd.grad(out[:,d._mcol("id")].sum(),vn)[0]
        return torch.cat([out.detach(),gi.detach()],dim=1).cpu().numpy()
    try:
        for _ in range(3): it()
        torch.cuda.synchronize(); ts=[]
        for _ in range(10):
            t0=time.perf_counter(); it(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
        m=np.median(ts); pk=torch.cuda.max_memory_allocated()/1e9
        print(f"{r}x{c:<5} {N:10d} {m*1e3:9.2f}ms {m/N*1e6:7.3f} {pk:7.2f} {m*41:8.2f}s")
    except RuntimeError as e:
        print(f"{r}x{c:<5} {N:10d}   OOM/err: {str(e)[:50]}")
