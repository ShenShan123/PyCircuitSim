"""V7.2.0 plan §3.2 / §3.4. Movement-INCLUSIVE per-NR-iteration cost of the NN eval, CPU vs GPU.

Every GPU number here is a full synchronous round trip measured with
torch.cuda.synchronize() at both ends:
    build input from host floats -> H2D -> forward+autograd -> D2H -> numpy
i.e. exactly what one NR iteration of a CPU-resident solver costs.

Three H2D/D2H styles are compared:
  today  : torch.tensor(list_of_tuples, device=cuda)  ... .tolist()
  naive  : torch.from_numpy(contig).to(cuda)          ... .cpu().numpy()
  pinned : reused pinned staging buffer, non_blocking  ... .cpu().numpy()

Run with CUDA_VISIBLE_DEVICES set to an IDLE gpu.
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
sys.path.insert(0, "/data2/shenshan/PyCircuitSim")

from pycircuitsim.models.mosfet_directnet import NMOS_NN  # noqa: E402

CKPT = ("/data2/shenshan/PyCircuitSim/external_compact_models/bsimar/"
        "checkpoints/tsmc5_dn_large_nmos_best.pt")
REPS = int(os.environ.get("REPS", "30"))
CAPS = os.environ.get("CAPS", "0") == "1"   # transient: 3 backwards


def build(dev_str):
    os.environ["PYCIRCUITSIM_NN_FORCE_DEVICE"] = dev_str
    d = NMOS_NN(name="M1", nodes=["d", "g", "s", "b"], model_path=CKPT,
                L=16e-9, NFIN=10.0, tech_code=0)
    d._nn_model.to(dev_str)
    d._device = torch.device(dev_str)
    for a in ("_geo_norm_t", "_tech_code_tensor", "_v_mean", "_v_std_t",
              "_v_min", "_v_max", "_clamp_beta"):
        setattr(d, a, getattr(d, a).to(dev_str))
    return d


def evaluate(dev, x_v_norm, x_g, tech, need_caps):
    x_v = x_v_norm.detach().requires_grad_(True)
    x_full = torch.cat([x_v, x_g], dim=1)
    with torch.enable_grad():
        out = dev._nn_model(x_full, tech_codes=tech)
        gi = torch.autograd.grad(out[:, dev._mcol("id")].sum(), x_v,
                                 retain_graph=need_caps)[0]
        gqg = gqd = None
        if need_caps:
            gqg = torch.autograd.grad(out[:, dev._mcol("qg")].sum(), x_v,
                                      retain_graph=True)[0]
            gqd = torch.autograd.grad(out[:, dev._mcol("qd")].sum(), x_v,
                                      retain_graph=False)[0]
    return out, gi, gqg, gqd


def timeit(fn, sync, reps=REPS):
    for _ in range(5):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


print(f"caps={CAPS} (DC=1 bwd, tran=3 bwd)   reps={REPS}")
print(f"{'N':>7} {'CPU end2end':>12} {'GPU today':>11} {'GPU naive':>11} "
      f"{'GPU pinned':>11} {'pinned vs CPU':>13}")

for N in (384, 1536, 6144, 24576):
    rng = np.random.default_rng(0)
    vdd = 0.7
    raw_np = np.stack([
        rng.choice([0.0, vdd], N) + rng.normal(0, .02, N),
        rng.choice([0.0, vdd], N) + rng.normal(0, .02, N),
        np.zeros(N), np.zeros(N)], axis=1).astype(np.float32)
    raw_list = [tuple(r) for r in raw_np.tolist()]

    # ---------- CPU ----------
    dc = build("cpu")
    xg_c = dc._geo_norm_t.unsqueeze(0).expand(N, -1)
    tc_c = dc._tech_code_tensor.expand(N)

    def cpu_iter():
        v = torch.tensor(raw_list, dtype=torch.float32)
        vn = dc._clamp_norm_voltages(v)
        out, gi, gqg, gqd = evaluate(dc, vn, xg_c, tc_c, CAPS)
        o = out.detach().numpy()
        g = gi.detach().numpy()
        return o, g

    t_cpu = timeit(cpu_iter, lambda: None, reps=max(5, REPS // 3))

    # ---------- GPU ----------
    dg = build("cuda")
    xg_g = dg._geo_norm_t.unsqueeze(0).expand(N, -1).contiguous()
    tc_g = dg._tech_code_tensor.expand(N).contiguous()
    sync = torch.cuda.synchronize
    stage = torch.empty((N, 4), dtype=torch.float32, pin_memory=True)

    def gpu_today():
        v = torch.tensor(raw_list, dtype=torch.float32, device="cuda")
        vn = dg._clamp_norm_voltages(v)
        out, gi, _, _ = evaluate(dg, vn, xg_g, tc_g, CAPS)
        return out.detach().tolist(), gi.detach().tolist()

    def gpu_naive():
        v = torch.from_numpy(raw_np).to("cuda")
        vn = dg._clamp_norm_voltages(v)
        out, gi, _, _ = evaluate(dg, vn, xg_g, tc_g, CAPS)
        return out.detach().cpu().numpy(), gi.detach().cpu().numpy()

    def gpu_pinned():
        stage.copy_(torch.from_numpy(raw_np))
        v = stage.to("cuda", non_blocking=True)
        vn = dg._clamp_norm_voltages(v)
        out, gi, _, _ = evaluate(dg, vn, xg_g, tc_g, CAPS)
        blk = torch.cat([out.detach(), gi.detach()], dim=1)
        return blk.cpu().numpy()

    t_today = timeit(gpu_today, sync, reps=max(5, REPS // 3))
    t_naive = timeit(gpu_naive, sync)
    t_pin = timeit(gpu_pinned, sync)

    print(f"{N:7d} {t_cpu*1e3:10.2f}ms {t_today*1e3:9.2f}ms "
          f"{t_naive*1e3:9.2f}ms {t_pin*1e3:9.2f}ms {t_cpu/t_pin:12.1f}x")

    del dg
    torch.cuda.empty_cache()
