"""V7.2.0 plan §2.2 / §8.1. Phase-2a feasibility probe: scalar `_unpack_eval` tail vs a vectorised
numpy equivalent, on the REAL production DirectNet checkpoint + real device
objects. Measures (a) speed, (b) BIT-IDENTITY of the vectorised form.

Not a gate; a plan-input measurement.
"""
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, "/data2/shenshan/PyCircuitSim")

from pycircuitsim.models.mosfet_directnet import NMOS_NN  # noqa: E402

N = int(os.environ.get("N", "6144"))
rng = np.random.default_rng(0)

CKPT = ("/data2/shenshan/PyCircuitSim/external_compact_models/bsimar/"
        "checkpoints/tsmc5_dn_large_nmos_best.pt")

dev = NMOS_NN(
    name="M1", nodes=["d", "g", "s", "b"], model_path=CKPT,
    L=16e-9, NFIN=10.0, tech_code=0,
)
print(f"device ok: {type(dev).__name__}  N={N}")

# --- Build a realistic batch of raw voltages (SRAM-ish: many off devices) ---
vdd = 0.7
v_d = rng.choice([0.0, vdd], size=N) + rng.normal(0, 0.02, N)
v_g = rng.choice([0.0, vdd], size=N) + rng.normal(0, 0.02, N)
v_s = np.zeros(N)
v_b = np.zeros(N)
raw = list(zip(v_d.tolist(), v_g.tolist(), v_s.tolist(), v_b.tolist()))

x_v = torch.tensor([[d, g, s, b] for d, g, s, b in raw], dtype=torch.float32)
v_norm = dev._clamp_norm_voltages(x_v)
x_v = v_norm.detach().requires_grad_(True)
x_g = dev._geo_norm_t.unsqueeze(0).expand(N, -1)
x_full = torch.cat([x_v, x_g], dim=1)
tech = dev._tech_code_tensor.expand(N)

t0 = time.perf_counter()
with torch.enable_grad():
    out = dev._nn_model(x_full, tech_codes=tech)
    grad_id = torch.autograd.grad(out[:, dev._mcol("id")].sum(), x_v)[0]
t_tensor = time.perf_counter() - t0
print(f"tensor (fwd+1 bwd), N={N}: {t_tensor*1e3:.1f} ms")

# --- A: scalar tail, exactly as shipped -------------------------------------
t0 = time.perf_counter()
res_scalar = []
for i in range(N):
    res_scalar.append(
        dev._unpack_eval(out[i], grad_id[i], None, None, raw[i][0], raw[i][2]))
t_scalar = time.perf_counter() - t0
print(f"scalar tail  : {t_scalar*1e3:8.1f} ms  ({t_scalar/N*1e6:.1f} us/device)")

# --- B: vectorised tail, numpy, float64 -------------------------------------
o = out.detach().cpu().numpy().astype(np.float64)
gi = grad_id.detach().cpu().numpy().astype(np.float64)
vds_arr = np.array([r[0] - r[2] for r in raw])


def vec_tail(o, gi, vds_arr, dev, exp_mode="np"):
    mc = dev._mcol
    sc = dev._stats_idx  # noqa: F841

    def denorm(name, val):
        u = val * dev._out_std_f[name] + dev._out_mean_f[name]
        if dev._asinh_f:
            return dev._asinh_f[name] * np.sinh(u)
        return u

    def denorm_deriv(out_name, in_col, dn, phys):
        in_std = dev._in_std_f[in_col]
        if in_std < 1e-12:
            return np.zeros_like(dn)
        out_std = dev._out_std_f[out_name]
        if dev._asinh_f:
            s = dev._asinh_f[out_name]
            fac = np.sqrt(s * s + phys * phys)
        else:
            fac = 1.0
        return dn * out_std * fac / in_std

    id_p = denorm("id", o[:, mc("id")])
    gm = -denorm_deriv("id", 1, gi[:, 1], id_p)
    gds = -denorm_deriv("id", 0, gi[:, 0], id_p)
    gmb = -denorm_deriv("id", 3, gi[:, 3], id_p)

    from pycircuitsim.models.mosfet_nn import _GDS_GUARD_K
    gds = np.where(gds > 0.0, gds, np.maximum(np.abs(id_p) * _GDS_GUARD_K, 1e-12))

    # --- Vds correction, vectorised ---
    VDD = dev._vdd_estimate
    VT = max(0.06 * VDD, 0.026)
    a = np.abs(vds_arr)
    normal = (vds_arr < 0.0) if dev._is_pmos else (vds_arr > 0.0)

    over = a - VDD
    m_ext = a > VDD
    g_max, x_ref = 1.0e-3, 0.5 * VDD
    x_cap = 5.0 * x_ref
    id_extra = np.where(
        over <= x_cap,
        0.5 * g_max * over * over / x_ref,
        0.5 * g_max * x_cap * x_cap / x_ref
        + (g_max * x_cap / x_ref) * (over - x_cap))
    g_extra = np.where(over <= x_cap, g_max * over / x_ref, g_max * x_cap / x_ref)
    m_ext_n = m_ext & normal
    if dev._is_pmos:
        id_p = np.where(m_ext_n, id_p + id_extra, id_p)
    else:
        id_p = np.where(m_ext_n, id_p - id_extra, id_p)
    gds = np.where(m_ext, np.maximum(gds, g_extra), gds)

    fast = normal & (a > 20.0 * VT)
    if exp_mode == "np":
        exp_sym = np.where(a <= 20.0 * VT, np.exp(-a / VT), 0.0)
    else:  # scalar libm loop, only where needed
        exp_sym = np.zeros_like(a)
        need = np.flatnonzero(a <= 20.0 * VT)
        import math
        arg = (-a[need] / VT).tolist()
        exp_sym[need] = [math.exp(v) for v in arg]
    f_sym = 1.0 - exp_sym

    x0, x1 = 0.20 * VDD, 0.30 * VDD
    u = np.clip((a - x0) / (x1 - x0), 0.0, 1.0)
    taper = np.where(a <= x0, 1.0, np.where(a >= x1, 0.0,
                                            1.0 - u * u * (3.0 - 2.0 * u)))
    f_id = np.where(normal, f_sym, f_sym * taper)

    id_raw = id_p
    id_new = id_raw * f_id
    gm_new = gm * f_id
    gmb_new = gmb * f_id
    gds_new = gds * f_sym + np.abs(id_raw) * exp_sym / VT
    gds_new = np.where(gds_new > 0.0, gds_new,
                       np.maximum(np.abs(id_new) * _GDS_GUARD_K, 1e-12))

    if dev._is_pmos:
        wrong = np.where(normal, id_new < 0.0, id_new > 0.0)
    else:
        wrong = np.where(normal, id_new > 0.0, id_new < 0.0)
    id_new = np.where(wrong, 0.0, id_new)
    gm_new = np.where(wrong, 0.0, gm_new)
    gmb_new = np.where(wrong, 0.0, gmb_new)

    # fast-path rows keep pre-correction values
    id_f = np.where(fast, id_raw, id_new)
    gm_f = np.where(fast, gm, gm_new)
    gmb_f = np.where(fast, gmb, gmb_new)
    gds_f = np.where(fast, gds, gds_new)
    return id_f, gm_f, gds_f, gmb_f


for mode in ("np", "libm"):
    t0 = time.perf_counter()
    idv, gmv, gdsv, gmbv = vec_tail(o, gi, vds_arr, dev, exp_mode=mode)
    t_vec = time.perf_counter() - t0
    ids = np.array([r["id"] for r in res_scalar])
    gms = np.array([r["gm"] for r in res_scalar])
    gdss = np.array([r["gds"] for r in res_scalar])
    exact = int((idv == ids).sum())
    print(f"vector tail ({mode:4s}): {t_vec*1e3:8.2f} ms "
          f"({t_scalar/t_vec:6.1f}x faster)  id bit-exact "
          f"{exact}/{N} ({100*exact/N:.2f}%)  "
          f"gm {int((gmv==gms).sum())}/{N}  gds {int((gdsv==gdss).sum())}/{N}  "
          f"max|rel id| {np.max(np.abs(idv-ids)/np.maximum(np.abs(ids),1e-30)):.2e}")
