#!/usr/bin/env python3
"""V6.7 decision gate — charge-derivative (autograd dQ/dV) cap fidelity.

The simulator's small-signal capacitances are NOT the network's predicted
``cgg/cgd/cdg/cdd`` *columns* — they are the **autograd** derivatives of the
predicted terminal charges ``qg/qd`` (``mosfet_nn._unpack_eval``):

    cgg_sim = +∂qg/∂Vg     cgd_sim = +∂qg/∂Vd
    cdg_sim = +∂qd/∂Vg     cdd_sim = +∂qd/∂Vd       (raw autograd, NO flip)

The AC/transient solvers consume exactly these. The OSDI ground-truth caps in
the dataset use the SPICE condensed convention (``pycmg/model.py:_condense_caps``
negates the off-diagonals):

    cgg_data = +∂Qg/∂Vg    cdd_data = +∂Qd/∂Vd        (diagonal, no flip)
    cgd_data = −∂Qg/∂Vd    cdg_data = −∂Qd/∂Vg        (off-diag, SPICE-negated)

So the *signed* comparison the simulator actually cares about is:

    autograd ∂qg/∂Vg  vs  +cgg_data      (no flip)
    autograd ∂qg/∂Vd  vs  −cgd_data      (FLIP)
    autograd ∂qd/∂Vg  vs  −cdg_data      (FLIP)
    autograd ∂qd/∂Vd  vs  +cdd_data      (no flip)

This diagnostic loads a trained DirectNet checkpoint + its training dataset and,
over (a) the full grid, (b) a saturation subset (the AC CS-amp pole region) and
(c) a low-Vds / charging subset (the switchcap transmission-gate region),
reports per-channel:

  * the **signed ratio** median(autograd / signed_target)  — should be ~+1 if
    the autograd cap matches OSDI; a value <1 means the NN UNDER-predicts the
    cap (the f3db>1 / switchcap-over-charge failure mode), a sign≠+ means a
    convention bug,
  * the median |rel err| of the autograd cap vs the signed OSDI target,
  * the median |rel err| of the DIRECT predicted column vs the |OSDI| target
    (to separate "the value head is fine but the autograd drifts" — the S10
    analog, which says a charge-Sobolev term is the fix — from "the value head
    itself is wrong" — which would say up-weight the columns instead).

A batched autograd evaluator replicates ``_unpack_eval``'s cap path exactly and
is sanity-checked against the real ``NMOS_NN.get_capacitances`` device at the top
of the run (asserts < 1e-4 rel agreement) so the bulk numbers are trustworthy.

Ground truth is ALWAYS the OSDI dataset caps (CLAUDE.md Validation rule).

Usage:
    python tests/diag_charge_cap_fidelity.py --tech tsmc5 --dev nmos --size medium
    python tests/diag_charge_cap_fidelity.py            # all techs/devs, medium
"""
from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

print = functools.partial(print, flush=True)  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from bsimar.config import CHECKPOINT_DIR, DATA_DIR, local_variant_code
from bsimar.data.normalize import NormStats, OUTPUT_COLUMN_ORDER
from bsimar.models.direct_net import DirectNet

_OC = {n: i for i, n in enumerate(OUTPUT_COLUMN_ORDER)}

# (channel name, charge head, voltage input col, dataset target col, sign)
#   autograd ∂q_head/∂V[vcol]  compared to  sign * data[target_col]
CAP_CHANNELS: List[Tuple[str, str, int, str, float]] = [
    ("cgg", "qg", 1, "cgg", +1.0),   # ∂qg/∂Vg  vs  +cgg
    ("cgd", "qg", 0, "cgd", -1.0),   # ∂qg/∂Vd  vs  −cgd
    ("cdg", "qd", 1, "cdg", -1.0),   # ∂qd/∂Vg  vs  −cdg
    ("cdd", "qd", 0, "cdd", +1.0),   # ∂qd/∂Vd  vs  +cdd
]


def _build_model(state: Dict[str, torch.Tensor]) -> DirectNet:
    net_keys = [k for k in state if k.startswith("net.") and k.endswith(".weight")]
    output_dim = state[net_keys[-1]].shape[0]
    hidden_dim = state[net_keys[-1]].shape[1]
    n_layers = len(net_keys) - 1
    num_tech_codes = state["tech_embedding.weight"].shape[0]
    tech_embed_dim = state["tech_embedding.weight"].shape[1]
    input_dim = state[net_keys[0]].shape[1] - tech_embed_dim
    m = DirectNet(
        input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers,
        output_dim=output_dim, num_tech_codes=num_tech_codes,
        tech_embed_dim=tech_embed_dim)
    m.load_state_dict(state)
    m.eval()
    return m


class _BatchedCapEval:
    """Faithful batched replica of ``_MOSFETNNBase`` cap autograd path."""

    def __init__(self, ckpt: Path, norm: Path):
        state = torch.load(str(ckpt), weights_only=True, map_location="cpu")
        self.model = _build_model(state)
        self.s = NormStats.load(str(norm))
        v_std = self.s.input_std[:4].copy(); v_std[v_std < 1e-12] = 1.0
        self.v_mean = torch.tensor(self.s.input_mean[:4], dtype=torch.float32)
        self.v_std = torch.tensor(v_std, dtype=torch.float32)
        self.v_min = torch.tensor(self.s.input_min[:4], dtype=torch.float32)
        self.v_max = torch.tensor(self.s.input_max[:4], dtype=torch.float32)
        vr = torch.clamp(self.v_max - self.v_min, min=0.01)
        self.beta = 1.0 / (0.05 * vr)
        gstd = self.s.input_std[4:7].copy(); gstd[gstd < 1e-12] = 1.0
        self.g_mean = self.s.input_mean[4:7]
        self.g_std = gstd

    def _clamp_norm(self, v_raw: torch.Tensor) -> torch.Tensor:
        beta = self.beta
        bx = beta * (v_raw - self.v_min)
        vc = self.v_min + torch.where(
            bx > 20.0, v_raw - self.v_min, torch.log1p(torch.exp(bx)) / beta)
        bx2 = beta * (self.v_max - vc)
        vc = self.v_max - torch.where(
            bx2 > 20.0, self.v_max - vc, torch.log1p(torch.exp(bx2)) / beta)
        return (vc - self.v_mean) / self.v_std

    def caps(self, V: np.ndarray, geo: np.ndarray, tcode: np.ndarray
             ) -> Dict[str, np.ndarray]:
        """V:(B,4) raw volts [Vd,Vg,Vs,Vb]; geo:(B,3) [nfin_log2,L,T];
        tcode:(B,) local codes. Returns autograd caps + direct columns + qg/qd."""
        v_raw = torch.tensor(V, dtype=torch.float32)
        x_v = self._clamp_norm(v_raw).requires_grad_(True)
        x_g = torch.tensor((geo - self.g_mean) / self.g_std, dtype=torch.float32)
        x = torch.cat([x_v, x_g], dim=1)
        tc = torch.tensor(tcode, dtype=torch.long)
        with torch.enable_grad():
            out = self.model(x, tech_codes=tc)
            gqg = torch.autograd.grad(out[:, _OC["qg"]].sum(), x_v,
                                      retain_graph=True)[0]
            gqd = torch.autograd.grad(out[:, _OC["qd"]].sum(), x_v)[0]
        out_np = out.detach().numpy()

        def denorm(name: str, val_norm: np.ndarray) -> np.ndarray:
            i = _OC[name]
            u = val_norm * self.s.output_std[i] + self.s.output_mean[i]
            if self.s.mode == "asinh":
                return self.s.asinh_scale[i] * np.sinh(u)
            return u

        def dderiv(name: str, ic: int, dn: np.ndarray, yp: np.ndarray) -> np.ndarray:
            i = _OC[name]
            in_std = float(self.s.input_std[ic])
            if in_std < 1e-12:
                return np.zeros_like(dn)
            out_std = float(self.s.output_std[i])
            if self.s.mode == "asinh":
                sc = float(self.s.asinh_scale[i])
                fac = np.sqrt(sc * sc + yp * yp)
            else:
                fac = np.ones_like(yp)
            return dn * out_std * fac / in_std

        qg = denorm("qg", out_np[:, _OC["qg"]])
        qd = denorm("qd", out_np[:, _OC["qd"]])
        gqg = gqg.numpy(); gqd = gqd.numpy()
        res = {
            "cgg": dderiv("qg", 1, gqg[:, 1], qg),
            "cgd": dderiv("qg", 0, gqg[:, 0], qg),
            "cdg": dderiv("qd", 1, gqd[:, 1], qd),
            "cdd": dderiv("qd", 0, gqd[:, 0], qd),
            "qg": qg, "qd": qd,
        }
        for c in ("cgg", "cgd", "cgs", "cdg", "cdd"):
            res[f"col_{c}"] = denorm(c, out_np[:, _OC[c]])
        return res


def _resolve(tech: str, dev: str, size: str) -> Tuple[Path, Path, Path]:
    ck = CHECKPOINT_DIR / f"{tech}_dn_{size}_{dev}_best.pt"
    nm = CHECKPOINT_DIR / f"{tech}_dn_{size}_{dev}_norm.npz"
    ds = DATA_DIR / f"{tech}_{dev}.npz"
    return ck, nm, ds


def _sanity_check(ckpt: Path, norm: Path, tech: str, dev: str) -> None:
    """Assert the batched evaluator matches the real device get_capacitances."""
    from pycircuitsim.models.mosfet_directnet import NMOS_NN, PMOS_NN
    cls = NMOS_NN if dev == "nmos" else PMOS_NN
    # Pick a representative on-state bias; device is at Vs=0 so the frame is the
    # dataset frame. Use NFIN=4, L=2e-8.
    L, NFIN = 2e-8, 4.0
    tcode = local_variant_code(tech, tech, _first_variant(tech))
    dev_obj = cls("Mx", ["d", "g", "s", "b"], str(ckpt), L=L, NFIN=NFIN,
                  temperature=300.15, tech_code=tcode)
    volts = {"d": 0.3, "g": 0.5, "s": 0.0, "b": 0.0}
    dcaps = dev_obj.get_capacitances(volts)
    ev = _BatchedCapEval(ckpt, norm)
    geo = np.array([[np.log2(NFIN), L, 300.15]])
    V = np.array([[0.3, 0.5, 0.0, 0.0]])
    bc = ev.caps(V, geo, np.array([tcode]))
    for c in ("cgg", "cgd", "cdg", "cdd"):
        a, b = dcaps[c], float(bc[c][0])
        rel = abs(a - b) / max(abs(a), 1e-20)
        assert rel < 1e-4, f"sanity FAIL {c}: device={a:.4e} batched={b:.4e} rel={rel:.2e}"
    print(f"  [sanity] batched evaluator matches device get_capacitances "
          f"(< 1e-4 rel) for {tech}/{dev}")


def _first_variant(tech: str) -> str:
    import sys as _s
    _s.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))
    from pycmg.nn_config import TECH_CONFIGS
    return TECH_CONFIGS[tech].variant_names[0]


def _summarize(name: str, autog: np.ndarray, target: np.ndarray,
               col: np.ndarray) -> str:
    """Per-channel signed-ratio + autograd-rel-err + column-rel-err."""
    # Mask to meaningful magnitudes (caps above 1% of the channel's median |.|).
    thr = 0.01 * np.median(np.abs(target))
    m = np.abs(target) > max(thr, 1e-20)
    if m.sum() < 10:
        return f"    {name:4s}: (insufficient rows)"
    a, t, c = autog[m], target[m], col[m]
    ratio = a / t                                   # signed; want ~+1
    rel_ag = np.abs(a - t) / np.abs(t)              # autograd vs signed OSDI
    rel_col = np.abs(np.abs(c) - np.abs(t)) / np.abs(t)   # |col| vs |OSDI|
    return (f"    {name:4s}: signed-ratio med={np.median(ratio):+.3f} "
            f"[p25={np.percentile(ratio,25):+.3f} p75={np.percentile(ratio,75):+.3f}]  "
            f"autograd relerr med={np.median(rel_ag)*100:5.1f}%  "
            f"col relerr med={np.median(rel_col)*100:5.1f}%")


def run_one(tech: str, dev: str, size: str, n_sample: int = 80000) -> None:
    ck, nm, ds = _resolve(tech, dev, size)
    if not ck.exists():
        print(f"  SKIP {tech}/{dev}/{size}: checkpoint missing ({ck.name})")
        return
    print(f"\n=== {tech} / {dev} / {size} ===")
    _sanity_check(ck, nm, tech, dev)

    data = np.load(str(ds), allow_pickle=True)
    inp, geo15, out = data["inputs"], data["geometry"], data["outputs"]
    vdd = float(data["meta_vdd"])
    # Build local tech codes from the universal labels file.
    from bsimar.eval.loo_labels import get_or_build_tech_variant_labels
    from bsimar.config import CODE_TO_TECH_VARIANT, LOCAL_VARIANT_CODES, \
        LOCAL_UNKNOWN_CODE_ID
    uni = get_or_build_tech_variant_labels(str(ds), dev, verbose=False)
    tbl = LOCAL_VARIANT_CODES[tech]; unk = LOCAL_UNKNOWN_CODE_ID[tech]
    local = np.array([tbl.get(CODE_TO_TECH_VARIANT.get(int(c), ("", "")), unk)
                      for c in uni], dtype=np.int64)

    n = len(inp)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(n_sample, n), replace=False)
    Vd, Vg, Vs, Vb = (inp[idx, 0], inp[idx, 1], inp[idx, 2], inp[idx, 3])
    geo = np.column_stack([np.log2(np.clip(geo15[idx, 0], 1.0, None)),
                           geo15[idx, 1], geo15[idx, 2]])
    V = np.column_stack([Vd, Vg, Vs, Vb])
    tc = local[idx]

    ev = _BatchedCapEval(ck, nm)
    # Evaluate in chunks (autograd graph memory).
    caps: Dict[str, List[np.ndarray]] = {k: [] for k in
                                         ("cgg", "cgd", "cdg", "cdd",
                                          "col_cgg", "col_cgd", "col_cdg", "col_cdd")}
    B = 8192
    for s0 in range(0, len(V), B):
        sl = slice(s0, s0 + B)
        r = ev.caps(V[sl], geo[sl], tc[sl])
        for k in caps:
            caps[k].append(r[k])
    for k in caps:
        caps[k] = np.concatenate(caps[k])

    # Regions in the dataset (Vs=0) frame.
    sat = (Vd > 0.4 * vdd) & (Vg > 0.30 * vdd) & (Vg < 0.95 * vdd)  # AC CS-amp pole
    tri = (Vd >= 0.0) & (Vd < 0.30 * vdd) & (Vg > 0.45 * vdd)        # SC charging / triode
    regions = [("ALL grid", np.ones(len(V), dtype=bool)),
               ("SAT (AC pole)", sat),
               ("TRIODE (switchcap)", tri)]

    for rname, rmask in regions:
        print(f"  -- {rname}  (n={int(rmask.sum())}) --")
        for cname, qh, vcol, tcol, sign in CAP_CHANNELS:
            tgt = sign * out[idx, _OC[tcol]]
            a = caps[cname]
            col = caps[f"col_{cname}"]
            print(_summarize(cname, a[rmask], tgt[rmask], col[rmask]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", default="tsmc5,tsmc7,tsmc12,tsmc16")
    ap.add_argument("--dev", default="nmos,pmos")
    ap.add_argument("--size", default="medium")
    args = ap.parse_args()
    print("=" * 80)
    print("Charge-derivative (autograd dQ/dV) cap-fidelity diagnostic")
    print("  signed-ratio = median(autograd_cap / signed_OSDI_target); want ~+1.0")
    print("  ratio<1  => NN under-predicts cap (f3db>1 / switchcap over-charge)")
    print("=" * 80)
    for tech in [t.strip() for t in args.tech.split(",")]:
        for dev in [d.strip() for d in args.dev.split(",")]:
            try:
                run_one(tech, dev, args.size)
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR {tech}/{dev}: {exc!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
