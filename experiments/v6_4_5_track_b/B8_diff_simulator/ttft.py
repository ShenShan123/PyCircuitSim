"""B8 — Test-time fine-tuning (TTFT) of the TSMC7 DirectNet through the
differentiable ring oscillator.

Trainable: a LoRA delta (rank r) on every trunk Linear of the NMOS and PMOS
models. A delta is cheaper than full-weight tuning and — being zero at init —
preserves the inverter while it is small; we also add a small L2 anchor on
the delta and a stop on inverter break (checked by the production scorer
offline).

Loss = (measured_period_ps - OSDI_ps)^2 / OSDI_ps^2
       + alpha * waveform_MSE_vs_OSDI
       + beta  * ||delta||^2     (delta-magnitude anchor)

OSDI reference waveform/period come from the harness ``run_ngspice_ro`` (saved
to /tmp/osdi_ro_tsmc7.npz by the caller).

Checkpoints every ``--ckpt-every`` steps to disk under NON-canonical stems
``b8_ttft_tsmc7_{nmos,pmos}_best.pt`` (+ base ``_norm.npz`` copy) so a crash
does not lose progress; the merged (base+delta) plain DirectNet state_dict is
saved (loadable by the inference path / production scorer).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
PROJECT_ROOT = _HERE.parents[3]

from ro_torch import RingOscTorch, soft_period  # noqa: E402
from build import load_pair, CKPT_DIR, TSMC7_VDD  # noqa: E402

OSDI_PS = 46.64143635801447


# ---------------------------------------------------------------------------
# LoRA delta wrapper for an nn.Linear (additive low-rank delta-W = B @ A)
# ---------------------------------------------------------------------------
class LoRADelta(nn.Module):
    def __init__(self, lin: nn.Linear, rank: int, dtype, device):
        super().__init__()
        self.lin = lin  # frozen
        out_f, in_f = lin.weight.shape
        self.A = nn.Parameter(torch.zeros(rank, in_f, dtype=dtype, device=device))
        self.B = nn.Parameter(torch.zeros(out_f, rank, dtype=dtype, device=device))
        nn.init.normal_(self.A, std=1e-3)
        # B stays zero at init → delta = 0 → exact base behaviour.

    def forward(self, x):
        base = self.lin(x)
        delta = (x @ self.A.t()) @ self.B.t()
        return base + delta

    def delta_norm_sq(self):
        return (self.B @ self.A).pow(2).sum()


def inject_lora(model, rank: int, dtype, device) -> List[LoRADelta]:
    """Replace each trunk Linear in model.net with lin+LoRA; freeze base."""
    for p in model.parameters():
        p.requires_grad_(False)
    deltas: List[LoRADelta] = []
    for i, m in enumerate(model.net):
        if isinstance(m, nn.Linear):
            ld = LoRADelta(m, rank, dtype, device)
            model.net[i] = ld
            deltas.append(ld)
    return deltas


def merge_lora_state_dict(model) -> Dict[str, torch.Tensor]:
    """Return a plain DirectNet state_dict with LoRA deltas folded into the
    trunk Linear weights (so the saved checkpoint is a stock DirectNet)."""
    sd = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRADelta):
            # mod is model.net[i]; its base linear is mod.lin
            W = (mod.lin.weight.data + mod.B.data @ mod.A.data).to(torch.float32)
            b = mod.lin.bias.data.to(torch.float32)
            # name like 'net.0' → keys 'net.0.weight'/'net.0.bias'
            sd[f"{name}.weight"] = W
            sd[f"{name}.bias"] = b
    # everything else (tech_embedding, mono.*) straight from the base model
    base_sd = model.state_dict()
    for k, vv in base_sd.items():
        if ".lin." in k:
            # skip LoRA-internal names; handled above via merged net.<i>.weight
            continue
        if k.endswith(".A") or k.endswith(".B"):
            continue
        if k not in sd:
            sd[k] = vv.detach().cpu().to(torch.float32)
    return sd


def save_merged(nmos, pmos, nmos_stem_src, pmos_stem_src):
    for model, dev, src in ((nmos.model, "nmos", nmos_stem_src),
                            (pmos.model, "pmos", pmos_stem_src)):
        sd = merge_lora_state_dict(model)
        out_pt = CKPT_DIR / f"b8_ttft_tsmc7_{dev}_best.pt"
        torch.save(sd, str(out_pt))
        shutil.copy2(CKPT_DIR / f"{src}_norm.npz",
                     CKPT_DIR / f"b8_ttft_tsmc7_{dev}_norm.npz")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmos-src", default="tsmc7_dn_medium_nmos")
    ap.add_argument("--pmos-src", default="tsmc7_dn_medium_pmos")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=2.0,
                    help="waveform-MSE weight")
    ap.add_argument("--beta", type=float, default=0.0,
                    help="delta L2 anchor weight")
    ap.add_argument("--tstop", type=float, default=0.5e-9)
    ap.add_argument("--settle", type=float, default=0.25e-9)
    ap.add_argument("--newton", type=int, default=18)
    ap.add_argument("--ckpt-every", type=int, default=20)
    ap.add_argument("--kill-step", type=int, default=200,
                    help="report KILL if RO not <=kill-thresh by here")
    ap.add_argument("--kill-thresh", type=float, default=7.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    print(f"device={device} dtype={dtype} rank={args.rank} lr={args.lr} "
          f"alpha={args.alpha} tstop={args.tstop} settle={args.settle}",
          flush=True)

    nmos, pmos = load_pair(args.nmos_src, args.pmos_src, device=device, dtype=dtype)
    deltas = inject_lora(nmos.model, args.rank, dtype, device)
    deltas += inject_lora(pmos.model, args.rank, dtype, device)
    params = [p for d in deltas for p in (d.A, d.B)]
    n_trainable = sum(p.numel() for p in params)
    print(f"trainable LoRA params: {n_trainable}", flush=True)

    ro = RingOscTorch(nmos, pmos, vdd=TSMC7_VDD, device=device, dtype=dtype)
    opt = torch.optim.Adam(params, lr=args.lr)

    # OSDI reference (resampled onto the torch post-settle grid each eval).
    ref = np.load("/tmp/osdi_ro_tsmc7.npz")
    ref_t = ref["t"]
    ref_v = ref["v"]
    mid = TSMC7_VDD / 2.0

    history = []
    best_err = float("inf")
    t_start = time.time()
    for step in range(args.steps + 1):
        opt.zero_grad()
        t, v = ro.simulate(tstep=2e-12, tstop=args.tstop,
                           n_newton=args.newton, keep_graph_from=args.settle)
        per = soft_period(t, v, mid, settle=args.settle)
        if not torch.isfinite(per):
            print(f"step {step}: period=NaN (no oscillation) — abort", flush=True)
            break
        per_ps = per * 1e12
        # waveform MSE vs OSDI on the post-settle grid (align by interpolation
        # of the OSDI ref onto the torch times; OSDI ref is detached const).
        keep = (t >= args.settle)
        tk = t[keep]
        vk = v[keep]
        ref_interp = torch.tensor(
            np.interp(tk.detach().cpu().numpy(), ref_t, ref_v),
            dtype=dtype, device=device)
        wave_mse = torch.mean((vk - ref_interp) ** 2)

        per_loss = (per_ps - OSDI_PS) ** 2 / OSDI_PS ** 2
        delta_l2 = sum(d.delta_norm_sq() for d in deltas)
        loss = per_loss + args.alpha * wave_mse + args.beta * delta_l2

        cur_err = abs(float(per_ps) - OSDI_PS) / OSDI_PS * 100.0
        rec = {"step": step, "period_ps": float(per_ps), "ro_err_pct": cur_err,
               "per_loss": float(per_loss), "wave_mse": float(wave_mse),
               "loss": float(loss), "wall_s": round(time.time() - t_start, 1)}
        history.append(rec)
        print(f"step {step:3d}  per={float(per_ps):.3f}ps  ROerr={cur_err:.3f}%  "
              f"wave_mse={float(wave_mse):.3e}  loss={float(loss):.5f}  "
              f"[{rec['wall_s']}s]", flush=True)

        if cur_err < best_err:
            best_err = cur_err
            save_merged(nmos, pmos, args.nmos_src, args.pmos_src)
            print(f"   -> saved best (ROerr={cur_err:.3f}%)", flush=True)

        if step == args.steps:
            break
        loss.backward()
        # clip for stability
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % args.ckpt_every == 0 and step > 0:
            with open(_HERE.parent / "ttft_history.json", "w") as fh:
                json.dump(history, fh, indent=2)

    with open(_HERE.parent / "ttft_history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"\nBEST torch-sim RO err = {best_err:.3f}%  "
          f"(KILL thresh {args.kill_thresh}% by step {args.kill_step})",
          flush=True)
    print("Saved candidate stems: b8_ttft_tsmc7_nmos / b8_ttft_tsmc7_pmos",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
