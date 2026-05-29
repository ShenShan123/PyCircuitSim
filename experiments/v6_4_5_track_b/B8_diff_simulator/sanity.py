"""B8 sanity gate — torch RO period vs production RO period (base weights).

Make-or-break: with the base TSMC7 canonical weights the differentiable
torch RO must reproduce the production DirectNet RO period (≈50.8 ps) within
a few %. If not, the simulator is wrong — fix before TTFT.
"""
from __future__ import annotations

import argparse
import sys
import time as _time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from ro_torch import RingOscTorch, soft_period  # noqa: E402
from build import load_pair, TSMC7_VDD  # noqa: E402

# production reference (scripts/eval_v6_4_5_candidate.py on canonical TSMC7)
PROD_DN_PERIOD_PS = 50.828513969374406
OSDI_PERIOD_PS = 46.64143635801447


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmos", default="tsmc7_dn_medium_nmos")
    ap.add_argument("--pmos", default="tsmc7_dn_medium_pmos")
    ap.add_argument("--tstop", type=float, default=1.2e-9)
    ap.add_argument("--tstep", type=float, default=2e-12)
    ap.add_argument("--newton", type=int, default=8)
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available())
                          else "cpu")
    print(f"device={device}")
    nmos, pmos = load_pair(args.nmos, args.pmos, device=device)
    ro = RingOscTorch(nmos, pmos, vdd=TSMC7_VDD, device=device)

    t0 = _time.time()
    with torch.no_grad():
        t, v = ro.simulate(tstep=args.tstep, tstop=args.tstop,
                           n_newton=args.newton, keep_graph_from=1e9)
    wall = _time.time() - t0
    mid = TSMC7_VDD / 2.0
    per = soft_period(t, v, mid, settle=0.3e-9)
    per_ps = float(per) * 1e12
    print(f"  torch RO period = {per_ps:.3f} ps   (wall {wall:.1f}s)")
    print(f"  production DN  period = {PROD_DN_PERIOD_PS:.3f} ps")
    print(f"  OSDI ground-truth period = {OSDI_PERIOD_PS:.3f} ps")
    if np.isfinite(per_ps):
        gap_prod = abs(per_ps - PROD_DN_PERIOD_PS) / PROD_DN_PERIOD_PS * 100
        gap_osdi = abs(per_ps - OSDI_PERIOD_PS) / OSDI_PERIOD_PS * 100
        print(f"  torch vs production gap = {gap_prod:.2f}%")
        print(f"  torch vs OSDI period err = {gap_osdi:.2f}%")
        v_np = v.detach().cpu().numpy()
        print(f"  waveform: min={v_np.min():.4f} max={v_np.max():.4f} "
              f"last={v_np[-1]:.4f}")
        # crude swing check
        keep = (t.cpu().numpy() >= 0.3e-9)
        print(f"  post-settle swing: [{v_np[keep].min():.4f}, "
              f"{v_np[keep].max():.4f}]")
    else:
        print("  torch RO did NOT oscillate (period=NaN) — sim is wrong")
    return 0


if __name__ == "__main__":
    sys.exit(main())
