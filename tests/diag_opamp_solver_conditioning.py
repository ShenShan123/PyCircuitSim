"""V6.5.6 — opamp solver-conditioning probe: is the T1-created high-gain OP REACHABLE?

T1 (the net-node KCL-residual fine-tune) made the L72 high-gain OP a residual zero
of the NN current map (tsmc7 vo1i F_rel 0.128 -> 0.007 on the k2_c checkpoint) —
P0-1 reclassified it EXISTENCE -> CONTRACTION. The 1c basin-seed showed the cold
*and* L72-seeded sweeps still rail (0% gain). But the high-gain OP has, by nature,
a near-singular Jacobian (high gain == tiny effective output conductance), so a
tiny residual gets amplified into a railing Newton step. LM damping is already
auto-active in the NR loop, so the open question this probe answers is the
DECISIVE one for routing the next lever:

  **Does an exact high-gain DC fixed point EXIST near the L72 OP on the NN
  surface, reachable by a better-conditioned / better-seeded solve?**

If YES -> a DC-SAFE solver lever (seed the opamp gate from a mid-rail guess,
optionally GMIN homotopy) recovers the opamp with NO further surface change ->
NO ring/preservation risk. Pairs with a gentle T1.
If NO  -> 0.7% is a small residual but not an exact zero; there is no high-gain
solution to find -> route to the retrain track (ring-anchored harder existence +
localized N2 contraction term).

METHOD (at the L72 high-gain crossing vin*):
  multi-start the DC solve from a GRID of mid-rail seeds (vout swept 0.1..0.9 VDD,
  internal nodes from the L72 OP) PLUS the exact L72 OP, each solved with the
  stock damped+LM solver AND with GMIN homotopy. Classify every CONVERGED solution
  as RAILED (|vout-rail| small) or HIGH-GAIN (vout in the mid band). For any
  high-gain solution, measure the actual small-signal gain by warm ±dVin re-solve.

Reading:
  - >=1 converged HIGH-GAIN solution with measured gain >~50  => REACHABLE
    (DC-safe solver/seed lever wins). Report the seed that finds it.
  - only RAILED converged solutions                           => NO nearby zero
    (retrain track).

Run on the T1 checkpoint (k2_c) AND production (negative control, F_rel 0.128):
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NGSPICE_BIN="$PWD/tools/ngspice-45.2/bin/ngspice" \
    PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS=tsmc7_dn_kcl2_c \
    PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS=tsmc7_dn_kcl2_c \
      conda run -n pycircuitsim python tests/diag_opamp_solver_conditioning.py --tech TSMC7
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))

from tests.common.complex import BENCH, RESULTS_BASE, BenchTech  # noqa: E402
from tests.verify_complex_opamp import _bias, _gain_trip  # noqa: E402
from tests.diag_opamp_op_decomp import _run_nn_opamp, _run_l72_opamp, _build_op  # noqa: E402
from tests.diag_opamp_basin_seed import _find_inp_source, _node_keys  # noqa: E402
from pycircuitsim.solver import DCSolver  # noqa: E402

HIGH_GAIN_MIN = 50.0     # |dVout/dVin| above this at vin* = genuine high-gain branch


def _solve(circuit, guess: Dict[str, float], use_gmin: bool) -> Tuple[Optional[Dict], bool]:
    try:
        s = DCSolver(circuit, initial_guess=guess, use_source_stepping=False,
                     use_gmin_stepping=use_gmin)
        sol = s.solve()
        conv = bool(getattr(s, "_last_solve_converged", True))
        return sol, conv
    except Exception:  # noqa: BLE001
        return None, False


def _measure_gain(circuit, inp_src, vin: float, sol: Dict[str, float],
                  dvin: float = 1e-3) -> float:
    """Warm ±dVin re-solve from `sol` to measure |dVout/dVin| at vin."""
    outs = {}
    for sign in (-1.0, +1.0):
        inp_src.value = vin + sign * dvin
        s2, ok = _solve(circuit, sol, use_gmin=False)
        outs[sign] = s2.get("vout", float("nan")) if (s2 and ok) else float("nan")
    inp_src.value = vin
    if any(np.isnan(v) for v in outs.values()):
        return float("nan")
    return abs((outs[+1.0] - outs[-1.0]) / (2.0 * dvin))


def run(name: str) -> Optional[Dict]:
    bt: BenchTech = BENCH[name]
    vdd = bt.vdd
    vcm, vbn, vbp = _bias(bt)
    work = RESULTS_BASE / "diag_opamp_cond" / name
    work.mkdir(parents=True, exist_ok=True)

    logging.disable(logging.CRITICAL)
    try:
        l_res, _ = _run_l72_opamp(bt, work)
        nn_res, nn_circ = _run_nn_opamp(bt, work)
        sw_l = np.asarray(l_res["inp"]); vo_l = np.asarray(l_res["vout"])
        g_l, _, _ = _gain_trip(sw_l, vo_l, vdd)
        vin_star = float(sw_l[int(np.argmax(np.abs(np.gradient(vo_l, sw_l))))])
        op_l = _build_op(l_res, sw_l, vin_star, vcm, vbn, vbp, vdd)
        nkeys = _node_keys(l_res)
        inp_src = _find_inp_source(nn_circ)
        inp_src.value = vin_star

        # seed grid: the exact L72 OP + mid-rail vout sweeps (internal nodes
        # carried from the L72 OP so only the output basin is varied).
        base = {k: float(np.asarray(l_res[k])[
            int(np.argmin(np.abs(sw_l - vin_star)))]) for k in nkeys}
        seeds: List[Tuple[str, Dict[str, float]]] = [("L72-OP", dict(op_l))]
        for f in np.linspace(0.1, 0.9, 9):
            g = dict(base)
            g["vout"] = float(f * vdd)
            if "vo1i" in g:
                g["vo1i"] = float(np.clip(op_l.get("vo1i", 0.5 * vdd), 0.05 * vdd, 0.95 * vdd))
            seeds.append((f"vout={f:.2f}VDD", g))

        railed, highgain = [], []
        for use_gmin in (False, True):
            for label, g in seeds:
                sol, conv = _solve(nn_circ, g, use_gmin)
                if not sol or not conv:
                    continue
                vout = float(sol.get("vout", float("nan")))
                if np.isnan(vout):
                    continue
                tag = f"{label}{'+gmin' if use_gmin else ''}"
                if 0.20 * vdd <= vout <= 0.80 * vdd:
                    gain = _measure_gain(nn_circ, inp_src, vin_star, sol)
                    highgain.append((tag, vout, gain))
                else:
                    railed.append((tag, vout))
    finally:
        logging.disable(logging.NOTSET)

    print(f"\n===== {name} (VDD={vdd}) opamp solver-conditioning @ vin*={vin_star:.4f} =====")
    print(f"  L72 gain (truth) = {g_l:.1f}   seeds tried = {len(seeds)}×2(gmin) = {2*len(seeds)}")
    print(f"  CONVERGED RAILED solutions: {len(railed)}  "
          f"(vout∈{{{min((v for _,v in railed), default=float('nan')):.3f}.."
          f"{max((v for _,v in railed), default=float('nan')):.3f}}})")
    print(f"  CONVERGED HIGH-GAIN (mid-rail) solutions: {len(highgain)}")
    best_gain = 0.0
    for tag, vout, gain in sorted(highgain, key=lambda t: -(t[2] if np.isfinite(t[2]) else 0))[:6]:
        print(f"     seed {tag:18s} -> vout={vout:.4f}  measured|gain|={gain:7.1f}")
        if np.isfinite(gain):
            best_gain = max(best_gain, gain)

    reachable = best_gain >= HIGH_GAIN_MIN
    if reachable:
        verdict = (f"REACHABLE — a high-gain DC fixed point EXISTS near the L72 OP "
                   f"(best measured |gain|={best_gain:.0f}) and is found by a mid-rail "
                   f"seed{''.join('' for _ in ())}. => DC-SAFE solver lever: seed the opamp "
                   f"gate from mid-rail (± GMIN). NO retrain / NO ring risk.")
    elif highgain:
        verdict = (f"PARTIAL — converged mid-rail solutions exist but measured gain "
                   f"<{HIGH_GAIN_MIN:.0f} (best {best_gain:.0f}); the branch is shallow, "
                   f"not the true high-gain OP. Lean retrain track.")
    else:
        verdict = ("NO nearby high-gain zero — every converged solution is RAILED. "
                   "The 0.7% residual is not an exact zero; there is no high-gain "
                   "solution to seed. => route to the RETRAIN track (ring-anchored "
                   "harder existence + localized N2 contraction term).")
    print(f"  >> VERDICT: {verdict}")
    return {"tech": name, "g_l72": g_l, "n_railed": len(railed),
            "n_highgain": len(highgain), "best_gain": best_gain, "reachable": reachable}


def main() -> int:
    ap = argparse.ArgumentParser(description="Opamp solver-conditioning reachability probe")
    ap.add_argument("--tech", default="TSMC7")
    args = ap.parse_args()
    print("=" * 78)
    print("Opamp solver-conditioning — does a high-gain zero EXIST near the L72 OP?")
    print("  multi-start (mid-rail seeds) × {stock, GMIN}; classify converged solutions")
    print("=" * 78)
    rows = []
    for t in [x.strip() for x in args.tech.split(",")]:
        if t not in BENCH:
            print(f"  SKIP unknown tech {t}"); continue
        try:
            r = run(t)
            if r:
                rows.append(r)
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"  {t}: ERROR {exc!r}"); traceback.print_exc()
    if rows:
        print("\n" + "=" * 78)
        print("SUMMARY:")
        for r in rows:
            print(f"  {r['tech']:7s} high-gain solns={r['n_highgain']:2d} "
                  f"best|gain|={r['best_gain']:6.0f} reachable={r['reachable']} "
                  f"(L72={r['g_l72']:.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
