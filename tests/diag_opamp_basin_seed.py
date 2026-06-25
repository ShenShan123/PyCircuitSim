"""Opamp basin-reachability probe: is the high-gain OP STABLE on the NN surface?

V6.5.5 Tier-1c, fired by 1b's BASIN verdict for tsmc7 (diag_opamp_op_decomp.py:
at the L72 true OP the NN has gain 142 ≈ 0.58×L72, but the NN DC sweep lands a
RAILED fixed point — vo1i/vout ≈ 0 — i.e. a different basin, not a flat surface).

This is the decisive, zero-commit experiment that routes the fix. The DC sweep
warm-starts every point from point-0's OP, so the whole transfer is set by which
basin point 0 lands in. Re-solve the NN sweep three ways:

  COLD          : the gate path (cold OP at point 0, warm-continue). Baseline.
  SEED-EACH-PT  : seed EVERY point from the L72 ground-truth solution at that Vinp.
  SEED-PT0-WARM : seed ONLY point 0 from the L72 OP, then warm-continue (prev pt).

Reading:
  * SEED-EACH-PT recovers high gain  -> the high-gain branch EXISTS & is locally
    stable on the NN surface => a solver-seed / homotopy lever can fix tsmc7
    WITHOUT a retrain (cheap). SEED-PT0-WARM also high => the branch self-sustains
    once entered (a point-0 OP seed suffices — cheapest of all).
  * SEED-EACH-PT STILL rails (NN won't hold the high-gain OP even when handed it)
    -> the high-gain fixed point is UNSTABLE on the NN value surface => not a seed
    problem; needs the shape/corridor lever (2b). (This is the V6.4.8-S2 outcome.)

Usage:
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      conda run -n pycircuitsim python tests/diag_opamp_basin_seed.py --tech TSMC7,TSMC12
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))

from tests.common.complex import BENCH, RESULTS_BASE, BenchTech  # noqa: E402
from tests.verify_complex_opamp import _gain_trip  # noqa: E402
from tests.diag_opamp_op_decomp import _run_nn_opamp, _run_l72_opamp  # noqa: E402
from pycircuitsim.solver import DCSolver  # noqa: E402


def _node_keys(results: Dict) -> List[str]:
    return [k for k in results if not k.startswith("i(") and
            np.asarray(results[k]).ndim == 1]


def _find_inp_source(circuit):
    for c in circuit.components:
        if c.__class__.__name__ == "VoltageSource" and c.nodes[0] == "inp":
            return c
    raise KeyError("Vinp source (node inp) not found")


def _seeded_sweep(circuit, inp_src, vin_axis: np.ndarray,
                  seeds: List[Dict[str, float]], warm_only_pt0: bool) -> np.ndarray:
    """Re-solve the NN sweep, seeding from `seeds`. If warm_only_pt0, seed point
    0 from seeds[0] then warm-continue from the previous solution."""
    vout = np.full(vin_axis.shape, np.nan)
    prev: Optional[Dict[str, float]] = None
    for i, vin in enumerate(vin_axis):
        inp_src.value = float(vin)
        if warm_only_pt0:
            guess = seeds[0] if i == 0 else prev
        else:
            guess = seeds[i]
        try:
            sol = DCSolver(circuit, initial_guess=guess,
                           use_source_stepping=False, use_gmin_stepping=False).solve()
        except Exception:  # noqa: BLE001
            sol = prev or {}
        vout[i] = sol.get("vout", float("nan"))
        prev = sol
    return vout


def run(name: str) -> Optional[Dict]:
    bt: BenchTech = BENCH[name]
    vdd = bt.vdd
    work = RESULTS_BASE / "diag_opamp_basin" / name
    work.mkdir(parents=True, exist_ok=True)

    logging.disable(logging.CRITICAL)
    try:
        l_res, _ = _run_l72_opamp(bt, work)
        nn_res, nn_circ = _run_nn_opamp(bt, work)

        vin_axis = np.asarray(l_res["inp"])
        nkeys = _node_keys(l_res)
        seeds = [{k: float(np.asarray(l_res[k])[i]) for k in nkeys}
                 for i in range(len(vin_axis))]
        inp_src = _find_inp_source(nn_circ)

        vout_each = _seeded_sweep(nn_circ, inp_src, vin_axis, seeds, warm_only_pt0=False)
        vout_pt0 = _seeded_sweep(nn_circ, inp_src, vin_axis, seeds, warm_only_pt0=True)
    finally:
        logging.disable(logging.NOTSET)

    g_l, _, _ = _gain_trip(vin_axis, np.asarray(l_res["vout"]), vdd)
    g_cold, _, _ = _gain_trip(np.asarray(nn_res["inp"]), np.asarray(nn_res["vout"]), vdd)
    g_each, _, _ = _gain_trip(vin_axis, vout_each, vdd)
    g_pt0, _, _ = _gain_trip(vin_axis, vout_pt0, vdd)

    print(f"\n===== {name} (VDD={vdd}) opamp basin-reachability =====")
    print(f"  L72 gain (truth)              : {g_l:7.1f}")
    print(f"  NN COLD (gate path)           : {g_cold:7.1f}   {'PASS' if g_cold>=0.9*g_l else 'FAIL (railed basin)'}")
    print(f"  NN SEED-EACH-PT (from L72)     : {g_each:7.1f}")
    print(f"  NN SEED-PT0-WARM (from L72 OP) : {g_pt0:7.1f}")

    frac_each = g_each / g_l if g_l > 1e-9 else float("nan")
    if frac_each >= 0.5:
        if g_pt0 >= 0.5 * g_l:
            verdict = ("REACHABLE & SELF-SUSTAINING — a point-0 OP seed (PTC/homotopy) "
                       "recovers high gain. CHEAP solver-seed fix for tsmc7, NO retrain. "
                       "Build the OP-seed lever (Tier-2, solver-side).")
        else:
            verdict = ("REACHABLE but METASTABLE — holds only when every point is "
                       "seeded; a point-0 seed decays. Needs per-point homotopy "
                       "guidance (harder solver lever) OR the corridor (2b).")
    else:
        verdict = ("UNSTABLE on the NN surface — the high-gain OP rails even when "
                   "handed the ground-truth seed. NOT a seed problem; route to the "
                   "shape/corridor lever (2b). (Matches the V6.4.8-S2 finding.)")
    print(f"  >> SEED-EACH-PT recovers {frac_each*100:.0f}% of L72 gain -> {verdict}")
    return {"tech": name, "g_l72": g_l, "g_cold": g_cold,
            "g_seed_each": g_each, "g_seed_pt0": g_pt0, "frac_each": frac_each}


def main() -> int:
    ap = argparse.ArgumentParser(description="Opamp basin-reachability probe")
    ap.add_argument("--tech", default="TSMC7,TSMC12")
    args = ap.parse_args()
    print("=" * 78)
    print("Opamp basin-reachability — does the NN HOLD the high-gain OP when seeded?")
    print("=" * 78)
    for t in [x.strip() for x in args.tech.split(",")]:
        if t not in BENCH:
            print(f"  SKIP unknown tech {t}"); continue
        try:
            run(t)
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"  {t}: ERROR {exc!r}"); traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
