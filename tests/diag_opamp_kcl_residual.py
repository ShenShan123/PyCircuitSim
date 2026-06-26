"""P0-1 (decisive): tsmc7 opamp G1 — EXISTENCE vs CONTRACTION routing probe.

The V6.5.5 1c diagnostic (`diag_opamp_basin_seed.py`) proved the high-gain opamp
OP is UNSTABLE on the NN surface — seeding the NN sweep from the L72 ground-truth
OP at every point still rails to gain 0. But 1c runs the full damped DCSolver, so
it CONFLATES two distinct failure modes the fix-class depends on:

  (i)  EXISTENCE failure — the L72 OP is NOT a residual zero of KCL-with-NN-currents
       (the net NN current into the balance nodes vo1i/vout is large at the L72 OP).
       => the surface lacks the fixed point. The ONLY admissible lever is a net-node
          KCL-residual loss (T1) — the one quantity the corridor never pinned (it
          supervised each device's ABSOLUTE id, never the DIFFERENCE i_Mn2 - i_Mp4).
  (ii) CONTRACTION failure — the L72 OP IS a residual zero (F ~ 0) but the Newton
       map repels it. => the surface HAS the fixed point but it is unstable.
          T1 is inert; the lever is a contraction penalty at labeled OPs (N2/T3).

DECISIVE MEASUREMENT. At the L72-converged OP `op_l` (true high-gain node voltages),
assemble the net nonlinear MOSFET current into each free node {vtail,n1,vo1i,vout}
using the solver's own convention (`solver.py:303-309`):
    i_leaving = -i_ds if PMOS else i_ds ;  F[drain] += i_leaving ; F[source] -= i_leaving
At a true DC solution F(free node) == 0. We compute F for BOTH circuits at op_l:
  - F_L72 ~ 0  is a BUILT-IN SELF-CHECK (validates the sign assembly AND that op_l is
    a genuine KCL zero of L72 currents).
  - F_NN at the SAME op_l is the existence residual. Normalize by the per-node arm
    current |i_leaving| to read it as a fraction:
        F_rel(vo1i) = |F_NN(vo1i)| / max(|i_Mn2|,|i_Mp4|)
    F_rel large (>~5%) at vo1i/vout  => EXISTENCE  (route T1)
    F_rel small (<~1%)               => CONTRACTION (route N2/T3)

Run TSMC7 (open gate) with TSMC12 (passing) as a positive control: TSMC12 should
show BOTH a tiny F_L72 self-check AND a small F_NN (the OP exists on its surface).

Usage:
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NGSPICE_BIN="$PWD/tools/ngspice-45.2/bin/ngspice" \
      conda run -n pycircuitsim python tests/diag_opamp_kcl_residual.py --tech TSMC7,TSMC12
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))

# Reuse the gate-faithful opamp builders + OP extraction from the 1b diagnostic.
from tests.diag_opamp_op_decomp import (  # noqa: E402
    _run_nn_opamp, _run_l72_opamp, _build_op,
)
from tests.common.complex import BENCH, RESULTS_BASE  # noqa: E402
from tests.verify_complex_opamp import _bias, _gain_trip  # noqa: E402
from pycircuitsim.solver import _is_pmos, _is_mosfet  # noqa: E402

# Free internal nodes of the two-stage Miller opamp (the unknowns the sweep solves
# for; the balance nodes vo1i/vout are where an existence gap shows).
FREE_NODES = ("vtail", "n1", "vo1i", "vout")
EXISTENCE_FRAC = 0.05   # F_rel above this at vo1i/vout => existence failure
CONTRACTION_FRAC = 0.01  # F_rel below this everywhere => contraction failure
SELFCHECK_TOL_A = 1e-7   # |F_L72| must be below this (nA-µA solver residual) to trust


def _node_residuals(circuit, op: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Net nonlinear MOSFET current INTO each node at fixed OP `op`.

    Generic KCL assembly mirroring `solver._stamp_mosfet_dc:303-309`: the drain node
    receives +i_leaving, the source node -i_leaving; gate/bulk carry no DC current.
    Returns (F, arm) where F[node] is the residual current and arm[node] is the
    largest single-device current magnitude touching that node (the normalizer).
    """
    F: Dict[str, float] = {}
    arm: Dict[str, float] = {}
    for c in circuit.components:
        if not _is_mosfet(c):
            continue
        nodes = c.nodes
        drain, source = nodes[0], nodes[2]
        i_ds = c.calculate_current(op)
        i_lev = -i_ds if _is_pmos(c) else i_ds
        for nd, contrib in ((drain, i_lev), (source, -i_lev)):
            F[nd] = F.get(nd, 0.0) + contrib
            arm[nd] = max(arm.get(nd, 0.0), abs(i_lev))
    return F, arm


def run(name: str) -> Optional[Dict]:
    bt = BENCH[name]
    vdd = bt.vdd
    vcm, vbn, vbp = _bias(bt)
    work = RESULTS_BASE / "diag_opamp_kcl" / name
    work.mkdir(parents=True, exist_ok=True)

    logging.disable(logging.CRITICAL)
    try:
        nn_res, nn_circ = _run_nn_opamp(bt, work)
        l_res, l_circ = _run_l72_opamp(bt, work)
    finally:
        logging.disable(logging.NOTSET)

    sw_nn = np.asarray(nn_res["inp"]); vo_nn = np.asarray(nn_res["vout"])
    sw_l = np.asarray(l_res["inp"]); vo_l = np.asarray(l_res["vout"])
    g_nn, _, _ = _gain_trip(sw_nn, vo_nn, vdd)
    g_l, _, _ = _gain_trip(sw_l, vo_l, vdd)

    # the true high-gain crossing = where L72 has peak |dVout/dVin|
    vin_star = float(sw_l[int(np.argmax(np.abs(np.gradient(vo_l, sw_l))))])
    op_l = _build_op(l_res, sw_l, vin_star, vcm, vbn, vbp, vdd)

    # Residuals of BOTH circuits at the SAME L72-converged OP.
    F_l, arm_l = _node_residuals(l_circ, op_l)   # self-check: must be ~0
    F_nn, arm_nn = _node_residuals(nn_circ, op_l)  # the existence residual

    def _rel(F, arm, nd):
        a = arm.get(nd, 0.0)
        return abs(F.get(nd, 0.0)) / a if a > 1e-15 else float("nan")

    print(f"\n===== {name} (VDD={vdd}) opamp KCL-residual @ L72 OP (vin*={vin_star:.3f}) =====")
    print(f"  swept gain: NN={g_nn:7.1f}  L72={g_l:7.1f}")
    print(f"  {'node':>6} {'F_L72(µA)':>11} {'F_NN(µA)':>11} {'arm_NN(µA)':>11} "
          f"{'F_rel_NN':>9}")
    selfcheck_ok = True
    for nd in FREE_NODES:
        fl = F_l.get(nd, 0.0); fn = F_nn.get(nd, 0.0)
        an = arm_nn.get(nd, 0.0); rel = _rel(F_nn, arm_nn, nd)
        if abs(fl) > SELFCHECK_TOL_A:
            selfcheck_ok = False
        print(f"  {nd:>6} {fl*1e6:11.4f} {fn*1e6:11.4f} {an*1e6:11.4f} {rel:9.3f}")

    rel_vo1i = _rel(F_nn, arm_nn, "vo1i")
    rel_vout = _rel(F_nn, arm_nn, "vout")
    # PRIMARY discriminator = vo1i, the stage-1 diff-pair/mirror BALANCE node (the
    # i_Mn2 - i_Mp4 difference that sets stage-1 gain — exactly what T1 supervises).
    # vout (stage-2 CS output) is reported as a SECONDARY diagnostic only: the passing
    # TSMC12 control shows a large vout F_rel (~0.19) yet passes, so a vout residual is
    # NOT predictive of gain collapse and must not drive the verdict.
    balance_rel = rel_vo1i

    sc = "OK (|F_L72|<%.0e A)" % SELFCHECK_TOL_A if selfcheck_ok else \
         "FAILED — op_l is not a clean L72 KCL zero; signs/columns suspect"
    print(f"  self-check F_L72: {sc}")

    gate_pass = g_l > 1e-9 and abs(g_nn - g_l) / g_l <= 0.10
    if gate_pass:
        verdict = (f"PASSING (positive control) — stage-1 balance residual small "
                   f"(vo1i F_rel={balance_rel:.3f}); the high-gain OP exists on the NN surface.")
    elif balance_rel >= EXISTENCE_FRAC:
        verdict = (f"EXISTENCE failure — the L72 OP is NOT a KCL zero of NN currents at the "
                   f"stage-1 balance node (vo1i F_rel={balance_rel:.3f} ≥ {EXISTENCE_FRAC}). The "
                   f"NN net node current at vo1i is a large fraction of the arm current. "
                   f"=> route T1 (net-node KCL-residual loss). N2/T3 alone are insufficient.")
    elif balance_rel <= CONTRACTION_FRAC:
        verdict = (f"CONTRACTION failure — the L72 OP IS ~a KCL zero of NN currents at vo1i "
                   f"(F_rel={balance_rel:.3f} ≤ {CONTRACTION_FRAC}) yet 1c showed it repels. The "
                   f"surface HAS the fixed point but it is unstable. "
                   f"=> route N2/T3 (contraction penalty). T1 is INERT.")
    else:
        verdict = (f"MIXED — vo1i F_rel={balance_rel:.3f} is between the thresholds "
                   f"({CONTRACTION_FRAC}..{EXISTENCE_FRAC}). Partial existence gap; T1 is the "
                   f"primary lever but a contraction term (N2) may be needed too.")
    print(f"  >> stage-1 balance (vo1i) F_rel={rel_vo1i:.3f}  [secondary: vout F_rel={rel_vout:.3f}]")
    print(f"  >> VERDICT: {verdict}")
    return {"tech": name, "g_nn": g_nn, "g_l72": g_l,
            "selfcheck_ok": selfcheck_ok, "rel_vo1i": rel_vo1i, "rel_vout": rel_vout,
            "balance_rel": balance_rel}


def main() -> int:
    ap = argparse.ArgumentParser(description="Opamp existence-vs-contraction KCL probe")
    ap.add_argument("--tech", default="TSMC7,TSMC12",
                    help="comma-separated; include a passing tech as control")
    args = ap.parse_args()
    print("=" * 78)
    print("P0-1 opamp G1 — EXISTENCE (F_NN large at L72 OP) vs CONTRACTION (F_NN ~ 0)")
    print("  Net nonlinear MOSFET current into free nodes at the L72-converged OP")
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
        print("SUMMARY (route the G1 effort by the TSMC7 balance residual):")
        for r in rows:
            print(f"  {r['tech']:7} selfcheck={'ok' if r['selfcheck_ok'] else 'FAIL':4} "
                  f"balance_F_rel={r['balance_rel']:.3f}  (NN gain {r['g_nn']:.0f} / "
                  f"L72 {r['g_l72']:.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
