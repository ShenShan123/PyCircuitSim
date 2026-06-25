"""Opamp gain->0 decomposition: BASIN (wrong DC fixed point) vs VALUE-SURFACE.

V6.5.5 Tier-1b routing diagnostic for the open tsmc7 opamp gate (DC gain->0).
The native-L72 control proves the gate is well-posed (L72-in-PyCircuitSim finds
gain~163-188 via the continuation-first path). The open-loop gain is a SINGLE-OP
property: gain = (gm1*ro1)*(gm6*ro6) of the diff-pair + 2nd stage at the
high-gain crossing. The solver reads gm/gds from the AUTOGRAD Jacobian of the NN
id head (mosfet_nn.py), not the predicted columns. gain->0 therefore has two
DISJOINT causes the Vout-curve gate cannot separate:

  (a) BASIN — the NN DC sweep converges to a DIFFERENT fixed point than L72
      (diff-pair off-balance / mirror out of saturation -> low gm there), or
  (b) VALUE-SURFACE — the NN lands the SAME OP as L72 but its autograd gm/gds
      surface is genuinely flat (gm*ro ~ 0) at that correct OP.

DECISIVE TEST: evaluate the NN devices' autograd gm/gds AT THE L72-CONVERGED OP
(the true high-gain node voltages) and recompute the analytic two-stage gain:
  A1 = gm_n1 / (gds_n2 + gds_p4)      (stage-1 diff-pair over vo1i conductance)
  A2 = gm_p6 / (gds_p6 + gds_n7)      (stage-2 CS gain)
  - A_NN(at L72 OP) high  AND  NN's own OP far from L72's  -> BASIN  (route 1c)
  - A_NN(at L72 OP) ~ 0                                    -> VALUE  (route 2b)
Plus the node-voltage basin delta (NN's own converged OP vs L72's).

Run TSMC7 (fail) with TSMC12 (pass) as a positive control.

Usage:
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NGSPICE_BIN="$PWD/tools/ngspice-45.2/bin/ngspice" \
      conda run -n pycircuitsim python tests/diag_opamp_op_decomp.py --tech TSMC7,TSMC12
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

from tests.common.complex import (  # noqa: E402
    BENCH, RESULTS_BASE, BenchTech, render_directnet_netlist, parse_netlist,
)
from tests.diag_l72_complex_control import _merged_card, _parse_l72  # noqa: E402
from tests.verify_complex_opamp import _bias, _gain_trip  # noqa: E402

OPAMP_TEMPLATE = PROJECT_ROOT / "examples" / "complex" / "miller_opamp_directnet.sp"
BASIN_TOL = 0.030  # V — node-voltage divergence that flags a different basin


# ── runs (gate-faithful solve paths) ────────────────────────────────────────

def _run_nn_opamp(bt: BenchTech, work: Path):
    from pycircuitsim.simulation import run_dc_sweep
    from pycircuitsim.visualizer import Visualizer
    vcm, vbn, vbp = _bias(bt)
    netlist = render_directnet_netlist(OPAMP_TEMPLATE, bt, work / f"opamp_nn_{bt.name}.sp")
    text = netlist.read_text()
    text = text.replace("Vbn vbn 0 0.36", f"Vbn vbn 0 {vbn}")
    text = text.replace("Vbp vbp 0 0.44", f"Vbp vbp 0 {vbp}")
    text = text.replace("Vinn inn 0 0.44", f"Vinn inn 0 {vcm}")
    text = text.replace("Vinp inp 0 0.44", f"Vinp inp 0 {vcm}")
    lo, hi = round(vcm - 0.15, 3), round(vcm + 0.15, 3)
    text = text.replace(".dc Vinp 0.29 0.59 0.002", f".dc Vinp {lo} {hi} 0.002")
    netlist.write_text(text)
    parser = parse_netlist(netlist)
    out_dir = work / "nn_dcsweep"; out_dir.mkdir(parents=True, exist_ok=True)
    results = run_dc_sweep(parser.circuit, parser.analysis_params,
                           Visualizer(), out_dir, f"nn_opamp_{bt.name}")
    return results, parser.circuit


def _run_l72_opamp(bt: BenchTech, work: Path):
    from pycircuitsim.simulation import run_dc_sweep
    from pycircuitsim.visualizer import Visualizer
    import pycircuitsim.simulation as _sim
    merged, n, p = _merged_card(bt, work)
    vdd, tfin_nm = bt.vdd, bt.tfin * 1e9
    ln, lp, nf = bt.l_nmos * 1e9, bt.l_pmos * 1e9, bt.nfin
    vcm, vbn, vbp = _bias(bt)
    lo, hi = round(vcm - 0.15, 3), round(vcm + 0.15, 3)
    deck = f"""* L72 opamp control ({bt.name})
Vdd vdd 0 {vdd}
Vbn vbn 0 {vbn}
Vbp vbp 0 {vbp}
Vinn inn 0 {vcm}
Vinp inp 0 {vcm}
Mn1 n1   inp vtail 0   {n} L={ln:.0f}n NFIN={nf} TFIN={tfin_nm:.1f}n
Mn2 vo1i inn vtail 0   {n} L={ln:.0f}n NFIN={nf} TFIN={tfin_nm:.1f}n
Mp3 n1   n1  vdd   vdd {p} L={lp:.0f}n NFIN={nf} TFIN={tfin_nm:.1f}n
Mp4 vo1i n1  vdd   vdd {p} L={lp:.0f}n NFIN={nf} TFIN={tfin_nm:.1f}n
Mn5 vtail vbn 0    0   {n} L={ln:.0f}n NFIN={nf} TFIN={tfin_nm:.1f}n
Mp6 vout vo1i vdd vdd {p} L={lp:.0f}n NFIN={nf} TFIN={tfin_nm:.1f}n
Mn7 vout vbn  0   0   {n} L={ln:.0f}n NFIN={nf} TFIN={tfin_nm:.1f}n
Cc vo1i vout 20f
CL vout 0 50f
.model {n} NMOS (LEVEL=72)
.model {p} PMOS (LEVEL=72)
.dc Vinp {lo} {hi} 0.002
.end
"""
    parser = _parse_l72(deck, work / f"opamp_l72_{bt.name}.sp", merged, n, p)
    out_dir = work / "l72_dcsweep"; out_dir.mkdir(parents=True, exist_ok=True)
    _orig = _sim._circuit_has_nn
    _sim._circuit_has_nn = lambda _c: True   # force the continuation-first path
    try:
        results = run_dc_sweep(parser.circuit, parser.analysis_params,
                               Visualizer(), out_dir, f"l72_opamp_{bt.name}")
    finally:
        _sim._circuit_has_nn = _orig
    return results, parser.circuit


# ── per-device small-signal extraction at a fixed OP ────────────────────────

def _dev_by_drain(circuit, drain: str, kind: str):
    for c in circuit.components:
        nodes = getattr(c, "nodes", None)
        if not nodes or len(nodes) < 4 or nodes[0] != drain:
            continue
        cls = c.__class__.__name__
        if kind == "N" and "NMOS" in cls:
            return c
        if kind == "P" and "PMOS" in cls:
            return c
    raise KeyError(f"no {kind}MOS with drain={drain}")


def _eval_any(dev, volts: Dict[str, float]) -> Dict[str, float]:
    fn = getattr(dev, "_eval", None) or getattr(dev, "_eval_dc")
    return fn(volts)


def _stage_gains(circuit, op: Dict[str, float]) -> Tuple[float, float, Dict]:
    """Analytic two-stage gain from the device autograd/OSDI Jacobian at OP.

    A1 = gm_n1 / (gds_n2 + gds_p4) ; A2 = gm_p6 / (gds_p6 + gds_n7).
    """
    mn1 = _dev_by_drain(circuit, "n1", "N")
    mn2 = _dev_by_drain(circuit, "vo1i", "N")
    mp4 = _dev_by_drain(circuit, "vo1i", "P")
    mp6 = _dev_by_drain(circuit, "vout", "P")
    mn7 = _dev_by_drain(circuit, "vout", "N")
    e = {k: _eval_any(d, op) for k, d in
         (("n1", mn1), ("n2", mn2), ("p4", mp4), ("p6", mp6), ("n7", mn7))}
    g1 = abs(e["n2"]["gds"]) + abs(e["p4"]["gds"])
    g2 = abs(e["p6"]["gds"]) + abs(e["n7"]["gds"])
    a1 = abs(e["n1"]["gm"]) / g1 if g1 > 0 else float("inf")
    a2 = abs(e["p6"]["gm"]) / g2 if g2 > 0 else float("inf")
    detail = {
        "gm1": abs(e["n1"]["gm"]), "gds_o1": g1, "A1": a1,
        "gm6": abs(e["p6"]["gm"]), "gds_o2": g2, "A2": a2,
    }
    return a1, a2, detail


def _nodes_at(results: Dict, sweep: np.ndarray, vin: float) -> Dict[str, float]:
    out = {}
    for k, arr in results.items():
        a = np.asarray(arr)
        if a.ndim != 1 or a.shape[0] != sweep.shape[0]:
            continue  # skip metadata / mismatched columns
        out[k] = float(np.interp(vin, sweep, a))
    return out


# ── driver ──────────────────────────────────────────────────────────────────

def run(name: str) -> Optional[Dict]:
    bt = BENCH[name]
    vdd = bt.vdd
    vcm, vbn, vbp = _bias(bt)
    work = RESULTS_BASE / "diag_opamp_op" / name
    work.mkdir(parents=True, exist_ok=True)

    logging.disable(logging.CRITICAL)
    try:
        nn_res, nn_circ = _run_nn_opamp(bt, work)
        l_res, l_circ = _run_l72_opamp(bt, work)
    finally:
        logging.disable(logging.NOTSET)

    sw_nn = np.asarray(nn_res["inp"]); vo_nn = np.asarray(nn_res["vout"])
    sw_l = np.asarray(l_res["inp"]); vo_l = np.asarray(l_res["vout"])
    g_nn, trip_nn, _ = _gain_trip(sw_nn, vo_nn, vdd)
    g_l, trip_l, _ = _gain_trip(sw_l, vo_l, vdd)

    # the true high-gain crossing = where L72 has peak |dVout/dVin|
    vin_star = float(sw_l[int(np.argmax(np.abs(np.gradient(vo_l, sw_l))))])

    op_l = _build_op(l_res, sw_l, vin_star, vcm, vbn, vbp, vdd)
    op_nn = _build_op(nn_res, sw_nn, vin_star, vcm, vbn, vbp, vdd)

    # value-surface test: NN devices at the TRUE (L72) OP
    a1_nn, a2_nn, det_nn = _stage_gains(nn_circ, op_l)
    a1_l, a2_l, det_l = _stage_gains(l_circ, op_l)
    A_nn, A_l = a1_nn * a2_nn, a1_l * a2_l

    # basin test: NN's own converged internal nodes vs L72's at vin_star
    basin_delta = max(abs(op_nn[k] - op_l[k]) for k in ("vtail", "n1", "vo1i"))

    print(f"\n===== {name} (VDD={vdd}) opamp gain->0 decomposition =====")
    print(f"  gate gains:  NN={g_nn:7.1f} (trip {trip_nn:.3f})   "
          f"L72={g_l:7.1f} (trip {trip_l:.3f})   vin*={vin_star:.3f}")
    print(f"  internal OP @ vin*:   {'node':>6} {'NN':>9} {'L72':>9} {'|Δ|(mV)':>9}")
    for k in ("vtail", "n1", "vo1i", "vout"):
        print(f"                       {k:>6} {op_nn[k]:9.4f} {op_l[k]:9.4f} "
              f"{abs(op_nn[k]-op_l[k])*1e3:9.1f}")
    print(f"  basin delta (max |Δ| over vtail/n1/vo1i) = {basin_delta*1e3:.1f} mV "
          f"(>{BASIN_TOL*1e3:.0f} mV ⇒ different basin)")

    print(f"\n  small-signal gain at the TRUE (L72) OP — analytic gm/(Σgds):")
    print(f"    {'':10} {'gm1(uS)':>9} {'gds_o1':>9} {'A1':>7} | "
          f"{'gm6(uS)':>9} {'gds_o2':>9} {'A2':>7} | {'A_tot':>8}")
    print(f"    {'L72':10} {det_l['gm1']*1e6:9.2f} {det_l['gds_o1']*1e6:9.3f} "
          f"{a1_l:7.2f} | {det_l['gm6']*1e6:9.2f} {det_l['gds_o2']*1e6:9.3f} "
          f"{a2_l:7.2f} | {A_l:8.1f}")
    print(f"    {'NN@L72OP':10} {det_nn['gm1']*1e6:9.2f} {det_nn['gds_o1']*1e6:9.3f} "
          f"{a1_nn:7.2f} | {det_nn['gm6']*1e6:9.2f} {det_nn['gds_o2']*1e6:9.3f} "
          f"{a2_nn:7.2f} | {A_nn:8.1f}")

    surf_ratio = A_nn / A_l if A_l > 1e-9 else float("nan")
    gate_pass = g_l > 1e-9 and abs(g_nn - g_l) / g_l <= 0.10
    if gate_pass:
        verdict = ("PASSING (positive control) — NN sits at the true OP "
                   f"(Δ={basin_delta*1e3:.0f}mV) with high swept gain ({g_nn:.0f}).")
    elif np.isfinite(surf_ratio) and surf_ratio >= 0.5:
        if basin_delta > BASIN_TOL:
            verdict = ("CANDIDATE-BASIN — surface HAS gain at the true OP (A_NN≈A_L72) "
                       "but the NN sweep lands a railed fixed point. CONFIRM WITH 1c "
                       "diag_opamp_basin_seed.py: if seeding from the L72 OP recovers "
                       "gain -> seedable (solver lever); if it still rails -> the OP is "
                       "UNSTABLE on the NN surface -> VALUE-SURFACE/corridor (2b).")
        else:
            verdict = ("INCONCLUSIVE — surface has gain AND NN sits at the true OP, "
                       "yet the swept gain is low. Inspect the sweep/measurement.")
    else:
        verdict = ("VALUE-SURFACE — the NN autograd gm/gds is genuinely flat at "
                   "the correct OP (A_NN≪A_L72). Route -> 2b trip-OP corridor "
                   "retrain (the only lever with a chance), medium, multi-seed.")
    print(f"  >> A_NN/A_L72 at true OP = {surf_ratio:.3f} ; basin Δ={basin_delta*1e3:.0f}mV")
    print(f"  >> VERDICT: {verdict}")
    return {"tech": name, "g_nn": g_nn, "g_l72": g_l, "A_nn_at_l72op": A_nn,
            "A_l72": A_l, "surf_ratio": surf_ratio, "basin_delta_mV": basin_delta * 1e3}


def _build_op(results: Dict, sweep: np.ndarray, vin: float,
              vcm: float, vbn: float, vbp: float, vdd: float) -> Dict[str, float]:
    nodes = _nodes_at(results, sweep, vin)
    # pin the fixed sources the sweep results may not carry as columns
    nodes.setdefault("vdd", vdd)
    nodes.update({"inp": vin, "inn": vcm, "vbn": vbn, "vbp": vbp,
                  "vdd": vdd, "0": 0.0})
    return nodes


def main() -> int:
    ap = argparse.ArgumentParser(description="Opamp basin-vs-value decomposition")
    ap.add_argument("--tech", default="TSMC7,TSMC12",
                    help="comma-separated; include a passing tech as control")
    args = ap.parse_args()
    print("=" * 78)
    print("Opamp gain->0 decomposition — BASIN (wrong OP) vs VALUE-SURFACE (flat gm·ro)")
    print("  NN devices evaluated at the L72-converged true high-gain OP")
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
