"""Ring-osc period decomposition: CONDUCTION (gm under-drive) vs CAP (dQ/dV).

V6.5.5 Tier-1a routing diagnostic for the open tsmc5 ring-osc gate (period
12.66% too LONG = inverter too slow at the switching edge). The native-L72
control (tests/diag_l72_complex_control.py) already proved the gate is well-posed
(L72-in-PyCircuitSim period err ~0%), so the residual is the NN value surface.
This probe localizes WHICH part of the surface owns it, so the (deferred,
A/B-gated) corridor retrain can be ROUTED instead of guessed.

Ring period = Σ ∫ C_node·dV / I_drive. "Too slow" has exactly two mechanisms:
  (a) CONDUCTION — the drain drive current I_drive at the VDD/2 edge is
      under-predicted (autograd gm / id too low), or
  (b) CAP — the intrinsic transcap load C_node = autograd dQ/dV is over-predicted
      (more charge to move per volt).

DECISIVE GLOBAL SPLIT (Part A): toggle the MOSFET intrinsic-charge transient
stamp OFF (TransientSolver._stamp_mosfet_transient -> _stamp_mosfet_dc, the
switchcap-diag pattern). With charge OFF only the explicit Cl=0.5f remains, so
the NN-vs-L72 period difference is PURE conduction. Run BOTH models BOTH ways:
  conduction_err = |T_NN_off - T_L72_off| / T_L72_off   (charge removed)
  total_err      = |T_NN_on  - T_L72_on | / T_L72_on    (≈ 12.66%, the gate)
  conduction_err ≈ total_err  -> CONDUCTION-owned  (route to corridor 2a)
  conduction_err ≈ 0          -> CAP-owned         (charge model injects it)

LOCALIZATION (Part B): a static stage-1 inverter drive comparison at the
high-gain crossing (vout = VDD/2), sweeping vin across the switching band — the
per-device NN `_eval` vs OSDI `_eval_dc` {id, gm} that sets the edge rate.

Usage:
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NGSPICE_BIN="$PWD/tools/ngspice-45.2/bin/ngspice" \
      conda run -n pycircuitsim python tests/diag_nn_ring_trajectory.py --tech TSMC5
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

from tests.common.complex import (  # noqa: E402
    BENCH, RESULTS_BASE, BenchTech, render_directnet_netlist, parse_netlist,
)
from tests.diag_l72_complex_control import (  # noqa: E402
    _merged_card, _parse_l72, _period, RO_TSTEP, RO_TSTOP, RO_SETTLE,
)
from pycircuitsim.solver import (  # noqa: E402
    DCSolver, TransientSolver, _stamp_mosfet_dc,
)
from pycircuitsim.models.passive import VoltageSource  # noqa: E402

RING_TEMPLATE = PROJECT_ROOT / "examples" / "complex" / "ring_osc_5stage_directnet.sp"


# ── circuit builders (identical topology, only the model differs) ───────────

def _build_nn_ring(bt: BenchTech, work: Path):
    netlist = render_directnet_netlist(RING_TEMPLATE, bt, work / f"ring_nn_{bt.name}.sp")
    text = netlist.read_text().replace(
        ".tran 1p 5n",
        f".tran {RO_TSTEP*1e12:.0f}p {RO_TSTOP*1e9:.2f}n")
    netlist.write_text(text)
    return parse_netlist(netlist).circuit


def _build_l72_ring(bt: BenchTech, work: Path):
    merged, n, p = _merged_card(bt, work)
    vdd, tfin_nm = bt.vdd, bt.tfin * 1e9
    ln, lp, nf = bt.l_nmos * 1e9, bt.l_pmos * 1e9, bt.nfin
    nd = [f"n{i}" for i in range(1, 6)]
    lines = [f"* L72 ring control ({bt.name})", f"Vdd vdd 0 {vdd}",
             f".ic V(n1)=0.0 V(n2)={vdd} V(n3)=0.0 V(n4)={vdd} V(n5)=0.0"]
    for i in range(5):
        inp = nd[i - 1]  # i=0 -> n5 feedback wrap
        lines += [
            f"Mp{i} {nd[i]} {inp} vdd vdd {p} L={lp:.0f}n NFIN={nf} TFIN={tfin_nm:.1f}n",
            f"Mn{i} {nd[i]} {inp} 0 0 {n} L={ln:.0f}n NFIN={nf} TFIN={tfin_nm:.1f}n",
            f"Cl{i} {nd[i]} 0 0.5f"]
    lines += [f".model {n} NMOS (LEVEL=72)", f".model {p} PMOS (LEVEL=72)",
              f".tran {RO_TSTEP:.0e} {RO_TSTOP:.0e}", ".end"]
    circuit = _parse_l72("\n".join(lines) + "\n", work / "ring_l72.sp",
                         merged, n, p).circuit
    return circuit


# ── shared solve (uic-OP + transient, optional charge toggle) ───────────────

def _uic_op_retry(circuit):
    """uic-pinned DC OP mirroring run_directnet_transient (fast + GMIN retry)."""
    guess = circuit.initial_conditions or None
    temps: List[VoltageSource] = []
    if circuit.initial_conditions:
        vsc = set()
        for comp in circuit.components:
            if isinstance(comp, VoltageSource):
                if comp.nodes[1] in ("0", "GND"):
                    vsc.add(comp.nodes[0])
                elif comp.nodes[0] in ("0", "GND"):
                    vsc.add(comp.nodes[1])
        nm = circuit.get_node_map()
        for node, val in circuit.initial_conditions.items():
            if node not in ("0", "GND") and node not in vsc and node in nm:
                vs = VoltageSource(f"_V_uic_{node}", [node, "0"], val)
                circuit.components.append(vs); temps.append(vs)
    try:
        op = DCSolver(circuit, initial_guess=guess, use_source_stepping=True,
                      use_gmin_stepping=False)
        try:
            sol = op.solve()
            if not getattr(op, "_last_solve_converged", True):
                raise RuntimeError("OP fast path did not converge")
        except (RuntimeError, np.linalg.LinAlgError):
            op = DCSolver(circuit, initial_guess=guess, use_source_stepping=True,
                          use_gmin_stepping=True)
            sol = op.solve()
    finally:
        for vs in temps:
            circuit.components.remove(vs)
    return sol


def _solve_ring(circuit, charge_on: bool) -> Dict[str, np.ndarray]:
    op = _uic_op_retry(circuit)
    orig = TransientSolver._stamp_mosfet_transient
    if not charge_on:
        def _dc_only(self, mosfet, mna, rhs, node_map, voltages):
            _stamp_mosfet_dc(mosfet, mna, rhs, node_map, voltages, self.gmin)
        TransientSolver._stamp_mosfet_transient = _dc_only
    try:
        solver = TransientSolver(
            circuit, t_stop=RO_TSTOP, dt=RO_TSTEP, initial_guess=op,
            use_gmin_stepping=True, gmin_initial=1e-9, gmin_final=1e-12,
            gmin_steps=5, use_pseudo_transient=True, pseudo_transient_steps=5,
            pseudo_transient_cap=1e-12, debug=False, nr_tolerance=1e-7)
        res = solver.solve()
    finally:
        TransientSolver._stamp_mosfet_transient = orig
    return {k: np.asarray(v) for k, v in res.items()}


def _ring_period(res: Dict[str, np.ndarray], vdd: float) -> float:
    return _period(res["time"], res["n5"], vdd / 2.0, RO_SETTLE)


# ── Part B: static stage-1 drive comparison NN vs OSDI ──────────────────────

def _find_inv_devices(circuit, out_node: str):
    nmos = pmos = None
    for c in circuit.components:
        nodes = getattr(c, "nodes", None)
        if not nodes or len(nodes) < 4 or nodes[0] != out_node:
            continue
        cls = c.__class__.__name__
        if "NMOS" in cls:
            nmos = c
        elif "PMOS" in cls:
            pmos = c
    return nmos, pmos


def _eval_any(dev, volts: Dict[str, float]) -> Dict[str, float]:
    fn = getattr(dev, "_eval", None) or getattr(dev, "_eval_dc")
    return fn(volts)


def _drive_sweep(circuit_nn, circuit_l72, vdd: float) -> List[Tuple]:
    """At the high-gain crossing vout=VDD/2, sweep vin across the switching
    band; dump per-device id/gm (NN vs OSDI) — the drive that sets edge rate."""
    nn_n, nn_p = _find_inv_devices(circuit_nn, "n1")
    l_n, l_p = _find_inv_devices(circuit_l72, "n1")
    rows = []
    vout = vdd / 2.0
    for frac in (0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70):
        vin = frac * vdd
        volts = {"n1": vout, "n5": vin, "vdd": vdd, "0": 0.0}
        en_nn = _eval_any(nn_n, volts); ep_nn = _eval_any(nn_p, volts)
        en_l = _eval_any(l_n, volts); ep_l = _eval_any(l_p, volts)
        rows.append((vin,
                     en_nn["id"], en_l["id"], ep_nn["id"], ep_l["id"],
                     en_nn["gm"], en_l["gm"], ep_nn["gm"], ep_l["gm"]))
    return rows


# ── driver ──────────────────────────────────────────────────────────────────

def run(name: str) -> Optional[Dict]:
    bt = BENCH[name]
    vdd = bt.vdd
    work = RESULTS_BASE / "diag_ring_traj" / name
    work.mkdir(parents=True, exist_ok=True)

    logging.disable(logging.CRITICAL)
    try:
        nn_on = _build_nn_ring(bt, work)
        T_nn_on = _ring_period(_solve_ring(nn_on, charge_on=True), vdd)
        nn_off = _build_nn_ring(bt, work)
        T_nn_off = _ring_period(_solve_ring(nn_off, charge_on=False), vdd)

        l_on = _build_l72_ring(bt, work)
        T_l_on = _ring_period(_solve_ring(l_on, charge_on=True), vdd)
        l_off = _build_l72_ring(bt, work)
        T_l_off = _ring_period(_solve_ring(l_off, charge_on=False), vdd)

        # fresh circuits for the static device dump (avoid cached eval state)
        drive = _drive_sweep(_build_nn_ring(bt, work), _build_l72_ring(bt, work), vdd)
    finally:
        logging.disable(logging.NOTSET)

    def _err(a, b):
        return abs(a - b) / b * 100.0 if (np.isfinite(a) and b > 0) else float("nan")

    total_err = _err(T_nn_on, T_l_on)
    cond_err = _err(T_nn_off, T_l_off)

    print(f"\n===== {name} (VDD={vdd}) ring period decomposition =====")
    print(f"  {'':14s} | {'NN (ps)':>9s} | {'L72 (ps)':>9s} | {'err vs L72':>10s}")
    print(f"  {'charge ON':14s} | {T_nn_on*1e12:9.2f} | {T_l_on*1e12:9.2f} | "
          f"{total_err:9.2f}%   <- the gate (~12.66%)")
    print(f"  {'charge OFF':14s} | {T_nn_off*1e12:9.2f} | {T_l_off*1e12:9.2f} | "
          f"{cond_err:9.2f}%   <- pure conduction")

    # verdict: how much of the total error survives with charge removed
    share = (cond_err / total_err * 100.0
             if np.isfinite(cond_err) and np.isfinite(total_err) and total_err > 1e-9
             else float("nan"))
    if np.isfinite(share):
        if share >= 60.0:
            verdict = ("CONDUCTION-owned (gm/id under-drive). Route -> ring-edge "
                       "corridor retrain (Tier-2a); EXPECT a tsmc5 ring<->opamp trade.")
        elif share <= 25.0:
            verdict = ("CAP-owned (intrinsic dQ/dV over-prediction). Charge-band "
                       "densification (contraindicated by prior under-drive evidence "
                       "-- re-examine).")
        else:
            verdict = "MIXED conduction+cap. Inspect Part B before committing GPU."
        print(f"  >> conduction explains {share:.0f}% of the period error -> {verdict}")
    else:
        print("  >> degenerate (non-oscillating) -- inspect raw periods above.")

    print(f"\n  [Part B] stage-1 drive at vout=VDD/2 (the high-gain crossing) "
          f"-- id & gm, NN vs OSDI:")
    print(f"  {'vin':>6} | {'idN_NN':>9} {'idN_L72':>9} {'r':>5} | "
          f"{'idP_NN':>9} {'idP_L72':>9} {'r':>5} | {'gmN_NN':>8} {'gmN_L72':>8} {'r':>5}")
    band_ratios = []
    for (vin, idnnn, idnl, idpnn, idpl, gmnnn, gmnl, gmpnn, gmpl) in drive:
        rN = idnnn / idnl if abs(idnl) > 1e-15 else float("nan")
        rP = idpnn / idpl if abs(idpl) > 1e-15 else float("nan")
        rgN = gmnnn / gmnl if abs(gmnl) > 1e-15 else float("nan")
        if 0.40 * vdd <= vin <= 0.60 * vdd and np.isfinite(rN):
            band_ratios.append(rN)
        print(f"  {vin:6.3f} | {idnnn*1e6:9.3f} {idnl*1e6:9.3f} {rN:5.2f} | "
              f"{idpnn*1e6:9.3f} {idpl*1e6:9.3f} {rP:5.2f} | "
              f"{gmnnn*1e6:8.2f} {gmnl*1e6:8.2f} {rgN:5.2f}")
    if band_ratios:
        med = float(np.median(band_ratios))
        print(f"  >> median NMOS id ratio (NN/OSDI) in [0.4,0.6]*VDD switching band "
              f"= {med:.3f}  ({'UNDER-drive' if med < 0.95 else 'OK/over'} ; "
              f"under-drive slows the edge => longer period)")
    print("  (id/gm in uA / uS; PyCMG sign: NMOS id<0 when ON. r = NN/OSDI ratio.)")
    return {"tech": name, "total_err": total_err, "cond_err": cond_err,
            "cond_share_pct": share, "T_nn_on": T_nn_on, "T_l72_on": T_l_on}


def main() -> int:
    ap = argparse.ArgumentParser(description="Ring conduction-vs-cap decomposition")
    ap.add_argument("--tech", default="TSMC5",
                    help="comma-separated techs (TSMC5/7/12/16)")
    args = ap.parse_args()
    print("=" * 78)
    print("Ring-osc period decomposition — CONDUCTION (gm under-drive) vs CAP")
    print("  charge-OFF isolates conduction; native-L72 is the in-solver reference")
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
