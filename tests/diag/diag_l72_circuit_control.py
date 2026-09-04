"""Native-LEVEL=72 controls for the ring-oscillator and opamp gates.

The project's most productive discipline (force_ic, switchcap) is: *run the
EXACT benchmark circuit through PyCircuitSim's OWN solver with the ground-truth
BSIM-CMG (LEVEL=72) OSDI model — no NN — and compare to NGSPICE.* If the L72
control ALSO misses the gate, the gap is a SOLVER / HARNESS artifact, not an NN
model gap (a gate that rejects ground-truth physics is mis-specified). If the
L72 control MATCHES NGSPICE, the gap is genuinely the NN value-surface.

This control was recommended in plan V6.5.2.4 ("apply the L72 control to the
ring-osc timing gaps before any NN lever — they may be partly solver-owned
too") but had never been built. It reuses the gate's own NGSPICE reference
builders so the L72-in-PyCircuitSim run and the NGSPICE run are byte-identical
circuits — only the simulator differs.

Usage:
    NGSPICE_BIN=$PWD/tools/ngspice-45.2/bin/ngspice \
      conda run -n pycircuitsim python tests/diag/diag_l72_circuit_control.py \
      --circuit ring,opamp --tech TSMC5,TSMC7,TSMC12,TSMC16
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "bsim_cmg" / "tests"))

from tests.common.circuit_benchmarks import (  # noqa: E402
    BENCH,
    RESULTS_BASE,
    BenchTech,
    OpAmpParams,
    gain_trip,
    opamp_bias,
)
from tests.common.base import template_deck, render_template  # noqa: E402
from pycircuitsim.parser import Parser  # noqa: E402
from pycircuitsim.solver import DCSolver, TransientSolver  # noqa: E402
from pycircuitsim.models.passive import VoltageSource  # noqa: E402

# ring-osc gate constants (mirror verify_circuit_ring_osc.py)
RO_TSTEP, RO_TSTOP, RO_SETTLE = 2e-12, 1.2e-9, 0.3e-9


def _merged_card(bt: BenchTech, work: Path):
    prof = bt.profile
    vp = prof.get_vt_pair(bt.vt)
    nmos_src = prof.get_nmos_modelcard(vp, bt.l_nmos)
    pmos_src = prof.get_pmos_modelcard(vp, bt.l_pmos)
    merged = work / f"merged_{bt.name}.lib"
    merged.write_text(nmos_src.read_text() + "\n" + pmos_src.read_text())
    return merged, vp.nmos_model, vp.pmos_model


def _parse_l72(deck_text: str, deck_path: Path, merged: Path, n: str, p: str):
    deck_path.write_text(deck_text)
    parser = Parser(modelcard_path=str(merged),
                    model_name_map={"NMOS": n, "PMOS": p})
    parser.parse_file(str(deck_path))
    return parser


def _uic_op(circuit):
    """OP with .ic nodes pinned (mirror run_directnet_transient V6.4.7 S5b)."""
    guess = circuit.initial_conditions or None
    temps = []
    if circuit.initial_conditions:
        vs_constrained = set()
        for comp in circuit.components:
            if isinstance(comp, VoltageSource):
                if comp.nodes[1] in ("0", "GND"):
                    vs_constrained.add(comp.nodes[0])
                elif comp.nodes[0] in ("0", "GND"):
                    vs_constrained.add(comp.nodes[1])
        node_map = circuit.get_node_map()
        for node, val in circuit.initial_conditions.items():
            if (node not in ("0", "GND") and node not in vs_constrained
                    and node in node_map):
                vs = VoltageSource(f"_V_uic_{node}", [node, "0"], val)
                circuit.components.append(vs)
                temps.append(vs)
        if temps:
            circuit.invalidate_topology()
    try:
        op = DCSolver(circuit, initial_guess=guess,
                      use_source_stepping=True).solve()
    finally:
        for vs in temps:
            circuit.components.remove(vs)
        if temps:
            circuit.invalidate_topology()
    return op


def _verdict(err: float, tol_pct: float,
             undefined_why: str = "control metric undefined") -> str:
    """Three-way ownership verdict for one control cell.

    audit B5h: a two-way ``"SOLVER-OWNED" if err > tol else "model-owned"`` is
    wrong for a non-finite ``err`` — ``float("nan") > tol`` is False in Python,
    so the cell that produced NO comparable number at all (ring that never
    oscillated, non-positive reference gain) printed the maximally reassuring
    "L72 matches NGSPICE". Both call sites share this helper so the NaN branch
    cannot drift apart again.
    """
    if not np.isfinite(err):
        return f"INCONCLUSIVE ({undefined_why}; NOT evidence of a model-owned gap)"
    if err > tol_pct:
        return "SOLVER-OWNED (L72 misses gate too)"
    return "model-owned (L72 matches NGSPICE)"


def _period(t, v, mid, settle):
    keep = t >= settle
    t, v = t[keep], v[keep]
    sign = np.sign(v - mid)
    cross = np.where((sign[:-1] < 0) & (sign[1:] >= 0))[0]
    if len(cross) < 3:
        return float("nan")
    times = []
    for i in cross:
        v0, v1 = v[i], v[i + 1]
        frac = (mid - v0) / (v1 - v0) if v1 != v0 else 0.0
        times.append(t[i] + frac * (t[i + 1] - t[i]))
    return float(np.mean(np.diff(times)))


def ring(bt: BenchTech) -> Dict:
    """L72-in-PyCircuitSim 5-stage ring osc vs NGSPICE."""
    from tests.simple_circuits.verify_circuit_ring_osc import run_ngspice_ro
    work = RESULTS_BASE / "l72_control" / "ring" / bt.name
    work.mkdir(parents=True, exist_ok=True)
    merged, n, p = _merged_card(bt, work)
    vdd, tfin_nm = bt.vdd, bt.tfin * 1e9
    ln, lp, nf = bt.l_nmos * 1e9, bt.l_pmos * 1e9, bt.nfin

    deck = render_template(template_deck("ring_oscillator.spice.tmpl"), {
        "MODEL_SETUP": (
            f".model {n} NMOS (LEVEL=72)\n"
            f".model {p} PMOS (LEVEL=72)"
        ),
        "TEMP": "27", "VDD": f"{vdd}",
        "N_PREFIX": "M", "P_PREFIX": "M",
        "N_DEVICE": (
            f"{n} L={ln:.0f}n NFIN={nf} TFIN={tfin_nm:.1f}n"
        ),
        "P_DEVICE": (
            f"{p} L={lp:.0f}n NFIN={nf} TFIN={tfin_nm:.1f}n"
        ),
        "RING_CLOAD": "0.5f",
        "ANALYSIS": f".tran {RO_TSTEP:.0e} {RO_TSTOP:.0e}",
    })

    logging.disable(logging.CRITICAL)
    try:
        parser = _parse_l72(deck, work / "ring.sp", merged, n, p)
        circuit = parser.circuit
        op = _uic_op(circuit)
        solver = TransientSolver(
            circuit, t_stop=RO_TSTOP, dt=RO_TSTEP, initial_guess=op,
            use_gmin_stepping=True, gmin_initial=1e-9, gmin_final=1e-12,
            gmin_steps=5, use_pseudo_transient=True, pseudo_transient_steps=5,
            pseudo_transient_cap=1e-12, debug=False, nr_tolerance=1e-7)
        res = solver.solve()
    finally:
        logging.disable(logging.NOTSET)

    t = np.asarray(res["time"]); v = np.asarray(res["n5"])
    l72_per = _period(t, v, vdd / 2.0, RO_SETTLE)
    ng = run_ngspice_ro(bt, work)
    ng_per = _period(ng["time"], ng["v(n5)"], vdd / 2.0, RO_SETTLE)
    err = (abs(l72_per - ng_per) / ng_per * 100.0
           if np.isfinite(l72_per) and ng_per > 0 else float("nan"))
    verdict = _verdict(err, 5.0,
                       "period undefined: <3 mid-crossings — the L72 or "
                       "NGSPICE control never oscillated")
    print(f"  RING {bt.name}: L72-PyCS period={l72_per*1e12:7.2f}ps | "
          f"NGSPICE={ng_per*1e12:7.2f}ps | err={err:6.2f}%  -> {verdict}")
    return {"tech": bt.name, "l72_per": l72_per, "ng_per": ng_per, "err": err}


def opamp(bt: BenchTech) -> Dict:
    """L72-in-PyCircuitSim two-stage Miller opamp DC gain vs NGSPICE."""
    from tests.simple_circuits.verify_circuit_opamp import run_ngspice_opamp
    from pycircuitsim.simulation import run_dc_sweep
    from pycircuitsim.visualizer import Visualizer
    work = RESULTS_BASE / "l72_control" / "opamp" / bt.name
    work.mkdir(parents=True, exist_ok=True)
    merged, n, p = _merged_card(bt, work)
    vdd, tfin_nm = bt.vdd, bt.tfin * 1e9
    ln, lp, nf = bt.l_nmos * 1e9, bt.l_pmos * 1e9, bt.nfin
    vcm, vbn, vbp = opamp_bias(bt, OpAmpParams())
    lo, hi = round(vcm - 0.15, 3), round(vcm + 0.15, 3)

    deck = render_template(template_deck("opamp_miller.spice.tmpl"), {
        "MODEL_SETUP": (
            f".model {n} NMOS (LEVEL=72)\n"
            f".model {p} PMOS (LEVEL=72)"
        ),
        "TEMP": "27", "VDD_SPEC": f"{vdd}",
        "VBN": f"{vbn}", "VBP": f"{vbp}",
        "VINP_SPEC": f"{vcm}", "VINN_SPEC": f"{vcm}",
        "N_PREFIX": "M", "P_PREFIX": "M",
        "N_DEVICE": (
            f"{n} L={ln:.0f}n NFIN={nf} TFIN={tfin_nm:.1f}n"
        ),
        "P_DEVICE": (
            f"{p} L={lp:.0f}n NFIN={nf} TFIN={tfin_nm:.1f}n"
        ),
        "CC": "20f", "CL": "50f",
        "ANALYSIS": f".dc Vinp {lo} {hi} 0.002",
    })
    # Force the SAME DC-sweep path the NN opamp uses: continuation-first
    # (warm start, source-stepping OFF) + GMIN retry. run_dc_sweep keys this on
    # _circuit_has_nn (True only for LEVEL>=73). For an apples-to-apples
    # ownership test we monkeypatch it True so the L72 ground-truth opamp takes
    # the identical solver path. Without this the L72 control uses the legacy
    # per-point source-stepping homotopy (which diverges here) — a DIFFERENT
    # experiment from the NN, so its divergence would NOT prove the gate
    # mis-specified. force_path in {"nn","legacy"}.
    import pycircuitsim.simulation as _sim
    _orig = _sim._circuit_has_nn
    logging.disable(logging.CRITICAL)
    try:
        parser = _parse_l72(deck, work / "opamp.sp", merged, n, p)
        circuit = parser.circuit
        out_dir = work / "dcsweep"
        out_dir.mkdir(parents=True, exist_ok=True)
        _sim._circuit_has_nn = lambda _c: True   # force continuation-first path
        try:
            results = run_dc_sweep(circuit, parser.analysis_params,
                                   Visualizer(), out_dir, f"l72_opamp_{bt.name}")
        finally:
            _sim._circuit_has_nn = _orig
    finally:
        logging.disable(logging.NOTSET)

    sweep = np.asarray(results["inp"]); vout = np.asarray(results["vout"])
    l72_gain, l72_trip, _ = gain_trip(sweep, vout, vdd)
    ng = run_ngspice_opamp(bt, work)
    ng_gain, ng_trip, _ = gain_trip(ng["sweep"], ng["vout"], vdd)
    err = abs(l72_gain - ng_gain) / ng_gain * 100.0 if ng_gain > 0 else float("nan")
    verdict = _verdict(err, 10.0,
                       "L72 or NGSPICE gain undefined/non-positive")
    print(f"  OPAMP {bt.name}: L72-PyCS gain={l72_gain:7.1f} trip={l72_trip:.4f} | "
          f"NGSPICE gain={ng_gain:7.1f} trip={ng_trip:.4f} | err={err:6.2f}%  -> {verdict}")
    return {"tech": bt.name, "l72_gain": l72_gain, "ng_gain": ng_gain, "err": err}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="ring,opamp")
    ap.add_argument("--tech", default="TSMC5,TSMC7,TSMC12,TSMC16")
    args = ap.parse_args()
    circuits = [c.strip() for c in args.circuit.split(",")]
    techs = [t.strip() for t in args.tech.split(",")]

    print("=" * 78)
    print("Native-L72 control: exact circuit benchmarks through PyCircuitSim's")
    print("own solver with ground-truth OSDI (no NN) vs NGSPICE.")
    print("  Gate that rejects ground-truth physics == mis-specified / solver-owned.")
    print("=" * 78)
    # audit B5h: a cell that raised, or that produced a non-finite err, yields NO
    # ownership verdict. Tally them so a reader cannot mistake a broken control
    # run for a "model-owned" result.
    n_cells = n_inconclusive = 0
    if "ring" in circuits:
        print("\n[Ring oscillator — period gate +/-5%]")
        for t in techs:
            n_cells += 1
            try:
                if not np.isfinite(ring(BENCH[t])["err"]):
                    n_inconclusive += 1
            except Exception as exc:  # noqa: BLE001
                n_inconclusive += 1
                import traceback; print(f"  RING {t}: ERROR {exc!r}"); traceback.print_exc()
    if "opamp" in circuits:
        print("\n[Two-stage Miller opamp — gain gate +/-10%]")
        for t in techs:
            n_cells += 1
            try:
                if not np.isfinite(opamp(BENCH[t])["err"]):
                    n_inconclusive += 1
            except Exception as exc:  # noqa: BLE001
                n_inconclusive += 1
                import traceback; print(f"  OPAMP {t}: ERROR {exc!r}"); traceback.print_exc()

    print(f"\n{n_inconclusive}/{n_cells} cells INCONCLUSIVE (no usable "
          f"L72-vs-NGSPICE comparison — read NO ownership verdict from them).")
    # Diagnostic, not a gate: the exit code stays 0 even when cells are
    # inconclusive, so no caller starts treating this script as a pass/fail.
    return 0


if __name__ == "__main__":
    sys.exit(main())
