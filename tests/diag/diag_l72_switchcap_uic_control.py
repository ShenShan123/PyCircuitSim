"""Native-L72 switchcap control — with uic pinning.

Its predecessor (``diag_l72_switchcap_control.py``, removed in V7.5.9) ran the
L72 ground-truth switchcap through a PLAIN DC op with no uic pinning, so the
high-impedance hold node vsamp seeded at the leakage equilibrium (~vin)
instead of the .ic 0 V — it answered a question about a circuit neither the NN
path nor NGSPICE runs, and this file superseded it on arrival. The NN path
(run_directnet_transient, V6.4.7 S5b) pins .ic nodes with temporary ideal
sources during the OP so the transient starts at .ic exactly — matching
NGSPICE `tran ... uic`. This control replicates THAT uic pinning with the L72
model, so the L72 control runs the SAME experiment as the NN path and the
NGSPICE ground truth. If the L72-with-uic result now matches NGSPICE, the
14.65% "solver floor" was a control-harness artifact (missing uic), not a
real PyCircuitSim transient-solver gap.
"""
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from tests.common.complex import BENCH  # noqa: E402
from pycircuitsim.parser import Parser  # noqa: E402
from pycircuitsim.solver import DCSolver, TransientSolver  # noqa: E402
from pycircuitsim.models.passive import VoltageSource  # noqa: E402

TRAN_TSTEP, TRAN_TSTOP = 5e-12, 12e-9
SAMPLE_END = 2.3e-9


def run(name="TSMC5", use_uic=True):
    bt = BENCH[name]
    prof = bt.profile
    vp = prof.get_vt_pair(bt.vt)
    nmos_src = prof.get_nmos_modelcard(vp, bt.l_nmos)
    pmos_src = prof.get_pmos_modelcard(vp, bt.l_pmos)
    work = PROJECT_ROOT / "results" / "v6_5_2_solver" / "l72_uic" / name
    work.mkdir(parents=True, exist_ok=True)
    merged = work / f"merged_{name}.lib"
    merged.write_text(nmos_src.read_text() + "\n" + pmos_src.read_text())

    n, p = vp.nmos_model, vp.pmos_model
    vdd = bt.vdd
    vin = round(vdd * 0.6, 3)
    tfin_nm = prof.tfin * 1e9
    ln, lp, nf = bt.l_nmos * 1e9, bt.l_pmos * 1e9, bt.nfin
    deck = f"""* L72 switchcap uic control ({name})
Vdd vdd 0 {vdd}
Vin vin 0 {vin}
Vphi phi 0 PULSE 0 {vdd} 0.5n 0.1n 0.1n 1.9n 4n
Mpc phib phi vdd vdd {p} L={lp:.0f}n NFIN={nf} TFIN={tfin_nm:.1f}n
Mnc phib phi 0 0 {n} L={ln:.0f}n NFIN={nf} TFIN={tfin_nm:.1f}n
Mnt vin phi vsamp 0 {n} L={ln:.0f}n NFIN={nf} TFIN={tfin_nm:.1f}n
Mpt vin phib vsamp vdd {p} L={lp:.0f}n NFIN={nf} TFIN={tfin_nm:.1f}n
Csample vsamp 0 100f
.ic V(vsamp)=0 V(phib)={vdd}
.model {n} NMOS (LEVEL=72)
.model {p} PMOS (LEVEL=72)
.tran {TRAN_TSTEP:.0e} {TRAN_TSTOP:.0e}
.end
"""
    deck_path = work / f"l72_sc_{name}.sp"
    deck_path.write_text(deck)

    logging.disable(logging.CRITICAL)
    try:
        parser = Parser(modelcard_path=str(merged),
                        model_name_map={"NMOS": n, "PMOS": p})
        parser.parse_file(str(deck_path))
        circuit = parser.circuit
        guess = circuit.initial_conditions or None

        # --- uic pinning (mirror run_directnet_transient V6.4.7 S5b) ---
        _temps = []
        if use_uic and circuit.initial_conditions:
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
                    _temps.append(vs)
        try:
            op = DCSolver(circuit, initial_guess=guess,
                          use_source_stepping=True).solve()
        finally:
            for vs in _temps:
                circuit.components.remove(vs)

        vsamp_seed = op.get("vsamp", float("nan"))
        solver = TransientSolver(
            circuit, t_stop=TRAN_TSTOP, dt=TRAN_TSTEP, initial_guess=op,
            use_gmin_stepping=True, gmin_initial=1e-9, gmin_final=1e-12,
            gmin_steps=5, use_pseudo_transient=True, pseudo_transient_steps=5,
            pseudo_transient_cap=1e-12, debug=False, nr_tolerance=1e-7)
        res = solver.solve()
    finally:
        logging.disable(logging.NOTSET)

    t = np.asarray(res["time"]); vs = np.asarray(res["vsamp"])
    v_at = float(np.interp(SAMPLE_END, t, vs))
    from tests.simple_circuits.verify_complex_switchcap import run_ngspice_sc, _at, SAMPLE_END as SE
    ng_data = run_ngspice_sc(bt, work)
    ng = _at(ng_data["time"], ng_data["vsamp"], SE)
    err = abs(v_at - ng) / vdd * 100.0
    tag = "uic-pinned" if use_uic else "plain-DCop"
    print(f"{name} [{tag}]: OP-seed vsamp={vsamp_seed:.4f} | "
          f"vsamp@2.3ns = {v_at:.4f} | NGSPICE = {ng:.4f} | err = {err:.2f}% VDD")
    return v_at, err


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", default="TSMC5,TSMC7,TSMC12,TSMC16")
    args = ap.parse_args()
    print("L72 switchcap control WITH uic pinning vs the plain-DCop control.\n")
    for t in [x.strip() for x in args.tech.split(",")]:
        try:
            run(t, use_uic=False)   # the OLD (flawed) control
            run(t, use_uic=True)    # the CORRECTED control
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"{t}: ERROR {exc!r}")
            traceback.print_exc()
