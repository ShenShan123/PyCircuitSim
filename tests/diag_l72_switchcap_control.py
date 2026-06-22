"""Native-LEVEL=72 switchcap control: PyCircuitSim solver + ground-truth BSIM-CMG.

Runs the EXACT switchcap (same topology/stimulus as verify_complex_switchcap.py's
NGSPICE deck) through PyCircuitSim's OWN transient solver but with the LEVEL=72
BSIM-CMG OSDI model — no NN. If this ALSO over-charges vs NGSPICE's 0.2948 V, the
~11-12% switchcap "gap" is a PyCircuitSim-solver-vs-NGSPICE difference, not the NN
(cf. the V6.4.7 force_ic harness-bug lesson: run the native-L72 control first).
"""
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from tests.common.complex import BENCH  # noqa: E402
from pycircuitsim.parser import Parser  # noqa: E402
from pycircuitsim.solver import DCSolver, TransientSolver  # noqa: E402

TRAN_TSTEP, TRAN_TSTOP = 5e-12, 12e-9
SAMPLE_END = 2.3e-9


def run(name="TSMC5"):
    bt = BENCH[name]
    prof = bt.profile
    vp = prof.get_vt_pair(bt.vt)
    nmos_src = prof.get_nmos_modelcard(vp, bt.l_nmos)
    pmos_src = prof.get_pmos_modelcard(vp, bt.l_pmos)
    work = PROJECT_ROOT / "results" / "v6_7_csob" / "l72ctrl"
    work.mkdir(parents=True, exist_ok=True)
    merged = work / f"merged_{name}.lib"
    merged.write_text(nmos_src.read_text() + "\n" + pmos_src.read_text())

    n, p = vp.nmos_model, vp.pmos_model
    vdd = bt.vdd
    vin = round(vdd * 0.6, 3)
    tfin_nm = prof.tfin * 1e9
    ln, lp, nf = bt.l_nmos * 1e9, bt.l_pmos * 1e9, bt.nfin
    deck = f"""* L72 switchcap control ({name})
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
        op = DCSolver(circuit, initial_guess=guess, use_source_stepping=True).solve()
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
    # True per-tech NGSPICE reference (same deck verify_complex_switchcap uses).
    from tests.verify_complex_switchcap import run_ngspice_sc, _at, SAMPLE_END as SE
    ng_data = run_ngspice_sc(bt, work)
    ng = _at(ng_data["time"], ng_data["vsamp"], SE)
    err = abs(v_at - ng) / vdd * 100.0
    print(f"{name}: L72(PyCircuitSim solver) vsamp@2.3ns = {v_at:.4f} V  "
          f"| NGSPICE = {ng:.4f} V  | err = {err:.2f}% of VDD")
    print(f"  >> If err is large, the switchcap gap is SOLVER-owned (PyCircuitSim "
          f"transient vs NGSPICE), NOT the NN.")
    return v_at, err


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Native-L72 switchcap control: solver-vs-NN ownership check")
    ap.add_argument("--tech", default="TSMC5")
    args = ap.parse_args()
    print("Switchcap solver-ownership control — ground-truth L72 in PyCircuitSim "
          "vs NGSPICE.\nIf the ground-truth L72 ALSO over-charges, the gate is "
          "solver-bounded and NOT NN-fixable.\n")
    for t in [x.strip() for x in args.tech.split(",")]:
        try:
            run(t)
        except Exception as exc:  # noqa: BLE001
            print(f"{t}: ERROR {exc!r}")
