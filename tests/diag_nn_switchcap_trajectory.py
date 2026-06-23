"""NN switchcap: localize the over-charge that conduction does NOT explain.

The NN UNDER-conducts on the pass-gate trajectory (TG conduction integral
~0.284 V for tsmc5) yet the NN switchcap transient over-charges to ~0.372 V.
The +0.09 V excess is not conduction. This probe dumps, for the FULL NN
switchcap (clock inverter + TG + Csample):

  * the DC-OP seed vsamp(0)            (is uic pinning honored?)
  * vsamp(t) trajectory
  * the same with the MOSFET intrinsic-charge stamp DISABLED (does the NN
    charge / transcapacitance model inject the excess?)
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
from pycircuitsim.solver import DCSolver, TransientSolver, _stamp_mosfet_dc  # noqa: E402
from pycircuitsim.models.passive import VoltageSource  # noqa: E402

SAMPLE_END = 2.3e-9


def build_circuit(name):
    bt = BENCH[name]
    vdd = bt.vdd
    vin = round(vdd * 0.6, 3)
    work = PROJECT_ROOT / "results" / "v6_5_2_solver" / "nn_sc" / name
    work.mkdir(parents=True, exist_ok=True)
    ln, lp, nf = bt.l_nmos * 1e9, bt.l_pmos * 1e9, bt.nfin
    deck = f"""* NN switchcap ({name})
Vdd vdd 0 {vdd}
Vin vin 0 {vin}
Vphi phi 0 PULSE 0 {vdd} 0.5n 0.1n 0.1n 1.9n 4n
Mpc phib phi vdd vdd pmos_nn L={lp:.0f}n NFIN={nf}
Mnc phib phi 0 0 nmos_nn L={ln:.0f}n NFIN={nf}
Mnt vin phi vsamp 0 nmos_nn L={ln:.0f}n NFIN={nf}
Mpt vin phib vsamp vdd pmos_nn L={lp:.0f}n NFIN={nf}
Csample vsamp 0 100f
.ic V(vsamp)=0 V(phib)={vdd}
.model nmos_nn NMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.vt})
.model pmos_nn PMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.vt})
.tran 5p 4n
.end
"""
    dp = work / "nn_sc.sp"; dp.write_text(deck)
    parser = Parser()
    parser.parse_file(str(dp))
    return parser.circuit, vdd, vin


def solve(name, disable_charge=False):
    circuit, vdd, vin = build_circuit(name)
    guess = circuit.initial_conditions or None
    # uic pinning (mirror run_directnet_transient)
    temps = []
    vsc = set()
    for comp in circuit.components:
        if isinstance(comp, VoltageSource):
            if comp.nodes[1] in ("0", "GND"):
                vsc.add(comp.nodes[0])
            elif comp.nodes[0] in ("0", "GND"):
                vsc.add(comp.nodes[1])
    nm = circuit.get_node_map()
    for node, val in (circuit.initial_conditions or {}).items():
        if node not in ("0", "GND") and node not in vsc and node in nm:
            vs = VoltageSource(f"_V_uic_{node}", [node, "0"], val)
            circuit.components.append(vs); temps.append(vs)
    try:
        op = DCSolver(circuit, initial_guess=guess, use_source_stepping=True,
                      use_gmin_stepping=False).solve()
    finally:
        for vs in temps:
            circuit.components.remove(vs)
    seed = op.get("vsamp", float("nan"))
    orig = TransientSolver._stamp_mosfet_transient
    if disable_charge:
        def _dc_only(self, mosfet, mna, rhs, node_map, voltages):
            _stamp_mosfet_dc(mosfet, mna, rhs, node_map, voltages, self.gmin)
        TransientSolver._stamp_mosfet_transient = _dc_only
    try:
        solver = TransientSolver(circuit, t_stop=4e-9, dt=5e-12, initial_guess=op,
                                 use_gmin_stepping=True, gmin_initial=1e-9,
                                 gmin_final=1e-12, gmin_steps=5,
                                 use_pseudo_transient=True, pseudo_transient_steps=5,
                                 pseudo_transient_cap=1e-12, nr_tolerance=1e-7)
        res = solver.solve()
    finally:
        TransientSolver._stamp_mosfet_transient = orig
    t = np.asarray(res["time"]); vs = np.asarray(res["vsamp"])
    return seed, t, vs, vin


def run(name="TSMC5"):
    logging.disable(logging.CRITICAL)
    seed, t, vs, vin = solve(name, disable_charge=False)
    seed2, t2, vs2, _ = solve(name, disable_charge=True)
    logging.disable(logging.NOTSET)
    print(f"\n===== {name} (vin={vin}) =====")
    print(f"  OP-seed vsamp(0) = {seed:.5f}  (uic target 0.0)")
    print(f"  {'t(ns)':>6} | {'vsamp(charge ON)':>16} | {'vsamp(charge OFF)':>17}")
    for tk in (0.5e-9, 0.6e-9, 1.0e-9, 1.6e-9, 2.3e-9):
        v1 = float(np.interp(tk, t, vs))
        v2 = float(np.interp(tk, t2, vs2))
        print(f"  {tk*1e9:6.2f} | {v1:16.5f} | {v2:17.5f}")
    v_on = float(np.interp(SAMPLE_END, t, vs))
    v_off = float(np.interp(SAMPLE_END, t2, vs2))
    print(f"  >> charge ON  vsamp@2.3ns = {v_on:.4f}  (NN switchcap gate value)")
    print(f"  >> charge OFF vsamp@2.3ns = {v_off:.4f}  (conduction-only)")
    print(f"  >> conduction integral ~0.284 (tsmc5); NG ~0.2948. If charge-OFF "
          f"~conduction and charge-ON over-charges, the NN CHARGE model injects "
          f"the excess.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", default="TSMC5")
    args = ap.parse_args()
    for t in [x.strip() for x in args.tech.split(",")]:
        try:
            run(t)
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"{t}: ERROR {exc!r}"); traceback.print_exc()
