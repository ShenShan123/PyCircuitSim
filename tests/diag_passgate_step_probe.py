"""Single-step NR probe: gate held ON (DC), cap v_prev=0, one BE step.

If a clean single BE step from vsamp=0 lands at vin instead of a few mV,
the per-timestep implicit solve over-charges. Prints the NR iterates.
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


def build(name, dt):
    bt = BENCH[name]
    prof = bt.profile
    vp = prof.get_vt_pair(bt.vt)
    nmos_src = prof.get_nmos_modelcard(vp, bt.l_nmos)
    work = PROJECT_ROOT / "results" / "v6_5_2_solver" / name
    work.mkdir(parents=True, exist_ok=True)
    merged = work / "nmos.lib"
    merged.write_text(nmos_src.read_text())
    n = vp.nmos_model
    vdd = bt.vdd
    vin = round(vdd * 0.6, 3)
    tfin_nm = prof.tfin * 1e9
    ln, nf = bt.l_nmos * 1e9, bt.nfin
    # gate held ON at vdd (DC), no clock
    deck = f"""* 1-NMOS pass-gate gate-ON DC ({name})
Vin vin 0 {vin}
Vphi phi 0 {vdd}
Mnt vin phi vsamp 0 {n} L={ln:.0f}n NFIN={nf} TFIN={tfin_nm:.1f}n
Csample vsamp 0 100f
.ic V(vsamp)=0
.model {n} NMOS (LEVEL=72)
.tran {dt:.1e} {3*dt:.1e}
.end
"""
    dp = work / f"nmos_step_{name}.sp"
    dp.write_text(deck)
    parser = Parser(modelcard_path=str(merged), model_name_map={"NMOS": n})
    parser.parse_file(str(dp))
    return parser.circuit, vdd, vin


def run(name="TSMC5", dt=5e-12):
    logging.disable(logging.CRITICAL)
    circuit, vdd, vin = build(name, dt)
    mos = next(c for c in circuit.components if c.__class__.__name__ == "NMOS_CMG")
    cap = next(c for c in circuit.components if c.__class__.__name__ == "Capacitor")
    # DC op with gate ON but vsamp forced to .ic=0 via initial guess
    guess = circuit.initial_conditions or {}
    op = DCSolver(circuit, initial_guess=guess, use_source_stepping=True).solve()
    logging.disable(logging.NOTSET)
    print(f"\n===== {name} dt={dt:.1e} vin={vin} =====")
    print(f"  DC op: vsamp={op.get('vsamp', float('nan')):.5f}  "
          f"(this is the gate-ON steady state the .op finds)")

    # Manually do ONE backward-Euler step from vsamp=0.
    cap.v_prev = 0.0
    cap._i_prev = 0.0
    cap._use_trapezoidal = False
    cap._method = 'be'
    cap.get_companion_model(dt, cap.v_prev)
    mos.init_charge_state({"vin": vin, "phi": vdd, "vsamp": 0.0, "0": 0.0})

    nodes = circuit.get_nodes()
    node_map = circuit.get_node_map()
    num_nodes = len(nodes)
    nvs = circuit.count_voltage_sources()
    solver = TransientSolver(circuit, t_stop=3*dt, dt=dt, initial_guess=op,
                             use_gmin_stepping=False, use_pseudo_transient=False,
                             nr_tolerance=1e-7)
    solver._current_dt = dt
    solver._integration_method = 'be'
    init_v = {nname: (0.0 if nname == "vsamp" else op.get(nname, 0.0)) for nname in nodes}
    init_v["0"] = 0.0
    res = solver._solve_timestep_newton(
        nodes=nodes, node_map=node_map, num_nodes=num_nodes,
        num_voltage_sources=nvs, initial_voltages=init_v, time=dt,
        step_index=10, use_gmin=False)
    vsamp1 = res.get("vsamp")
    print(f"  After ONE BE step (dt={dt:.0e}) from vsamp=0:  vsamp={vsamp1:.6f}")
    # what SHOULD it be: I_mos(v) = C v/dt  -> v ~ I_mos(0)*dt/C for the 1st step
    i0 = mos._eval_dc({"vin": vin, "phi": vdd, "vsamp": 0.0, "0": 0.0}).get("is", 0.0)
    v_expect = i0 * dt / cap.capacitance
    print(f"  expected ~ I(0)*dt/C = {i0*1e6:.2f}uA * {dt:.0e}/100fF = {v_expect:.6f}")
    print(f"  >> If vsamp jumps to ~{vin} instead of ~{v_expect:.4f}, the "
          f"per-step implicit solve is the bug.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", default="TSMC5")
    ap.add_argument("--dt", type=float, default=5e-12)
    args = ap.parse_args()
    run(args.tech, args.dt)
