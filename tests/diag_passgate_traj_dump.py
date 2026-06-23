"""Dump vsamp(t) for the single-NMOS pass gate and compare to a faithful
implicit-Euler integration of the IDENTICAL static I-V curve.

Isolates WHERE in time PyCircuitSim's transient diverges from C dV/dt = I(V).
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

C_HOLD = 100e-15


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
    deck = f"""* 1-NMOS pass-gate ({name})
Vin vin 0 {vin}
Vphi phi 0 PULSE 0 {vdd} 0.5n 0.1n 0.1n 1.9n 4n
Mnt vin phi vsamp 0 {n} L={ln:.0f}n NFIN={nf} TFIN={tfin_nm:.1f}n
Csample vsamp 0 100f
.ic V(vsamp)=0
.model {n} NMOS (LEVEL=72)
.tran {dt:.1e} 4e-9
.end
"""
    dp = work / f"nmos_sc_traj_{name}.sp"
    dp.write_text(deck)
    parser = Parser(modelcard_path=str(merged), model_name_map={"NMOS": n})
    parser.parse_file(str(dp))
    return parser.circuit, vdd, vin


def i_of_vsamp(mos, vin, vdd, v):
    res = mos._eval_dc({"vin": vin, "phi": vdd, "vsamp": float(v), "0": 0.0})
    return res.get("is", 0.0)


def run(name="TSMC5", dt=5e-12):
    logging.disable(logging.CRITICAL)
    circuit, vdd, vin = build(name, dt)
    mos = next(c for c in circuit.components if c.__class__.__name__ == "NMOS_CMG")
    guess = circuit.initial_conditions or None
    op = DCSolver(circuit, initial_guess=guess, use_source_stepping=True).solve()
    solver = TransientSolver(circuit, t_stop=4e-9, dt=dt, initial_guess=op,
                             use_gmin_stepping=False, use_pseudo_transient=False,
                             nr_tolerance=1e-7)
    res = solver.solve()
    logging.disable(logging.NOTSET)
    t = np.asarray(res["time"]); vs = np.asarray(res["vsamp"])

    # faithful implicit-Euler integration of the SAME static I-V, same dt,
    # honoring the gate pulse (gate on only after td+tr).
    td, tr, pw = 0.5e-9, 0.1e-9, 1.9e-9
    v_ref = np.zeros_like(t)
    vcur = 0.0
    for k in range(1, len(t)):
        tk = t[k]
        gate = 0.0
        if td <= tk < td + tr:
            gate = vdd * (tk - td) / tr
        elif td + tr <= tk < td + tr + pw:
            gate = vdd
        elif tk >= td + tr + pw:
            gate = 0.0
        # implicit Euler: find v s.t. I(v;gate) = C (v - vcur)/dt
        v = vcur
        for _ in range(60):
            res2 = mos._eval_dc({"vin": vin, "phi": gate, "vsamp": float(v), "0": 0.0})
            iv = res2.get("is", 0.0)
            g = abs(res2.get("gds", 0.0)) + 1e-13
            f = iv - C_HOLD * (v - vcur) / dt
            dfdv = -g - C_HOLD / dt
            step = -f / dfdv
            v += step
            if abs(step) < 1e-9:
                break
        vcur = v
        v_ref[k] = vcur

    print(f"\n===== {name} (dt={dt:.1e}) vin={vin} =====")
    print(f"  {'t(ns)':>7} | {'gate':>6} | {'vsamp_PCS':>10} | {'vsamp_ref':>10} | {'diff':>9}")
    for tk in (0.6e-9, 0.8e-9, 1.0e-9, 1.3e-9, 1.6e-9, 1.9e-9, 2.3e-9):
        vp = float(np.interp(tk, t, vs))
        vr = float(np.interp(tk, t, v_ref))
        print(f"  {tk*1e9:7.2f} | {'--':>6} | {vp:10.5f} | {vr:10.5f} | {vp-vr:9.5f}")
    print(f"  >> vsamp_ref is implicit-Euler of the SAME I-V at the SAME dt. "
          f"A large diff = PyCircuitSim transient solver injects extra charge.")
    return t, vs, v_ref, vin


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", default="TSMC5")
    ap.add_argument("--dt", type=float, default=5e-12)
    args = ap.parse_args()
    for tname in [x.strip() for x in args.tech.split(",")]:
        run(tname, args.dt)
