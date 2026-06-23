"""Pass-gate charging-trajectory I-V: PyCircuitSim-L72 vs NGSPICE-L72.

Decisive localization for the switchcap over-charge (V6.5.2 found it is
SOLVER-owned, not NN-owned). A single NMOS pass transistor charges a 100 fF
hold cap; gate held ON at VDD (no clock). The cap-charging current is

    I_into_vsamp(vsamp) = -is(d=vin, g=VDD, s=vsamp, b=0)

i.e. the MOSFET source-terminal current delivered to the hold node as a
function of the (rising) hold voltage. We compare this static I-V curve
between PyCircuitSim's own L72 eval and NGSPICE's L72 DC sweep, then
hand-integrate C dV/dt = I(V) over the 1.7 ns sample window.

Reading:
  * If the two I-V curves DIFFER (PyCircuitSim conducts more near vsamp->vin),
    the over-charge is a MODEL-EVAL corner (despite DC grid tests passing).
  * If the two I-V curves MATCH but the hand-integration over-charges only
    when run through PyCircuitSim's transient solver, the bug is in the
    transient INTEGRATION (charge companion / NR), not the conduction model.
"""
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from tests.common.complex import BENCH, run_ngspice_wrdata, get_baked_modelcard  # noqa: E402
from pycircuitsim.parser import Parser  # noqa: E402

C_HOLD = 100e-15
T_WINDOW = 1.7e-9   # 0.6ns (clk on) -> 2.3ns (SAMPLE_END)
N_VS = 121


def _pcs_nmos(name="TSMC5"):
    """Parse a 1-device deck; return (nmos_component, vdd, vin)."""
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
    deck = f"""* 1-NMOS probe ({name})
Vin vin 0 {vin}
Vg phi 0 {vdd}
Vs vsamp 0 0
Mnt vin phi vsamp 0 {n} L={ln:.0f}n NFIN={nf} TFIN={tfin_nm:.1f}n
.model {n} NMOS (LEVEL=72)
.op
.end
"""
    dp = work / "nmos_probe.sp"
    dp.write_text(deck)
    logging.disable(logging.CRITICAL)
    try:
        parser = Parser(modelcard_path=str(merged), model_name_map={"NMOS": n})
        parser.parse_file(str(dp))
    finally:
        logging.disable(logging.NOTSET)
    mos = next(c for c in parser.circuit.components
               if c.__class__.__name__ == "NMOS_CMG")
    return mos, vdd, vin, bt, work


def pcs_iv(name="TSMC5"):
    mos, vdd, vin, bt, work = _pcs_nmos(name)
    vs = np.linspace(0.0, vin, N_VS)
    i_into = np.zeros_like(vs)
    idd = np.zeros_like(vs)
    for k, v in enumerate(vs):
        volt = {"vin": vin, "phi": vdd, "vsamp": float(v), "0": 0.0}
        res = mos._eval_dc(volt)
        # PyCMG 'is' = current leaving the source terminal INTO the hold node
        # (positive while charging up from 0 toward vin).
        i_into[k] = res.get("is", res.get("isb", 0.0))
        idd[k] = res.get("id", 0.0)
    return vs, i_into, idd, vdd, vin, bt, work


def ng_iv(bt, work):
    """NGSPICE DC sweep of vsamp; i(Vsamp) = current into the hold node."""
    baked = get_baked_modelcard(bt, bt.nfin, work)
    n = bt.nmos_model
    vin = round(bt.vdd * 0.6, 3)
    body = [f'.include "{baked}"', ".temp 27",
            f"Vin vin 0 {vin}", f"Vg phi 0 {bt.vdd}", "Vs vsamp 0 0",
            f"Nnt vin phi vsamp 0 {n}"]
    data = run_ngspice_wrdata("\n".join(body), "i(Vs)", work, f"pg_{bt.name}",
                              f"dc Vs 0 {vin} {vin/(N_VS-1):.6e}")
    vsweep, iv = data[:, 0], data[:, 1]
    # i(Vs): current through Vs (vsamp->0). The MOSFET source delivers the
    # charging current into vsamp; it exits to ground through Vs, so i(Vs)
    # equals that charging current. Matched to PyCMG 'is' by sign empirically.
    return vsweep, iv


def integrate(vs, i_into, vin, vdd):
    """Hand-integrate C dV/dt = I_into(V) over the window; return final V."""
    # monotone interp of I(V)
    def Iof(v):
        return float(np.interp(v, vs, i_into, left=i_into[0], right=i_into[-1]))
    v = 0.0
    dt = 1e-13
    nsteps = int(T_WINDOW / dt)
    for _ in range(nsteps):
        # forward Euler (tiny dt) — RC integration of the static I-V curve
        iv = Iof(v)
        v += dt * iv / C_HOLD
        if v >= vin:
            v = vin
            break
        if v < 0:
            v = 0.0
    return v


def pcs_transient(name, disable_mos_charge=False, dt=5e-12):
    """Run the single-NMOS clocked pass-gate in PyCircuitSim transient.

    Returns vsamp interpolated at SAMPLE_END (2.3 ns). If disable_mos_charge,
    the MOSFET intrinsic-charge companion stamp is suppressed (only the DC
    conduction stamp + the explicit 100 fF Csample companion remain).
    """
    from pycircuitsim.solver import DCSolver, TransientSolver
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
    deck = f"""* 1-NMOS pass-gate switchcap ({name})
Vin vin 0 {vin}
Vphi phi 0 PULSE 0 {vdd} 0.5n 0.1n 0.1n 1.9n 4n
Mnt vin phi vsamp 0 {n} L={ln:.0f}n NFIN={nf} TFIN={tfin_nm:.1f}n
Csample vsamp 0 100f
.ic V(vsamp)=0
.model {n} NMOS (LEVEL=72)
.tran {dt:.0e} 4e-9
.end
"""
    dp = work / f"nmos_sc_{name}.sp"
    dp.write_text(deck)
    logging.disable(logging.CRITICAL)
    orig = TransientSolver._stamp_mosfet_transient
    try:
        parser = Parser(modelcard_path=str(merged), model_name_map={"NMOS": n})
        parser.parse_file(str(dp))
        circuit = parser.circuit
        guess = circuit.initial_conditions or None
        op = DCSolver(circuit, initial_guess=guess,
                      use_source_stepping=True).solve()
        if disable_mos_charge:
            from pycircuitsim.solver import _stamp_mosfet_dc

            def _dc_only(self, mosfet, mna, rhs, node_map, voltages):
                _stamp_mosfet_dc(mosfet, mna, rhs, node_map, voltages, self.gmin)
            TransientSolver._stamp_mosfet_transient = _dc_only
        solver = TransientSolver(
            circuit, t_stop=4e-9, dt=dt, initial_guess=op,
            use_gmin_stepping=False, use_pseudo_transient=False,
            debug=False, nr_tolerance=1e-7)
        res = solver.solve()
    finally:
        TransientSolver._stamp_mosfet_transient = orig
        logging.disable(logging.NOTSET)
    t = np.asarray(res["time"]); vs = np.asarray(res["vsamp"])
    return float(np.interp(2.3e-9, t, vs))


def run(name="TSMC5"):
    print(f"\n===== {name} =====")
    vs_p, i_p, idd, vdd, vin, bt, work = pcs_iv(name)
    vs_n, i_n = ng_iv(bt, work)
    # resample NGSPICE onto the PyCircuitSim grid
    i_n_rs = np.interp(vs_p, vs_n, i_n)
    print(f"  vdd={vdd}  vin={vin}  Chold=100fF  window={T_WINDOW*1e9:.1f}ns")
    print(f"  {'vsamp':>7} | {'I_pcs(uA)':>10} | {'I_ng(uA)':>10} | {'ratio':>7}")
    for frac in (0.0, 0.25, 0.5, 0.7, 0.85, 0.95, 1.0):
        v = frac * vin
        ip = float(np.interp(v, vs_p, i_p)) * 1e6
        ino = float(np.interp(v, vs_p, i_n_rs)) * 1e6
        ratio = ip / ino if abs(ino) > 1e-12 else float("nan")
        print(f"  {v:7.4f} | {ip:10.4f} | {ino:10.4f} | {ratio:7.3f}")
    vf_p = integrate(vs_p, i_p, vin, vdd)
    print(f"\n  [static I-V hand-integration over {T_WINDOW*1e9:.1f}ns] "
          f"final vsamp = {vf_p:.4f} V  (vin={vin})")
    print(f"  >> NGSPICE transient final ~0.2948 for tsmc5; hand-integration "
          f"of the IDENTICAL I-V should land near it.")
    print(f"\n  [PyCircuitSim single-NMOS transient]")
    v_full = pcs_transient(name, disable_mos_charge=False)
    v_nochg = pcs_transient(name, disable_mos_charge=True)
    print(f"    full (charge stamp ON):  vsamp@2.3ns = {v_full:.4f} V")
    print(f"    charge stamp DISABLED:   vsamp@2.3ns = {v_nochg:.4f} V")
    print(f"    >> If 'full' reaches vin but DISABLED matches hand-integration,"
          f" the MOSFET intrinsic-charge companion over-injects.")
    print(f"    >> If BOTH reach vin, the explicit-cap companion / NR step "
          f"over-charges (conduction integration).")
    return vs_p, i_p, i_n_rs, vin, vdd


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", default="TSMC5,TSMC7,TSMC12,TSMC16")
    args = ap.parse_args()
    for t in [x.strip() for x in args.tech.split(",")]:
        try:
            run(t)
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"{t}: ERROR {exc!r}")
            traceback.print_exc()
