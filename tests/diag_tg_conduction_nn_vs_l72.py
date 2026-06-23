"""Transmission-gate charging-current I-V: NN (L73) vs L72 vs NGSPICE.

Both pass devices ON (NMOS gate=vdd, PMOS gate=0). Sweep the hold voltage
vsamp 0..vin and report the total current delivered into the hold node:

    I_into_vsamp = I_src(Mnt) + I_src(Mpt)

In the solver's stamping, the source node receives +i_eq where i_eq≈i_leaving,
i_leaving = i_ds (NMOS) = -i_ds (PMOS). So per device the current INTO the
source node = (+calc_current for NMOS, -calc_current for PMOS). Validated
against NGSPICE i(Vs) for L72.

If the NN curve sits ABOVE L72/NGSPICE along the trajectory, the NN
over-conducts in the pass-gate corner -> the switchcap over-charge is
NN-model-owned (and fixable), not solver-owned.
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

N_VS = 81


def _l72_devices(name):
    bt = BENCH[name]
    prof = bt.profile
    vp = prof.get_vt_pair(bt.vt)
    nmos_src = prof.get_nmos_modelcard(vp, bt.l_nmos)
    pmos_src = prof.get_pmos_modelcard(vp, bt.l_pmos)
    work = PROJECT_ROOT / "results" / "v6_5_2_solver" / "tg" / name
    work.mkdir(parents=True, exist_ok=True)
    merged = work / "merged.lib"
    merged.write_text(nmos_src.read_text() + "\n" + pmos_src.read_text())
    n, p = vp.nmos_model, vp.pmos_model
    vdd = bt.vdd
    vin = round(vdd * 0.6, 3)
    tfin = prof.tfin * 1e9
    ln, lp, nf = bt.l_nmos * 1e9, bt.l_pmos * 1e9, bt.nfin
    deck = f"""* TG L72 probe ({name})
Vin vin 0 {vin}
Vphi phi 0 {vdd}
Vphib phib 0 0
Vdd vdd 0 {vdd}
Vs vsamp 0 0
Mnt vin phi vsamp 0 {n} L={ln:.0f}n NFIN={nf} TFIN={tfin:.1f}n
Mpt vin phib vsamp vdd {p} L={lp:.0f}n NFIN={nf} TFIN={tfin:.1f}n
.model {n} NMOS (LEVEL=72)
.model {p} PMOS (LEVEL=72)
.op
.end
"""
    dp = work / "tg_l72.sp"; dp.write_text(deck)
    parser = Parser(modelcard_path=str(merged),
                    model_name_map={"NMOS": n, "PMOS": p})
    parser.parse_file(str(dp))
    comps = {c.name: c for c in parser.circuit.components}
    return comps["Mnt"], comps["Mpt"], vdd, vin, bt, work


def _nn_devices(name):
    bt = BENCH[name]
    vdd = bt.vdd
    vin = round(vdd * 0.6, 3)
    work = PROJECT_ROOT / "results" / "v6_5_2_solver" / "tg" / name
    work.mkdir(parents=True, exist_ok=True)
    ln, lp, nf = bt.l_nmos * 1e9, bt.l_pmos * 1e9, bt.nfin
    deck = f"""* TG NN probe ({name})
Vin vin 0 {vin}
Vphi phi 0 {vdd}
Vphib phib 0 0
Vdd vdd 0 {vdd}
Vs vsamp 0 0
Mnt vin phi vsamp 0 nmos_nn L={ln:.0f}n NFIN={nf}
Mpt vin phib vsamp vdd pmos_nn L={lp:.0f}n NFIN={nf}
.model nmos_nn NMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.vt})
.model pmos_nn PMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.vt})
.op
.end
"""
    dp = work / "tg_nn.sp"; dp.write_text(deck)
    parser = Parser()
    parser.parse_file(str(dp))
    comps = {c.name: c for c in parser.circuit.components}
    return comps["Mnt"], comps["Mpt"], vdd, vin


def i_into(mnt, mpt, vin, vdd, v):
    volt = {"vin": vin, "phi": vdd, "phib": 0.0, "vdd": vdd,
            "vsamp": float(v), "0": 0.0}
    i_n = mnt.calculate_current(volt)        # +calc for NMOS source
    i_p = -mpt.calculate_current(volt)       # -calc for PMOS source
    return i_n + i_p, i_n, i_p


def ng_iv(bt, work, vin):
    baked = get_baked_modelcard(bt, bt.nfin, work)
    n, p = bt.nmos_model, bt.pmos_model
    body = [f'.include "{baked}"', ".temp 27",
            f"Vin vin 0 {vin}", f"Vphi phi 0 {bt.vdd}", "Vphib phib 0 0",
            f"Vdd vdd 0 {bt.vdd}", "Vs vsamp 0 0",
            f"Nnt vin phi vsamp 0 {n}", f"Npt vin phib vsamp vdd {p}"]
    data = run_ngspice_wrdata("\n".join(body), "i(Vs)", work, f"tg_{bt.name}",
                              f"dc Vs 0 {vin} {vin/(N_VS-1):.6e}")
    return data[:, 0], data[:, 1]


def run(name="TSMC5"):
    print(f"\n===== {name} =====")
    mnt5, mpt5, vdd, vin, bt, work = _l72_devices(name)
    mntN, mptN, _, _ = _nn_devices(name)
    vs = np.linspace(0, vin, N_VS)
    i_l72 = np.array([i_into(mnt5, mpt5, vin, vdd, v)[0] for v in vs])
    i_nn = np.array([i_into(mntN, mptN, vin, vdd, v)[0] for v in vs])
    vsn, ivn = ng_iv(bt, work, vin)
    i_ng = np.interp(vs, vsn, ivn)
    print(f"  vdd={vdd} vin={vin}")
    print(f"  {'vsamp':>7} | {'I_L72(uA)':>10} | {'I_NG(uA)':>10} | "
          f"{'I_NN(uA)':>10} | {'NN/NG':>7}")
    for frac in (0.0, 0.25, 0.5, 0.7, 0.85, 0.95):
        v = frac * vin
        il = float(np.interp(v, vs, i_l72)) * 1e6
        ig = float(np.interp(v, vs, i_ng)) * 1e6
        inn = float(np.interp(v, vs, i_nn)) * 1e6
        r = inn / ig if abs(ig) > 1e-12 else float("nan")
        print(f"  {v:7.4f} | {il:10.3f} | {ig:10.3f} | {inn:10.3f} | {r:7.2f}")

    # integrate each to a final vsamp over the 1.7ns window (forward Euler)
    def integ(iv):
        v = 0.0; dt = 1e-13
        for _ in range(int(1.7e-9 / dt)):
            cur = float(np.interp(v, vs, iv))
            v += dt * cur / 100e-15
            if v >= vin:
                v = vin; break
            if v < 0:
                v = 0.0
        return v
    print(f"  hand-integrated final vsamp:  L72={integ(i_l72):.4f}  "
          f"NG={integ(i_ng):.4f}  NN={integ(i_nn):.4f}  (vin={vin})")
    print(f"  >> If NN > NG (ratio>1) along the trajectory and NN integrates "
          f"higher, the switchcap over-charge is NN-conduction-owned.")


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
            print(f"{t}: ERROR {exc!r}"); traceback.print_exc()
