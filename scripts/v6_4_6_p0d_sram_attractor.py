#!/usr/bin/env python3
"""V6.4.6 P0-D — SRAM off-transistor attractor instrumentation (TSMC7).

At the stuck ``force_ic`` fixed point on TSMC7 (the unconstrained re-solve that
walks off the rail into q ~ 0.18/0.82), characterise every transistor — and the
OFF ones in particular — to decide whether the Phase-3 leak/skeleton family is
alive or dead.

For each of the 6 transistors in the 6T cell at the stuck solution:
  * terminal bias (Vgs, Vds, Vbs) and on/off classification
  * post-Rule-15 NN inference ``id``/``gm``/``gds`` via ``mosfet._eval``
  * analytic OSDI ``id``/``gm``/``gds`` at the SAME absolute terminal bias
  * Vgs vs the per-device OSDI turn-on VTH (constant-current definition)

Decision questions answered (printed):
  (a) off-leakage OVER-MODELLED (NN |id| >> OSDI |id|) or already ~0 ?
  (b) OFF device DEEP-OFF (Vgs << VTH) or MODERATE inversion (Vgs ~ VTH) ?

Run:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      conda run -n pycircuitsim python scripts/v6_4_6_p0d_sram_attractor.py \
      > results/v6_4_6/phase0_logs/p0d_attractor.log 2>&1

Ground truth is ALWAYS the OSDI BSIM-CMG binary via PyCMG (CLAUDE.md).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Append PyCMG paths (lower precedence) but keep PROJECT_ROOT at the FRONT so
# `tests.common` resolves to the project's tests package, not PyCMG/tests.
sys.path.append(str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))
sys.path.append(str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
else:
    sys.path.remove(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.complex import BENCH, parse_netlist  # noqa: E402
from tests.verify_complex_sram_snm import _directnet_6t_netlist  # noqa: E402
from pycircuitsim.solver import DCSolver, _is_mosfet  # noqa: E402

from pycmg.nn_config import TECH_CONFIGS  # noqa: E402
from pycmg.nn_generate import (  # noqa: E402
    _create_model_and_instance, eval_single_point)

VARIANT = "ulvt"          # TSMC7 VT
TEMP_K = 300.15           # sims run at .temp 27
NFIN = 2.0
L_NMOS = 16e-9
L_PMOS = 20e-9

# OSDI instance cache (one per device_type)
_OSDI_CACHE: Dict[str, object] = {}


def _osdi_inst(device_type: str):
    if device_type not in _OSDI_CACHE:
        tech = TECH_CONFIGS["tsmc7"]
        L = L_NMOS if device_type == "nmos" else L_PMOS
        model, inst, proc = _create_model_and_instance(
            tech, device_type, VARIANT, L, NFIN, TEMP_K)
        _OSDI_CACHE[device_type] = inst
    return _OSDI_CACHE[device_type]


def osdi_eval(device_type: str, vd: float, vg: float,
              vs: float, vb: float) -> Optional[Dict[str, float]]:
    """OSDI ground truth at ABSOLUTE terminal voltages."""
    return eval_single_point(_osdi_inst(device_type), vd, vg, vs, vb)


def osdi_vth(device_type: str, vdd: float) -> Tuple[float, float]:
    """Constant-current turn-on VTH from an OSDI Id-Vgs sweep at Vds=VDD.

    Definition: |Id| crosses Icrit = 1e-7 * NFIN A (the standard
    constant-current threshold, scaled by fin count). Returns (vth, icrit).
    Vs=Vb=0; Vd=VDD. For PMOS we sweep negative Vgs by mirroring (pass
    absolute terminal voltages with source at VDD frame handled by caller),
    but here both 6T NMOS pull-downs are the interesting OFF devices; we
    still report VTH for whichever device_type is asked, using the device's
    natural conducting polarity.
    """
    icrit = 1e-7 * NFIN
    if device_type == "nmos":
        vgs_grid = np.linspace(0.0, vdd, 121)
        vth = float("nan")
        prev = None
        for vg in vgs_grid:
            out = osdi_eval("nmos", vdd, vg, 0.0, 0.0)
            iabs = abs(out["id"]) if out else 0.0
            if iabs >= icrit and prev is not None:
                # linear-interp the crossing
                vg0, i0 = prev
                if iabs != i0:
                    vth = vg0 + (vg - vg0) * (icrit - i0) / (iabs - i0)
                else:
                    vth = vg
                break
            prev = (vg, iabs)
        return vth, icrit
    else:
        # PMOS: source at VDD, sweep gate from VDD down to 0; |Vgs| grows.
        vgs_grid = np.linspace(vdd, 0.0, 121)
        vth = float("nan")
        prev = None
        for vg in vgs_grid:
            out = osdi_eval("pmos", 0.0, vg, vdd, vdd)  # Vds=-VDD, Vgs=vg-VDD
            iabs = abs(out["id"]) if out else 0.0
            if iabs >= icrit and prev is not None:
                vg0, i0 = prev
                if iabs != i0:
                    vgs_cross = (vg0 + (vg - vg0) * (icrit - i0) / (iabs - i0)) - vdd
                else:
                    vgs_cross = vg - vdd
                vth = vgs_cross
                break
            prev = (vg, iabs)
        return vth, icrit


def solve_stuck(tech: str, q0: float, qb0: float,
                work_dir: Path):
    bt = BENCH[tech]
    netlist = _directnet_6t_netlist(
        bt, q0, qb0, work_dir / f"p0d_{tech}_{q0}_{qb0}.sp")
    logging.disable(logging.CRITICAL)
    try:
        parser = parse_netlist(netlist)
        circuit = parser.circuit
        guess = circuit.initial_conditions or None
        solver = DCSolver(circuit, initial_guess=guess,
                          use_source_stepping=True, force_ic=True)
        sol = solver.solve()
    finally:
        logging.disable(logging.NOTSET)
    return circuit, sol


def classify(device_type: str, vgs: float, vth: float) -> str:
    # NMOS: on when Vgs > VTH; PMOS: on when Vgs < VTH (both negative).
    if device_type == "nmos":
        on = vgs > vth
    else:
        on = vgs < vth
    margin = vgs - vth
    if device_type == "pmos":
        margin = -margin  # how far into conduction (positive = on)
    if on:
        return f"ON   (Vov={margin*1e3:+.1f}mV)"
    if margin > -0.05:
        return f"NEAR-VTH (Vov={margin*1e3:+.1f}mV)"
    return f"OFF  (Vov={margin*1e3:+.1f}mV)"


def main() -> int:
    work_dir = PROJECT_ROOT / "results" / "v6_4_6" / "phase0_logs" / "p0d_scratch"
    work_dir.mkdir(parents=True, exist_ok=True)
    bt = BENCH["TSMC7"]
    vdd = bt.vdd

    # Per-device-type VTH (constant current 1e-7*NFIN)
    vth_n, icrit = osdi_vth("nmos", vdd)
    vth_p, _ = osdi_vth("pmos", vdd)
    print("=" * 90)
    print(f"P0-D — SRAM off-transistor attractor instrumentation  (TSMC7, VDD={vdd}, NFIN={NFIN})")
    print(f"OSDI constant-current VTH (Icrit=|Id|>={icrit:.2e} A @ Vds=VDD):")
    print(f"  NMOS VTH = {vth_n*1e3:.1f} mV   PMOS VTH(Vgs) = {vth_p*1e3:.1f} mV")
    print("=" * 90)

    for tag, (q0, qb0) in (("state1", (vdd, 0.0)), ("state0", (0.0, vdd))):
        circuit, sol = solve_stuck("TSMC7", q0, qb0, work_dir)
        print(f"\n##### {tag}  (seed q={q0} qb={qb0}) — STUCK node voltages:")
        for n in ("vdd", "q", "qb", "bl", "blb", "wl"):
            print(f"    V({n:4s}) = {sol.get(n, float('nan'))*1e3:8.2f} mV")

        mosfets = [c for c in circuit.components if _is_mosfet(c)]
        print(f"\n  {'Dev':4s} {'type':4s} {'D':>3s}/{'G':>3s}/{'S':>3s}/{'B':>3s} "
              f"{'Vgs(mV)':>9s} {'Vds(mV)':>9s} {'Vbs(mV)':>9s}  {'class':<22s}"
              f" {'NN id(A)':>12s} {'OSDI id(A)':>12s} {'|NN/OSDI|':>10s}"
              f" {'NN gds':>11s} {'OSDI gds':>11s} {'OSDI gm':>11s}")
        for m in mosfets:
            d, g, s, b = m.nodes
            device_type = "pmos" if m._is_pmos else "nmos"
            vd = sol.get(d, 0.0); vg = sol.get(g, 0.0)
            vs = sol.get(s, 0.0); vb = sol.get(b, 0.0)
            vgs = vg - vs; vds = vd - vs; vbs = vb - vs
            vth = vth_p if device_type == "pmos" else vth_n
            cls = classify(device_type, vgs, vth)

            nn = m._eval(sol)
            nn_id = nn["id"]; nn_gds = nn["gds"]
            osdi = osdi_eval(device_type, vd, vg, vs, vb)
            osdi_id = osdi["id"] if osdi else float("nan")
            osdi_gds = osdi["gds"] if osdi else float("nan")
            osdi_gm = osdi["gm"] if osdi else float("nan")
            ratio = (abs(nn_id) / abs(osdi_id)) if (osdi_id and abs(osdi_id) > 0) else float("inf")
            print(f"  {m.name:4s} {device_type:4s} {d:>3s}/{g:>3s}/{s:>3s}/{b:>3s} "
                  f"{vgs*1e3:9.2f} {vds*1e3:9.2f} {vbs*1e3:9.2f}  {cls:<22s}"
                  f" {nn_id:12.4e} {osdi_id:12.4e} {ratio:10.3e}"
                  f" {nn_gds:11.4e} {osdi_gds:11.4e} {osdi_gm:11.4e}")

    print("\n" + "=" * 90)
    print("(See decision block in results/v6_4_6/phase0_D_sram_attractor.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
