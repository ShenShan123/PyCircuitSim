#!/usr/bin/env python3
"""V6.4.7 S17 (P9 diagnostic) — force_ic per-device NN-vs-OSDI decomposition.

Generalises V6.4.6 p0d to any tech and to the CURRENT code (post-S2 frame fix,
post-S7 reverse clamp) + the installed checkpoint (BSIMAR_CHECKPOINT_DIR). At
the stuck force_ic fixed point it prints, per 6T transistor: terminal bias,
on/off class vs the OSDI constant-current VTH, post-Rule-15 NN id/gds, OSDI
id/gds/gm, and the |NN/OSDI| ratio.

THE P9 QUESTION it answers: at the "0"-storage node (the one stuck ~21-46 mV
above ground), which device's NN current error holds it up?
  * If the dominant error is the OFF load PMOS (Mpl/Mpr) leakage  -> P9 (OFF
    core) has a target.
  * If it is the ON driver NMOS (Mnl/Mnr, under-pull) or the ON access NMOS
    (Mal/Mar, over-pull) -> P9 will NOT help (moderate/strong-inversion error).

It also prints the "0"-node KCL: sum of device currents into that node with NN
values (~0, converged) vs with OSDI values at the SAME node voltages (the
residual = the net current OSDI says should flow -> the direction the true
fixed point lies).

Run (with the promoted checkpoint installed at {tech}_dn_medium in an isolated
BSIMAR_CHECKPOINT_DIR):
    OMP_NUM_THREADS=1 conda run -n pycircuitsim \
      python scripts/v6_4_7_s17_forceic_decomp.py --tech TSMC16
Ground truth is ALWAYS the OSDI BSIM-CMG binary via PyCMG (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))
sys.path.append(str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.complex import BENCH, parse_netlist  # noqa: E402
from tests.verify_complex_sram_snm import _directnet_6t_netlist  # noqa: E402
from pycircuitsim.solver import DCSolver, _is_mosfet  # noqa: E402
from pycmg.nn_config import TECH_CONFIGS  # noqa: E402
from pycmg.nn_generate import _create_model_and_instance, eval_single_point  # noqa: E402

TEMP_K = 300.15
L_NMOS = 16e-9
L_PMOS = 20e-9
_OSDI_CACHE: Dict[str, object] = {}


def _osdi_inst(tech_lc: str, variant: str, nfin: float, device_type: str):
    key = f"{tech_lc}:{variant}:{device_type}"
    if key not in _OSDI_CACHE:
        tech = TECH_CONFIGS[tech_lc]
        L = L_NMOS if device_type == "nmos" else L_PMOS
        model, inst, proc = _create_model_and_instance(
            tech, device_type, variant, L, nfin, TEMP_K)
        _OSDI_CACHE[key] = inst
    return _OSDI_CACHE[key]


def osdi_eval(tech_lc, variant, nfin, device_type, vd, vg, vs, vb):
    return eval_single_point(_osdi_inst(tech_lc, variant, nfin, device_type),
                             vd, vg, vs, vb)


def osdi_vth(tech_lc, variant, nfin, device_type, vdd) -> Tuple[float, float]:
    icrit = 1e-7 * nfin
    if device_type == "nmos":
        grid = np.linspace(0.0, vdd, 121)
        prev = None
        for vg in grid:
            out = osdi_eval(tech_lc, variant, nfin, "nmos", vdd, vg, 0.0, 0.0)
            iabs = abs(out["id"]) if out else 0.0
            if iabs >= icrit and prev is not None:
                vg0, i0 = prev
                return (vg0 + (vg - vg0) * (icrit - i0) / (iabs - i0)
                        if iabs != i0 else vg), icrit
            prev = (vg, iabs)
        return float("nan"), icrit
    grid = np.linspace(vdd, 0.0, 121)
    prev = None
    for vg in grid:
        out = osdi_eval(tech_lc, variant, nfin, "pmos", 0.0, vg, vdd, vdd)
        iabs = abs(out["id"]) if out else 0.0
        if iabs >= icrit and prev is not None:
            vg0, i0 = prev
            cross = (vg0 + (vg - vg0) * (icrit - i0) / (iabs - i0)
                     if iabs != i0 else vg) - vdd
            return cross, icrit
        prev = (vg, iabs)
    return float("nan"), icrit


def classify(device_type, vgs, vth):
    on = vgs > vth if device_type == "nmos" else vgs < vth
    margin = vgs - vth
    if device_type == "pmos":
        margin = -margin
    if on:
        return f"ON  (Vov={margin*1e3:+.0f})"
    if margin > -0.05:
        return f"NEAR(Vov={margin*1e3:+.0f})"
    return f"OFF (Vov={margin*1e3:+.0f})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", default="TSMC16")
    args = ap.parse_args()
    tech = args.tech.upper()
    tech_lc = tech.lower()
    bt = BENCH[tech]
    vdd = bt.vdd
    variant = bt.vt
    nfin = float(bt.nfin)
    work_dir = PROJECT_ROOT / "results" / "v6_4_7" / "s17_scratch"
    work_dir.mkdir(parents=True, exist_ok=True)

    vth_n, icrit = osdi_vth(tech_lc, variant, nfin, "nmos", vdd)
    vth_p, _ = osdi_vth(tech_lc, variant, nfin, "pmos", vdd)
    print("=" * 100)
    print(f"S17/P9 force_ic decomposition — {tech} VDD={vdd} NFIN={nfin} VT={variant}")
    print(f"OSDI const-current VTH (Icrit={icrit:.1e}A): NMOS={vth_n*1e3:.0f}mV  PMOS(Vgs)={vth_p*1e3:.0f}mV")
    print("=" * 100)

    for tag, (q0, qb0) in (("state1", (vdd, 0.0)), ("state0", (0.0, vdd))):
        netlist = _directnet_6t_netlist(bt, q0, qb0, work_dir / f"{tech}_{tag}.sp")
        logging.disable(logging.CRITICAL)
        try:
            circuit = parse_netlist(netlist).circuit
            solver = DCSolver(circuit, initial_guess=circuit.initial_conditions or None,
                              use_source_stepping=True, force_ic=True)
            sol = solver.solve()
        finally:
            logging.disable(logging.NOTSET)

        # which storage node is the "0" node for this seed?
        zero_node = "qb" if qb0 < q0 else "q"
        print(f"\n##### {tag} (seed q={q0} qb={qb0}) — '0'-node = {zero_node}")
        for n in ("q", "qb"):
            print(f"    V({n}) = {sol.get(n, float('nan'))*1e3:7.1f} mV"
                  + ("   <-- stuck '0' node" if n == zero_node else ""))

        mosfets = [c for c in circuit.components if _is_mosfet(c)]
        print(f"  {'Dev':4s}{'typ':4s} {'D/G/S/B':14s} {'Vgs':>6s}{'Vds':>7s}{'Vbs':>7s} "
              f"{'class':<14s}{'NN id':>12s}{'OSDI id':>12s}{'NN/OSDI':>9s}")
        # accumulate KCL into the zero node: device drain/source current into node
        nn_into = 0.0
        osdi_into = 0.0
        contribs = []
        for m in mosfets:
            d, g, s, b = m.nodes
            dt = "pmos" if m._is_pmos else "nmos"
            vd, vg = sol.get(d, 0.0), sol.get(g, 0.0)
            vs, vb = sol.get(s, 0.0), sol.get(b, 0.0)
            vgs, vds, vbs = vg - vs, vd - vs, vb - vs
            vth = vth_p if dt == "pmos" else vth_n
            cls = classify(dt, vgs, vth)
            nn = m._eval(sol)
            nn_id = nn["id"]
            o = osdi_eval(tech_lc, variant, nfin, dt, vd, vg, vs, vb)
            o_id = o["id"] if o else float("nan")
            ratio = abs(nn_id) / abs(o_id) if (o_id and abs(o_id) > 0) else float("inf")
            tag_node = ""
            # device 'id' = current leaving drain (positive=leaving drain).
            # current INTO node: drain node receives -id; source node receives +id.
            if d == zero_node:
                nn_into += -nn_id; osdi_into += -o_id; tag_node = f" [->{zero_node} via D]"
                contribs.append((m.name, dt, "D", -nn_id, -o_id))
            elif s == zero_node:
                nn_into += nn_id; osdi_into += o_id; tag_node = f" [->{zero_node} via S]"
                contribs.append((m.name, dt, "S", nn_id, o_id))
            print(f"  {m.name:4s}{dt:4s} {d+'/'+g+'/'+s+'/'+b:14s} "
                  f"{vgs*1e3:6.0f}{vds*1e3:7.0f}{vbs*1e3:7.0f} {cls:<14s}"
                  f"{nn_id:12.3e}{o_id:12.3e}{ratio:9.2e}{tag_node}")
        print(f"  --- KCL into '{zero_node}': NN sum={nn_into:.3e} A (converged ~0) | "
              f"OSDI sum={osdi_into:.3e} A  (>0 pushes node UP, <0 pulls to ground)")
        # rank contributors by |NN-OSDI| current error into the zero node
        contribs.sort(key=lambda c: abs(c[3] - c[4]), reverse=True)
        print(f"  --- per-device current-error into '{zero_node}' (|NN-OSDI|, ranked):")
        for name, dt, term, nn_c, o_c in contribs:
            print(f"        {name} ({dt},{term}): NN={nn_c:+.3e}  OSDI={o_c:+.3e}  "
                  f"err={nn_c-o_c:+.3e} A")
    print("\n" + "=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
