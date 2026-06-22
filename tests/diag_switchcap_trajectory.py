#!/usr/bin/env python3
"""V6.7 decision gate (part 2) — NN vs OSDI along the EXACT switchcap trajectory.

The grid diagnostic (``diag_charge_cap_fidelity.py``) showed the NN's autograd
caps match OSDI to ~1% on the *sampled training distribution*. Yet tsmc5
switchcap over-charges 11.8%. The switchcap pass devices traverse a specific
1-D locus as the hold node ``vsamp`` charges from 0 toward ``vin`` — visiting
Vds→0 and (for the NMOS) increasingly negative Vbs — corners the random grid
sample covers sparsely. This script walks that exact locus and compares the
**real inference device** (``_MOSFETNNBase._eval`` — full source-shift + clamp +
Vds-correction, i.e. exactly what the transient solver consumes) against OSDI
``eval_dc`` ground truth, for both pass transistors, reporting id / gds / the 4
condensed caps.

Sign mapping (OSDI SPICE convention vs the device's raw-autograd caps):
    device cgg  vs  +osdi cgg      device cdd  vs  +osdi cdd
    device cgd  vs  −osdi cgd      device cdg  vs  −osdi cdg

Ground truth is ALWAYS OSDI (CLAUDE.md Validation rule).

Usage:
    python tests/diag_switchcap_trajectory.py --tech tsmc5,tsmc12
"""
from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

print = functools.partial(print, flush=True)  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))

from bsimar.config import CHECKPOINT_DIR, local_variant_code

# Per-tech switchcap geometry (matches tests/common/complex.py BenchTech +
# examples/complex/switchcap_unitcell_directnet.sp). vt = the checkpoint VT.
TECH = {
    "tsmc5":  dict(vdd=0.65, vt="lvt", ln=16e-9, lp=20e-9, nfin=2),
    "tsmc7":  dict(vdd=0.75, vt="ulvt", ln=16e-9, lp=20e-9, nfin=2),
    "tsmc12": dict(vdd=0.80, vt="svt", ln=16e-9, lp=20e-9, nfin=2),
    "tsmc16": dict(vdd=0.80, vt="svt", ln=16e-9, lp=20e-9, nfin=2),
}
CAP_SIGN = {"cgg": +1.0, "cdd": +1.0, "cgd": -1.0, "cdg": -1.0}


def _osdi_inst(tech: str, dev: str, variant: str, L: float, NFIN: float):
    from pycmg.nn_config import TECH_CONFIGS
    from pycmg.nn_generate import _create_model_and_instance
    out = _create_model_and_instance(
        TECH_CONFIGS[tech], dev, variant, L, NFIN, 300.15)
    if out is None:
        raise RuntimeError(f"OSDI instance build failed {tech}/{dev}/{variant}")
    return out[1]  # Instance


def _nn_device(tech: str, dev: str, L: float, NFIN: float):
    from pycircuitsim.models.mosfet_directnet import NMOS_NN, PMOS_NN
    cls = NMOS_NN if dev == "nmos" else PMOS_NN
    ck = CHECKPOINT_DIR / f"{tech}_dn_medium_{dev}_best.pt"
    tcode = local_variant_code(tech, tech, TECH[tech]["vt"])
    return cls("Mx", ["d", "g", "s", "b"], str(ck), L=L, NFIN=NFIN,
               temperature=300.15, tech_code=tcode)


def _relerr(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-18)


def run_pass(tech: str, dev: str, gate_v: float, body_v: float) -> None:
    """Walk vsamp 0->vin for one pass device; print NN-vs-OSDI per point."""
    cfg = TECH[tech]
    vdd, vin = cfg["vdd"], round(cfg["vdd"] * 0.6, 3)
    L = cfg["ln"] if dev == "nmos" else cfg["lp"]
    NFIN = cfg["nfin"]
    inst = _osdi_inst(tech, dev, cfg["vt"], L, NFIN)
    nn = _nn_device(tech, dev, L, NFIN)

    print(f"\n  --- {tech} {dev} pass  (vin={vin}, gate={gate_v}, body={body_v}, "
          f"L={L*1e9:.0f}n NFIN={NFIN}) ---")
    print(f"    {'vsamp':>6} | {'Vds':>6} {'Vgs':>6} {'Vbs':>6} | "
          f"{'id_NN':>10} {'id_OS':>10} {'idErr%':>7} | "
          f"{'cdd_NN':>9} {'cdd_OS':>9} {'cddE%':>6} | "
          f"{'cgg_NN':>9} {'cgg_OS':>9} {'cggE%':>6}")
    id_errs, cap_errs = [], []
    for vsamp in np.linspace(0.0, vin + 0.03, 16):
        # Actual node voltages; the NN device shifts by Vs internally.
        volts = {"d": vin, "g": gate_v, "s": float(vsamp), "b": body_v}
        r = nn._eval(volts)
        nn.clear_cache()
        vds, vgs, vbs = vin - vsamp, gate_v - vsamp, body_v - vsamp
        o = inst.eval_dc({"d": vds, "g": vgs, "s": 0.0, "e": vbs})
        ide = _relerr(r["id"], o["id"]) * 100.0
        # caps with sign mapping
        cdd_os, cgg_os = CAP_SIGN["cdd"] * o["cdd"], CAP_SIGN["cgg"] * o["cgg"]
        cdde = _relerr(r["cdd"], cdd_os) * 100.0
        cgge = _relerr(r["cgg"], cgg_os) * 100.0
        id_errs.append(ide)
        cap_errs.append(cdde)
        print(f"    {vsamp:6.3f} | {vds:6.3f} {vgs:6.3f} {vbs:6.3f} | "
              f"{r['id']:10.3e} {o['id']:10.3e} {ide:7.1f} | "
              f"{r['cdd']:9.2e} {cdd_os:9.2e} {cdde:6.1f} | "
              f"{r['cgg']:9.2e} {cgg_os:9.2e} {cgge:6.1f}")
    print(f"    >> median id err {np.median(id_errs):.1f}%  "
          f"median cdd err {np.median(cap_errs):.1f}%  "
          f"max id err {np.max(id_errs):.1f}%")


def run_tech(tech: str) -> None:
    cfg = TECH[tech]
    vdd = cfg["vdd"]
    print(f"\n=== {tech}  (VDD={vdd}) — switchcap pass-device trajectories ===")
    # NMOS pass: gate=phi=VDD, body=0.  PMOS pass: gate=phib=0, body=VDD.
    run_pass(tech, "nmos", gate_v=vdd, body_v=0.0)
    run_pass(tech, "pmos", gate_v=0.0, body_v=vdd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", default="tsmc5,tsmc12")
    args = ap.parse_args()
    print("=" * 100)
    print("Switchcap pass-device trajectory: real NN device (_eval) vs OSDI eval_dc")
    print("=" * 100)
    for tech in [t.strip() for t in args.tech.split(",")]:
        try:
            run_tech(tech)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  ERROR {tech}: {exc!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
