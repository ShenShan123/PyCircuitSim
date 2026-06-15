#!/usr/bin/env python3
"""V6.4.7 S11 (P3) — subthreshold/weak-inversion id-fidelity probe.

The SRAM force_ic kill-gate leading indicator. At the inboard attractor the
pinning NMOS over-predicts its weak-inversion current ~7.5x (P0-D); this probe
measures the NN-vs-OSDI |id| ratio on a FIXED, checkpoint-independent
subthreshold Id-Vgs grid (so control-v2 and a candidate compare apples to
apples — re-solving the attractor is self-defeating since a better model
moves the attractor). Reports the median |NN/OSDI| per OSDI-|id| decade; the
weak-inversion band [1e-9, 1e-6] A is the pinning-device band the kill-gate
("≥10x improvement") scores, and [<1e-9] is the hard-OFF suppression band.

Checkpoint selection: the caller points ``BSIMAR_CHECKPOINT_DIR`` at an
isolated dir holding the candidate as ``{tech}_dn_medium_{dev}_best.pt`` (the
per-tech production name that fires Rule-19 local-vocab scope detection), so
both control-v2 and the P3 candidate use the identical biases. The combined
gate ``v6_4_7_s11_sram_gate.py`` wires that up; this script is also runnable
standalone once the env is set. CUDA must be hidden (CPU inference):
``CUDA_VISIBLE_DEVICES=""``.

Ground truth is ALWAYS the OSDI BSIM-CMG binary via PyCMG (CLAUDE.md).

Run:
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 BSIMAR_CHECKPOINT_DIR=<iso> \
      conda run -n pycircuitsim python scripts/v6_4_7_s11_subvt_probe.py \
      --label control-v2 --techs TSMC5,TSMC7,TSMC12,TSMC16 --json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))
sys.path.append(str(PROJECT_ROOT / "external_compact_models" / "PyCMG" / "tests"))
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.complex import BENCH  # noqa: E402
from pycircuitsim.solver import _is_mosfet  # noqa: E402
from pycmg.nn_config import TECH_CONFIGS  # noqa: E402
from pycmg.nn_generate import (  # noqa: E402
    _create_model_and_instance, eval_single_point)

NFIN = 2.0
L_NMOS = 16e-9
L_PMOS = 20e-9
TEMP_K = 300.15

# OSDI-|id| decades the ratios are binned into. The pinning device lives in
# the weak-inversion band [1e-9, 1e-6]; the hard-OFF band is <1e-9.
_BANDS = [(0.0, 1e-9, "off"), (1e-9, 1e-7, "weak_lo"),
          (1e-7, 1e-6, "weak_hi"), (1e-6, 1e-4, "moderate")]


def _osdi_inst(tech: str, variant: str, device_type: str):
    cfg = TECH_CONFIGS[tech]
    L = L_NMOS if device_type == "nmos" else L_PMOS
    _, inst, _ = _create_model_and_instance(
        cfg, device_type, variant, L, NFIN, TEMP_K)
    return inst


def _one_device_netlist(nn_tech: str, vt: str, device_type: str,
                        path: Path) -> Path:
    mtype = "NMOS" if device_type == "nmos" else "PMOS"
    L = "16n" if device_type == "nmos" else "20n"
    txt = (
        "* s11 subvt single-device probe\n"
        f".model dev_nn {mtype} (LEVEL=73 TECH={nn_tech} VT={vt})\n"
        "Vd d 0 0\nVg g 0 0\nVs s 0 0\nVb b 0 0\n"
        f"M1 d g s b dev_nn L={L} NFIN=2\n"
        ".op\n.end\n")
    path.write_text(txt)
    return path


def _nn_mosfet(nn_tech: str, vt: str, device_type: str, work: Path):
    from tests.common.complex import parse_netlist
    nl = _one_device_netlist(nn_tech, vt, device_type,
                             work / f"probe_{nn_tech}_{device_type}.sp")
    logging.disable(logging.CRITICAL)
    try:
        parser = parse_netlist(nl)
    finally:
        logging.disable(logging.NOTSET)
    mos = [c for c in parser.circuit.components if _is_mosfet(c)]
    assert len(mos) == 1, f"expected 1 mosfet, got {len(mos)}"
    return mos[0]


def probe_device(tech: str, device_type: str, work: Path) -> Dict:
    bt = BENCH[tech]
    vdd = bt.vdd
    variant = bt.vt
    m = _nn_mosfet(bt.nn_tech, variant, device_type, work)
    inst = _osdi_inst(bt.nn_tech, variant, device_type)
    sign = 1.0 if device_type == "nmos" else -1.0  # conducting-gate polarity

    # Deep-OFF → weak-inversion Id-Vgs grid. Source at 0 (NMOS) / vdd (PMOS);
    # gate swept from below threshold (captures the |id|<1e-9 OFF leakage the
    # SRAM rails must suppress) up to onset of strong inversion. Small Vds
    # (~attractor node split), saturation, and full-VDD (the held-rail OFF
    # device sees Vds≈VDD); zero and reverse-body. Absolute terminal voltages.
    vgs_grid = np.linspace(-0.15, 0.92, 55) * vdd
    vds_levels = [0.06 * vdd / 0.8, 0.5 * vdd, vdd]
    vbs_levels = [0.0, -0.2]

    ratios: List[float] = []
    rows: List[Dict] = []
    for vbs in vbs_levels:
        for vds in vds_levels:
            for vgs in vgs_grid:
                if device_type == "nmos":
                    vs, vb = 0.0, vbs
                    vg, vd = vgs, vds
                else:  # PMOS: source at vdd, conduction for vg<vs
                    vs = vdd
                    vb = vdd + vbs
                    vg = vdd - vgs
                    vd = vdd - vds
                nn = m._eval({"d": vd, "g": vg, "s": vs, "b": vb})
                osdi = eval_single_point(inst, vd, vg, vs, vb)
                if osdi is None:
                    continue
                nn_id = abs(float(nn["id"]))
                os_id = abs(float(osdi["id"]))
                if os_id <= 0 or not np.isfinite(os_id):
                    continue
                r = max(nn_id, 1e-30) / os_id
                rows.append({"vgs": float(vgs), "vds": float(vds),
                             "vbs": float(vbs), "nn_id": nn_id,
                             "osdi_id": os_id, "ratio": r})

    # Bin by OSDI |id| decade band; report median |log10 ratio| (the
    # checkpoint-comparable fidelity) and median ratio (signed direction).
    out: Dict = {"tech": tech, "device": device_type, "vdd": vdd,
                 "n_points": len(rows), "bands": {}}
    for lo, hi, name in _BANDS:
        rs = [x["ratio"] for x in rows if lo <= x["osdi_id"] < hi]
        if rs:
            logr = np.abs(np.log10(np.clip(rs, 1e-30, None)))
            out["bands"][name] = {
                "n": len(rs),
                "median_ratio": float(np.median(rs)),
                "median_abs_log10": float(np.median(logr)),
                "max_abs_log10": float(np.max(logr)),
            }
        else:
            out["bands"][name] = {"n": 0}
    return out


def run_probe(techs: List[str], work: Path) -> List[Dict]:
    work.mkdir(parents=True, exist_ok=True)
    results: List[Dict] = []
    for tech in techs:
        for dev in ("nmos", "pmos"):
            r = probe_device(tech, dev, work)
            results.append(r)
            wl = r["bands"].get("weak_lo", {})
            wh = r["bands"].get("weak_hi", {})
            off = r["bands"].get("off", {})
            print(f"{tech:7s} {dev:4s} | "
                  f"weak_lo med|log10|={wl.get('median_abs_log10', float('nan')):.3f} "
                  f"(ratio {wl.get('median_ratio', float('nan')):.2f}) | "
                  f"weak_hi med|log10|={wh.get('median_abs_log10', float('nan')):.3f} "
                  f"(ratio {wh.get('median_ratio', float('nan')):.2f}) | "
                  f"off med|log10|={off.get('median_abs_log10', float('nan')):.3f}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="(unset)",
                    help="reporting label; the candidate is selected by the "
                         "caller via BSIMAR_CHECKPOINT_DIR")
    ap.add_argument("--techs", default="TSMC5,TSMC7,TSMC12,TSMC16")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    work = PROJECT_ROOT / "results" / "v6_4_7" / "s11_probe_scratch"
    techs = [t.strip() for t in args.techs.split(",") if t.strip()]
    results = run_probe(techs, work)
    if args.json:
        print("RESULT " + json.dumps(
            {"label": args.label, "devices": results}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
