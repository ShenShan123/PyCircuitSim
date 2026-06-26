#!/usr/bin/env python3
"""V6.5.5 T1 — harvest opamp net-node KCL groups (the lever P0-1 routed to).

P0-1 (tests/diag_opamp_kcl_residual.py) proved the tsmc7 opamp gain->0 is an
EXISTENCE failure: at the L72 ground-truth high-gain OP the NN's net KCL current
into the stage-1 balance node vo1i is 12.8% of the arm current — the high-gain
fixed point is NOT a residual zero of the NN-current map. The corridor only ever
supervised each device's ABSOLUTE id (asinh-compressed, s_id~2.6e-5) — never the
DIFFERENCE i_Mn2 - i_Mp4 at the balance node. T1 supervises exactly that
difference, in the NATIVE-µA frame, via a grouped net-node KCL-residual loss.

This script harvests the GROUP structure the loss needs: it runs the EXACT gate
opamp through PyCircuitSim's own solver with the ground-truth BSIM-CMG (LEVEL=72)
OSDI model (the L72-control path the diagnostics proved matches NGSPICE, gain
~163), and at each Vin sweep point near the high-gain crossing records, per
transistor:

  * the source-referenced NN-frame bias (Vd-Vs, Vg-Vs, 0, Vb-Vs) it visits,
  * NFIN / L / T geometry + tech_code (local vocab),
  * the free-node incidence (which of {vtail,n1,vo1i,vout} its drain/source
    touch) and the sign — exactly the solver's KCL assembly
    (solver._stamp_mosfet_dc:303-309): i_leaving = -id ; F[drain]+= -id ;
    F[source]+= +id  (for BOTH device types, Rule 2).

Per group it also stores the L72 ground-truth arm current per free node (the
F_rel normalizer = max |i_leaving| touching the node) and the L72 self-check
residual (must be ~0 — validates the sign assembly and that the OP is a genuine
KCL zero of L72 currents).

The fine-tune (scripts/v6_5_5_finetune_kcl.py) then jointly nudges the tsmc7
NMOS+PMOS surfaces so Σ(signed NN id) -> 0 at these true OPs, WITHOUT moving the
15 passing gates (anchored by the base-data MAE).

Usage:
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NGSPICE_BIN="$PWD/tools/ngspice-45.2/bin/ngspice" \
      conda run -n pycircuitsim python scripts/v6_5_5_harvest_kcl.py \
        --tech TSMC7 --band 0.06
"""
from __future__ import annotations

import argparse
import functools
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

print = functools.partial(print, flush=True)  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "external_compact_models" / "PyCMG" / "tests",
          ROOT / "external_compact_models" / "PyCMG",
          ROOT / "external_compact_models", ROOT):
    sp = str(p)
    if sp in sys.path:
        sys.path.remove(sp)
    sys.path.insert(0, sp)

from tests.common.complex import BENCH, BenchTech, RESULTS_BASE  # noqa: E402
from tests.diag_opamp_op_decomp import _run_l72_opamp  # noqa: E402
from tests.verify_complex_opamp import _bias, _gain_trip  # noqa: E402
from pycircuitsim.solver import _is_pmos, _is_mosfet  # noqa: E402
from bsimar.config import local_variant_code  # noqa: E402

OUT_DIR = ROOT / "results" / "v6_5_5" / "kcl_groups"
FREE_NODES = ("vtail", "n1", "vo1i", "vout")
# bench geometry the gate opamp instantiates (NMOS 16n / PMOS 20n, NFIN=2).
L_NMOS, L_PMOS = 16e-9, 20e-9
NFIN = 2.0
ROOM_T_K = 300.15
# opamp variant per tech (matches scripts/v6_5_5_harvest_corridor BENCH_VARIANT
# and the miller_opamp_directnet.sp VT the gate resolves).
BENCH_VARIANT = {"tsmc5": "lvt", "tsmc7": "ulvt", "tsmc12": "svt", "tsmc16": "svt"}
ARM_FLOOR_A = 1e-7   # floor the F_rel normalizer so railed points don't blow up


def _free_idx(node: str) -> int:
    return FREE_NODES.index(node) if node in FREE_NODES else -1


def harvest(tech: str, band: float) -> Dict:
    bt: BenchTech = BENCH[tech.upper()]
    vdd = bt.vdd
    vcm, vbn, vbp = _bias(bt)
    scope = tech.lower()
    variant = BENCH_VARIANT[scope]
    tcode = local_variant_code(scope, scope, variant)
    work = RESULTS_BASE / "kcl_harvest" / tech.upper()
    work.mkdir(parents=True, exist_ok=True)

    logging.disable(logging.CRITICAL)
    try:
        res, circ = _run_l72_opamp(bt, work)
    finally:
        logging.disable(logging.NOTSET)

    sw = np.asarray(res["inp"])
    vo = np.asarray(res["vout"])
    n_pts = sw.shape[0]
    g_l, _, _ = _gain_trip(sw, vo, vdd)
    vin_star = float(sw[int(np.argmax(np.abs(np.gradient(vo, sw))))])

    # fill fixed source nodes that may not be present as sweep columns.
    res = {k: np.asarray(v) for k, v in res.items()}
    for nd, val in (("vdd", vdd), ("vbn", vbn), ("vbp", vbp),
                    ("inn", vcm), ("0", 0.0)):
        if nd not in res or np.asarray(res[nd]).shape[0] != n_pts:
            res[nd] = np.full(n_pts, float(val))

    # device topology (ordered, fixed across groups).
    devs = [c for c in circ.components if _is_mosfet(c)]
    dev_is_pmos = np.array([1 if _is_pmos(c) else 0 for c in devs], dtype=np.int8)
    drain_free = np.array([_free_idx(c.nodes[0]) for c in devs], dtype=np.int64)
    source_free = np.array([_free_idx(c.nodes[2]) for c in devs], dtype=np.int64)
    dev_nfin = np.array([NFIN] * len(devs), dtype=np.float64)
    dev_L = np.array([L_PMOS if p else L_NMOS for p in dev_is_pmos],
                     dtype=np.float64)
    dev_T = np.array([ROOM_T_K] * len(devs), dtype=np.float64)
    dev_tcode = np.array([tcode] * len(devs), dtype=np.int64)
    dev_names = [c.name for c in devs]

    print(f"\n===== {tech.upper()} (VDD={vdd}) opamp KCL harvest =====")
    print(f"  L72 swept gain={g_l:.1f}  vin*={vin_star:.4f}  band=±{band}")
    print(f"  free nodes={FREE_NODES}  devices={len(devs)} variant={variant} "
          f"tech_code={tcode}")
    for c, df, sf in zip(devs, drain_free, source_free):
        kind = "PMOS" if _is_pmos(c) else "NMOS"
        print(f"    {c.name:4s} {kind} d={c.nodes[0]:>5s}(free{df:+d}) "
              f"g={c.nodes[1]:>5s} s={c.nodes[2]:>5s}(free{sf:+d}) b={c.nodes[3]}")

    nn_volts: List[np.ndarray] = []   # (G, ndev, 4) source-referenced phys V
    arm_scale: List[np.ndarray] = []  # (G, 4)
    l72_selfcheck: List[np.ndarray] = []  # (G, 4)
    vins: List[float] = []
    n_kept = 0
    max_selfcheck = 0.0
    for k in range(n_pts):
        if abs(float(sw[k]) - vin_star) > band:
            continue
        op = {nd: float(np.asarray(arr)[k]) for nd, arr in res.items()
              if np.asarray(arr).shape[0] == n_pts}
        # per-device source-referenced NN-frame bias + L72 i_leaving.
        volts = np.zeros((len(devs), 4), dtype=np.float64)
        F = np.zeros(4, dtype=np.float64)
        arm = np.zeros(4, dtype=np.float64)
        for j, c in enumerate(devs):
            d, g, s, b = c.nodes
            vs = op[s]
            volts[j] = [op[d] - vs, op[g] - vs, 0.0, op[b] - vs]
            i_ds = c.calculate_current(op)       # L72 ground truth
            i_lev = -i_ds if _is_pmos(c) else i_ds
            for nd_idx, coeff in ((drain_free[j], 1.0), (source_free[j], -1.0)):
                if nd_idx >= 0:
                    F[nd_idx] += coeff * i_lev
                    arm[nd_idx] = max(arm[nd_idx], abs(i_lev))
        nn_volts.append(volts)
        arm_scale.append(np.maximum(arm, ARM_FLOOR_A))
        l72_selfcheck.append(F)
        vins.append(float(sw[k]))
        max_selfcheck = max(max_selfcheck, float(np.max(np.abs(F))))
        n_kept += 1

    nn_volts = np.asarray(nn_volts)
    arm_scale = np.asarray(arm_scale)
    l72_selfcheck = np.asarray(l72_selfcheck)
    vins = np.asarray(vins)

    print(f"  groups kept={n_kept}  L72 self-check max|F|={max_selfcheck:.3e} A "
          f"({'OK <1e-7' if max_selfcheck < 1e-7 else 'WARN — sign/columns suspect'})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{tech.lower()}_opamp_kcl.npz"
    np.savez(
        out,
        nn_volts=nn_volts, arm_scale=arm_scale, l72_selfcheck=l72_selfcheck,
        vin=vins, vin_star=vin_star,
        dev_is_pmos=dev_is_pmos, drain_free=drain_free, source_free=source_free,
        dev_nfin=dev_nfin, dev_L=dev_L, dev_T=dev_T, dev_tcode=dev_tcode,
        free_nodes=np.array(FREE_NODES), dev_names=np.array(dev_names),
        meta_tech=tech.lower(), meta_variant=variant, meta_vdd=vdd,
        meta_l72_gain=g_l,
    )
    print(f"  saved -> {out}")
    return {"tech": tech, "n_groups": n_kept, "selfcheck": max_selfcheck,
            "path": str(out)}


def main() -> int:
    ap = argparse.ArgumentParser(description="V6.5.5 T1 opamp KCL-group harvest")
    ap.add_argument("--tech", default="TSMC7")
    ap.add_argument("--band", type=float, default=0.06,
                    help="keep sweep OPs within ±band V of the high-gain crossing")
    args = ap.parse_args()
    print("=" * 78)
    print("V6.5.5 T1 — opamp net-node KCL-group harvest (existence-failure lever)")
    print("=" * 78)
    rows = []
    for t in [x.strip() for x in args.tech.split(",")]:
        if t.upper() not in BENCH:
            print(f"  SKIP unknown tech {t}"); continue
        rows.append(harvest(t, args.band))
    print("\nSUMMARY:")
    for r in rows:
        print(f"  {r['tech']:7s} groups={r['n_groups']:4d} "
              f"selfcheck={r['selfcheck']:.2e} -> {Path(r['path']).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
