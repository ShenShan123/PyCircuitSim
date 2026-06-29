#!/usr/bin/env python3
"""V6.5.8 T3 — harvest the FULL opamp topology for the differentiable DC solver.

The V6.5.5 KCL harvest (``v6_5_5_harvest_kcl.py``) records only what the static
residual fine-tune needs: the source-referenced device bias at the FROZEN L72 OP
plus the drain/source free-node incidence. ``KCLGroups.residual()`` therefore
evaluates ``F`` at a FIXED ``V`` — it is ``F(V_L72; θ)``, not ``F(V; θ)``.

T3 (the differentiable unrolled DC solver, plan
``2026-06-28-tsmc7-opamp-T3-differentiable-solver.md``) needs the residual as a
*function of the free-node voltage vector* ``V = [vtail, n1, vo1i, vout]`` so it
can wrap a Newton solve around it and supervise the emergent transfer curve
``Vout(Vin)``. That requires the FULL per-terminal topology the static harvest
omits:

  * ``term_free``     (ndev, 4) — for each terminal [d,g,s,b], the free-node
    index (0..n_free-1) it connects to, or -1 for a fixed rail/bias node. The
    static harvest only stored drain/source; the GATE terminals (Mp3/Mp4 → n1,
    Mp6 → vo1i) and BULK terminals are also needed to rebuild the device bias as
    V varies.
  * ``term_is_vin``   (ndev, 4) — 1 where the terminal connects to the swept
    input node ``inp`` (the independent variable of the transfer curve).
  * ``term_fix_v``    (G, ndev, 4) — the absolute physical voltage of every
    terminal at each group's L72 OP (used for FIXED terminals; free / vin
    terminals are overwritten at solve time).
  * ``V_l72``         (G, n_free) — the absolute L72 OP free-node voltages
    (teacher-forcing init + the ``vout`` curve target).

Reuses the EXACT gate opamp (``_run_l72_opamp``, the L72-control path the
diagnostics proved matches NGSPICE at gain ~163) and the same sign / arm-scale
conventions as the static harvest, so the two npz cross-check.

Usage:
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NGSPICE_BIN="$PWD/tools/ngspice-45.2/bin/ngspice" \
      conda run -n pycircuitsim python scripts/v6_5_8_harvest_opamp_topology.py \
        --tech TSMC7 --band 0.08
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
VIN_NODE = "inp"   # the .dc-swept input node (Vinp); Vinn (inn) is held at vcm
L_NMOS, L_PMOS = 16e-9, 20e-9
NFIN = 2.0
ROOM_T_K = 300.15
BENCH_VARIANT = {"tsmc5": "lvt", "tsmc7": "ulvt", "tsmc12": "svt", "tsmc16": "svt"}
ARM_FLOOR_A = 1e-7


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

    res = {k: np.asarray(v) for k, v in res.items()}
    for nd, val in (("vdd", vdd), ("vbn", vbn), ("vbp", vbp),
                    ("inn", vcm), ("0", 0.0)):
        if nd not in res or np.asarray(res[nd]).shape[0] != n_pts:
            res[nd] = np.full(n_pts, float(val))

    devs = [c for c in circ.components if _is_mosfet(c)]
    ndev = len(devs)
    dev_is_pmos = np.array([1 if _is_pmos(c) else 0 for c in devs], dtype=np.int8)
    drain_free = np.array([_free_idx(c.nodes[0]) for c in devs], dtype=np.int64)
    source_free = np.array([_free_idx(c.nodes[2]) for c in devs], dtype=np.int64)
    # FULL per-terminal incidence [d, g, s, b].
    term_free = np.array([[_free_idx(c.nodes[t]) for t in range(4)]
                          for c in devs], dtype=np.int64)
    term_is_vin = np.array([[1 if c.nodes[t] == VIN_NODE else 0
                             for t in range(4)] for c in devs], dtype=np.int8)
    term_node = np.array([[str(c.nodes[t]) for t in range(4)] for c in devs])
    dev_nfin = np.array([NFIN] * ndev, dtype=np.float64)
    dev_L = np.array([L_PMOS if p else L_NMOS for p in dev_is_pmos],
                     dtype=np.float64)
    dev_T = np.array([ROOM_T_K] * ndev, dtype=np.float64)
    dev_tcode = np.array([tcode] * ndev, dtype=np.int64)
    dev_names = np.array([c.name for c in devs])
    n_free = len(FREE_NODES)

    print(f"\n===== {tech.upper()} (VDD={vdd}) opamp TOPOLOGY harvest =====")
    print(f"  L72 swept gain={g_l:.1f}  vin*={vin_star:.4f}  band=±{band}  "
          f"vcm={vcm} vbn={vbn} vbp={vbp}")
    print(f"  free nodes={FREE_NODES}  devices={ndev} variant={variant} "
          f"tech_code={tcode}")
    for j, c in enumerate(devs):
        kind = "PMOS" if _is_pmos(c) else "NMOS"
        tn = term_node[j]
        tf = term_free[j]
        vin_mark = "".join("*" if term_is_vin[j, t] else " " for t in range(4))
        print(f"    {c.name:4s} {kind}  d={tn[0]:>5s}(f{tf[0]:+d}) "
              f"g={tn[1]:>5s}(f{tf[1]:+d}) s={tn[2]:>5s}(f{tf[2]:+d}) "
              f"b={tn[3]:>5s}(f{tf[3]:+d})  vin@[{vin_mark}]")

    nn_volts: List[np.ndarray] = []     # (G, ndev, 4) source-ref phys V (legacy)
    arm_scale: List[np.ndarray] = []    # (G, n_free)
    l72_selfcheck: List[np.ndarray] = []  # (G, n_free)
    V_l72: List[np.ndarray] = []        # (G, n_free) absolute free-node V
    term_fix_v: List[np.ndarray] = []   # (G, ndev, 4) absolute terminal V
    vins: List[float] = []
    n_kept = 0
    max_selfcheck = 0.0
    for k in range(n_pts):
        if abs(float(sw[k]) - vin_star) > band:
            continue
        op = {nd: float(np.asarray(arr)[k]) for nd, arr in res.items()
              if np.asarray(arr).shape[0] == n_pts}
        volts = np.zeros((ndev, 4), dtype=np.float64)
        tfix = np.zeros((ndev, 4), dtype=np.float64)
        F = np.zeros(n_free, dtype=np.float64)
        arm = np.zeros(n_free, dtype=np.float64)
        for j, c in enumerate(devs):
            d, g, s, b = c.nodes
            vs = op[s]
            volts[j] = [op[d] - vs, op[g] - vs, 0.0, op[b] - vs]
            tfix[j] = [op[d], op[g], op[s], op[b]]   # absolute terminal V
            i_ds = c.calculate_current(op)
            i_lev = -i_ds if _is_pmos(c) else i_ds
            for nd_idx, coeff in ((drain_free[j], 1.0), (source_free[j], -1.0)):
                if nd_idx >= 0:
                    F[nd_idx] += coeff * i_lev
                    arm[nd_idx] = max(arm[nd_idx], abs(i_lev))
        nn_volts.append(volts)
        term_fix_v.append(tfix)
        arm_scale.append(np.maximum(arm, ARM_FLOOR_A))
        l72_selfcheck.append(F)
        V_l72.append(np.array([op[nd] for nd in FREE_NODES], dtype=np.float64))
        vins.append(float(sw[k]))
        max_selfcheck = max(max_selfcheck, float(np.max(np.abs(F))))
        n_kept += 1

    nn_volts = np.asarray(nn_volts)
    arm_scale = np.asarray(arm_scale)
    l72_selfcheck = np.asarray(l72_selfcheck)
    V_l72 = np.asarray(V_l72)
    term_fix_v = np.asarray(term_fix_v)
    vins = np.asarray(vins)
    vout_l72 = V_l72[:, FREE_NODES.index("vout")]

    print(f"  groups kept={n_kept}  L72 self-check max|F|={max_selfcheck:.3e} A "
          f"({'OK <1e-7' if max_selfcheck < 1e-7 else 'WARN — sign/cols suspect'})")
    print(f"  Vout(vin) curve over band: [{vout_l72.min():.4f}, "
          f"{vout_l72.max():.4f}]  (mid-rail VDD/2={vdd/2:.4f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{tech.lower()}_opamp_topo.npz"
    np.savez(
        out,
        # group-varying
        vin=vins, V_l72=V_l72, vout_l72=vout_l72,
        arm_scale=arm_scale, l72_selfcheck=l72_selfcheck,
        nn_volts=nn_volts, term_fix_v=term_fix_v,
        # topology (fixed across groups)
        free_nodes=np.array(FREE_NODES), dev_names=dev_names,
        dev_is_pmos=dev_is_pmos, drain_free=drain_free, source_free=source_free,
        term_free=term_free, term_is_vin=term_is_vin, term_node=term_node,
        dev_nfin=dev_nfin, dev_L=dev_L, dev_T=dev_T, dev_tcode=dev_tcode,
        # meta
        vin_star=vin_star, meta_tech=tech.lower(), meta_variant=variant,
        meta_vdd=vdd, meta_l72_gain=g_l,
    )
    print(f"  saved -> {out}")
    return {"tech": tech, "n_groups": n_kept, "selfcheck": max_selfcheck,
            "l72_gain": g_l, "path": str(out)}


def main() -> int:
    ap = argparse.ArgumentParser(description="V6.5.8 T3 opamp topology harvest")
    ap.add_argument("--tech", default="TSMC7")
    ap.add_argument("--band", type=float, default=0.08,
                    help="keep sweep OPs within ±band V of the high-gain crossing "
                         "(wider than the 0.06 KCL band so the peak-slope window "
                         "and its neighbours are fully covered)")
    args = ap.parse_args()
    print("=" * 78)
    print("V6.5.8 T3 — opamp FULL-topology harvest (differentiable DC solver)")
    print("=" * 78)
    rows = []
    for t in [x.strip() for x in args.tech.split(",")]:
        if t.upper() not in BENCH:
            print(f"  SKIP unknown tech {t}"); continue
        rows.append(harvest(t, args.band))
    print("\nSUMMARY:")
    for r in rows:
        print(f"  {r['tech']:7s} groups={r['n_groups']:4d} "
              f"selfcheck={r['selfcheck']:.2e} L72gain={r['l72_gain']:.1f} "
              f"-> {Path(r['path']).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
