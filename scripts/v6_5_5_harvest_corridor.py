#!/usr/bin/env python3
"""V6.5.5 — TARGETED trajectory-corridor harvest (diagnostic-routed).

Recreates the V6.4.7-S12 corridor harvest, but TARGETED per the V6.5.5 Tier-1
verdicts (docs/plans/2026-06-24-v6.5.5-diagnose-then-corridor.md):

  * tsmc5 RING  — the period gap is CONDUCTION-owned (NMOS pull-down under-drives
    ~23% at the VDD/2 switching edge). Harvest the L72 ring switching tube.
  * tsmc7 OPAMP — gain->0 is VALUE-SURFACE-owned (the high-gain OP is unstable on
    the NN surface even when seeded from ground truth). Harvest the L72 opamp
    diff-pair / trip OP locus.

For the chosen circuit it runs the EXACT gate circuit through PyCircuitSim's own
solver with the ground-truth BSIM-CMG (LEVEL=72) OSDI model (the L72-control path
my diagnostics already proved matches NGSPICE — ring ~0%, opamp gain 163), reads
the per-device (Vd,Vg,Vs,Vb) trajectory each transistor actually visits,
source-shifts into the NN Vs=0 frame, dedups on a 2mV grid, adds a +/-12mV jitter
tube (20 samples) so the NN must match in a NEIGHBORHOOD of every OP (what NR
convergence needs), OSDI-evaluates ground truth at every sample, and saves a
per-(tech,device) `traj_corridor` fragment. Appended by v6_5_5_append_corridor.py.

Usage:
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      conda run -n pycircuitsim python scripts/v6_5_5_harvest_corridor.py \
      --tech tsmc5 --circuit ring
    ... --tech tsmc7 --circuit opamp
"""
from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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

from tests.common.complex import BENCH, BenchTech  # noqa: E402
from tests.diag_nn_ring_trajectory import _build_l72_ring, _solve_ring  # noqa: E402
from tests.diag_opamp_op_decomp import _run_l72_opamp  # noqa: E402
from pycircuitsim.models.mosfet_cmg import NMOS_CMG, PMOS_CMG  # noqa: E402
from pycmg.nn_generate import _create_model_and_instance, eval_single_point  # noqa: E402
from pycmg.nn_config import TECH_CONFIGS  # noqa: E402
from pycmg.sweep import NN_OUTPUT_COLUMNS  # noqa: E402

OUT_DIR = ROOT / "results" / "v6_5_5" / "corridors"
BENCH_VARIANT = {"tsmc5": "lvt", "tsmc7": "ulvt", "tsmc12": "svt", "tsmc16": "svt"}
ROOM_T_K = 300.15
NFIN = 2
L_NMOS, L_PMOS = 16e-9, 20e-9

BIAS_GRID_V = 0.002       # 2 mV dedup grid in (vd,vg,vbs)
PRE_EVAL_CAP = 200_000    # safety cap on unique biases before OSDI eval
JITTER_N = 20             # tube samples per unique bias
JITTER_V = 0.012          # +/-12 mV tube
JITTER_SEED = 20260624


def _device_map(circuit) -> List[Tuple[str, bool, List[str]]]:
    return [(c.name, isinstance(c, PMOS_CMG), list(c.nodes))
            for c in circuit.components if isinstance(c, (NMOS_CMG, PMOS_CMG))]


def _n_points(results: Dict) -> int:
    return max((len(np.atleast_1d(v)) for v in results.values()
               if hasattr(v, "__len__")), default=0)


def _trace(results: Dict, node: str, n: int) -> np.ndarray:
    if node in ("0", "gnd", "GND"):
        return np.zeros(n)
    a = np.asarray(results[node], dtype=float)
    if a.shape[0] != n:
        raise ValueError(f"node {node!r} len {a.shape[0]} != {n}")
    return a


def _harvest_traj(results: Dict, devs, nmos_acc: Dict, pmos_acc: Dict) -> int:
    """Accumulate source-shifted 2mV-grid-dedup'd biases with residence counts."""
    n = _n_points(results)
    if n == 0:
        return 0
    raw = 0
    for _name, is_pmos, nodes in devs:
        d, g, s, b = nodes
        vd, vg, vs, vb = (_trace(results, x, n) for x in (d, g, s, b))
        sd, sg, sbs = vd - vs, vg - vs, vb - vs   # Vs == 0 frame
        acc = pmos_acc if is_pmos else nmos_acc
        for k in range(n):
            key = (round(float(sd[k]) / BIAS_GRID_V),
                   round(float(sg[k]) / BIAS_GRID_V),
                   round(float(sbs[k]) / BIAS_GRID_V))
            acc[key] = acc.get(key, 0) + 1
            raw += 1
    return raw


def harvest_circuit(tech: str, circuit_name: str, work: Path) -> Dict:
    bt: BenchTech = BENCH[tech.upper()]
    nmos_acc: Dict[Tuple[int, int, int], int] = {}
    pmos_acc: Dict[Tuple[int, int, int], int] = {}

    if circuit_name == "ring":
        circ = _build_l72_ring(bt, work)
        devs = _device_map(circ)
        res = _solve_ring(circ, charge_on=True)
        raw = _harvest_traj(res, devs, nmos_acc, pmos_acc)
        print(f"    [{tech}/ring] devs={len(devs)} pts={_n_points(res)} raw={raw}")
    elif circuit_name == "opamp":
        res, circ = _run_l72_opamp(bt, work)
        devs = _device_map(circ)
        # results carry node arrays; fixed sources (vdd/vbn/vbp/inn) may be
        # absent as columns — fill them so every terminal resolves.
        vcm = round(bt.vdd * 0.55, 3)
        vbn = round(bt.vdd * 0.45, 3)
        n_pts = _n_points(res)
        res = {k: np.asarray(v) for k, v in res.items()}
        for nd, val in (("vdd", bt.vdd), ("vbn", vbn), ("inn", vcm)):
            res.setdefault(nd, np.full(n_pts, float(val)))
        raw = _harvest_traj(res, devs, nmos_acc, pmos_acc)
        print(f"    [{tech}/opamp] devs={len(devs)} pts={n_pts} raw={raw}")
    else:
        raise ValueError(f"unknown circuit {circuit_name!r}")

    print(f"  [{tech}/{circuit_name}] unique NMOS={len(nmos_acc)} "
          f"unique PMOS={len(pmos_acc)}")
    return {"nmos": nmos_acc, "pmos": pmos_acc, "bt": bt}


def eval_and_save(tech: str, dev: str, acc: Dict, circuit_name: str) -> Dict:
    is_pmos = dev == "pmos"
    L = L_PMOS if is_pmos else L_NMOS
    variant = BENCH_VARIANT[tech]
    cfg = TECH_CONFIGS[tech]
    built = _create_model_and_instance(cfg, dev, variant, L, float(NFIN), ROOM_T_K)
    if built is None:
        raise RuntimeError(f"OSDI instance build failed {tech}/{dev}/{variant}")
    _model, inst, proc = built
    geo = np.array([float(NFIN), L, ROOM_T_K] + proc.as_array(), dtype=np.float64)

    items = sorted(acc.items(), key=lambda kv: -kv[1])
    if len(items) > PRE_EVAL_CAP:
        items = items[:PRE_EVAL_CAP]

    inputs, outputs, residence = [], [], []
    n_fail = 0
    rng = np.random.default_rng(JITTER_SEED + (1 if is_pmos else 0))
    for (kd, kg, kbs), res in items:
        cvd, cvg, cvbs = kd * BIAS_GRID_V, kg * BIAS_GRID_V, kbs * BIAS_GRID_V
        samples = [(cvd, cvg, cvbs)]
        if JITTER_N > 0:
            for dvd, dvg, dvbs in rng.uniform(-JITTER_V, JITTER_V, size=(JITTER_N, 3)):
                samples.append((cvd + dvd, cvg + dvg, cvbs + dvbs))
        for vd, vg, vbs in samples:
            out = eval_single_point(inst, vd=vd, vg=vg, vs=0.0, vb=vbs)
            if out is None:
                n_fail += 1
                continue
            inputs.append([vd, vg, 0.0, vbs])
            outputs.append([out[k] for k in NN_OUTPUT_COLUMNS])
            residence.append(res)

    inputs = np.asarray(inputs, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64)
    residence = np.asarray(residence, dtype=np.int64)
    geometry = np.tile(geo, (len(inputs), 1))
    id_idx = list(NN_OUTPUT_COLUMNS).index("id")
    idmag = np.abs(outputs[:, id_idx]) if len(outputs) else np.array([])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frag = OUT_DIR / f"{tech}_{circuit_name}_{dev}_corridor.npz"
    np.savez(frag, inputs=inputs, geometry=geometry, outputs=outputs,
             residence=residence, idmag=idmag,
             meta_tech=tech, meta_device=dev, meta_variant=variant,
             meta_circuit=circuit_name, meta_L=L, meta_NFIN=NFIN, meta_T=ROOM_T_K,
             meta_output_columns=np.array(list(NN_OUTPUT_COLUMNS)))
    nz = idmag[idmag > 0]
    decs = np.floor(np.log10(nz)).astype(int) if len(nz) else np.array([])
    dec_hist = {int(d): int((decs == d).sum()) for d in np.unique(decs)} if len(decs) else {}
    print(f"    [{tech}/{dev}] saved {len(inputs)} rows (fail={n_fail}) -> "
          f"{frag.name}  |id| decades={dec_hist}")
    return {"tech": tech, "dev": dev, "rows": int(len(inputs)), "frag": str(frag)}


def main() -> int:
    ap = argparse.ArgumentParser(description="V6.5.5 targeted corridor harvest")
    ap.add_argument("--tech", required=True, help="tsmc5 | tsmc7 | ...")
    ap.add_argument("--circuit", required=True, choices=["ring", "opamp"])
    args = ap.parse_args()
    tech = args.tech.strip().lower()

    print(f"=== V6.5.5 harvest {tech} / {args.circuit} ===")
    t0 = time.time()
    import tempfile
    work = Path(tempfile.mkdtemp(prefix=f"v655_{tech}_{args.circuit}_"))
    accs = harvest_circuit(tech, args.circuit, work)
    summary = []
    for dev in ("nmos", "pmos"):
        summary.append(eval_and_save(tech, dev, accs[dev], args.circuit))
    print(f"  done in {time.time()-t0:.0f}s")
    for r in summary:
        print(f"  {r['tech']:6s} {r['dev']:4s} rows={r['rows']:7d} -> {Path(r['frag']).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
