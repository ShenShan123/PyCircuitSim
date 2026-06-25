#!/usr/bin/env python3
"""V6.5.5 — append a targeted trajectory-corridor fragment to the clean dataset.

Merges the per-(tech,device) `traj_corridor` fragment harvested by
v6_5_5_harvest_corridor.py into the CURRENT clean V6.5.4 dataset
(`tsmc{X}_{dev}.npz`) as sample_class code 12 (traj_corridor), writing a NEW
`tsmc{X}_cor_{dev}.npz` (the clean original stays pristine). Pre-seeds the
tech-variant LABEL CACHE so the off-PDK-grid corridor rows (NMOS L=16n) are
never fingerprinted: base rows via the labeller, corridor rows via the known
bench-variant universal code (the loader then remaps to the per-tech local vocab
exactly like the base rows). Validates column agreement, geometry, decade
coverage, cache length, and a reload before any training.

Usage:
    conda run -n pycircuitsim python scripts/v6_5_5_append_corridor.py \
        --tech tsmc5 --circuit ring
    ... --tech tsmc7 --circuit opamp
"""
from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

print = functools.partial(print, flush=True)  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "external_compact_models" / "PyCMG",
          ROOT / "external_compact_models", ROOT):
    sp = str(p)
    if sp in sys.path:
        sys.path.remove(sp)
    sys.path.insert(0, sp)

from bsimar.eval.loo_labels import get_or_build_tech_variant_labels  # noqa: E402
from bsimar.config import tech_variant_to_code  # noqa: E402
from pycmg.nn_generate import SAMPLE_CLASS_CODES  # noqa: E402

DATA_DIR = ROOT / "external_compact_models" / "bsimar" / "data" / "datasets"
FRAG_DIR = ROOT / "results" / "v6_5_5" / "corridors"
BENCH_VARIANT = {"tsmc5": "lvt", "tsmc7": "ulvt", "tsmc12": "svt", "tsmc16": "svt"}
CORRIDOR_CODE = SAMPLE_CLASS_CODES["traj_corridor"]  # 12


def _as_str_list(arr) -> List[str]:
    return [a.decode() if isinstance(a, bytes) else str(a) for a in arr]


def append_one(tech: str, circuit: str, dev: str) -> Dict:
    base_path = DATA_DIR / f"{tech}_{dev}.npz"
    frag_path = FRAG_DIR / f"{tech}_{circuit}_{dev}_corridor.npz"
    out_path = DATA_DIR / f"{tech}_cor_{dev}.npz"
    for pth in (base_path, frag_path):
        if not pth.exists():
            raise FileNotFoundError(pth)

    base = np.load(base_path, allow_pickle=True)
    fr = np.load(frag_path, allow_pickle=True)

    base_cols = _as_str_list(base["meta_output_columns"])
    fr_cols = _as_str_list(fr["meta_output_columns"])
    if base_cols != fr_cols:
        raise ValueError(f"output_columns mismatch {tech}/{dev}")

    base_in, base_geo, base_out = base["inputs"], base["geometry"], base["outputs"]
    fr_in, fr_geo, fr_out = fr["inputs"], fr["geometry"], fr["outputs"]
    if fr_in.shape[1] != 4 or fr_geo.shape[1] != 15 or fr_out.shape[1] != 13:
        raise ValueError(f"fragment shape bad {tech}/{dev}: "
                         f"{fr_in.shape} {fr_geo.shape} {fr_out.shape}")
    n_base, n_cor = len(base_in), len(fr_in)

    exp_L = 20e-9 if dev == "pmos" else 16e-9
    g0 = fr_geo[0]
    if abs(g0[0] - 2.0) > 1e-9 or abs(g0[1] - exp_L) > 1e-12:
        raise ValueError(f"corridor geo NFIN/L wrong {tech}/{dev}: NFIN={g0[0]} L={g0[1]}")
    if not np.allclose(fr_geo, g0, rtol=0, atol=0):
        raise ValueError(f"corridor geometry not uniform {tech}/{dev}")

    inputs = np.concatenate([base_in, fr_in], axis=0)
    geometry = np.concatenate([base_geo, fr_geo], axis=0)
    outputs = np.concatenate([base_out, fr_out], axis=0)
    if "sample_class" in base.files:
        base_sc = np.asarray(base["sample_class"], dtype=np.int8)
    else:
        base_sc = np.full(n_base, 4, dtype=np.int8)  # 'grid' fallback
    cor_sc = np.full(n_cor, CORRIDOR_CODE, dtype=np.int8)
    sample_class = np.concatenate([base_sc, cor_sc], axis=0)

    meta = {k: base[k] for k in base.files if k.startswith("meta_")}

    np.savez(out_path, inputs=inputs, geometry=geometry, outputs=outputs,
             sample_class=sample_class, **meta)

    # pre-seed the label cache: base labels (via labeller) ++ corridor code
    base_labels = get_or_build_tech_variant_labels(str(base_path), dev, verbose=False)
    if len(base_labels) != n_base:
        raise ValueError(f"base label len {len(base_labels)} != {n_base}")
    cor_code = tech_variant_to_code(tech, BENCH_VARIANT[tech])
    all_labels = np.concatenate(
        [np.asarray(base_labels), np.full(n_cor, cor_code, np.asarray(base_labels).dtype)])
    cache_path = out_path.with_name(out_path.stem + "_tech_variant_labels.npy")
    np.save(cache_path, all_labels)

    # validation
    chk = np.load(out_path, allow_pickle=True)
    assert chk["inputs"].shape == inputs.shape
    assert len(chk["geometry"]) == n_base + n_cor
    assert _as_str_list(chk["meta_sample_class_names"])[CORRIDOR_CODE] == "traj_corridor"
    assert int((np.asarray(chk["sample_class"]) == CORRIDOR_CODE).sum()) == n_cor
    assert len(np.load(cache_path)) == n_base + n_cor
    frac = 100.0 * n_cor / (n_base + n_cor)
    id_idx = base_cols.index("id")
    idmag = np.abs(fr_out[:, id_idx])
    nz = idmag[idmag > 0]
    decs = np.floor(np.log10(nz)).astype(int) if len(nz) else np.array([])
    dec_hist = {int(d): int((decs == d).sum()) for d in np.unique(decs)} if len(decs) else {}
    print(f"  [{tech}/{dev}] base={n_base} + corridor={n_cor} = {n_base+n_cor} "
          f"({frac:.3f}% corridor)  cor_code={cor_code}  |id|decades={dec_hist} "
          f"-> {out_path.name}")
    return {"tech": tech, "dev": dev, "n_base": n_base, "n_cor": n_cor, "frac_pct": frac}


def main() -> int:
    ap = argparse.ArgumentParser(description="V6.5.5 append corridor to clean data")
    ap.add_argument("--tech", required=True)
    ap.add_argument("--circuit", required=True, choices=["ring", "opamp"])
    args = ap.parse_args()
    tech = args.tech.strip().lower()
    rows = []
    for dev in ("nmos", "pmos"):
        t0 = time.time()
        rows.append(append_one(tech, args.circuit, dev))
        print(f"    ({time.time()-t0:.0f}s)")
    print("\n=== SUMMARY ===")
    for r in rows:
        print(f"  {r['tech']:6s} {r['dev']:4s}  +{r['n_cor']:6d} rows ({r['frac_pct']:.3f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
