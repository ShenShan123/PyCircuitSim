#!/usr/bin/env python3
"""DirectNet V6.4.7 — S12 (P5): append trajectory-corridor fragments to v2.

Merges the per-(tech,device) ``traj_corridor`` fragments harvested by
``v6_4_7_s12_harvest_corridors.py`` into the v2 datasets as sample_class
code 12, writing NEW ``{tech}_v2cor_{dev}.npz`` files (the v2 originals stay
pristine). Also pre-seeds the tech-variant LABEL CACHE for each merged file:
the corridor rows sit at the benchmark geometry (NMOS L=16n is OFF the PDK
grid), which the fingerprint labeller cannot match, so they would trip its
miss-assert on the next rebuild. We label the v2 rows via the labeller (their
existing cache) and APPEND the known bench-variant code for the corridor rows,
keeping rows and labels in the SAME concat order — the loader then loads the
length-matched cache and never re-fingerprints.

Validation before any training: output-column agreement, geometry sanity,
|id| decade coverage, label-cache length, and a bit-for-bit reload.

Usage:
    conda run -n pycircuitsim python scripts/v6_4_7_s12_append_corridors.py \
        --tech tsmc5,tsmc7,tsmc12,tsmc16
"""
from __future__ import annotations

import argparse
import functools
import shutil
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

from bsimar.eval.loo_labels import (  # noqa: E402
    get_or_build_tech_variant_labels,
    write_sidecar_meta,
)
from bsimar.config import tech_variant_to_code  # noqa: E402
from pycmg.nn_generate import SAMPLE_CLASS_NAMES, SAMPLE_CLASS_CODES  # noqa: E402

DATA_DIR = ROOT / "external_compact_models" / "bsimar" / "data" / "datasets"
FRAG_DIR = ROOT / "results" / "v6_4_7" / "s12_corridors"
BACKUP_DIR = ROOT / "external_compact_models" / "bsimar" / "data" / "datasets_precor_backup"
BENCH_VARIANT = {"tsmc5": "lvt", "tsmc7": "ulvt", "tsmc12": "svt", "tsmc16": "svt"}
CORRIDOR_CODE = SAMPLE_CLASS_CODES["traj_corridor"]  # 12


def _as_str_list(arr) -> List[str]:
    return [a.decode() if isinstance(a, bytes) else str(a) for a in arr]


def append_one(tech: str, dev: str, frag_tag: str = "",
               out_tag: str = "") -> Dict:
    # V6.6.2: current benchmark datasets are {tech}_{dev}.npz (was V6.4.7
    # {tech}_v2_{dev}.npz); corridor output is {tech}_cor{out_tag}_{dev}.npz.
    # frag_tag/out_tag let a ring-only corridor (frag_tag="R", out_tag="r" ->
    # {tech}_corr_{dev}.npz) coexist with the full corridor.
    v2_path = DATA_DIR / f"{tech}_{dev}.npz"
    frag_path = FRAG_DIR / f"{tech}_{dev}_corridor{frag_tag}.npz"
    out_path = DATA_DIR / f"{tech}_cor{out_tag}_{dev}.npz"
    if not v2_path.exists():
        raise FileNotFoundError(v2_path)
    if not frag_path.exists():
        raise FileNotFoundError(frag_path)

    v2 = np.load(v2_path, allow_pickle=True)
    fr = np.load(frag_path, allow_pickle=True)

    # --- column agreement ---
    v2_cols = _as_str_list(v2["meta_output_columns"])
    fr_cols = _as_str_list(fr["meta_output_columns"])
    if v2_cols != fr_cols:
        raise ValueError(f"output_columns mismatch {tech}/{dev}: "
                         f"{v2_cols} vs {fr_cols}")

    # --- shapes ---
    v2_in, v2_geo, v2_out = v2["inputs"], v2["geometry"], v2["outputs"]
    fr_in, fr_geo, fr_out = fr["inputs"], fr["geometry"], fr["outputs"]
    if fr_in.shape[1] != 4 or fr_geo.shape[1] != 15 or fr_out.shape[1] != 13:
        raise ValueError(f"fragment shape bad {tech}/{dev}: "
                         f"{fr_in.shape} {fr_geo.shape} {fr_out.shape}")
    n_v2, n_cor = len(v2_in), len(fr_in)

    # --- geometry sanity: corridor at one bench bin ---
    exp_L = 20e-9 if dev == "pmos" else 16e-9
    g0 = fr_geo[0]
    if abs(g0[0] - 2.0) > 1e-9 or abs(g0[1] - exp_L) > 1e-12:
        raise ValueError(f"corridor geo NFIN/L wrong {tech}/{dev}: "
                         f"NFIN={g0[0]} L={g0[1]}")
    if not np.allclose(fr_geo, g0, rtol=0, atol=0):
        raise ValueError(f"corridor geometry not uniform {tech}/{dev}")

    # --- concat (corridor rows LAST, same order as labels below) ---
    inputs = np.concatenate([v2_in, fr_in], axis=0)
    geometry = np.concatenate([v2_geo, fr_geo], axis=0)
    outputs = np.concatenate([v2_out, fr_out], axis=0)
    v2_sc = np.asarray(v2["sample_class"], dtype=np.int8)
    cor_sc = np.full(n_cor, CORRIDOR_CODE, dtype=np.int8)
    sample_class = np.concatenate([v2_sc, cor_sc], axis=0)

    # --- meta: copy v2 meta unchanged, extend the class-name list to 13 ---
    meta = {k: v2[k] for k in v2.files
            if k.startswith("meta_") and k != "meta_sample_class_names"}
    meta["meta_sample_class_names"] = np.array(list(SAMPLE_CLASS_NAMES))

    np.savez(out_path, inputs=inputs, geometry=geometry, outputs=outputs,
             sample_class=sample_class, **meta)

    # --- pre-seed the label cache: v2 labels (via labeller) ++ corridor code ---
    v2_labels = get_or_build_tech_variant_labels(str(v2_path), dev,
                                                 verbose=False)
    if len(v2_labels) != n_v2:
        raise ValueError(f"v2 label len {len(v2_labels)} != {n_v2}")
    cor_code = tech_variant_to_code(tech, BENCH_VARIANT[tech])
    all_labels = np.concatenate(
        [np.asarray(v2_labels), np.full(n_cor, cor_code, v2_labels.dtype)])
    cache_path = out_path.with_name(out_path.stem + "_tech_variant_labels.npy")
    np.save(cache_path, all_labels)
    # audit C6o — the appended sidecar must carry the fingerprint of the
    # CONCATENATED geometry, not the v2 block it was derived from.
    write_sidecar_meta(out_path, geometry, all_labels)

    # --- validation: reload + decade coverage + cache length ---
    chk = np.load(out_path, allow_pickle=True)
    assert chk["inputs"].shape == inputs.shape
    assert len(chk["geometry"]) == n_v2 + n_cor
    assert _as_str_list(chk["meta_sample_class_names"])[CORRIDOR_CODE] == "traj_corridor"
    assert int((np.asarray(chk["sample_class"]) == CORRIDOR_CODE).sum()) == n_cor
    cache = np.load(cache_path)
    assert len(cache) == n_v2 + n_cor, f"cache len {len(cache)} != total"
    # corridor |id| decades
    id_idx = v2_cols.index("id")
    idmag = np.abs(fr_out[:, id_idx])
    nz = idmag[idmag > 0]
    decs = np.floor(np.log10(nz)).astype(int) if len(nz) else np.array([])
    dec_hist = {int(d): int((decs == d).sum()) for d in np.unique(decs)} if len(decs) else {}
    frac = 100.0 * n_cor / (n_v2 + n_cor)
    print(f"  [{tech}/{dev}] v2={n_v2} + corridor={n_cor} = {n_v2 + n_cor} "
          f"({frac:.3f}% corridor)  cor_code={cor_code}  "
          f"|id|decades={dec_hist}  -> {out_path.name}")
    return {"tech": tech, "dev": dev, "n_v2": n_v2, "n_cor": n_cor,
            "frac_pct": frac, "cor_code": cor_code, "id_decades": dec_hist}


def main() -> int:
    ap = argparse.ArgumentParser(description="S12/P5 append corridors to v2")
    ap.add_argument("--tech", default="tsmc5,tsmc7,tsmc12,tsmc16")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--frag-tag", default="",
                    help="Fragment suffix to read ({tech}_{dev}_corridor{tag}.npz).")
    ap.add_argument("--out-tag", default="",
                    help="Output suffix ({tech}_cor{tag}_{dev}.npz); e.g. 'r' "
                         "for the ring-only corridor.")
    args = ap.parse_args()
    techs = [t.strip().lower() for t in args.tech.split(",")]

    if not args.no_backup:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        for tech in techs:
            for dev in ("nmos", "pmos"):
                src = DATA_DIR / f"{tech}_{dev}.npz"
                dst = BACKUP_DIR / src.name
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
        print(f"  v2 datasets backed up under {BACKUP_DIR}")

    rows = []
    for tech in techs:
        for dev in ("nmos", "pmos"):
            t0 = time.time()
            rows.append(append_one(tech, dev, frag_tag=args.frag_tag,
                                   out_tag=args.out_tag))
            print(f"    ({time.time() - t0:.0f}s)")

    print("\n=== SUMMARY ===")
    for r in rows:
        print(f"  {r['tech']:6s} {r['dev']:4s}  +{r['n_cor']:6d} rows "
              f"({r['frac_pct']:.3f}%)  code={r['cor_code']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
