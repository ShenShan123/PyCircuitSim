#!/usr/bin/env python3
"""B6 fast augmented dataset builder and retrainer.

Directly uses the residuals numpy arrays (already computed) to select top-K
operating points, evaluates PyCMG for the full 13-output row, and writes the
augmented dataset. Uses vectorized numpy operations to avoid slow Python loops.

Usage:
    cd /data2/home/shenshan/NN_SPICE-trackb
    CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
        conda run -n pycircuitsim python \\
        experiments/v6_4_5_track_b/B6_fast_augment.py [--seeds 42,7,17]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BSIMAR_ROOT = PROJECT_ROOT / "external_compact_models" / "bsimar"
PYCMG_ROOT = PROJECT_ROOT / "external_compact_models" / "PyCMG"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))
sys.path.insert(0, str(PYCMG_ROOT))

HARVEST_DIR = PROJECT_ROOT / "results" / "v6_4_5_track_b" / "B6_harvest"
DATA_DIR = BSIMAR_ROOT / "data" / "datasets"
CKPT_DIR = BSIMAR_ROOT / "checkpoints"


def build_augmented_dataset_fast(device_type: str, top_k: int) -> Optional[Path]:
    """Build augmented dataset using vectorized ops to pick top-K residual points."""
    from pycmg.nn_config import TECH_CONFIGS
    from pycmg.nn_generate import _create_model_and_instance, eval_single_point
    from pycmg.sweep import NN_OUTPUT_COLUMNS, save_npz

    residuals_path = HARVEST_DIR / f"residuals_{device_type}.npz"
    cache_path = HARVEST_DIR / "harvest_cache.npz"

    if not residuals_path.exists() or not cache_path.exists():
        print(f"  [ERROR] Missing files. Harvest step must run first.", flush=True)
        return None

    # Load residuals (already unique points)
    rdat = np.load(str(residuals_path))
    residuals = rdat["residuals"]      # (N,) per unique operating point
    nn_inputs = rdat["nn_inputs"]      # (N, 4) [vd, vg, vs, vb]
    valid = rdat["valid"].astype(bool) # (N,) bool

    # Load cache to get L, NFIN, T per point
    cache = np.load(str(cache_path))
    prefix = device_type
    c_vd = cache[f"{prefix}_vd"]
    c_vg = cache[f"{prefix}_vg"]
    c_vs = cache[f"{prefix}_vs"]
    c_vb = cache[f"{prefix}_vb"]
    c_L = cache[f"{prefix}_L"]
    c_NFIN = cache[f"{prefix}_NFIN"]
    c_T = cache[f"{prefix}_T"]

    # Fast numpy-based deduplication using structured array
    # Round to 4 decimal places for dedup
    c_vd_r = np.round(c_vd, 4)
    c_vg_r = np.round(c_vg, 4)
    c_vs_r = np.round(c_vs, 4)
    c_vb_r = np.round(c_vb, 4)
    c_nfin_i = c_NFIN.astype(int)

    # Use structured array for fast dedup
    dt = np.dtype([("vd", np.float32), ("vg", np.float32),
                   ("vs", np.float32), ("vb", np.float32),
                   ("nfin", np.int32)])
    sa = np.empty(len(c_vd), dtype=dt)
    sa["vd"] = c_vd_r.astype(np.float32)
    sa["vg"] = c_vg_r.astype(np.float32)
    sa["vs"] = c_vs_r.astype(np.float32)
    sa["vb"] = c_vb_r.astype(np.float32)
    sa["nfin"] = c_nfin_i

    # Get unique indices preserving first-occurrence order
    _, first_idx = np.unique(sa, return_index=True)
    first_idx = np.sort(first_idx)  # restore original order

    # Reconstruct unique arrays
    u_vd = c_vd[first_idx]
    u_vg = c_vg[first_idx]
    u_vs = c_vs[first_idx]
    u_vb = c_vb[first_idx]
    u_L = c_L[first_idx]
    u_NFIN = c_NFIN[first_idx]
    u_T = c_T[first_idx]

    n_unique = len(u_vd)
    print(f"  {device_type.upper()}: {len(c_vd)} raw -> {n_unique} unique pts",
          flush=True)

    # The valid/residuals arrays index into unique_pts in the order they were
    # computed. The residuals saved by Step 2 used the same dedup order.
    # Since the residuals array has len=n_unique (after dedup), map directly.
    n_res = len(residuals)
    if n_res > n_unique:
        print(f"  [WARN] residuals({n_res}) > unique({n_unique}), truncating",
              flush=True)
        residuals = residuals[:n_unique]
        nn_inputs = nn_inputs[:n_unique]
        valid = valid[:n_unique]

    valid_mask = valid
    valid_resid = residuals[valid_mask]

    n_harvest = min(top_k, int(valid_mask.sum()))
    top_within_valid = np.argsort(valid_resid)[-n_harvest:]
    valid_idx_global = np.where(valid_mask)[0]
    harvest_global_idx = valid_idx_global[top_within_valid]

    h_vd = u_vd[harvest_global_idx]
    h_vg = u_vg[harvest_global_idx]
    h_vs = u_vs[harvest_global_idx]
    h_vb = u_vb[harvest_global_idx]
    h_L = u_L[harvest_global_idx]
    h_NFIN = u_NFIN[harvest_global_idx]
    h_T = u_T[harvest_global_idx]

    print(f"  Top-{n_harvest} residual points selected.", flush=True)
    print(f"  Unique (L, NFIN) combos: "
          f"{len(set(zip(h_L.tolist(), h_NFIN.astype(int).tolist())))}", flush=True)

    # Build PyCMG instances per (L, NFIN) combo
    tech = TECH_CONFIGS["tsmc7"]
    inst_cache: Dict[tuple, object] = {}
    ln_combos = set(zip(h_L.tolist(), h_NFIN.astype(int).tolist()))
    for L_val, NFIN_val in ln_combos:
        key = (round(L_val, 14), int(NFIN_val))
        result = _create_model_and_instance(tech, device_type, "ulvt",
                                            L_val, NFIN_val, 300.15)
        if result is not None:
            inst_cache[key] = result[1]
        else:
            print(f"  [WARN] instance failed for L={L_val*1e9:.1f}nm "
                  f"NFIN={NFIN_val}", flush=True)

    # Load original dataset for geometry reference
    ds_path = DATA_DIR / f"tsmc7_{device_type}.npz"
    orig_data = np.load(str(ds_path), allow_pickle=True)
    orig_inputs = orig_data["inputs"]
    orig_geometry = orig_data["geometry"]
    orig_outputs = orig_data["outputs"]
    orig_sc = orig_data.get("sample_class",
                             np.zeros(len(orig_inputs), dtype=np.int8))
    orig_nfin = orig_geometry[:, 0]
    orig_l = orig_geometry[:, 1]

    # Pre-compute geometry reference rows for each unique (L, NFIN) combo
    # (avoids the O(N*M) argmin loop inside the per-point loop)
    geom_ref: Dict[tuple, np.ndarray] = {}
    for L_v, NFIN_v in ln_combos:
        key = (round(L_v, 14), int(NFIN_v))
        dists = np.abs(orig_l - L_v) + np.abs(orig_nfin - NFIN_v) * 1e-10
        ref_idx = int(np.argmin(dists))
        geom_row = orig_geometry[ref_idx].copy()
        geom_row[0] = float(NFIN_v)
        geom_row[1] = float(L_v)
        geom_row[2] = 300.15
        geom_ref[key] = geom_row

    # Evaluate PyCMG
    print(f"  Evaluating PyCMG for {n_harvest} points...", flush=True)
    t0 = time.time()

    harvest_inputs_list: List[np.ndarray] = []
    harvest_geometry_list: List[np.ndarray] = []
    harvest_outputs_list: List[np.ndarray] = []

    failed = 0
    succeeded = 0
    for i in range(n_harvest):
        if i % 5000 == 0:
            print(f"    [{i}/{n_harvest}] elapsed={time.time()-t0:.1f}s",
                  flush=True)

        L_v = float(h_L[i])
        NFIN_v = float(h_NFIN[i])
        key = (round(L_v, 14), int(NFIN_v))
        vd = float(h_vd[i])
        vg = float(h_vg[i])
        vs = float(h_vs[i])
        vb = float(h_vb[i])

        inst = inst_cache.get(key)
        if inst is None:
            failed += 1
            continue

        try:
            result = eval_single_point(inst, vd=vd, vg=vg, vs=vs, vb=vb)
            if result is None:
                failed += 1
                continue
            row = np.array([result[k] for k in NN_OUTPUT_COLUMNS], dtype=np.float64)
        except Exception:
            failed += 1
            continue

        harvest_inputs_list.append(np.array([vd, vg, vs, vb], dtype=np.float64))
        harvest_geometry_list.append(geom_ref[key].copy())
        harvest_outputs_list.append(row)
        succeeded += 1

    elapsed = time.time() - t0
    print(f"  Done: {succeeded} OK, {failed} failed in {elapsed:.1f}s", flush=True)

    if succeeded == 0:
        print("  [ERROR] No successful PyCMG evals.", flush=True)
        return None

    harvest_inputs = np.array(harvest_inputs_list, dtype=np.float64)
    harvest_geometry = np.array(harvest_geometry_list, dtype=np.float64)
    harvest_outputs = np.array(harvest_outputs_list, dtype=np.float64)
    harvest_sc = np.full(len(harvest_inputs), 11, dtype=np.int8)

    # Concatenate
    aug_inputs = np.concatenate([orig_inputs, harvest_inputs], axis=0)
    aug_geometry = np.concatenate([orig_geometry, harvest_geometry], axis=0)
    aug_outputs = np.concatenate([orig_outputs, harvest_outputs], axis=0)
    aug_sc = np.concatenate([orig_sc, harvest_sc], axis=0)

    # Metadata — preserve all meta_ keys from original dataset
    meta: Dict = {}
    for k in orig_data.files:
        if k.startswith("meta_"):
            v = orig_data[k]
            # Keep as-is (could be string, list, scalar, or array)
            try:
                if v.ndim == 0:
                    # scalar: convert to python native
                    raw = v.item()
                    meta[k[5:]] = raw
                else:
                    meta[k[5:]] = v.tolist()
            except Exception:
                meta[k[5:]] = str(v)
    meta["b6_harvest_size"] = succeeded

    out_path = DATA_DIR / f"tsmc7_{device_type}_b6.npz"
    save_npz(aug_inputs, aug_geometry, aug_outputs, out_path,
             metadata=meta, sample_class=aug_sc)
    print(f"  {device_type.upper()} augmented: {len(orig_inputs)} + {succeeded} "
          f"= {len(aug_inputs)} rows -> {out_path}", flush=True)
    return out_path


def retrain_one(device_type: str, data_path: Path, seed: int,
                exp_stem: str, gpu_id: int) -> bool:
    """Retrain one (device_type, seed) pair with the augmented dataset."""
    import os
    stem = f"{exp_stem}_{device_type}"
    cmd = [
        "conda", "run", "-n", "pycircuitsim",
        "python", "-u", "-m", "bsimar.cli.train",
        "--model", "direct",
        "--size", "medium",
        "--device-type", device_type,
        "--tech-scope", "tsmc7",
        "--cuda",
        "--overwrite",
        "--data", str(data_path),
        "--exp-name", stem,
        "--seed", str(seed),
    ]
    env = os.environ.copy()
    # PYTHONPATH must include the external_compact_models dir so `bsimar` is
    # importable when running as `python -m bsimar.cli.train` in a subprocess.
    ext_path = str(PROJECT_ROOT / "external_compact_models")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{ext_path}:{existing_pp}" if existing_pp else ext_path
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    })
    print(f"\n  Training {stem} seed={seed} GPU={gpu_id} ...", flush=True)
    t0 = time.time()
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True,
        cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [ERROR] {stem} rc={result.returncode}:", flush=True)
        print(result.stderr[-3000:], flush=True)
        return False

    lines = result.stdout.strip().split("\n")
    for line in lines[-20:]:
        print(f"    {line}", flush=True)
    ckpt = CKPT_DIR / f"{stem}_best.pt"
    ok = ckpt.exists()
    print(f"  {'OK' if ok else 'MISSING CKPT'} in {elapsed:.0f}s -> {ckpt}", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7,17")
    ap.add_argument("--top-k", type=int, default=20000)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--skip-augment", action="store_true",
                    help="Use existing b6.npz datasets")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    print("=" * 70, flush=True)
    print("B6 fast augment + retrain (TSMC7)", flush=True)
    print(f"  top_k={args.top_k}  seeds={seeds}  gpu={args.gpu}", flush=True)
    print("=" * 70, flush=True)

    # Build augmented datasets
    aug_paths: Dict[str, Path] = {}
    if not args.skip_augment:
        for dev in ("nmos", "pmos"):
            print(f"\n=== Building augmented {dev.upper()} dataset ===", flush=True)
            p = build_augmented_dataset_fast(dev, top_k=args.top_k)
            aug_paths[dev] = p if p is not None else DATA_DIR / f"tsmc7_{dev}.npz"
    else:
        for dev in ("nmos", "pmos"):
            p = DATA_DIR / f"tsmc7_{dev}_b6.npz"
            aug_paths[dev] = p if p.exists() else DATA_DIR / f"tsmc7_{dev}.npz"
            print(f"  {dev}: {aug_paths[dev]}", flush=True)

    # Retrain
    print(f"\n=== Retraining {len(seeds)} seeds x 2 devices ===", flush=True)
    trained_ok: List[str] = []
    for seed in seeds:
        exp_stem = f"b6_tsmc7_s{seed}"
        for dev in ("nmos", "pmos"):
            data_path = aug_paths[dev]
            ok = retrain_one(dev, data_path, seed, exp_stem, args.gpu)
            if ok:
                trained_ok.append(f"{exp_stem}_{dev}")

    print(f"\n=== Done. Trained: {trained_ok} ===", flush=True)
    print("\nScore commands:", flush=True)
    for seed in seeds:
        n = f"b6_tsmc7_s{seed}_nmos"
        p = f"b6_tsmc7_s{seed}_pmos"
        if CKPT_DIR.joinpath(f"{n}_best.pt").exists():
            print(f"  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' "
                  f"conda run -n pycircuitsim python scripts/eval_v6_4_5_candidate.py "
                  f"--tech TSMC7 --nmos {n} --pmos {p} --json", flush=True)


if __name__ == "__main__":
    main()
