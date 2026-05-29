#!/usr/bin/env python3
"""B6 Step 2: Build augmented dataset from saved residuals + retrain.

Uses the residuals already saved by B6_harvest_retrain.py Step 2 to
avoid re-running the expensive PyCMG residual computation.

Usage:
    cd /data2/home/shenshan/NN_SPICE-trackb
    CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
        conda run -n pycircuitsim python \\
        experiments/v6_4_5_track_b/B6_augment_and_train.py [--seed 42]
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


def _get_pycmg_instance(device_type: str, L: float, NFIN: float,
                        temperature_k: float = 300.15):
    """Build a PyCMG Instance for TSMC7 ulvt at (L, NFIN, T)."""
    from pycmg.nn_config import TECH_CONFIGS
    from pycmg.nn_generate import _create_model_and_instance

    tech = TECH_CONFIGS["tsmc7"]
    result = _create_model_and_instance(
        tech, device_type, "ulvt", L, NFIN, temperature_k)
    if result is None:
        raise RuntimeError(
            f"PyCMG instance failed for {device_type} L={L} NFIN={NFIN}")
    _, inst, _ = result
    return inst


def eval_pycmg_full(device_type: str, L: float, NFIN: float,
                    vd: float, vg: float, vs: float, vb: float,
                    inst=None) -> Optional[np.ndarray]:
    """Evaluate PyCMG at one bias point. Returns (13,) row or None."""
    from pycmg.nn_generate import eval_single_point
    from pycmg.sweep import NN_OUTPUT_COLUMNS
    try:
        if inst is None:
            inst = _get_pycmg_instance(device_type, L, NFIN)
        result = eval_single_point(inst, vd=vd, vg=vg, vs=vs, vb=vb)
        if result is None:
            return None
        return np.array([result[k] for k in NN_OUTPUT_COLUMNS], dtype=np.float64)
    except Exception:
        return None


def build_augmented_dataset(
    device_type: str,
    top_k: int = 20000,
) -> Optional[Path]:
    """Build tsmc7_{device_type}_b6.npz augmented with top-K harvested points.

    Loads residuals from the saved residuals_{device_type}.npz and re-evaluates
    PyCMG for the top-K highest-residual points to get the full 13-output row.
    """
    residuals_path = HARVEST_DIR / f"residuals_{device_type}.npz"
    if not residuals_path.exists():
        print(f"  [ERROR] residuals file not found: {residuals_path}")
        return None

    cache_path = HARVEST_DIR / "harvest_cache.npz"
    if not cache_path.exists():
        print(f"  [ERROR] harvest cache not found: {cache_path}")
        return None

    # Load residuals
    rdat = np.load(str(residuals_path))
    residuals = rdat["residuals"]
    nn_inputs = rdat["nn_inputs"]  # (N, 4) [vd, vg, vs, vb]
    valid = rdat["valid"]

    # Load full harvest cache for geometry (L, NFIN, T)
    cache = np.load(str(cache_path))
    cache_pts = []
    n_cache = len(cache[f"{device_type}_vd"])
    for i in range(n_cache):
        cache_pts.append({
            "vd": float(cache[f"{device_type}_vd"][i]),
            "vg": float(cache[f"{device_type}_vg"][i]),
            "vs": float(cache[f"{device_type}_vs"][i]),
            "vb": float(cache[f"{device_type}_vb"][i]),
            "NFIN": float(cache[f"{device_type}_NFIN"][i]),
            "L": float(cache[f"{device_type}_L"][i]),
            "temperature": float(cache[f"{device_type}_T"][i]),
        })

    # Deduplicate cache_pts by (vd, vg, vs, vb, NFIN, L) -> the unique_pts
    # matching the residuals array order
    seen: Dict[tuple, int] = {}  # key -> first index in unique list
    unique_pts: List[Dict] = []
    for pt in cache_pts:
        key = (round(pt["vd"], 4), round(pt["vg"], 4), round(pt["vs"], 4),
               round(pt["vb"], 4), int(pt["NFIN"]), round(pt["L"], 14))
        if key not in seen:
            seen[key] = len(unique_pts)
            unique_pts.append(pt)

    n_unique = len(unique_pts)
    if n_unique != len(residuals):
        print(f"  [WARN] unique_pts={n_unique} != residuals={len(residuals)}; "
              f"using min")

    valid_resid = residuals[valid]
    # The valid array maps 1:1 to the unique_pts ordering (same as residuals)
    valid_idx = np.where(valid)[0]
    valid_pts = [unique_pts[i] for i in valid_idx if i < len(unique_pts)]

    n_harvest = min(top_k, len(valid_pts))
    top_indices = np.argsort(valid_resid)[-n_harvest:]
    harvest_pts = [valid_pts[i] for i in top_indices]

    print(f"\n  {device_type.upper()}: evaluating PyCMG full output for "
          f"{len(harvest_pts)} top-residual points...", flush=True)

    # Load original dataset
    ds_path = DATA_DIR / f"tsmc7_{device_type}.npz"
    orig_data = np.load(str(ds_path), allow_pickle=True)
    orig_inputs = orig_data["inputs"]
    orig_geometry = orig_data["geometry"]
    orig_outputs = orig_data["outputs"]
    orig_sc = orig_data.get("sample_class",
                             np.zeros(len(orig_inputs), dtype=np.int8))

    # Build geometry ref lookup: index rows by (NFIN, L) for fast matching
    orig_nfin = orig_geometry[:, 0]
    orig_l = orig_geometry[:, 1]

    harvest_inputs_list: List[np.ndarray] = []
    harvest_geometry_list: List[np.ndarray] = []
    harvest_outputs_list: List[np.ndarray] = []

    # Build one PyCMG instance per (L, NFIN) combo (cached)
    inst_cache: Dict[tuple, object] = {}

    failed = 0
    succeeded = 0
    for i, pt in enumerate(harvest_pts):
        if i % 2000 == 0:
            print(f"    [{i}/{len(harvest_pts)}] PyCMG eval for augmentation...",
                  flush=True)

        L = pt["L"]
        NFIN = pt["NFIN"]
        T = pt.get("temperature", 300.15)
        vd, vg, vs, vb = pt["vd"], pt["vg"], pt["vs"], pt["vb"]

        inst_key = (round(L, 14), int(NFIN))
        if inst_key not in inst_cache:
            try:
                inst_cache[inst_key] = _get_pycmg_instance(device_type, L, NFIN, T)
            except Exception:
                inst_cache[inst_key] = None

        inst = inst_cache[inst_key]
        cmg_row = eval_pycmg_full(device_type, L, NFIN, vd, vg, vs, vb, inst)
        if cmg_row is None:
            failed += 1
            continue

        harvest_inputs_list.append(np.array([vd, vg, vs, vb], dtype=np.float64))

        # Geometry: find closest row in orig dataset by NFIN and L
        dists = np.abs(orig_l - L) + np.abs(orig_nfin - NFIN) * 1e-10
        ref_idx = int(np.argmin(dists))
        geom_row = orig_geometry[ref_idx].copy()
        geom_row[0] = float(NFIN)
        geom_row[1] = float(L)
        geom_row[2] = float(T)
        harvest_geometry_list.append(geom_row)
        harvest_outputs_list.append(cmg_row)
        succeeded += 1

    if succeeded == 0:
        print(f"  [ERROR] All {failed} PyCMG augmentation evals failed.")
        return None

    print(f"  Harvested: {succeeded} OK, {failed} failed.", flush=True)

    harvest_inputs = np.array(harvest_inputs_list, dtype=np.float64)
    harvest_geometry = np.array(harvest_geometry_list, dtype=np.float64)
    harvest_outputs = np.array(harvest_outputs_list, dtype=np.float64)
    harvest_sc = np.full(len(harvest_inputs), 11, dtype=np.int8)

    # Concatenate
    aug_inputs = np.concatenate([orig_inputs, harvest_inputs], axis=0)
    aug_geometry = np.concatenate([orig_geometry, harvest_geometry], axis=0)
    aug_outputs = np.concatenate([orig_outputs, harvest_outputs], axis=0)
    aug_sc = np.concatenate([orig_sc, harvest_sc], axis=0)

    # Gather metadata from original
    metadata: Dict = {}
    for k in orig_data.files:
        if k.startswith("meta_"):
            v = orig_data[k]
            metadata[k[5:]] = v.tolist() if v.ndim > 0 else float(v)
    metadata["b6_harvest_size"] = succeeded

    out_path = DATA_DIR / f"tsmc7_{device_type}_b6.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(PYCMG_ROOT))
    from pycmg.sweep import save_npz
    save_npz(aug_inputs, aug_geometry, aug_outputs, out_path,
             metadata=metadata, sample_class=aug_sc)

    print(f"  {device_type.upper()} augmented: {len(orig_inputs)} + {succeeded} "
          f"= {len(aug_inputs)} rows → {out_path}", flush=True)
    return out_path


def retrain_one(device_type: str, data_path: Path, seed: int,
                exp_name: str, gpu_id: int) -> bool:
    """Retrain TSMC7 medium DirectNet for one (device_type, seed)."""
    import os
    stem = f"{exp_name}_{device_type}"
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
        "--exp-name", f"{exp_name}_{device_type}",
        "--seed", str(seed),
    ]
    env = os.environ.copy()
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
    else:
        lines = result.stdout.strip().split("\n")
        for line in lines[-20:]:
            print(f"    {line}", flush=True)
        ckpt = CKPT_DIR / f"{stem}_best.pt"
        print(f"  OK in {elapsed:.0f}s -> {ckpt}", flush=True)
        return ckpt.exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7,17")
    ap.add_argument("--top-k", type=int, default=20000)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--skip-augment", action="store_true")
    args = ap.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    print("=" * 70, flush=True)
    print("B6 augment + retrain", flush=True)
    print(f"  top_k={args.top_k}  seeds={seeds}  gpu={args.gpu}", flush=True)
    print("=" * 70, flush=True)

    aug_paths: Dict[str, Path] = {}
    if not args.skip_augment:
        for dev in ("nmos", "pmos"):
            p = build_augmented_dataset(dev, top_k=args.top_k)
            if p is not None:
                aug_paths[dev] = p
            else:
                # fallback to original
                aug_paths[dev] = DATA_DIR / f"tsmc7_{dev}.npz"
                print(f"  [WARN] Using original dataset for {dev} fallback")
    else:
        for dev in ("nmos", "pmos"):
            p = DATA_DIR / f"tsmc7_{dev}_b6.npz"
            if p.exists():
                aug_paths[dev] = p
                print(f"  Using existing {p}", flush=True)
            else:
                aug_paths[dev] = DATA_DIR / f"tsmc7_{dev}.npz"
                print(f"  [WARN] {p} not found, using original", flush=True)

    print(f"\n=== Retraining {len(seeds)} seeds x 2 devices ===", flush=True)
    trained_stems: List[str] = []
    for seed in seeds:
        exp_name = f"b6_tsmc7_s{seed}"
        for dev in ("nmos", "pmos"):
            data_path = aug_paths.get(dev, DATA_DIR / f"tsmc7_{dev}.npz")
            ok = retrain_one(dev, data_path, seed, exp_name, args.gpu)
            stem = f"{exp_name}_{dev}"
            if ok:
                trained_stems.append(stem)

    print(f"\n=== Training complete ===", flush=True)
    print(f"  Trained stems: {trained_stems}", flush=True)
    print(f"\nTo score:", flush=True)
    for seed in seeds:
        s_n = f"b6_tsmc7_s{seed}_nmos"
        s_p = f"b6_tsmc7_s{seed}_pmos"
        if CKPT_DIR.joinpath(f"{s_n}_best.pt").exists():
            print(f"  conda run -n pycircuitsim python scripts/eval_v6_4_5_candidate.py "
                  f"--tech TSMC7 --nmos {s_n} --pmos {s_p} --json")


if __name__ == "__main__":
    main()
