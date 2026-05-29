#!/usr/bin/env python3
"""B6 — Adversarial harvest-then-retrain for TSMC7 DirectNet.

Step 1: Run TSMC7 5-stage ring oscillator (transient) + TSMC7 SRAM force_ic
        (.op, both states) with the canonical V6.4.4 NN, capturing every
        DirectNet operating point (vd,vg,vs,vb in NN frame, NFIN, L,
        device_type) and the NN's predicted `id`. Evaluate PyCMG (BSIM-CMG)
        at the same points to compute the residual |id_NN − id_CMG|.

Step 2: Histogram analysis — are worst residuals inside or outside the
        training box? This is the B6 falsifier.

Step 3 (if in-box): Augment the TSMC7 dataset with top-K harvested points
        evaluated by PyCMG (full 13-output row). Write tsmc7_{nmos,pmos}_b6.npz.

Step 4: Retrain TSMC7 NMOS+PMOS at seeds {42, 7, 17} with augmented dataset.
        Checkpoints saved as b6_tsmc7_s{seed}_{nmos,pmos} (non-canonical).

Usage:
    cd /data2/home/shenshan/NN_SPICE-trackb
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \\
        conda run -n pycircuitsim python experiments/v6_4_5_track_b/B6_harvest_retrain.py \\
        [--skip-harvest] [--skip-retrain] [--top-k 20000]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYCMG_ROOT = PROJECT_ROOT / "external_compact_models" / "PyCMG"
BSIMAR_ROOT = PROJECT_ROOT / "external_compact_models" / "bsimar"

# Path order matters: PROJECT_ROOT must be FIRST so 'tests' resolves to the
# project's tests/ package, not PyCMG's tests/ directory.
sys.path.insert(0, str(PYCMG_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

print = __builtins__.__dict__["print"]  # type: ignore[attr-defined]


# ── Harvest storage ───────────────────────────────────────────────────────────
# Collected per device_type ("nmos" | "pmos")
_COLLECTED: Dict[str, List[Dict]] = defaultdict(list)
_PYCMG_INSTANCE_CACHE: Dict[str, object] = {}  # key = "nmos|L|NFIN|T"


def _flush(msg: str = "") -> None:
    import sys
    sys.stdout.flush()
    if msg:
        print(msg, flush=True)


# ── PyCMG model factory for TSMC7 ────────────────────────────────────────────
def _get_pycmg_instance(device_type: str, L: float, NFIN: float,
                        temperature_k: float = 300.15):
    """Return (or build+cache) a PyCMG Instance for TSMC7 ulvt at (L, NFIN, T)."""
    cache_key = f"{device_type}|{L:.6e}|{int(NFIN)}|{temperature_k:.2f}"
    if cache_key in _PYCMG_INSTANCE_CACHE:
        return _PYCMG_INSTANCE_CACHE[cache_key]

    from pycmg.nn_config import TECH_CONFIGS
    from pycmg.nn_generate import _create_model_and_instance

    tech = TECH_CONFIGS["tsmc7"]
    variant = "ulvt"  # TSMC7 benchmark variant per CLAUDE.md
    result = _create_model_and_instance(
        tech, device_type, variant, L, NFIN, temperature_k)
    if result is None:
        raise RuntimeError(
            f"PyCMG instance creation failed for {device_type} L={L} NFIN={NFIN}")
    _, inst, _ = result
    _PYCMG_INSTANCE_CACHE[cache_key] = inst
    return inst


def eval_pycmg_full(device_type: str, L: float, NFIN: float,
                    vd: float, vg: float, vs: float, vb: float,
                    temperature_k: float = 300.15) -> Optional[np.ndarray]:
    """Evaluate PyCMG at one bias point. Returns (13,) output row or None."""
    from pycmg.nn_generate import eval_single_point
    from pycmg.sweep import NN_OUTPUT_COLUMNS

    try:
        inst = _get_pycmg_instance(device_type, L, NFIN, temperature_k)
        result = eval_single_point(inst, vd=vd, vg=vg, vs=vs, vb=vb)
        if result is None:
            return None
        return np.array([result[k] for k in NN_OUTPUT_COLUMNS], dtype=np.float64)
    except Exception as exc:
        return None


# ── Monkeypatch to instrument _MOSFETNNBase._eval ─────────────────────────────
def _install_harvest_hook(device_type: str) -> None:
    """Patch mosfet_nn._MOSFETNNBase._eval to capture NN evals for device_type."""
    from pycircuitsim.models import mosfet_nn as _mnn

    original_eval = _mnn._MOSFETNNBase._eval

    def _patched_eval(self, voltages: Dict[str, float]) -> Dict:  # type: ignore
        # Call the original to get the NN result.
        result = original_eval(self, voltages)

        # Infer device type from the subclass flag.
        dev_t = "pmos" if self._is_pmos else "nmos"
        if dev_t != device_type:
            return result

        # Collect the operating point in the NN frame (source-relative).
        vd_nn, vg_nn, vs_nn, vb_nn = self._raw_voltages(voltages)
        # Absolute frame for PyCMG eval.
        _COLLECTED[dev_t].append({
            "vd": float(vd_nn),
            "vg": float(vg_nn),
            "vs": float(vs_nn),
            "vb": float(vb_nn),
            "NFIN": float(self.NFIN),
            "L": float(self.L),
            "id_nn": float(result["id"]),
            "temperature": float(self.temperature),
        })
        return result

    _mnn._MOSFETNNBase._eval = _patched_eval  # type: ignore[method-assign]


def _install_all_harvest_hooks() -> None:
    """Install hooks for both nmos and pmos."""
    from pycircuitsim.models import mosfet_nn as _mnn

    original_eval = _mnn._MOSFETNNBase._eval

    def _patched_eval_both(self, voltages: Dict[str, float]) -> Dict:  # type: ignore
        result = original_eval(self, voltages)
        dev_t = "pmos" if self._is_pmos else "nmos"
        vd_nn, vg_nn, vs_nn, vb_nn = self._raw_voltages(voltages)
        _COLLECTED[dev_t].append({
            "vd": float(vd_nn),
            "vg": float(vg_nn),
            "vs": float(vs_nn),
            "vb": float(vb_nn),
            "NFIN": float(self.NFIN),
            "L": float(self.L),
            "id_nn": float(result["id"]),
            "temperature": float(self.temperature),
        })
        return result

    _mnn._MOSFETNNBase._eval = _patched_eval_both  # type: ignore[method-assign]


# ── Circuit runners ───────────────────────────────────────────────────────────
def run_ring_osc_tsmc7(work_dir: Path) -> None:
    """Run TSMC7 5-stage ring oscillator transient to collect NN eval points."""
    import tempfile
    from tests.common.complex import BENCH, RESULTS_BASE
    from tests.verify_complex_ring_osc import (
        run_directnet_ro, SETTLE, TRAN_TSTEP,
    )

    bt = BENCH["TSMC7"]
    ro_work = work_dir / "ring_osc_harvest"
    ro_work.mkdir(parents=True, exist_ok=True)

    print(f"  Running TSMC7 ring oscillator transient (harvest)...", flush=True)
    n_before = sum(len(v) for v in _COLLECTED.values())
    try:
        _dn, _partial, _warn = run_directnet_ro(bt, ro_work)
    except Exception as exc:
        print(f"  [WARN] RO transient raised: {exc}", flush=True)
    n_after = sum(len(v) for v in _COLLECTED.values())
    print(f"  Collected {n_after - n_before} new NN eval points from RO.",
          flush=True)


def run_sram_force_ic_tsmc7(work_dir: Path) -> None:
    """Run TSMC7 SRAM 6T force_ic .op for both states to collect NN eval points."""
    from tests.common.complex import BENCH

    bt = BENCH["TSMC7"]
    ic_work = work_dir / "sram_harvest"
    ic_work.mkdir(parents=True, exist_ok=True)

    print(f"  Running TSMC7 SRAM force_ic (both states, harvest)...", flush=True)

    for tag, (q0, qb0) in (("state1", (bt.vdd, 0.0)), ("state0", (0.0, bt.vdd))):
        # Build the 6T netlist
        netlist_path = ic_work / f"sram6t_tsmc7_{tag}.sp"
        n_l = bt.l_nmos * 1e9
        p_l = bt.l_pmos * 1e9
        netlist_path.write_text(
            f"* 6T SRAM cell harvest — TSMC7 {tag}\n"
            f"Vdd vdd 0 {bt.vdd}\n"
            f"Vwl wl 0 {bt.vdd}\n"
            f"Vbl bl 0 {bt.vdd}\n"
            f"Vblb blb 0 {bt.vdd}\n"
            f".ic V(q)={q0} V(qb)={qb0}\n"
            f"Mpl qb q vdd vdd pmos_nn L={p_l:.0f}n NFIN={bt.nfin}\n"
            f"Mnl qb q 0   0   nmos_nn L={n_l:.0f}n NFIN={bt.nfin}\n"
            f"Mpr q qb vdd vdd pmos_nn L={p_l:.0f}n NFIN={bt.nfin}\n"
            f"Mnr q qb 0   0   nmos_nn L={n_l:.0f}n NFIN={bt.nfin}\n"
            f"Mal bl  wl q  0 nmos_nn L={n_l:.0f}n NFIN={bt.nfin}\n"
            f"Mar blb wl qb 0 nmos_nn L={n_l:.0f}n NFIN={bt.nfin}\n"
            f".model nmos_nn NMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.vt})\n"
            f".model pmos_nn PMOS (LEVEL=73 TECH={bt.nn_tech} VT={bt.vt})\n"
            f".op\n.end\n"
        )

        n_before = sum(len(v) for v in _COLLECTED.values())
        try:
            from pycircuitsim.parser import Parser
            from pycircuitsim.solver import DCSolver

            p = Parser()
            p.parse_file(str(netlist_path))
            circuit = p.circuit
            guess = circuit.initial_conditions or None
            solver = DCSolver(circuit, initial_guess=guess,
                              use_source_stepping=True, force_ic=True)
            sol = solver.solve()
            q_v = sol.get("q", float("nan"))
            qb_v = sol.get("qb", float("nan"))
            print(f"    {tag}: q={q_v:.4f} qb={qb_v:.4f}", flush=True)
        except Exception as exc:
            print(f"    [WARN] SRAM {tag} raised: {exc}", flush=True)

        n_after = sum(len(v) for v in _COLLECTED.values())
        print(f"    Collected {n_after - n_before} new NN eval points from {tag}.",
              flush=True)


# ── Residual computation ──────────────────────────────────────────────────────
def compute_residuals(
    collected_pts: List[Dict],
    device_type: str,
    max_pts: int = 50000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute |id_NN - id_CMG| for collected operating points.

    Returns:
        residuals: (N,) absolute residual |id_NN - id_CMG|
        nn_inputs: (N, 4) [vd, vg, vs, vb] in NN frame (source-relative)
        valid_mask: (N,) bool — points where PyCMG converged
    """
    # Deduplicate (round to 4 sig figs to merge nearly-identical NR iters)
    seen = set()
    unique_pts: List[Dict] = []
    for pt in collected_pts:
        key = (round(pt["vd"], 4), round(pt["vg"], 4), round(pt["vs"], 4),
               round(pt["vb"], 4), int(pt["NFIN"]), round(pt["L"], 14))
        if key not in seen:
            seen.add(key)
            unique_pts.append(pt)

    if len(unique_pts) > max_pts:
        # Randomly subsample to keep PyCMG eval tractable
        rng = np.random.default_rng(42)
        idx = rng.choice(len(unique_pts), size=max_pts, replace=False)
        unique_pts = [unique_pts[i] for i in idx]

    print(f"  Computing residuals for {len(unique_pts)} unique {device_type} "
          f"operating points...", flush=True)

    residuals = []
    nn_inputs = []
    id_nn_list = []
    id_cmg_list = []
    valid_flags = []

    for i, pt in enumerate(unique_pts):
        if i % 2000 == 0:
            print(f"    [{i}/{len(unique_pts)}] ...", flush=True)

        vd, vg, vs, vb = pt["vd"], pt["vg"], pt["vs"], pt["vb"]
        L = pt["L"]
        NFIN = pt["NFIN"]
        T = pt.get("temperature", 300.15)
        id_nn = pt["id_nn"]

        # PyCMG eval — in source-relative frame (vs=0 by definition in NN frame)
        cmg_row = eval_pycmg_full(device_type, L, NFIN, vd, vg, vs, vb, T)

        if cmg_row is not None:
            id_cmg = float(cmg_row[0])  # index 0 = "id"
            resid = abs(id_nn - id_cmg)
        else:
            resid = float("nan")

        residuals.append(resid)
        nn_inputs.append([vd, vg, vs, vb])
        id_nn_list.append(id_nn)
        id_cmg_list.append(id_cmg if cmg_row is not None else float("nan"))
        valid_flags.append(cmg_row is not None)

    residuals_arr = np.array(residuals, dtype=np.float64)
    nn_inputs_arr = np.array(nn_inputs, dtype=np.float64)
    valid_arr = np.array(valid_flags, dtype=bool)

    return residuals_arr, nn_inputs_arr, valid_arr, unique_pts


# ── Training box check ────────────────────────────────────────────────────────
def check_training_box(
    nn_inputs: np.ndarray,
    dataset_path: Path,
    device_type: str,
    residuals: np.ndarray,
    valid_mask: np.ndarray,
    top_k_frac: float = 0.05,
) -> Dict:
    """Check whether top-residual points are inside/outside the training box.

    The training box is defined by the per-dim min/max of the canonical dataset.
    """
    data = np.load(str(dataset_path), allow_pickle=True)
    box_min = data["inputs"].min(axis=0)  # (4,) [vd, vg, vs, vb]
    box_max = data["inputs"].max(axis=0)

    valid_resid = residuals[valid_mask]
    valid_inputs = nn_inputs[valid_mask]

    if len(valid_resid) == 0:
        return {"error": "no valid PyCMG evals"}

    n_top = max(1, int(len(valid_resid) * top_k_frac))
    top_idx = np.argsort(valid_resid)[-n_top:]
    top_inputs = valid_inputs[top_idx]
    top_resids = valid_resid[top_idx]

    # Check each point: is it inside [box_min, box_max] on all 4 voltage dims?
    in_box = np.all(
        (top_inputs >= box_min[None, :]) & (top_inputs <= box_max[None, :]),
        axis=1)

    result = {
        "device_type": device_type,
        "n_total_valid": int(valid_mask.sum()),
        "n_top": int(n_top),
        "top_resid_mean_uA": float(top_resids.mean() * 1e6),
        "top_resid_max_uA": float(top_resids.max() * 1e6),
        "top_resid_p50_uA": float(np.median(top_resids) * 1e6),
        "frac_in_box": float(in_box.mean()),
        "n_in_box": int(in_box.sum()),
        "n_out_of_box": int((~in_box).sum()),
        "box_min": box_min.tolist(),
        "box_max": box_max.tolist(),
        "verdict": "in-box" if in_box.mean() > 0.5 else "out-of-box",
    }

    # Bin the worst points by voltage dimension to find the key axis.
    dim_names = ["Vd", "Vg", "Vs", "Vb"]
    out_dims: Dict[str, int] = {}
    for d, name in enumerate(dim_names):
        out_count = int(((top_inputs[:, d] < box_min[d]) |
                         (top_inputs[:, d] > box_max[d])).sum())
        out_dims[name] = out_count
    result["out_of_box_dims"] = out_dims

    return result


# ── Augmented dataset creation ────────────────────────────────────────────────
def build_augmented_dataset(
    device_type: str,
    unique_pts: List[Dict],
    residuals: np.ndarray,
    valid_mask: np.ndarray,
    orig_dataset_path: Path,
    out_path: Path,
    top_k: int = 20000,
) -> int:
    """Build tsmc7_{device_type}_b6.npz augmented with top-K harvested points."""
    from pycmg.nn_config import TECH_CONFIGS
    from pycmg.sweep import NN_OUTPUT_COLUMNS

    print(f"\n  Building augmented {device_type} dataset...", flush=True)

    valid_pts = [pt for pt, v in zip(unique_pts, valid_mask) if v]
    valid_resid = residuals[valid_mask]

    if len(valid_pts) == 0:
        print("  ERROR: No valid PyCMG-evaluated points — cannot augment.",
              flush=True)
        return 0

    n_harvest = min(top_k, len(valid_pts))
    top_idx = np.argsort(valid_resid)[-n_harvest:]
    harvest_pts = [valid_pts[i] for i in top_idx]

    print(f"  Evaluating PyCMG full 13-output for {len(harvest_pts)} top-K "
          f"points...", flush=True)

    # Load original dataset to get geometry template and metadata
    orig_data = np.load(str(orig_dataset_path), allow_pickle=True)
    orig_inputs = orig_data["inputs"]
    orig_geometry = orig_data["geometry"]
    orig_outputs = orig_data["outputs"]
    orig_sc = orig_data.get("sample_class",
                             np.zeros(len(orig_inputs), dtype=np.int8))

    # Build geometry row for TSMC7 ulvt (extract from existing rows closest to L)
    from pycmg.nn_config import TECH_CONFIGS, extract_process_params
    from pycmg.model import Model, Instance

    # Build a reference instance to get process params for TSMC7 ulvt
    tech_cfg = TECH_CONFIGS["tsmc7"]

    # Get a representative geometry row from orig dataset for the L=16nm/20nm bins
    harvest_inputs_list: List[np.ndarray] = []
    harvest_geometry_list: List[np.ndarray] = []
    harvest_outputs_list: List[np.ndarray] = []

    failed = 0
    succeeded = 0
    for i, pt in enumerate(harvest_pts):
        if i % 2000 == 0:
            print(f"    [{i}/{len(harvest_pts)}] PyCMG eval...", flush=True)

        L = pt["L"]
        NFIN = pt["NFIN"]
        T = pt.get("temperature", 300.15)
        vd, vg, vs, vb = pt["vd"], pt["vg"], pt["vs"], pt["vb"]

        cmg_row = eval_pycmg_full(device_type, L, NFIN, vd, vg, vs, vb, T)
        if cmg_row is None:
            failed += 1
            continue

        harvest_inputs_list.append(np.array([vd, vg, vs, vb], dtype=np.float64))

        # Build geometry row: find closest row in orig by matching L and NFIN
        # (first 3 cols: NFIN, L, T — col 0 is raw NFIN, col 1 is L)
        l_col = orig_geometry[:, 1]
        nfin_col = orig_geometry[:, 0]
        dists = np.abs(l_col - L) + np.abs(nfin_col - NFIN) * 1e-10
        ref_row_idx = int(np.argmin(dists))
        geom_row = orig_geometry[ref_row_idx].copy()
        geom_row[0] = float(NFIN)
        geom_row[1] = float(L)
        geom_row[2] = float(T)
        harvest_geometry_list.append(geom_row)

        harvest_outputs_list.append(cmg_row)
        succeeded += 1

    if succeeded == 0:
        print(f"  ERROR: All {failed} PyCMG augmentation evals failed.",
              flush=True)
        return 0

    print(f"  Harvested: {succeeded} OK, {failed} failed.", flush=True)

    harvest_inputs = np.array(harvest_inputs_list, dtype=np.float64)
    harvest_geometry = np.array(harvest_geometry_list, dtype=np.float64)
    harvest_outputs = np.array(harvest_outputs_list, dtype=np.float64)
    # Sample class 11 = "harvested" (beyond the 10 existing classes)
    harvest_sc = np.full(len(harvest_inputs), 11, dtype=np.int8)

    # Concatenate
    aug_inputs = np.concatenate([orig_inputs, harvest_inputs], axis=0)
    aug_geometry = np.concatenate([orig_geometry, harvest_geometry], axis=0)
    aug_outputs = np.concatenate([orig_outputs, harvest_outputs], axis=0)
    aug_sc = np.concatenate([orig_sc, harvest_sc], axis=0)

    # Save with 2x weight tag in the metadata (the trainer uses LDS anyway;
    # we mark via a new sample_class and set the weight in the trainer via env).
    metadata: Dict = {}
    for k in orig_data.files:
        if k.startswith("meta_"):
            metadata[k[5:]] = orig_data[k].tolist() if orig_data[k].ndim > 0 else float(orig_data[k])
    metadata["b6_harvest_size"] = succeeded

    out_path.parent.mkdir(parents=True, exist_ok=True)
    from pycmg.sweep import save_npz
    save_npz(aug_inputs, aug_geometry, aug_outputs, out_path,
             metadata=metadata, sample_class=aug_sc)

    print(f"  Augmented dataset: {len(orig_inputs)} orig + {succeeded} harvested "
          f"= {len(aug_inputs)} total rows → {out_path}", flush=True)
    return succeeded


# ── Training ──────────────────────────────────────────────────────────────────
def retrain_tsmc7(
    device_type: str,
    data_path: Path,
    seed: int,
    exp_name: str,
    gpu_id: int = 3,
) -> None:
    """Retrain TSMC7 DirectNet medium on the augmented dataset at one seed."""
    import subprocess

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
    env_overrides = {
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }
    import os
    env = os.environ.copy()
    env.update(env_overrides)

    stem = f"{exp_name}_{device_type}"
    print(f"\n  Launching retrain: {stem} seed={seed} ...", flush=True)
    t0 = time.time()
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True,
        cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [ERROR] retrain {stem} failed (rc={result.returncode}):",
              flush=True)
        print(result.stderr[-2000:], flush=True)
    else:
        print(f"  retrain {stem} OK in {elapsed:.0f}s", flush=True)
        # Show last lines of output for LDS / epoch info
        lines = result.stdout.strip().split("\n")
        for line in lines[-15:]:
            print(f"    {line}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="B6 harvest-then-retrain")
    parser.add_argument("--skip-harvest", action="store_true",
                        help="Load previously saved harvest (skip circuit runs)")
    parser.add_argument("--skip-retrain", action="store_true",
                        help="Skip the retrain step (analysis only)")
    parser.add_argument("--top-k", type=int, default=20000,
                        help="Number of top-residual points for augmentation")
    parser.add_argument("--seeds", type=str, default="42,7,17",
                        help="Comma-separated seeds for retraining")
    parser.add_argument("--gpu", type=int, default=3,
                        help="CUDA device ID for training (default: 3)")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    WORK_DIR = PROJECT_ROOT / "results" / "v6_4_5_track_b" / "B6_harvest"
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    HARVEST_CACHE = WORK_DIR / "harvest_cache.npz"
    DATA_DIR = BSIMAR_ROOT / "data" / "datasets"
    CKPT_DIR = BSIMAR_ROOT / "checkpoints"

    print("=" * 70, flush=True)
    print("B6 — Adversarial harvest-then-retrain (TSMC7)", flush=True)
    print(f"  top_k={args.top_k}  seeds={seeds}  gpu={args.gpu}", flush=True)
    print("=" * 70, flush=True)

    # ── Step 1: Harvest ────────────────────────────────────────────────────
    if not args.skip_harvest or not HARVEST_CACHE.exists():
        print("\n=== Step 1: Harvest operating points ===", flush=True)
        _install_all_harvest_hooks()

        run_ring_osc_tsmc7(WORK_DIR)
        run_sram_force_ic_tsmc7(WORK_DIR)

        # Persist collected points
        all_nmos = _COLLECTED["nmos"]
        all_pmos = _COLLECTED["pmos"]
        np.savez(
            str(HARVEST_CACHE),
            nmos_vd=np.array([p["vd"] for p in all_nmos]),
            nmos_vg=np.array([p["vg"] for p in all_nmos]),
            nmos_vs=np.array([p["vs"] for p in all_nmos]),
            nmos_vb=np.array([p["vb"] for p in all_nmos]),
            nmos_NFIN=np.array([p["NFIN"] for p in all_nmos]),
            nmos_L=np.array([p["L"] for p in all_nmos]),
            nmos_id_nn=np.array([p["id_nn"] for p in all_nmos]),
            nmos_T=np.array([p.get("temperature", 300.15) for p in all_nmos]),
            pmos_vd=np.array([p["vd"] for p in all_pmos]),
            pmos_vg=np.array([p["vg"] for p in all_pmos]),
            pmos_vs=np.array([p["vs"] for p in all_pmos]),
            pmos_vb=np.array([p["vb"] for p in all_pmos]),
            pmos_NFIN=np.array([p["NFIN"] for p in all_pmos]),
            pmos_L=np.array([p["L"] for p in all_pmos]),
            pmos_id_nn=np.array([p["id_nn"] for p in all_pmos]),
            pmos_T=np.array([p.get("temperature", 300.15) for p in all_pmos]),
        )
        print(f"\n  Saved harvest cache: NMOS={len(all_nmos)} PMOS={len(all_pmos)}",
              flush=True)
    else:
        print("\n=== Step 1: Loading saved harvest ===", flush=True)
        cache = np.load(str(HARVEST_CACHE))
        for dev in ("nmos", "pmos"):
            n = len(cache[f"{dev}_vd"])
            for i in range(n):
                _COLLECTED[dev].append({
                    "vd": float(cache[f"{dev}_vd"][i]),
                    "vg": float(cache[f"{dev}_vg"][i]),
                    "vs": float(cache[f"{dev}_vs"][i]),
                    "vb": float(cache[f"{dev}_vb"][i]),
                    "NFIN": float(cache[f"{dev}_NFIN"][i]),
                    "L": float(cache[f"{dev}_L"][i]),
                    "id_nn": float(cache[f"{dev}_id_nn"][i]),
                    "temperature": float(cache[f"{dev}_T"][i]),
                })
        print(f"  Loaded: NMOS={len(_COLLECTED['nmos'])} "
              f"PMOS={len(_COLLECTED['pmos'])}", flush=True)

    # ── Step 2: Residuals + in-box analysis ───────────────────────────────
    print("\n=== Step 2: Residual computation + training-box analysis ===",
          flush=True)

    all_results: Dict[str, Dict] = {}
    all_residuals: Dict[str, np.ndarray] = {}
    all_nn_inputs: Dict[str, np.ndarray] = {}
    all_valid: Dict[str, np.ndarray] = {}
    all_unique_pts: Dict[str, List[Dict]] = {}

    for dev in ("nmos", "pmos"):
        pts = _COLLECTED[dev]
        if not pts:
            print(f"  [WARN] No {dev} points collected.", flush=True)
            continue

        print(f"\n--- {dev.upper()} ({len(pts)} raw pts) ---", flush=True)
        residuals, nn_inputs, valid_mask, unique_pts = compute_residuals(
            pts, dev, max_pts=100000)
        all_residuals[dev] = residuals
        all_nn_inputs[dev] = nn_inputs
        all_valid[dev] = valid_mask
        all_unique_pts[dev] = unique_pts

        ds_path = DATA_DIR / f"tsmc7_{dev}.npz"
        box_result = check_training_box(
            nn_inputs, ds_path, dev, residuals, valid_mask)
        all_results[dev] = box_result

        valid_resid = residuals[valid_mask]
        if len(valid_resid) > 0:
            print(f"\n  {dev.upper()} residual stats (n={valid_mask.sum()}):",
                  flush=True)
            print(f"    mean |Δid| = {valid_resid.mean()*1e6:.3f} µA", flush=True)
            print(f"    p50        = {np.percentile(valid_resid,50)*1e6:.3f} µA",
                  flush=True)
            print(f"    p90        = {np.percentile(valid_resid,90)*1e6:.3f} µA",
                  flush=True)
            print(f"    max |Δid|  = {valid_resid.max()*1e6:.3f} µA", flush=True)
            print(f"\n  Top-{box_result.get('n_top','?')} worst points:", flush=True)
            print(f"    in-box     = {box_result.get('n_in_box','?')} "
                  f"({box_result.get('frac_in_box',float('nan')):.1%})", flush=True)
            print(f"    out-of-box = {box_result.get('n_out_of_box','?')}",
                  flush=True)
            print(f"  >> VERDICT (top-5%): {box_result.get('verdict','?').upper()}",
                  flush=True)
            if "out_of_box_dims" in box_result:
                print(f"  Out-of-box by dim: {box_result['out_of_box_dims']}",
                      flush=True)

    # Save residual arrays for offline analysis
    for dev in all_residuals:
        np.savez(
            str(WORK_DIR / f"residuals_{dev}.npz"),
            residuals=all_residuals[dev],
            nn_inputs=all_nn_inputs[dev],
            valid=all_valid[dev],
        )
        print(f"  Saved residuals_{dev}.npz", flush=True)

    # ── Step 3: In-box vs out-of-box falsifier decision ───────────────────
    print("\n=== Step 3: Falsifier decision ===", flush=True)

    in_box_verdict = {}
    for dev in ("nmos", "pmos"):
        v = all_results.get(dev, {}).get("verdict", "unknown")
        in_box_verdict[dev] = v
        print(f"  {dev.upper()}: {v}", flush=True)

    # If BOTH are fully in-box (interpolation failure), retrain alone cannot help.
    all_in_box = all(v == "in-box" for v in in_box_verdict.values()
                     if v != "unknown")
    some_out_of_box = any(v == "out-of-box" for v in in_box_verdict.values())

    if all_in_box and not some_out_of_box:
        print("\n  [FALSIFIER] ALL worst-residual points are INSIDE the training box.",
              flush=True)
        print("  Interpolation failure — adding more in-distribution data cannot",
              flush=True)
        print("  fix the model's attractor. Recommend Tier-3 (B9 monotone lattice",
              flush=True)
        print("  / B7 physics skeleton). Proceeding with retrain attempt anyway",
              flush=True)
        print("  (to confirm the falsifier empirically).", flush=True)

    if args.skip_retrain:
        print("\n  [skip-retrain flag] Stopping after analysis.", flush=True)
        _write_report(WORK_DIR, all_results, all_residuals, all_valid,
                      in_box_verdict, {}, seeds, args.top_k)
        return

    # ── Step 4a: Build augmented datasets ─────────────────────────────────
    print("\n=== Step 4a: Build augmented datasets ===", flush=True)
    aug_paths: Dict[str, Path] = {}
    for dev in ("nmos", "pmos"):
        if dev not in all_unique_pts:
            continue
        ds_path = DATA_DIR / f"tsmc7_{dev}.npz"
        aug_path = DATA_DIR / f"tsmc7_{dev}_b6.npz"
        n_ok = build_augmented_dataset(
            dev,
            all_unique_pts[dev],
            all_residuals[dev],
            all_valid[dev],
            ds_path,
            aug_path,
            top_k=args.top_k,
        )
        if n_ok > 0:
            aug_paths[dev] = aug_path
        else:
            aug_paths[dev] = ds_path  # fallback to original

    # ── Step 4b: Retrain ───────────────────────────────────────────────────
    print("\n=== Step 4b: Retrain TSMC7 NMOS+PMOS (seeds {}) ===".format(
        seeds), flush=True)

    trained_stems: Dict[str, List[str]] = {"nmos": [], "pmos": []}
    for seed in seeds:
        exp_name = f"b6_tsmc7_s{seed}"
        for dev in ("nmos", "pmos"):
            data_path = aug_paths.get(dev, DATA_DIR / f"tsmc7_{dev}_b6.npz")
            if not data_path.exists():
                print(f"  [SKIP] {dev} augmented dataset not found: {data_path}",
                      flush=True)
                continue
            retrain_tsmc7(dev, data_path, seed, exp_name, gpu_id=args.gpu)
            stem = f"{exp_name}_{dev}"
            ckpt = CKPT_DIR / f"{stem}_best.pt"
            if ckpt.exists():
                trained_stems[dev].append(stem)
                print(f"  Checkpoint saved: {ckpt}", flush=True)
            else:
                print(f"  [WARN] Expected checkpoint not found: {ckpt}", flush=True)

    # ── Step 5: Summary ────────────────────────────────────────────────────
    print("\n=== Step 5: Summary ===", flush=True)
    print(f"  Trained NMOS stems: {trained_stems['nmos']}", flush=True)
    print(f"  Trained PMOS stems: {trained_stems['pmos']}", flush=True)
    print(f"\n  Run eval_v6_4_5_candidate.py for each seed pair to score.", flush=True)

    _write_report(WORK_DIR, all_results, all_residuals, all_valid,
                  in_box_verdict, trained_stems, seeds, args.top_k)


def _write_report(
    work_dir: Path,
    all_results: Dict,
    all_residuals: Dict[str, np.ndarray],
    all_valid: Dict[str, np.ndarray],
    in_box_verdict: Dict,
    trained_stems: Dict,
    seeds: List[int],
    top_k: int,
) -> None:
    """Write the preliminary B6 report (harvest section)."""
    lines = [
        "# B6 Harvest Analysis Report (preliminary — before eval scoring)",
        "",
        f"**Date:** 2026-05-29 · **top_k:** {top_k}",
        "",
        "## Harvest collection",
        "",
    ]
    for dev in ("nmos", "pmos"):
        n = len(all_residuals.get(dev, []))
        n_valid = int(all_valid.get(dev, np.array([])).sum())
        lines.append(f"- **{dev.upper()}**: {n} unique OPs, {n_valid} PyCMG-converged")

    lines += [
        "",
        "## Training-box analysis (top-5% worst residuals)",
        "",
        "| Device | n_valid | top5% n | in-box | out-of-box | verdict |",
        "|--------|---------|---------|--------|------------|---------|",
    ]
    for dev in ("nmos", "pmos"):
        r = all_results.get(dev, {})
        lines.append(
            f"| {dev.upper()} | {r.get('n_total_valid','?')} | "
            f"{r.get('n_top','?')} | "
            f"{r.get('n_in_box','?')} ({r.get('frac_in_box',float('nan')):.0%}) | "
            f"{r.get('n_out_of_box','?')} | "
            f"**{r.get('verdict','?').upper()}** |"
        )

    lines += [
        "",
        "## Residual statistics",
        "",
    ]
    for dev in ("nmos", "pmos"):
        r = all_results.get(dev, {})
        resid = all_residuals.get(dev, np.array([]))
        valid = all_valid.get(dev, np.array([], dtype=bool))
        if len(resid) > 0 and valid.sum() > 0:
            v = resid[valid]
            lines.append(
                f"- **{dev.upper()}**: mean={v.mean()*1e6:.3f} µA  "
                f"p50={np.median(v)*1e6:.3f} µA  "
                f"p90={np.percentile(v,90)*1e6:.3f} µA  "
                f"max={v.max()*1e6:.3f} µA"
            )
            lines.append(
                f"  top-5% mean={r.get('top_resid_mean_uA',float('nan')):.3f} µA  "
                f"max={r.get('top_resid_max_uA',float('nan')):.3f} µA"
            )

    lines += [
        "",
        "## Trained checkpoint stems",
        "",
    ]
    for dev in ("nmos", "pmos"):
        stems = trained_stems.get(dev, [])
        lines.append(f"- **{dev.upper()}**: {stems if stems else '(none yet)'}")

    lines += [
        "",
        "## Falsifier verdict",
        "",
        f"- NMOS in-box/out-of-box: **{in_box_verdict.get('nmos','?')}**",
        f"- PMOS in-box/out-of-box: **{in_box_verdict.get('pmos','?')}**",
        "",
        "*(Score each retrain candidate with eval_v6_4_5_candidate.py — "
        "results appended to the full B6_harvest_retrain.md after eval.)*",
        "",
    ]

    rpt = work_dir / "B6_harvest_analysis.md"
    rpt.write_text("\n".join(lines))
    print(f"\n  Preliminary report saved: {rpt}", flush=True)


if __name__ == "__main__":
    main()
