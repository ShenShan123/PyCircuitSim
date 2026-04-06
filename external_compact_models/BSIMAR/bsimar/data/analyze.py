"""Analyze NN training dataset: distribution, outliers, and data quality.

Usage:
    conda run -n pycircuitsim python -m bsimar.data.analyze
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from bsimar.config import (
    DATA_DIR, OUTPUT_COLUMNS, PROCESS_PARAM_NAMES, INPUT_COLUMNS,
)

# Column groups
INPUT_NAMES = ["Vd", "Vg", "Vs", "Vb"]
GEOMETRY_NAMES = ["NFIN", "T"] + PROCESS_PARAM_NAMES


def load_dataset(path: Path) -> Dict[str, np.ndarray]:
    """Load .npz dataset and return inputs/geometry/outputs."""
    data = np.load(path, allow_pickle=True)
    return {
        "inputs": data["inputs"],
        "geometry": data["geometry"],
        "outputs": data["outputs"],
    }


def basic_stats(arr: np.ndarray, names: List[str], title: str) -> None:
    """Print basic statistics for each column."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  {'Column':>8s}  {'Min':>12s}  {'Max':>12s}  {'Mean':>12s}  "
          f"{'Std':>12s}  {'Median':>12s}  {'NaN':>5s}  {'Inf':>5s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  "
          f"{'-'*5}  {'-'*5}")
    for i, name in enumerate(names):
        col = arr[:, i]
        n_nan = int(np.isnan(col).sum())
        n_inf = int(np.isinf(col).sum())
        clean = col[np.isfinite(col)]
        if len(clean) == 0:
            print(f"  {name:>8s}  {'ALL NaN/Inf':>12s}")
            continue
        print(f"  {name:>8s}  {clean.min():>+12.4e}  {clean.max():>+12.4e}  "
              f"{clean.mean():>+12.4e}  {clean.std():>12.4e}  "
              f"{np.median(clean):>+12.4e}  {n_nan:>5d}  {n_inf:>5d}")


def check_outliers(arr: np.ndarray, names: List[str], title: str,
                   z_thresh: float = 5.0) -> int:
    """Detect outliers using z-score threshold. Returns count."""
    print(f"\n{'='*80}")
    print(f"  Outlier Analysis ({title}) — |z-score| > {z_thresh}")
    print(f"{'='*80}")
    total_outliers = 0
    for i, name in enumerate(names):
        col = arr[:, i]
        clean = col[np.isfinite(col)]
        if len(clean) == 0 or clean.std() == 0:
            continue
        z = np.abs((clean - clean.mean()) / clean.std())
        n_outliers = int((z > z_thresh).sum())
        if n_outliers > 0:
            outlier_vals = clean[z > z_thresh]
            print(f"  {name:>8s}: {n_outliers:>6d} outliers "
                  f"(range: [{outlier_vals.min():+.4e}, {outlier_vals.max():+.4e}])")
            total_outliers += n_outliers
    if total_outliers == 0:
        print(f"  No outliers found (all columns within {z_thresh}σ)")
    return total_outliers


def check_physical_constraints(outputs: np.ndarray) -> None:
    """Check physically meaningful constraints on device outputs."""
    print(f"\n{'='*80}")
    print(f"  Physical Constraint Checks")
    print(f"{'='*80}")

    id_col = outputs[:, OUTPUT_COLUMNS.index("id")]
    gm_col = outputs[:, OUTPUT_COLUMNS.index("gm")]
    gds_col = outputs[:, OUTPUT_COLUMNS.index("gds")]

    id_abs = np.abs(id_col)
    print(f"\n  |id| range: [{id_abs.min():.4e}, {id_abs.max():.4e}]")
    n_large_id = int((id_abs > 0.1).sum())
    n_very_large_id = int((id_abs > 1.0).sum())
    print(f"  |id| > 100mA: {n_large_id} ({100*n_large_id/len(id_col):.3f}%)")
    print(f"  |id| > 1A:    {n_very_large_id} (should be 0)")

    n_neg_gds = int((gds_col < 0).sum())
    print(f"\n  gds < 0: {n_neg_gds} ({100*n_neg_gds/len(gds_col):.3f}%)")
    if n_neg_gds > 0:
        neg_gds = gds_col[gds_col < 0]
        print(f"    Negative gds range: [{neg_gds.min():.4e}, {neg_gds.max():.4e}]")

    n_zero_id = int((id_abs < 1e-15).sum())
    print(f"\n  |id| < 1e-15 (near-zero): {n_zero_id} ({100*n_zero_id/len(id_col):.3f}%)")

    qg = outputs[:, OUTPUT_COLUMNS.index("qg")]
    qd = outputs[:, OUTPUT_COLUMNS.index("qd")]
    qs = outputs[:, OUTPUT_COLUMNS.index("qs")]
    qb = outputs[:, OUTPUT_COLUMNS.index("qb")]
    q_sum = qg + qd + qs + qb
    print(f"\n  Charge sum (qg+qd+qs+qb):")
    print(f"    Mean: {q_sum.mean():.4e}")
    print(f"    Std:  {q_sum.std():.4e}")
    print(f"    Max |sum|: {np.abs(q_sum).max():.4e}")

    cgg = outputs[:, OUTPUT_COLUMNS.index("cgg")]
    cgd = outputs[:, OUTPUT_COLUMNS.index("cgd")]
    cgs = outputs[:, OUTPUT_COLUMNS.index("cgs")]
    cap_err = cgg + cgd + cgs
    print(f"\n  Capacitance check (cgg + cgd + cgs):")
    print(f"    Mean: {cap_err.mean():.4e}")
    print(f"    Max |err|: {np.abs(cap_err).max():.4e}")


def distribution_percentiles(arr: np.ndarray, names: List[str], title: str) -> None:
    """Print percentile distribution for key columns."""
    print(f"\n{'='*80}")
    print(f"  Percentile Distribution ({title})")
    print(f"{'='*80}")
    pcts = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
    header = f"  {'Column':>8s}" + "".join(f"  {p:>7.1f}%" for p in pcts)
    print(header)
    print(f"  {'-'*8}" + "  -------" * len(pcts))
    for i, name in enumerate(names):
        col = arr[:, i]
        clean = col[np.isfinite(col)]
        if len(clean) == 0:
            continue
        vals = np.percentile(clean, pcts)
        row = f"  {name:>8s}" + "".join(f"  {v:>+8.2e}" for v in vals)
        print(row)


def check_duplicate_points(inputs: np.ndarray, geometry: np.ndarray) -> None:
    """Check for duplicate bias points (same V + geometry)."""
    print(f"\n{'='*80}")
    print(f"  Duplicate Point Check")
    print(f"{'='*80}")
    combined = np.hstack([inputs, geometry])
    _, unique_idx, counts = np.unique(
        combined, axis=0, return_index=True, return_counts=True
    )
    n_dups = int((counts > 1).sum())
    total_dup_rows = int(counts[counts > 1].sum() - n_dups)
    print(f"  Total points: {len(inputs)}")
    print(f"  Unique points: {len(unique_idx)}")
    print(f"  Duplicate groups: {n_dups}")
    print(f"  Extra duplicate rows: {total_dup_rows}")
    if n_dups > 0 and n_dups <= 20:
        print(f"\n  Top duplicate groups (count > 1):")
        dup_mask = counts > 1
        dup_counts = counts[dup_mask]
        dup_indices = unique_idx[dup_mask]
        sort_order = np.argsort(-dup_counts)[:20]
        for rank, idx in enumerate(sort_order):
            row = combined[dup_indices[idx]]
            print(f"    #{rank+1}: count={dup_counts[idx]}, "
                  f"Vd={row[0]:.3f} Vg={row[1]:.3f} NFIN={row[4]:.0f}")


def per_tech_summary(geometry: np.ndarray, outputs: np.ndarray) -> None:
    """Summarize dataset per technology (using process params as discriminator)."""
    print(f"\n{'='*80}")
    print(f"  Per-Technology Summary")
    print(f"{'='*80}")
    phig_idx = PROCESS_PARAM_NAMES.index("PHIG")
    u0_idx = PROCESS_PARAM_NAMES.index("U0")
    eot_idx = PROCESS_PARAM_NAMES.index("EOT")

    phig = geometry[:, 2 + phig_idx]
    u0 = geometry[:, 2 + u0_idx]
    eot = geometry[:, 2 + eot_idx]

    combos = np.column_stack([phig, u0, eot])
    unique_combos = np.unique(combos, axis=0)

    id_col = outputs[:, OUTPUT_COLUMNS.index("id")]

    print(f"  {'PHIG':>8s}  {'U0':>10s}  {'EOT':>10s}  {'Count':>8s}  "
          f"{'id_min':>12s}  {'id_max':>12s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*12}  {'-'*12}")

    for combo in unique_combos:
        mask = np.all(combos == combo, axis=1)
        n = int(mask.sum())
        ids = id_col[mask]
        print(f"  {combo[0]:>8.4f}  {combo[1]:>10.4e}  {combo[2]:>10.4e}  "
              f"{n:>8d}  {ids.min():>+12.4e}  {ids.max():>+12.4e}")

    print(f"\n  Total unique (PHIG, U0, EOT) combos: {len(unique_combos)}")


def analyze_dataset(path: Path, device_type: str) -> None:
    """Full analysis of a dataset file."""
    print(f"\n{'#'*80}")
    print(f"#  DATASET ANALYSIS: {path.name}  ({device_type.upper()})")
    print(f"#  File size: {path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"{'#'*80}")

    data = load_dataset(path)
    inputs = data["inputs"]
    geometry = data["geometry"]
    outputs = data["outputs"]

    print(f"\n  Shape — inputs: {inputs.shape}, geometry: {geometry.shape}, "
          f"outputs: {outputs.shape}")

    basic_stats(inputs, INPUT_NAMES, f"Input Voltages ({device_type.upper()})")
    basic_stats(geometry, GEOMETRY_NAMES, f"Geometry + Process Params ({device_type.upper()})")
    basic_stats(outputs, OUTPUT_COLUMNS, f"Output Columns ({device_type.upper()})")

    distribution_percentiles(outputs, OUTPUT_COLUMNS,
                             f"Outputs ({device_type.upper()})")

    check_outliers(outputs, OUTPUT_COLUMNS,
                   f"Outputs ({device_type.upper()})", z_thresh=5.0)

    check_physical_constraints(outputs)
    check_duplicate_points(inputs, geometry)
    per_tech_summary(geometry, outputs)


def main() -> None:
    for device_type in ["nmos", "pmos"]:
        path = DATA_DIR / f"universal_{device_type}.npz"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        analyze_dataset(path, device_type)


if __name__ == "__main__":
    main()
