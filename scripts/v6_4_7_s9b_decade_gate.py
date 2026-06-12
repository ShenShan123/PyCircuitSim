#!/usr/bin/env python3
"""V6.4.7 S9b regen-v2 acceptance gate: per-decade id occupancy.

Plan ruling 5 (rev 3): every id decade in (1e-12, 1e-6] A must hold
>= 1k rows per (tech, device) cell. This script is the orchestrator's
acceptance gate for the regenerated datasets.

Usage:
    conda run -n pycircuitsim python scripts/v6_4_7_s9b_decade_gate.py \
        external_compact_models/bsimar/data/datasets/tsmc7_nmos.npz [...]

Prints per-decade row counts for |id| over (1e-12, 1e-11] .. (1e-7, 1e-6],
plus the exact-zero count and total. Exits non-zero if any decade in any
file falls below --min-rows (default 1000).

npz layout: ``outputs`` is (N, 13) in OUTPUT_COLUMN_ORDER with ``id`` at
column 0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

ID_COLUMN: int = 0  # OUTPUT_COLUMN_ORDER[0] == "id"

# Six gate decades: (1e-12, 1e-11] .. (1e-7, 1e-6] A.
DECADES: List[Tuple[float, float]] = [
    (10.0 ** e, 10.0 ** (e + 1)) for e in range(-12, -6)
]


def check_file(path: Path, min_rows: int) -> bool:
    """Print the decade histogram for one npz; return True if it passes."""
    with np.load(path, allow_pickle=True) as data:
        if "outputs" not in data:
            print(f"\n{path}\n  ERROR: no 'outputs' key", flush=True)
            return False
        id_abs = np.abs(data["outputs"][:, ID_COLUMN])

    n_total = int(id_abs.size)
    n_zero = int(np.sum(id_abs == 0.0))
    print(f"\n{path}")
    print(f"  total rows: {n_total:>12,}")
    print(f"  id == 0.0 : {n_zero:>12,} ({100.0 * n_zero / max(n_total, 1):.2f} %)")

    all_pass = True
    for lo, hi in DECADES:
        n = int(np.sum((id_abs > lo) & (id_abs <= hi)))
        ok = n >= min_rows
        all_pass &= ok
        print(f"  ({lo:7.0e}, {hi:7.0e}] A: {n:>10,}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"  => {'PASS' if all_pass else 'FAIL'} "
          f"(gate: >= {min_rows:,} rows per decade)")
    return all_pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="S9b decade-occupancy acceptance gate for NN datasets"
    )
    parser.add_argument("npz", nargs="+", type=Path,
                        help="Dataset .npz path(s), one per (tech, device) cell")
    parser.add_argument("--min-rows", type=int, default=1000,
                        help="Minimum rows per decade per cell (default 1000)")
    args = parser.parse_args()

    overall = True
    for path in args.npz:
        if not path.exists():
            print(f"\n{path}\n  ERROR: file not found", flush=True)
            overall = False
            continue
        overall &= check_file(path, args.min_rows)

    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'} "
          f"({len(args.npz)} file(s))")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
