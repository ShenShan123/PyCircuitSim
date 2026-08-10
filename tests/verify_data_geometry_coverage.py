"""Guard: every benchmark geometry must be resolved by the training grid.

V7.4.2. The complex-circuit gates pin NMOS L=16 nm / PMOS L=20 nm, but the
NN datasets sampled L only at each PDK length bin's *lower corner* — and
short-channel bins are wide (TSMC5's shortest spans L in [6, 20] nm). For
TSMC5/6/7 the benchmark NMOS therefore sat deep inside an unsampled bin
interior, where nothing constrained the fit. Higher capacity converged the
knots and let the interpolant between them drift: measured against L72, the
TSMC7 `xl` NMOS was 13 % weak at 16 nm while landing within 0.2 % at 8, 11,
20 and 36 nm. That is what produced the "capacity hurts BSIM-AR" artifact —
the ring oscillator ran monotonically slower with every tier.

The check is deliberately **bin-aware**. A knot in the neighbouring bin does
not help: crossing a bin boundary changes the modelcard, so L=20 nm tells
the model nothing about L=16 nm inside the [6, 20] bin. Only same-bin knots
count.

Fails loud, so the defect cannot silently return the next time datasets are
regenerated.

Usage:
    conda run -n pycircuitsim python tests/verify_data_geometry_coverage.py
    python tests/verify_data_geometry_coverage.py --max-l-ratio 1.35
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# PROJECT_ROOT must stay ahead of PyCMG: both ship a top-level ``tests``
# package, and PyCMG's has no ``common`` submodule.
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))
sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.complex import BENCH, BENCH_TECHS  # noqa: E402
from bsimar.config import DATA_DIR, TECH_CONFIGS  # noqa: E402
from pycmg.parser import _scan_all_variants  # noqa: E402
from pycmg.tech import _resolve_path  # noqa: E402

# Same ratio the V7.4.2 datasets were generated at. A benchmark length more
# than this far from the nearest same-bin knot is an unresolved interior.
DEFAULT_MAX_L_RATIO = 1.35
# NFIN is sampled at each bin's {nfinmin, nfinmax}; the benchmarks use
# NFIN=2, which is a corner everywhere. Kept as an explicit check so an
# NFIN-side regression is caught by the same guard.
DEFAULT_MAX_NFIN_RATIO = 2.0


def _bin_containing(
    pdk_path: str, pdk_device: str, L: float, nfin: float,
) -> Optional[Tuple[float, float]]:
    """(lmin, lmax) of the PDK length bin holding (L, NFIN); None if absent."""
    for v in _scan_all_variants(pdk_path, pdk_device):
        if v.lmin <= L <= v.lmax:
            if v.nfinmin is None or v.nfinmax is None:
                return (v.lmin, v.lmax)
            if v.nfinmin <= nfin <= v.nfinmax:
                return (v.lmin, v.lmax)
    return None


def _nearest_ratio(target: float, knots: Sequence[float]) -> Optional[float]:
    """max(target/k, k/target) for the closest knot; None if there are none."""
    if len(knots) == 0 or target <= 0:
        return None
    return min(max(target / k, k / target) for k in knots if k > 0)


def _dataset_geometry(tech: str, dev: str) -> Optional[Tuple[np.ndarray,
                                                             np.ndarray]]:
    path = DATA_DIR / f"{tech}_{dev}.npz"
    if not path.exists():
        return None
    with np.load(str(path), allow_pickle=True) as d:
        g = d["geometry"]
        return np.unique(g[:, 1]), np.unique(g[:, 0])


def check(max_l_ratio: float, max_nfin_ratio: float) -> List[Tuple[str, bool,
                                                                  str]]:
    results: List[Tuple[str, bool, str]] = []
    for name in BENCH_TECHS:
        bt = BENCH[name]
        cfg = TECH_CONFIGS[bt.nn_tech]
        # PDK paths in the registry are PyCMG-relative.
        pdk = str(_resolve_path(str(cfg.pycmg_tech.pdk_path)))
        for dev, L_bench, nfin_bench, vt in (
            ("nmos", bt.l_nmos, float(bt.nfin), bt.effective_nmos_vt),
            ("pmos", bt.l_pmos, float(bt.effective_nfin_p),
             bt.effective_pmos_vt),
        ):
            label = f"{bt.nn_tech}:{dev} L={L_bench * 1e9:.0f}nm NFIN={nfin_bench:g}"
            geo = _dataset_geometry(bt.nn_tech, dev)
            if geo is None:
                results.append((label, False,
                                f"dataset {bt.nn_tech}_{dev}.npz not found"))
                continue
            l_knots, nfin_knots = geo

            pdk_device = cfg.pycmg_tech.get_device(f"{dev}_{vt}").pdk_device
            span = _bin_containing(pdk, pdk_device, L_bench, nfin_bench)
            if span is None:
                results.append((label, False,
                                "no PDK bin contains the benchmark geometry"))
                continue
            lo, hi = span
            # Only knots inside the SAME bin constrain this length — the
            # half-open interval [lo, hi). A knot sitting exactly on `hi` is
            # the NEXT bin's lower corner: `_find_length_variant` resolves
            # that length to the next variant, so the row was generated
            # under a different modelcard and says nothing about this bin's
            # interior. Counting it is precisely how the V7.4.0 grid looked
            # covered while [6, 20] nm held a single sample at 6 nm.
            same_bin = l_knots[(l_knots >= lo * (1 - 1e-9))
                               & (l_knots < hi * (1 - 1e-9))]
            r_l = _nearest_ratio(L_bench, same_bin)
            r_n = _nearest_ratio(nfin_bench, nfin_knots)
            if r_l is None:
                results.append((label, False,
                                f"bin [{lo*1e9:.0f}, {hi*1e9:.0f}] nm has no "
                                f"sampled L at all"))
                continue
            ok = r_l <= max_l_ratio and (r_n is not None
                                         and r_n <= max_nfin_ratio)
            results.append((
                label, ok,
                f"bin [{lo*1e9:5.1f},{hi*1e9:6.1f}] nm, {len(same_bin)} knot(s) "
                f"{np.round(same_bin * 1e9, 2).tolist()} -> L ratio {r_l:.3f} "
                f"(<= {max_l_ratio}), NFIN ratio "
                f"{'n/a' if r_n is None else f'{r_n:.3f}'}"))
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-l-ratio", type=float, default=DEFAULT_MAX_L_RATIO)
    p.add_argument("--max-nfin-ratio", type=float,
                   default=DEFAULT_MAX_NFIN_RATIO)
    a = p.parse_args()

    print("=" * 96)
    print("Benchmark-geometry coverage in the NN training grid (V7.4.2)")
    print("  A benchmark length is resolved only by knots in its OWN PDK bin —")
    print("  a neighbouring bin uses a different modelcard.")
    print("=" * 96)

    results = check(a.max_l_ratio, a.max_nfin_ratio)
    for label, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:34s} {detail}")

    n_pass = sum(1 for _, ok, _ in results if ok)
    print("\n" + "=" * 96)
    print(f"RESULT: {n_pass}/{len(results)} PASS")
    if n_pass != len(results):
        print("A FAIL means the benchmark circuits ask the model for a "
              "geometry the training data never sampled inside the relevant\n"
              "PDK bin. Regenerate with `--max-l-ratio "
              f"{a.max_l_ratio}` (see docs/plans/2026-08-10-v742-bsimar-"
              "capacity.md).")
    print("=" * 96)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
