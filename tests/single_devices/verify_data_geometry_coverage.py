"""Guard: every benchmark geometry must be resolved by the training grid.

V7.4.2. The circuit benchmark gates pin NMOS L=16 nm / PMOS L=20 nm, but the
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
    conda run -n pycircuitsim python tests/single_devices/verify_data_geometry_coverage.py
    python tests/single_devices/verify_data_geometry_coverage.py --max-l-ratio 1.35
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# PROJECT_ROOT must stay ahead of PyCMG: both ship a top-level ``tests``
# package, and PyCMG's has no ``common`` submodule.
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "bsim_cmg"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))
sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.circuit_benchmarks import BENCH, BENCH_TECHS  # noqa: E402
from neural_network.config import (  # noqa: E402
    DATA_DIR,
    TECH_CONFIGS,
    TECH_VARIANT_CODES,
)
from neural_network.eval.loo_labels import (  # noqa: E402
    get_or_build_tech_variant_labels,
)
from tests.common.nn_sweep import (  # noqa: E402
    NN_TECHS,
    build_dc_parametric,
    build_inv_parametric,
    make_dc_baseline,
    make_inv_baseline,
)
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
) -> Optional[Tuple[float, float, float, float]]:
    """Joint L/NFIN bounds of the PDK bin holding the target geometry."""
    for v in _scan_all_variants(pdk_path, pdk_device):
        if v.lmin <= L <= v.lmax:
            if v.nfinmin is None or v.nfinmax is None:
                return (v.lmin, v.lmax, -np.inf, np.inf)
            if v.nfinmin <= nfin <= v.nfinmax:
                return (v.lmin, v.lmax, v.nfinmin, v.nfinmax)
    return None


def _nearest_ratio(target: float, knots: Sequence[float]) -> Optional[float]:
    """max(target/k, k/target) for the closest knot; None if there are none."""
    if len(knots) == 0 or target <= 0:
        return None
    return min(max(target / k, k / target) for k in knots if k > 0)


@lru_cache(maxsize=None)
def _dataset_arrays(
    tech: str,
    dev: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    path = DATA_DIR / f"{tech}_{dev}.npz"
    if not path.exists():
        return None
    with np.load(str(path), allow_pickle=True) as data:
        geometry = data["geometry"]
    labels = get_or_build_tech_variant_labels(
        str(path), dev, verbose=False,
    )
    return geometry, labels


@lru_cache(maxsize=None)
def _dataset_geometry(
    tech: str,
    dev: str,
    vt: str,
    temperature_k: float,
) -> Optional[np.ndarray]:
    arrays = _dataset_arrays(tech, dev)
    if arrays is None:
        return None
    geometry, labels = arrays
    code = TECH_VARIANT_CODES.get((tech, vt))
    if code is None:
        return np.empty((0, geometry.shape[1]), dtype=geometry.dtype)
    mask = ((labels == code)
            & np.isclose(geometry[:, 2], temperature_k, atol=1e-6))
    return geometry[mask]


@dataclass(frozen=True)
class EvalGeometry:
    """One device geometry/variant/temperature requested by a scored gate."""

    label: str
    tech: str
    dev: str
    vt: str
    length: float
    nfin: float
    temperature_k: float


def _evaluation_geometries() -> List[EvalGeometry]:
    points: List[EvalGeometry] = []
    for tech_key in NN_TECHS:
        for dev in ("nmos", "pmos"):
            for cfg in [make_dc_baseline(tech_key, dev),
                        *build_dc_parametric(tech_key, dev)]:
                tech = cfg.tech
                vt = tech.nn_vt if dev == "nmos" else tech.effective_pmos_vt
                length = tech.l_nmos if dev == "nmos" else tech.effective_l_pmos
                points.append(EvalGeometry(
                    cfg.label, tech.nn_tech_key, dev, vt, length,
                    float(tech.nfin), tech.temperature_c + 273.15,
                ))
        for analysis in ("vtc", "tran"):
            for cfg in [make_inv_baseline(tech_key, analysis),
                        *build_inv_parametric(tech_key, analysis)]:
                tech = cfg.tech
                points.extend((
                    EvalGeometry(
                        f"{cfg.label}:nmos", tech.nn_tech_key, "nmos",
                        tech.nn_vt, tech.effective_inv_l_nmos,
                        float(tech.effective_inv_nfin),
                        tech.temperature_c + 273.15,
                    ),
                    EvalGeometry(
                        f"{cfg.label}:pmos", tech.nn_tech_key, "pmos",
                        tech.effective_pmos_vt, tech.effective_inv_l_pmos,
                        float(tech.effective_inv_nfin_p),
                        tech.temperature_c + 273.15,
                    ),
                ))
    return points


def check(max_l_ratio: float, max_nfin_ratio: float) -> List[Tuple[str, bool,
                                                                  str]]:
    results: List[Tuple[str, bool, str]] = []
    points = _evaluation_geometries()
    for name in BENCH_TECHS:
        bt = BENCH[name]
        points.extend((
            EvalGeometry(
                f"{name}:complex:nmos", bt.nn_tech, "nmos",
                bt.effective_nmos_vt, bt.l_nmos, float(bt.nfin), 300.15,
            ),
            EvalGeometry(
                f"{name}:complex:pmos", bt.nn_tech, "pmos",
                bt.effective_pmos_vt, bt.l_pmos,
                float(bt.effective_nfin_p), 300.15,
            ),
        ))

    for point in points:
        cfg = TECH_CONFIGS[point.tech]
        # PDK paths in the registry are PyCMG-relative.
        pdk = str(_resolve_path(str(cfg.pycmg_tech.pdk_path)))
        label = (f"{point.label} VT={point.vt} T={point.temperature_k:g}K "
                 f"L={point.length * 1e9:.0f}nm NFIN={point.nfin:g}")
        geo = _dataset_geometry(
            point.tech, point.dev, point.vt, point.temperature_k,
        )
        if geo is None:
            results.append((label, False,
                            f"dataset {point.tech}_{point.dev}.npz not found"))
            continue
        pdk_device = cfg.pycmg_tech.get_device(
            f"{point.dev}_{point.vt}").pdk_device
        span = _bin_containing(
            pdk, pdk_device, point.length, point.nfin,
        )
        if span is None:
            results.append((label, False, "no PDK bin contains the geometry"))
            continue
        lo, hi, nlo, nhi = span
        same_bin = geo[
            (geo[:, 1] >= lo * (1 - 1e-9))
            & (geo[:, 1] < hi * (1 - 1e-9))
            & (geo[:, 0] >= nlo * (1 - 1e-9))
            & (geo[:, 0] <= nhi * (1 + 1e-9))
        ]
        if not len(same_bin):
            results.append((label, False, "variant/temperature PDK bin has no rows"))
            continue
        ratios = np.column_stack((
            np.maximum(point.length / same_bin[:, 1],
                       same_bin[:, 1] / point.length),
            np.maximum(point.nfin / same_bin[:, 0],
                       same_bin[:, 0] / point.nfin),
        ))
        best = int(np.argmin(np.max(ratios, axis=1)))
        r_l, r_n = ratios[best]
        ok = r_l <= max_l_ratio and r_n <= max_nfin_ratio
        results.append((
            label, ok,
            f"joint knot L={same_bin[best, 1]*1e9:.2f}nm "
            f"NFIN={same_bin[best, 0]:g}; ratios L={r_l:.3f}, NFIN={r_n:.3f}",
        ))
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
