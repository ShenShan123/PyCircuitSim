#!/usr/bin/env python3
"""Generate NN training data (.npz) from PyCMG BSIM-CMG sweeps.

Usage (from PyCMG root):
    python scripts/generate_nn_data.py --device both --universal
    python scripts/generate_nn_data.py --device nmos --tech asap7
    python scripts/generate_nn_data.py --device both --universal --n-workers 8
    python scripts/generate_nn_data.py --device both --universal \
        --finetune-size 8000

Phase D rewrite features:
    --n-workers N           : parallelize bins via multiprocessing.Pool (D4)
    --temperatures T1,T2..  : Kelvin temperature sweep (D1)
    --n-lhs-samples N       : LHS sample budget per bin (D3)
    --voltage-box-factor F  : Vg/Vd/Vbs box width in units of VDD (D3)
    --finetune-size N       : also write a stratified finetune split (D8)

Output goes to --data-dir (default: ../../bsimar/data/datasets/).
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pycmg.nn_config import TECH_CONFIGS
from pycmg.nn_generate import (
    DEFAULT_GRID_PER_AXIS,
    DEFAULT_HOT_PER_AXIS,
    DEFAULT_JITTER_SIGMA_FRAC,
    DEFAULT_LHS_SAMPLES_PER_BIN,
    DEFAULT_SAMPLER,
    DEFAULT_TEMPERATURES_K,
    DEFAULT_VBS_LEVELS,
    DEFAULT_VOLTAGE_BOX_FACTOR,
    generate_dataset,
    generate_universal_dataset,
)
from pycmg.sweep import save_npz


def _default_data_dir() -> Path:
    pycmg_root = Path(__file__).resolve().parents[1]
    project_root = pycmg_root.parents[1]
    return project_root / "external_compact_models" / "bsimar" / "data" / "datasets"


def _parse_temperatures(arg: str) -> tuple:
    return tuple(float(x.strip()) for x in arg.split(",") if x.strip())


def _save_finetune_split(
    full: dict,
    out_path: Path,
    n_samples: int,
    seed: int,
) -> None:
    """Write a stratified random subset of `full` for paper-style fine-tune.

    "Stratified" here means a uniform random shuffle (the LHS sampler
    already produced uniform coverage; further stratification by
    (variant, T) bin is overkill for an N=1k–8k subset). Reuses the
    same metadata as the parent dataset.
    """
    n_total = full["inputs"].shape[0]
    n = min(n_samples, n_total)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)[:n]

    sample_class = full.get("sample_class")
    if sample_class is not None:
        sample_class = sample_class[idx]

    save_npz(
        full["inputs"][idx],
        full["geometry"][idx],
        full["outputs"][idx],
        out_path,
        metadata=full["metadata"],
        sample_class=sample_class,
    )
    print(f"  Fine-tune split: {n:,} samples -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NN training data (.npz) from PyCMG BSIM-CMG"
    )
    parser.add_argument("--device", choices=["nmos", "pmos", "both"], default="nmos")
    parser.add_argument("--tech", choices=list(TECH_CONFIGS.keys()) + ["all"],
                        default="asap7")
    parser.add_argument("--variants", default="all",
                        help="Comma-separated variant names (default: all)")
    parser.add_argument("--universal", action="store_true",
                        help="Generate universal dataset across all techs/variants")

    # D1
    parser.add_argument(
        "--temperatures", type=_parse_temperatures,
        default=DEFAULT_TEMPERATURES_K,
        help="Comma-separated temperatures in Kelvin (default: -25, 27, 125 °C)",
    )

    # D3
    parser.add_argument("--n-lhs-samples", type=int,
                        default=DEFAULT_LHS_SAMPLES_PER_BIN,
                        help="LHS samples per (variant, L, NFIN, T) bin "
                             "(only used when --sampler=lhs)")
    parser.add_argument("--voltage-box-factor", type=float,
                        default=DEFAULT_VOLTAGE_BOX_FACTOR,
                        help="Voltage box width in units of VDD "
                             "(2.0 = [0, 2]·VDD; covers NR overshoot)")

    # B1 (v5 plan §4): hybrid uniform-grid sampler.
    parser.add_argument("--sampler", choices=["grid", "lhs"],
                        default=DEFAULT_SAMPLER,
                        help="Bulk-sample sampler: 'grid' = hybrid "
                             "uniform-grid + jitter + hot densification "
                             "(default, B1); 'lhs' = legacy Latin Hypercube")
    parser.add_argument("--grid-per-axis", type=int,
                        default=DEFAULT_GRID_PER_AXIS,
                        help="[grid] base 2D grid size per axis (Vgs, Vds)")
    parser.add_argument("--vbs-levels", type=int,
                        default=DEFAULT_VBS_LEVELS,
                        help="[grid] number of Vbs levels {0,±0.25,±0.5}·VDD")
    parser.add_argument("--hot-per-axis", type=int,
                        default=DEFAULT_HOT_PER_AXIS,
                        help="[grid] hot-region densification grid size "
                             "(0 to disable)")
    parser.add_argument("--jitter-sigma-frac", type=float,
                        default=DEFAULT_JITTER_SIGMA_FRAC,
                        help="[grid] Gaussian jitter sigma in fractions of VDD")

    # D4
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel worker count (1 = serial)")
    parser.add_argument("--seed", type=int, default=42)

    # D8
    parser.add_argument("--finetune-size", type=int, default=0,
                        help="If >0, also write finetune_<base>.npz with a "
                             "stratified random subset of N samples (D8)")

    # v5 plan §4-B5: dataset versioning + tech exclusion.
    parser.add_argument(
        "--version", default="",
        help="Version tag prefixed onto output filenames "
             "(e.g. 'v5' -> universal_v5_{nmos,pmos}.npz). "
             "Empty string preserves the legacy unversioned name.",
    )
    parser.add_argument(
        "--exclude-techs", default="",
        help="Comma-separated tech names to exclude from generation "
             "(case-insensitive). Common v5 use: --exclude-techs asap7.",
    )

    # v5p (V5'): inv_trip overlay is now opt-in. Default off matches the
    # V4 B1 base sampler. When set, nn_generate.py additionally gates
    # the overlay to TSMC5 only.
    parser.add_argument(
        "--enable-inv-trip", action="store_true", default=False,
        help="Enable v5 plan §4-B1 inverter-trip overlay. In V5' this "
             "is additionally gated to TSMC5 only inside nn_generate.py.",
    )

    # V6.4.7 S9b: subthreshold/OFF densification + DC-solve floor fix.
    parser.add_argument(
        "--enable-subvt-off", action="store_true", default=False,
        help="Enable the V6.4.7 S9b subthreshold/OFF |id|-space band probe "
             "(sample_class='subvt_off'). Fills the 1e-12..1e-6 A id decades "
             "for the decade-occupancy acceptance gate. Requires the "
             "DC-solve floor fix (see --dc-solve-tol).",
    )
    parser.add_argument(
        "--dc-solve-tol", type=float, default=1e-12,
        help="OSDI internal-node NR tolerance for generated rows, exported "
             "as NN_DC_SOLVE_TOL (default 1e-12). The legacy 1e-9 default "
             "returned EXACT 0 for true |id|<~1e-9 A (the 6-8%% zero-row "
             "artifact); 1e-12 resolves the sub-nA band. 1e-14 is FP-limited.",
    )

    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Output directory for .npz files")
    args = parser.parse_args()

    # V6.4.7 S9b generator floor fix: export the tightened internal-node NR
    # tolerance so every generated row (including multiprocessing workers,
    # which inherit the parent environment at spawn) resolves sub-nA |id|
    # instead of returning EXACT 0. Instance.eval_dc's own default is
    # unchanged when this env var is absent.
    os.environ["NN_DC_SOLVE_TOL"] = repr(float(args.dc_solve_tol))
    print(f"[gen] NN_DC_SOLVE_TOL={os.environ['NN_DC_SOLVE_TOL']} "
          f"(DC internal-node NR floor)")

    data_dir = args.data_dir or _default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    devices = ["nmos", "pmos"] if args.device == "both" else [args.device]

    # v5 plan §4-B5: parse exclude-techs once, validate.
    exclude_techs = sorted({
        t.strip().lower() for t in args.exclude_techs.split(",") if t.strip()
    })
    for t in exclude_techs:
        if t not in TECH_CONFIGS:
            raise SystemExit(
                f"--exclude-techs: unknown tech {t!r}; "
                f"valid options: {sorted(TECH_CONFIGS.keys())}"
            )

    # v5 plan §4-B5: optional version tag. Empty -> legacy name.
    version_tag = args.version.strip().strip("_")

    def _versioned(stem: str) -> str:
        return f"{stem}_{version_tag}" if version_tag else stem

    common_kw = dict(
        temperatures=args.temperatures,
        n_lhs_samples=args.n_lhs_samples,
        voltage_box_factor=args.voltage_box_factor,
        n_workers=args.n_workers,
        seed=args.seed,
        verbose=True,
        sampler=args.sampler,
        grid_per_axis=args.grid_per_axis,
        vbs_levels=args.vbs_levels,
        hot_per_axis=args.hot_per_axis,
        jitter_sigma_frac=args.jitter_sigma_frac,
        enable_inv_trip=args.enable_inv_trip,
        enable_subvt_off=args.enable_subvt_off,
    )

    if args.universal:
        for device_type in devices:
            data = generate_universal_dataset(
                device_type,
                exclude_techs=exclude_techs,
                **common_kw,
            )
            out = data_dir / f"{_versioned('universal')}_{device_type}.npz"
            save_npz(data["inputs"], data["geometry"], data["outputs"],
                     out, metadata=data["metadata"],
                     sample_class=data.get("sample_class"))
            print(f"  Wrote {out} ({data['inputs'].shape[0]:,} rows)")
            if args.finetune_size > 0:
                ft_out = data_dir / (
                    f"finetune_{_versioned('universal')}_{device_type}.npz"
                )
                _save_finetune_split(
                    data, ft_out, args.finetune_size, seed=args.seed,
                )
        return

    # Per-tech path: --exclude-techs prunes the explicit list too.
    if args.tech == "all":
        techs = [t for n, t in TECH_CONFIGS.items() if n not in exclude_techs]
    elif args.tech in exclude_techs:
        raise SystemExit(
            f"--tech {args.tech} conflicts with --exclude-techs {exclude_techs}"
        )
    else:
        techs = [TECH_CONFIGS[args.tech]]
    variant_names = None if args.variants == "all" \
        else [v.strip() for v in args.variants.split(",")]

    for tech in techs:
        for device_type in devices:
            data = generate_dataset(
                tech, device_type,
                variant_names=variant_names,
                **common_kw,
            )
            out = data_dir / (
                f"{_versioned(tech.name.lower())}_{device_type}.npz"
            )
            save_npz(data["inputs"], data["geometry"], data["outputs"],
                     out, metadata=data["metadata"],
                     sample_class=data.get("sample_class"))
            print(f"  Wrote {out} ({data['inputs'].shape[0]:,} rows)")
            if args.finetune_size > 0:
                ft_out = data_dir / (
                    f"finetune_{_versioned(tech.name.lower())}_{device_type}.npz"
                )
                _save_finetune_split(
                    data, ft_out, args.finetune_size, seed=args.seed,
                )


if __name__ == "__main__":
    main()
