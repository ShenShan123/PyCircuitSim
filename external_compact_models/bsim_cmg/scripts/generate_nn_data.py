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

Output goes to --data-dir (default: ../../neural_network/data/datasets/).
"""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from pycmg.nn_config import TECH_CONFIGS
from pycmg.nn_generate import (
    DEFAULT_GRID_PER_AXIS,
    DEFAULT_HOT_PER_AXIS,
    DEFAULT_JITTER_SIGMA_FRAC,
    DEFAULT_LHS_SAMPLES_PER_BIN,
    DEFAULT_MAX_L_RATIO,
    DEFAULT_SAMPLER,
    DEFAULT_TEMPERATURES_K,
    DEFAULT_VBS_LEVELS,
    DEFAULT_VOLTAGE_BOX_FACTOR,
    generate_dataset,
    generate_universal_dataset,
)
from pycmg.sweep import save_npz

from neural_network.data.sampling import stratified_sample_indices
from neural_network.data.contracts import (
    FULL_TERMINAL_OUTPUT_CONTRACT,
    REDUCED_OUTPUT_CONTRACT,
    dataset_filename,
)


def _default_data_dir() -> Path:
    pycmg_root = Path(__file__).resolve().parents[1]
    project_root = pycmg_root.parents[1]
    return (project_root / "external_compact_models" / "neural_network"
            / "data" / "datasets")


def _parse_temperatures(arg: str) -> tuple:
    return tuple(float(x.strip()) for x in arg.split(",") if x.strip())


def _save_finetune_split(
    full: dict,
    parent_path: Path,
    out_path: Path,
    n_samples: int,
    seed: int,
) -> None:
    """Write a subset stratified by geometry, process variant, and class."""
    n_total = full["inputs"].shape[0]
    n = min(n_samples, n_total)
    sample_class_full = full.get("sample_class")
    if sample_class_full is None:
        sample_class_full = np.full(n_total, -1, dtype=np.int8)
    strata = np.column_stack([full["geometry"], sample_class_full])
    idx = stratified_sample_indices(strata, n, seed)

    sample_class = full.get("sample_class")
    if sample_class is not None:
        sample_class = sample_class[idx]

    metadata = dict(full["metadata"])
    metadata.update({
        "dataset_variant": (
            f"{metadata.get('dataset_variant', 'generated')}_finetune_stratified"
        ),
        "requested_rows": np.int64(len(idx)),
        "kept_rows": np.int64(len(idx)),
        "rejected_rows": np.int64(0),
        "dropped_bins": np.int64(0),
        "allow_rejected_points": np.bool_(False),
        "parent_dataset": parent_path.name,
        "parent_dataset_sha256": _sha256(parent_path),
    })
    save_npz(
        full["inputs"][idx],
        full["geometry"][idx],
        full["outputs"][idx],
        out_path,
        metadata=metadata,
        sample_class=sample_class,
    )
    _write_completion_marker(out_path, len(idx), metadata)
    print(f"  Fine-tune split: {n:,} samples -> {out_path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_completion_marker(
    dataset_path: Path,
    row_count: int,
    metadata: dict,
) -> None:
    marker = dataset_path.with_suffix(dataset_path.suffix + ".complete")
    marker.write_text(json.dumps({
        "dataset": dataset_path.name,
        "dataset_sha256": _sha256(dataset_path),
        "rows": row_count,
        "source_commit": str(metadata.get("source_commit", "unknown")),
        "source_dirty": bool(metadata.get("source_dirty", True)),
        "generator_release": str(metadata.get("generator_release", "unknown")),
    }, sort_keys=True, indent=2) + "\n")


def _add_run_provenance(data: dict) -> None:
    project_root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root,
        capture_output=True, text=True, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=project_root, capture_output=True, text=True, check=False,
    )
    data["metadata"].update({
        "source_commit": commit.stdout.strip() if commit.returncode == 0 else "unknown",
        "source_dirty": status.returncode != 0 or bool(status.stdout.strip()),
        "generator_command": shlex.join([sys.executable, *sys.argv]),
    })


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
        help="Version tag inserted after the scope in output filenames "
             "(e.g. 'v5' -> universal_v5_nmos.npz, or "
             "universal_v5_dnf_nmos.npz for full-terminal data). Empty "
             "preserves the unversioned name.",
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

    # V7.4.2: intra-bin L sampling. The PDK grid gives one L per length
    # bin (its lower corner), and short-channel bins are wide — TSMC5's
    # shortest spans L in [6, 20] nm. Nothing constrains the model between
    # knots, and higher capacity lets that interpolant drift further, which
    # is what produced the "capacity hurts BSIM-AR" artifact (docs/plans/
    # 2026-08-10-v742-bsimar-capacity.md). Default off = legacy grid.
    parser.add_argument(
        "--max-l-ratio", type=float, default=DEFAULT_MAX_L_RATIO,
        help="Sample inside each PDK length bin so no adjacent pair of L "
             "knots differs by more than this ratio (default: 1.35). Costs roughly "
             "log(bin span)/log(ratio) times the rows.",
    )
    parser.add_argument(
        "--allow-rejected-points", action="store_true",
        help="Write a diagnostic artifact despite rejected points/bins. "
             "Canonical datasets fail instead.",
    )
    parser.add_argument(
        "--allow-safety-rejections", action="store_true",
        help="Keep a canonical dataset after excluding only the declared "
             "NaN/Inf, >1 A terminal-current, or internal-node-solve safety "
             "failures. Dropped bins and other failures remain fatal.",
    )
    parser.add_argument(
        "--output-contract",
        choices=[REDUCED_OUTPUT_CONTRACT, FULL_TERMINAL_OUTPUT_CONTRACT],
        default=REDUCED_OUTPUT_CONTRACT,
        help="Training targets: legacy reduced 13-head outputs or the "
             "V7.6.0 six-surface full-terminal contract.",
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
        max_l_ratio=args.max_l_ratio,
        allow_rejected_points=args.allow_rejected_points,
        allow_safety_rejections=args.allow_safety_rejections,
        output_contract=args.output_contract,
    )

    if args.universal:
        for device_type in devices:
            data = generate_universal_dataset(
                device_type,
                exclude_techs=exclude_techs,
                **common_kw,
            )
            _add_run_provenance(data)
            out = data_dir / dataset_filename(
                "universal", device_type, args.output_contract, version_tag,
            )
            save_npz(data["inputs"], data["geometry"], data["outputs"],
                     out, metadata=data["metadata"],
                     sample_class=data.get("sample_class"))
            _write_completion_marker(out, data["inputs"].shape[0], data["metadata"])
            print(f"  Wrote {out} ({data['inputs'].shape[0]:,} rows)")
            if args.finetune_size > 0:
                ft_out = data_dir / f"finetune_{out.name}"
                _save_finetune_split(
                    data, out, ft_out, args.finetune_size, seed=args.seed,
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
            _add_run_provenance(data)
            out = data_dir / dataset_filename(
                tech.name.lower(), device_type, args.output_contract,
                version_tag,
            )
            save_npz(data["inputs"], data["geometry"], data["outputs"],
                     out, metadata=data["metadata"],
                     sample_class=data.get("sample_class"))
            _write_completion_marker(out, data["inputs"].shape[0], data["metadata"])
            print(f"  Wrote {out} ({data['inputs'].shape[0]:,} rows)")
            if args.finetune_size > 0:
                ft_out = data_dir / f"finetune_{out.name}"
                _save_finetune_split(
                    data, out, ft_out, args.finetune_size, seed=args.seed,
                )


if __name__ == "__main__":
    main()
