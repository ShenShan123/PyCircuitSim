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

from cli_options import generation_parser, _parse_temperatures
from pycmg.nn_config import TECH_CONFIGS
from pycmg.nn_generate import (
    generate_dataset,
    generate_universal_dataset,
)
from pycmg.sweep import save_npz

from neural_network.data.sampling import stratified_sample_indices
from neural_network.data.contracts import (
    dataset_filename,
)


def _default_data_dir() -> Path:
    pycmg_root = Path(__file__).resolve().parents[1]
    project_root = pycmg_root.parents[1]
    return (project_root / "external_compact_models" / "neural_network"
            / "data" / "datasets")



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
        ["git", "status", "--porcelain"],
        cwd=project_root, capture_output=True, text=True, check=False,
    )
    data["metadata"].update({
        "source_commit": commit.stdout.strip() if commit.returncode == 0 else "unknown",
        "source_dirty": status.returncode != 0 or bool(status.stdout.strip()),
        "generator_command": shlex.join([sys.executable, *sys.argv]),
    })


def main() -> None:
    parser = generation_parser(tuple(TECH_CONFIGS))
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
        verbose=not args.quiet,
        sampler=args.sampler,
        grid_per_axis=args.grid_per_axis,
        vbs_levels=args.vbs_levels,
        hot_per_axis=args.hot_per_axis,
        jitter_sigma_frac=args.jitter_sigma_frac,
        overshoot_per_axis=args.overshoot_per_axis,
        n_vbs_lhs=args.n_vbs_lhs,
        enable_inv_trip=args.enable_inv_trip,
        enable_subvt_off=args.enable_subvt_off,
        max_l_ratio=args.max_l_ratio,
        allow_rejected_points=args.allow_rejected_points,
        allow_safety_rejections=args.allow_safety_rejections,
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
                "universal", device_type, version_tag,
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
                tech.name.lower(), device_type, version_tag,
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
