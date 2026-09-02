#!/usr/bin/env python3
"""Append an OSDI-evaluated trajectory tube to a full-terminal dataset.

The trajectory fragment supplies only source-relative bias coordinates and
geometry.  Every six-surface target is re-evaluated on the identical BSIM-CMG
OSDI model before append; legacy reduced-current targets are never converted
or treated as truth.  The parent dataset remains untouched, and the output is
campaign-ready only after its label sidecars and completion marker exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for _path in (
    ROOT / "external_compact_models" / "bsim_cmg",
    ROOT / "external_compact_models",
    ROOT,
):
    _text = str(_path)
    if _text in sys.path:
        sys.path.remove(_text)
    sys.path.insert(0, _text)

from neural_network.config import tech_variant_to_code  # noqa: E402
from neural_network.data.contracts import (  # noqa: E402
    FULL_TERMINAL_OUTPUT_COLUMN_ORDER,
    FULL_TERMINAL_OUTPUT_CONTRACT,
)
from neural_network.data.dataset import validate_canonical_dataset  # noqa: E402
from neural_network.eval.loo_labels import (  # noqa: E402
    get_or_build_tech_variant_labels,
    write_sidecar_meta,
)
from pycmg.nn_config import TECH_CONFIGS  # noqa: E402
from pycmg.nn_generate import (  # noqa: E402
    SAMPLE_CLASS_CODES,
    SAMPLE_CLASS_NAMES,
    _create_model_and_instance,
    _eval_single_point_with_reason,
)
from pycmg.sweep import save_npz  # noqa: E402


GENERATOR_RELEASE = "V7.6.4"
CORRIDOR_NAME = "traj_corridor"
CORRIDOR_CODE = SAMPLE_CLASS_CODES[CORRIDOR_NAME]
OUTPUT_COLUMNS = tuple(FULL_TERMINAL_OUTPUT_COLUMN_ORDER)
PointEvaluator = Callable[
    [float, float, float, float],
    Tuple[Optional[Mapping[str, float]], str],
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(value: object, name: str) -> object:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar")
    return array.reshape(()).item()


def _strings(value: object) -> list[str]:
    return [
        item.decode() if isinstance(item, bytes) else str(item)
        for item in np.asarray(value).tolist()
    ]


def _metadata(data: np.lib.npyio.NpzFile) -> Dict[str, object]:
    return {
        key.removeprefix("meta_"): np.asarray(data[key]).copy()
        for key in data.files if key.startswith("meta_")
    }


def evaluate_terminal_rows(
    inputs: np.ndarray,
    evaluator: PointEvaluator,
) -> np.ndarray:
    """Evaluate six independent full-terminal surfaces in contract order."""
    values = np.asarray(inputs, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"corridor inputs must have shape (N, 4), got {values.shape}")
    outputs = np.empty((len(values), len(OUTPUT_COLUMNS)), dtype=np.float64)
    for index, (vd, vg, vs, vb) in enumerate(values):
        result, reason = evaluator(float(vd), float(vg), float(vs), float(vb))
        if result is None:
            raise RuntimeError(
                f"OSDI rejected corridor row {index}: {reason or 'unknown reason'}"
            )
        try:
            outputs[index] = [float(result[name]) for name in OUTPUT_COLUMNS]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"OSDI corridor row {index} does not satisfy the six-surface "
                "output contract"
            ) from exc
    if not np.all(np.isfinite(outputs)):
        raise RuntimeError("OSDI corridor outputs contain NaN/Inf")
    return outputs


def validate_fragment_arrays(
    inputs: np.ndarray,
    geometry: np.ndarray,
    *,
    nfin: float,
    length: float,
    temperature: float,
) -> None:
    """Validate the harvested bias-frame and its one true circuit geometry."""
    values = np.asarray(inputs, dtype=np.float64)
    geom = np.asarray(geometry, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or not len(values):
        raise ValueError(f"corridor inputs must have non-empty shape (N, 4), got {values.shape}")
    if geom.shape != (len(values), 15):
        raise ValueError(
            f"corridor geometry must have shape ({len(values)}, 15), got {geom.shape}"
        )
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(geom)):
        raise ValueError("corridor inputs/geometry contain NaN/Inf")
    if not np.allclose(values[:, 2], 0.0, rtol=0.0, atol=0.0):
        raise ValueError("corridor inputs must use the source-relative Vs=0 frame")
    if not np.allclose(geom, geom[0], rtol=0.0, atol=0.0):
        raise ValueError("corridor rows must have one uniform geometry")
    expected = np.asarray([nfin, length, temperature], dtype=np.float64)
    if not np.allclose(geom[0, :3], expected, rtol=0.0, atol=1e-15):
        raise ValueError(
            "corridor metadata and geometry disagree: "
            f"row={geom[0, :3].tolist()} metadata={expected.tolist()}"
        )


def appended_metadata(
    parent: Mapping[str, object],
    *,
    parent_path: Path,
    fragment_path: Path,
    corridor_rows: int,
    tech: str,
    device: str,
    variant: str,
    length: float,
    nfin: float,
    temperature: float,
    source_commit: str,
    generator_command: str,
) -> Dict[str, object]:
    """Return audited metadata for a parent plus one truth overlay."""
    if corridor_rows <= 0:
        raise ValueError("corridor_rows must be positive")
    if len(source_commit) != 40:
        raise ValueError("source_commit must be a 40-character Git commit")
    result = dict(parent)
    requested = int(_scalar(parent["requested_rows"], "requested_rows"))
    kept = int(_scalar(parent["kept_rows"], "kept_rows"))
    rejected = int(_scalar(parent["rejected_rows"], "rejected_rows"))
    manifest = json.loads(str(_scalar(parent["manifest_json"], "manifest_json")))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("parent manifest_json must contain a non-empty list")
    manifest.append({
        "status": "complete",
        "requested": corridor_rows,
        "kept": corridor_rows,
        "rejected": 0,
        "failure_reason_counts": {},
        "failed_coordinates": [],
        "tech": tech,
        "device": device,
        "variant": variant,
        "L": length,
        "NFIN": nfin,
        "temperature_k": temperature,
        "sample_class": CORRIDOR_NAME,
        "source_fragment": fragment_path.name,
        "source_fragment_sha256": _sha256(fragment_path),
    })
    prior_classes = _strings(
        parent.get("externally_appended_sample_class_names", np.asarray([], dtype=str))
    )
    if CORRIDOR_NAME not in prior_classes:
        prior_classes.append(CORRIDOR_NAME)
    parent_variant = str(_scalar(parent["dataset_variant"], "dataset_variant"))
    result.update({
        "dataset_variant": f"{parent_variant}_plus_{CORRIDOR_NAME}",
        "generator_release": GENERATOR_RELEASE,
        "generator_command": generator_command,
        "source_commit": source_commit,
        "source_dirty": False,
        "requested_rows": np.int64(requested + corridor_rows),
        "kept_rows": np.int64(kept + corridor_rows),
        "rejected_rows": np.int64(rejected),
        "manifest_json": json.dumps(manifest, sort_keys=True),
        "externally_appended_sample_class_names": np.asarray(
            prior_classes, dtype=str
        ),
        "parent_dataset": parent_path.name,
        "parent_dataset_sha256": _sha256(parent_path),
        "corridor_fragment": fragment_path.name,
        "corridor_fragment_sha256": _sha256(fragment_path),
        "corridor_rows": np.int64(corridor_rows),
    })
    return result


def _clean_commit() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "full-terminal corridor generation requires a clean tracked worktree"
        )
    if len(commit) != 40:
        raise RuntimeError("could not resolve a full Git commit")
    return commit


def _write_completion_marker(
    output: Path,
    *,
    rows: int,
    source_commit: str,
) -> None:
    marker = output.with_suffix(output.suffix + ".complete")
    marker.write_text(json.dumps({
        "dataset": output.name,
        "dataset_sha256": _sha256(output),
        "generator_release": GENERATOR_RELEASE,
        "rows": rows,
        "source_commit": source_commit,
        "source_dirty": False,
    }, sort_keys=True, indent=2) + "\n")


def append_corridor(
    parent_path: Path,
    fragment_path: Path,
    output_path: Path,
    *,
    generator_command: str,
) -> None:
    """Build one isolated canonical full-terminal corridor dataset."""
    parent_path = parent_path.resolve()
    fragment_path = fragment_path.resolve()
    output_path = output_path.resolve()
    if output_path.suffix != ".npz":
        raise ValueError("output path must end in .npz")
    if output_path == parent_path:
        raise ValueError("corridor output must not overwrite its parent dataset")
    if output_path.exists() or output_path.with_suffix(
        output_path.suffix + ".complete"
    ).exists():
        raise FileExistsError(f"corridor output already exists: {output_path}")
    validate_canonical_dataset(parent_path)
    source_commit = _clean_commit()

    with np.load(fragment_path, allow_pickle=False) as fragment:
        required = {
            "inputs", "geometry", "meta_tech", "meta_device", "meta_variant",
            "meta_L", "meta_NFIN", "meta_T",
        }
        missing = sorted(required.difference(fragment.files))
        if missing:
            raise ValueError(f"corridor fragment is missing {missing}")
        corridor_inputs = np.asarray(fragment["inputs"], dtype=np.float64)
        corridor_geometry = np.asarray(fragment["geometry"], dtype=np.float64)
        tech = str(_scalar(fragment["meta_tech"], "meta_tech")).lower()
        device = str(_scalar(fragment["meta_device"], "meta_device")).lower()
        variant = str(_scalar(fragment["meta_variant"], "meta_variant")).lower()
        length = float(_scalar(fragment["meta_L"], "meta_L"))
        nfin = float(_scalar(fragment["meta_NFIN"], "meta_NFIN"))
        temperature = float(_scalar(fragment["meta_T"], "meta_T"))
    if tech not in TECH_CONFIGS:
        raise ValueError(f"unknown corridor technology {tech!r}")
    if device not in {"nmos", "pmos"}:
        raise ValueError(f"unknown corridor device {device!r}")
    validate_fragment_arrays(
        corridor_inputs, corridor_geometry, nfin=nfin, length=length,
        temperature=temperature,
    )

    built = _create_model_and_instance(
        TECH_CONFIGS[tech], device, variant, length, nfin, temperature,
    )
    if built is None:
        raise RuntimeError(
            f"could not construct OSDI teacher for {tech}/{device}/{variant}"
        )
    _model, instance, process = built
    expected_geometry = np.asarray(
        [nfin, length, temperature, *process.as_array()], dtype=np.float64
    )
    if not np.allclose(
        corridor_geometry[0], expected_geometry, rtol=1e-12, atol=1e-18,
    ):
        raise ValueError("corridor process fingerprint does not match OSDI teacher")

    def _evaluate(
        vd: float, vg: float, vs: float, vb: float,
    ) -> Tuple[Optional[Mapping[str, float]], str]:
        return _eval_single_point_with_reason(
            instance, vd=vd, vg=vg, vs=vs, vb=vb, _silent=True,
            output_contract=FULL_TERMINAL_OUTPUT_CONTRACT,
        )

    corridor_outputs = evaluate_terminal_rows(corridor_inputs, _evaluate)

    with np.load(parent_path, allow_pickle=False) as parent:
        if str(_scalar(parent["meta_output_contract"], "meta_output_contract")) != (
            FULL_TERMINAL_OUTPUT_CONTRACT
        ):
            raise ValueError("parent dataset is not full-terminal")
        if tuple(_strings(parent["meta_output_columns"])) != OUTPUT_COLUMNS:
            raise ValueError("parent dataset has the wrong full-terminal columns")
        names = _strings(parent["meta_sample_class_names"])
        if len(names) <= CORRIDOR_CODE or names[CORRIDOR_CODE] != CORRIDOR_NAME:
            raise ValueError("parent sample-class vocabulary lacks traj_corridor")
        parent_inputs = np.asarray(parent["inputs"], dtype=np.float64)
        parent_geometry = np.asarray(parent["geometry"], dtype=np.float64)
        parent_outputs = np.asarray(parent["outputs"], dtype=np.float64)
        parent_classes = np.asarray(parent["sample_class"], dtype=np.int8)
        metadata = appended_metadata(
            _metadata(parent), parent_path=parent_path,
            fragment_path=fragment_path, corridor_rows=len(corridor_inputs),
            tech=tech, device=device, variant=variant, length=length,
            nfin=nfin, temperature=temperature, source_commit=source_commit,
            generator_command=generator_command,
        )

    inputs = np.concatenate([parent_inputs, corridor_inputs], axis=0)
    geometry = np.concatenate([parent_geometry, corridor_geometry], axis=0)
    outputs = np.concatenate([parent_outputs, corridor_outputs], axis=0)
    classes = np.concatenate([
        parent_classes,
        np.full(len(corridor_inputs), CORRIDOR_CODE, dtype=np.int8),
    ])
    parent_labels = get_or_build_tech_variant_labels(
        str(parent_path), device, verbose=True,
    )
    labels = np.concatenate([
        np.asarray(parent_labels, dtype=np.int64),
        np.full(
            len(corridor_inputs), tech_variant_to_code(tech, variant),
            dtype=np.int64,
        ),
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f"{output_path.stem}.{os.getpid()}.partial.npz"
    )
    save_npz(
        inputs, geometry, outputs, temporary,
        metadata=metadata, sample_class=classes,
    )
    os.replace(temporary, output_path)
    label_path = output_path.with_name(
        output_path.stem + "_tech_variant_labels.npy"
    )
    label_tmp = label_path.with_name(
        f"{label_path.stem}.{os.getpid()}.partial.npy"
    )
    np.save(label_tmp, labels)
    os.replace(label_tmp, label_path)
    write_sidecar_meta(output_path, geometry, labels)
    _write_completion_marker(
        output_path, rows=len(inputs), source_commit=source_commit,
    )
    validate_canonical_dataset(output_path)
    print(
        f"  Canonical full-terminal corridor dataset: {output_path} "
        f"({len(parent_inputs)} + {len(corridor_inputs)} rows)"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Append an OSDI truth trajectory to a full-terminal dataset"
    )
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    command = shlex.join([sys.executable, *sys.argv])
    append_corridor(
        args.parent, args.fragment, args.output,
        generator_command=command,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
