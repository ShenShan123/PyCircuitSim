#!/usr/bin/env python3
"""Generate an all-geometry, frozen-parent active Hermite overlay.

The artifact keeps all queried candidates.  ``role`` identifies the 16
point-disjoint validation rows and 16 largest-parent-error training rows in
each geometry bin; the remaining candidates stay in the artifact for audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for import_root in (
    ROOT,
    ROOT / "external_compact_models",
    ROOT / "external_compact_models" / "bsim_cmg",
):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from neural_network.config import (  # noqa: E402
    CODE_TO_TECH_VARIANT,
    local_variant_code,
)
from neural_network.eval.loo_labels import (  # noqa: E402
    get_or_build_tech_variant_labels,
)
from pycircuitsim.models.mosfet_directnet_full import (  # noqa: E402
    _load_artifacts,
)
from pycmg.nn_config import TECH_CONFIGS  # noqa: E402
from pycmg.nn_generate import _create_model_and_instance  # noqa: E402


OUTPUT_COLUMNS = ("i_d", "i_g", "i_b", "qd", "qg", "qb")
TERMINAL_ROWS = (0, 1, 3)
VOLTAGE_COLUMNS = (0, 1, 3)
ROLE_UNUSED = np.int8(0)
ROLE_ACTIVE = np.int8(1)
ROLE_VALIDATION = np.int8(2)


@dataclass(frozen=True)
class BinJob:
    """One independently evaluable geometry-bin query."""

    bin_id: int
    tech: str
    device: str
    variant: str
    length: float
    nfin: float
    temperature: float
    source_rows: np.ndarray
    sample_class: np.ndarray
    inputs: np.ndarray
    expected_outputs: np.ndarray
    verify_fd: bool


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 for a provenance-bound file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coordinate_seed(
    base_seed: int,
    domain: str,
    tech: str,
    device: str,
    variant: str,
    nfin: float,
    length: float,
    temperature: float,
) -> int:
    """Hash exact bin coordinates into an order-independent NumPy seed."""
    payload = "\x1f".join((
        str(int(base_seed)), domain, tech, device, variant,
        float(nfin).hex(), float(length).hex(), float(temperature).hex(),
    )).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _derived_seed(base_seed: int, domain: str, value: int) -> int:
    payload = f"{int(base_seed)}\x1f{domain}\x1f{int(value)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def balanced_indices(
    indices: np.ndarray,
    sample_class: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    """Choose an exact, prefix-balanced set across available sample classes."""
    rows = np.asarray(indices, dtype=np.int64)
    if count < 0 or count > len(rows):
        raise ValueError(f"cannot choose {count} rows from {len(rows)}")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    groups = [
        rng.permutation(rows[np.asarray(sample_class)[rows] == code])
        for code in np.unique(np.asarray(sample_class)[rows])
    ]
    if not groups:
        raise ValueError("balanced selection has no sample classes")
    offsets = np.zeros(len(groups), dtype=np.int64)
    order: list[int] = []
    while len(order) < count:
        active = np.flatnonzero([
            offsets[index] < len(group) for index, group in enumerate(groups)
        ])
        if not len(active):
            raise ValueError("balanced selection exhausted its input rows")
        for group_index in rng.permutation(active):
            index = int(group_index)
            order.append(int(groups[index][offsets[index]]))
            offsets[index] += 1
            if len(order) == count:
                break
    return np.asarray(order, dtype=np.int64)


def assign_roles(
    source_rows: np.ndarray,
    validation_order: np.ndarray,
    scores: np.ndarray,
    *,
    validation_count: int,
    active_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reserve validation first, then rank remaining rows by score and row ID."""
    rows = np.asarray(source_rows, dtype=np.int64)
    validation_order = np.asarray(validation_order, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(rows) != len(scores) or set(rows) != set(validation_order):
        raise ValueError("role assignment inputs do not name the same rows")
    if validation_count + active_count > len(rows):
        raise ValueError("role counts exceed queried candidates")
    roles = np.full(len(rows), ROLE_UNUSED, dtype=np.int8)
    ranks = np.zeros(len(rows), dtype=np.int16)
    row_to_position = {int(row): index for index, row in enumerate(rows)}
    validation_rows = validation_order[:validation_count]
    for row in validation_rows:
        roles[row_to_position[int(row)]] = ROLE_VALIDATION
    remaining_positions = np.flatnonzero(roles == ROLE_UNUSED)
    remaining_rows = rows[remaining_positions]
    order = np.lexsort((remaining_rows, -scores[remaining_positions]))
    active_positions = remaining_positions[order[:active_count]]
    roles[active_positions] = ROLE_ACTIVE
    ranks[active_positions] = np.arange(1, active_count + 1, dtype=np.int16)
    return roles, ranks


def fixed_replay_indices(
    bin_ids: np.ndarray,
    sample_class: np.ndarray,
    excluded: np.ndarray,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    """Choose a fixed equal-bin, class-balanced replay set without leakage."""
    bins = np.asarray(bin_ids, dtype=np.int32)
    classes = np.asarray(sample_class, dtype=np.int8)
    if len(bins) != len(classes):
        raise ValueError("bin IDs and sample classes must align")
    if count < 0 or count > len(bins) - len(np.unique(excluded)):
        raise ValueError("replay count exceeds available non-candidate rows")
    blocked = np.zeros(len(bins), dtype=bool)
    blocked[np.asarray(excluded, dtype=np.int64)] = True
    order = np.argsort(bins, kind="stable")
    ordered_bins = bins[order]
    unique_bins, starts, sizes = np.unique(
        ordered_bins, return_index=True, return_counts=True,
    )
    allocation = np.full(len(unique_bins), count // len(unique_bins), dtype=int)
    remainder = count % len(unique_bins)
    bin_order = np.random.default_rng(
        _derived_seed(seed, "replay-allocation", len(unique_bins)),
    ).permutation(len(unique_bins))
    allocation[bin_order[:remainder]] += 1
    selected: list[np.ndarray] = []
    for ordinal, (bin_id, start, size, take) in enumerate(zip(
        unique_bins, starts, sizes, allocation,
    )):
        candidates = order[start:start + size]
        candidates = candidates[~blocked[candidates]]
        if take > len(candidates):
            raise ValueError(
                f"bin {int(bin_id)} has {len(candidates)} replay rows; "
                f"{int(take)} required"
            )
        selected.append(balanced_indices(
            candidates, classes, int(take),
            _derived_seed(seed, "replay-bin", int(bin_id)),
        ))
    replay = np.concatenate(selected).astype(np.int64, copy=False)
    rng = np.random.default_rng(_derived_seed(seed, "replay-order", count))
    replay = rng.permutation(replay)
    if len(replay) != count or len(np.unique(replay)) != count:
        raise RuntimeError("replay selection did not produce exact unique rows")
    if np.any(blocked[replay]):
        raise RuntimeError("replay selection leaked queried candidates")
    return replay


def expected_plan_counts(
    bins: int,
    candidates_per_bin: int,
    validation_per_bin: int,
    active_per_bin: int,
) -> dict[str, int]:
    """Return the fixed experiment denominator implied by the bin plan."""
    if validation_per_bin + active_per_bin > candidates_per_bin:
        raise ValueError("active and validation rows exceed candidates")
    return {
        "bins": bins,
        "queried_rows": bins * candidates_per_bin,
        "active_rows": bins * active_per_bin,
        "validation_rows": bins * validation_per_bin,
        "unused_rows": bins * (
            candidates_per_bin - validation_per_bin - active_per_bin
        ),
    }


def head_balanced_scores(
    per_head_error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average dimensionless within-head errors for active selection."""
    errors = np.asarray(per_head_error, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[1] != 3:
        raise ValueError("current-J errors must have three current heads")
    if not np.all(np.isfinite(errors)) or np.any(errors < 0.0):
        raise ValueError("current-J errors must be finite and non-negative")
    scale = errors.mean(axis=0)
    safe_scale = scale.copy()
    safe_scale[safe_scale <= 0.0] = 1.0
    return np.mean(errors / safe_scale[None, :], axis=1), scale


def terminal_values(result: dict[str, float]) -> np.ndarray:
    """Convert OSDI values to the six independent LEVEL=75 surfaces."""
    required = ("id", "ig", "is", "ie", "qd", "qg", "qs", "qb")
    values = np.asarray([float(result[name]) for name in required])
    if not np.all(np.isfinite(values)):
        raise RuntimeError("OSDI Hermite evaluation produced NaN/Inf")
    currents = np.asarray([-result["id"], -result["ig"], -result["ie"]])
    if np.max(np.abs(np.asarray([-result["id"], -result["ig"],
                                 -result["is"], -result["ie"]]))) > 1.0:
        raise RuntimeError("OSDI Hermite terminal current exceeds 1 A")
    return np.asarray([
        *currents, result["qd"], result["qg"], result["qb"],
    ], dtype=np.float64)


def terminal_jacobian(instance: Any) -> np.ndarray:
    """Return solver-positive value derivatives for (Vd, Vg, Vb)."""
    current = -np.asarray(instance.condense_last_jacobian())[
        np.ix_(TERMINAL_ROWS, VOLTAGE_COLUMNS)
    ]
    charge = np.asarray(instance.condense_last_react())[
        np.ix_(TERMINAL_ROWS, VOLTAGE_COLUMNS)
    ]
    result = np.concatenate((current, charge), axis=0).astype(np.float64)
    if result.shape != (6, 3) or not np.all(np.isfinite(result)):
        raise RuntimeError("OSDI Hermite Jacobian is invalid")
    return result


def _evaluate(instance: Any, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result = instance.eval_dc({
        "d": float(point[0]), "g": float(point[1]),
        "s": 0.0, "e": float(point[3]),
    })
    return terminal_values(result), terminal_jacobian(instance)


def _finite_difference_error(
    instance: Any,
    point: np.ndarray,
    analytical: np.ndarray,
) -> float:
    """Return worst error/tolerance under the NGSPICE-backed current-J gate."""
    step = 1e-6
    numerical = np.empty((3, 3), dtype=np.float64)
    for target_column, input_column in enumerate(VOLTAGE_COLUMNS):
        forward = point.copy()
        backward = point.copy()
        forward[input_column] += step
        backward[input_column] -= step
        y_forward, _ = _evaluate(instance, forward)
        y_backward, _ = _evaluate(instance, backward)
        numerical[:, target_column] = (
            y_forward[:3] - y_backward[:3]
        ) / (2.0 * step)
    tolerance = 1e-6 + 1e-2 * np.maximum(
        np.abs(numerical), np.abs(analytical[:3]),
    )
    return float(np.max(np.abs(numerical - analytical[:3]) / tolerance))


def _run_bin(job: BinJob) -> dict[str, Any]:
    tech = TECH_CONFIGS[job.tech]
    built = _create_model_and_instance(
        tech, job.device, job.variant, job.length, job.nfin,
        job.temperature,
    )
    if built is None:
        raise RuntimeError(f"failed to build Hermite bin {job.bin_id}")
    _model, instance, _process = built
    values = np.empty((len(job.inputs), 6), dtype=np.float64)
    jacobians = np.empty((len(job.inputs), 6, 3), dtype=np.float64)
    for index, point in enumerate(job.inputs):
        values[index], jacobians[index] = _evaluate(instance, point)
    if not np.allclose(
        values[:, :3], job.expected_outputs[:, :3], rtol=2e-6, atol=1e-15,
    ):
        raise RuntimeError(f"Hermite current values disagree in bin {job.bin_id}")
    if not np.allclose(
        values[:, 3:], job.expected_outputs[:, 3:], rtol=2e-6, atol=1e-24,
    ):
        raise RuntimeError(f"Hermite charge values disagree in bin {job.bin_id}")
    fd_error = np.nan
    if job.verify_fd:
        strong = int(np.argmax(np.max(np.abs(values[:, :3]), axis=1)))
        fd_error = _finite_difference_error(
            instance, job.inputs[strong], jacobians[strong],
        )
        if fd_error > 1.0:
            raise RuntimeError(
                f"Hermite current-J sign/scale failed in bin {job.bin_id}: "
                f"{fd_error:.3f}x tolerance"
            )
    return {
        "bin_id": np.full(len(values), job.bin_id, dtype=np.int32),
        "inputs": job.inputs,
        "outputs": values,
        "jacobians": jacobians,
        "source_rows": job.source_rows,
        "sample_class": job.sample_class,
        "tech_codes": np.full(
            len(values), local_variant_code(job.tech, job.tech, job.variant),
            dtype=np.int64,
        ),
        "variant": np.full(len(values), job.variant),
        "length": np.full(len(values), job.length, dtype=np.float64),
        "nfin": np.full(len(values), job.nfin, dtype=np.float64),
        "temperature": np.full(
            len(values), job.temperature, dtype=np.float64,
        ),
        "fd_error": fd_error,
    }


def _source_identity() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return commit, bool(dirty)


def _unique_coordinate_rows(
    rows: np.ndarray,
    inputs: np.ndarray,
) -> np.ndarray:
    """Keep the smallest source row for every exact voltage coordinate."""
    local = np.asarray(inputs[np.asarray(rows, dtype=np.int64)], dtype=np.float64)
    _unique, first = np.unique(local, axis=0, return_index=True)
    return np.asarray(rows, dtype=np.int64)[np.sort(first)]


def _load_dataset(
    dataset_path: Path,
    device: str,
) -> dict[str, np.ndarray]:
    labels = get_or_build_tech_variant_labels(
        str(dataset_path), device, verbose=False,
    )
    with np.load(dataset_path, allow_pickle=False) as data:
        arrays = {
            "inputs": np.asarray(data["inputs"]),
            "geometry": np.asarray(data["geometry"]),
            "outputs": np.asarray(data["outputs"]),
            "sample_class": np.asarray(data["sample_class"], dtype=np.int8),
        }
        for name in (
            "meta_osdi_sha256", "meta_modelcard_sha256_json",
            "meta_source_commit", "meta_generator_release",
        ):
            if name in data.files:
                arrays[name] = np.asarray(data[name])
    arrays["labels"] = np.asarray(labels, dtype=np.int64)
    geometry = arrays["geometry"]
    keys = np.column_stack((arrays["labels"], geometry[:, :3]))
    unique_keys, bin_ids = np.unique(keys, axis=0, return_inverse=True)
    arrays["bin_keys"] = unique_keys
    arrays["bin_ids"] = bin_ids.astype(np.int32)
    return arrays


def _build_jobs(
    arrays: dict[str, np.ndarray],
    device: str,
    *,
    expected_bins: int,
    candidates_per_bin: int,
    seed: int,
    fd_bins: int,
) -> list[BinJob]:
    keys = arrays["bin_keys"]
    if len(keys) != expected_bins:
        raise ValueError(f"expected {expected_bins} bins, found {len(keys)}")
    if np.any(keys[:, 1] < 2.0):
        raise ValueError("active Hermite input unexpectedly contains NFIN<2")
    order = np.argsort(arrays["bin_ids"], kind="stable")
    ordered_bins = arrays["bin_ids"][order]
    _ids, starts, sizes = np.unique(
        ordered_bins, return_index=True, return_counts=True,
    )
    fd_order = np.random.default_rng(
        _derived_seed(seed, "fd-bins", len(keys)),
    ).permutation(len(keys))
    fd_selected = set(int(value) for value in fd_order[:fd_bins])
    jobs: list[BinJob] = []
    for bin_id, (key, start, size) in enumerate(zip(keys, starts, sizes)):
        code, nfin, length, temperature = key
        tech, variant = CODE_TO_TECH_VARIANT[int(code)]
        if tech != "tsmc5":
            raise ValueError(f"active Hermite scope is TSMC5, found {tech}")
        candidates = _unique_coordinate_rows(
            order[start:start + size], arrays["inputs"],
        )
        if len(candidates) < candidates_per_bin:
            raise ValueError(
                f"bin {bin_id} has {len(candidates)} unique coordinates; "
                f"{candidates_per_bin} required"
            )
        selected = balanced_indices(
            candidates, arrays["sample_class"], candidates_per_bin,
            coordinate_seed(
                seed, "candidates", tech, device, variant,
                float(nfin), float(length), float(temperature),
            ),
        )
        jobs.append(BinJob(
            bin_id=bin_id, tech=tech, device=device, variant=variant,
            length=float(length), nfin=float(nfin),
            temperature=float(temperature), source_rows=selected,
            sample_class=arrays["sample_class"][selected],
            inputs=arrays["inputs"][selected],
            expected_outputs=arrays["outputs"][selected],
            verify_fd=bin_id in fd_selected,
        ))
    return jobs


def _concatenate(parts: Iterable[dict[str, Any]], name: str) -> np.ndarray:
    return np.concatenate([np.asarray(part[name]) for part in parts], axis=0)


def _normalization_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(
        checkpoint.name.replace("_best.pt", "_norm.npz"),
    )


def _normalization_target_jacobian(
    jacobians: torch.Tensor,
    physical_outputs: torch.Tensor,
    stats: Any,
    device: torch.device,
) -> torch.Tensor:
    input_std = torch.as_tensor(
        np.asarray(stats.input_std)[list(VOLTAGE_COLUMNS)],
        dtype=torch.float32, device=device,
    )
    output_std = torch.as_tensor(
        np.asarray(stats.output_std)[:3], dtype=torch.float32, device=device,
    )
    if stats.mode == "asinh":
        if stats.asinh_scale is None:
            raise ValueError("asinh parent is missing output scales")
        scale = torch.as_tensor(
            np.asarray(stats.asinh_scale)[:3],
            dtype=torch.float32, device=device,
        )
        factor = torch.sqrt(scale[None, :] ** 2 + physical_outputs[:, :3] ** 2)
    elif stats.mode == "zscore":
        factor = torch.ones_like(physical_outputs[:, :3])
    else:
        raise ValueError(f"unsupported parent normalizer {stats.mode!r}")
    return (
        jacobians[:, :3, :]
        * input_std[None, None, :]
        / output_std[None, :, None]
        / factor[:, :, None]
    )


def _score_parent(
    checkpoint: Path,
    arrays: dict[str, np.ndarray],
    parts: Sequence[dict[str, Any]],
    *,
    device_name: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model, stats, _num_codes, output_columns = _load_artifacts(checkpoint)
    if tuple(output_columns) != OUTPUT_COLUMNS:
        raise ValueError("parent is not the six-surface DirectNet contract")
    device = torch.device(device_name)
    model = model.to(device).eval()
    inputs = _concatenate(parts, "inputs")
    outputs = _concatenate(parts, "outputs")
    jacobians = _concatenate(parts, "jacobians")
    codes = _concatenate(parts, "tech_codes")
    nfin = _concatenate(parts, "nfin")
    length = _concatenate(parts, "length")
    temperature = _concatenate(parts, "temperature")
    raw = np.column_stack((
        inputs[:, 0], inputs[:, 1], np.zeros(len(inputs)), inputs[:, 3],
        np.log2(np.maximum(nfin, 1.0)), length, temperature,
    ))
    scale = np.asarray(stats.input_std, dtype=np.float64).copy()
    scale[scale < 1e-12] = 1.0
    normalized = ((raw - np.asarray(stats.input_mean)) / scale).astype(
        np.float32,
    )
    per_head_parts: list[np.ndarray] = []
    for start in range(0, len(inputs), batch_size):
        stop = min(start + batch_size, len(inputs))
        x = torch.as_tensor(
            normalized[start:stop], device=device,
        ).detach().clone().requires_grad_(True)
        code = torch.as_tensor(codes[start:stop], device=device)
        prediction = model(x, tech_codes=code)
        predicted_j: list[torch.Tensor] = []
        for head in range(3):
            gradient = torch.autograd.grad(
                prediction[:, head].sum(), x,
                retain_graph=head < 2,
            )[0][:, list(VOLTAGE_COLUMNS)]
            predicted_j.append(gradient)
        predicted = torch.stack(predicted_j, dim=1)
        target = _normalization_target_jacobian(
            torch.as_tensor(
                jacobians[start:stop], dtype=torch.float32, device=device,
            ),
            torch.as_tensor(
                outputs[start:stop], dtype=torch.float32, device=device,
            ),
            stats, device,
        )
        per_head_parts.append(
            torch.mean(torch.abs(predicted - target), dim=2)
            .detach().cpu().numpy().astype(np.float64)
        )
    per_head = np.concatenate(per_head_parts, axis=0)
    return per_head.mean(axis=1), per_head


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON provenance file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON provenance object: {path}")
    return value


def _publish_overlay(
    output: Path,
    arrays: dict[str, np.ndarray],
    marker_fields: dict[str, Any],
    *,
    overwrite: bool,
) -> dict[str, Any]:
    """Atomically publish one guarded overlay artifact and completion marker."""
    marker_path = output.with_suffix(output.suffix + ".complete")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = marker_path.with_name(marker_path.name + ".lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"overlay publication is already active: {lock_path}") \
            from exc
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}.npz")
    marker_temporary = marker_path.with_name(
        marker_path.name + f".tmp-{os.getpid()}")
    output_backup = output.with_name(output.name + f".bak-{os.getpid()}")
    marker_backup = marker_path.with_name(
        marker_path.name + f".bak-{os.getpid()}")
    try:
        if (output.exists() or marker_path.exists()) and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite {output} or {marker_path}")
        if output_backup.exists() or marker_backup.exists():
            raise FileExistsError("stale overlay publication backup exists")
        np.savez(temporary, **arrays)
        marker = {
            "artifact": output.name,
            "artifact_sha256": sha256_file(temporary),
            **marker_fields,
        }
        marker_temporary.write_text(
            json.dumps(marker, sort_keys=True, indent=2) + "\n")
        moved_output = False
        moved_marker = False
        try:
            if output.exists():
                os.replace(output, output_backup)
                moved_output = True
            if marker_path.exists():
                os.replace(marker_path, marker_backup)
                moved_marker = True
        except BaseException:
            if moved_output:
                os.replace(output_backup, output)
            if moved_marker:
                os.replace(marker_backup, marker_path)
            raise
        try:
            os.replace(temporary, output)
            os.replace(marker_temporary, marker_path)
        except BaseException:
            output.unlink(missing_ok=True)
            marker_path.unlink(missing_ok=True)
            if moved_output:
                os.replace(output_backup, output)
            if moved_marker:
                os.replace(marker_backup, marker_path)
            raise
        output_backup.unlink(missing_ok=True)
        marker_backup.unlink(missing_ok=True)
        return marker
    finally:
        temporary.unlink(missing_ok=True)
        marker_temporary.unlink(missing_ok=True)
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    dataset_marker = dataset.with_suffix(dataset.suffix + ".complete")
    checkpoint_marker = Path(f"{checkpoint}.complete")
    norm = _normalization_path(checkpoint)
    marker_path = output.with_suffix(output.suffix + ".complete")
    overwrite = bool(getattr(args, "overwrite", False))
    if (output.exists() or marker_path.exists()) \
            and not args.plan_only and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {output} or {marker_path}")
    for path in (dataset, dataset_marker, checkpoint, checkpoint_marker, norm):
        if not path.is_file():
            raise FileNotFoundError(path)
    dataset_marker_data = _read_json(dataset_marker)
    checkpoint_marker_data = _read_json(checkpoint_marker)
    dataset_hash = sha256_file(dataset)
    if dataset_marker_data.get("dataset_sha256") != dataset_hash:
        raise ValueError("dataset completion marker checksum mismatch")
    if (checkpoint_marker_data.get("dataset_sha256") is not None
            and checkpoint_marker_data["dataset_sha256"] != dataset_hash):
        raise ValueError("parent checkpoint was trained on another dataset")
    _load_artifacts(checkpoint)
    arrays = _load_dataset(dataset, args.device)
    jobs = _build_jobs(
        arrays, args.device, expected_bins=args.expected_bins,
        candidates_per_bin=args.candidates_per_bin, seed=args.seed,
        fd_bins=args.fd_bins,
    )
    plan = expected_plan_counts(
        len(jobs), args.candidates_per_bin, args.validation_per_bin,
        args.active_per_bin,
    )
    queried_rows = np.concatenate([job.source_rows for job in jobs])
    replay = fixed_replay_indices(
        arrays["bin_ids"], arrays["sample_class"], queried_rows,
        count=args.replay_rows, seed=args.seed,
    )
    plan["replay_rows"] = len(replay)
    if args.plan_only:
        return {
            **plan,
            "dataset_sha256": dataset_hash,
            "checkpoint_sha256": sha256_file(checkpoint),
            "normalization_sha256": sha256_file(norm),
        }

    source_commit, source_dirty = _source_identity()
    if source_dirty:
        raise RuntimeError("Hermite generator source has tracked changes")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        parts = list(executor.map(_run_bin, jobs))
    if sum(len(part["outputs"]) for part in parts) != plan["queried_rows"]:
        raise RuntimeError("Hermite OSDI evaluation returned an incomplete set")
    raw_scores, per_head_scores = _score_parent(
        checkpoint, arrays, parts, device_name=args.score_device,
        batch_size=args.score_batch_size,
    )
    scores, score_head_scale = head_balanced_scores(per_head_scores)
    source_rows = _concatenate(parts, "source_rows")
    sample_class = arrays["sample_class"]
    roles = np.empty(len(source_rows), dtype=np.int8)
    ranks = np.empty(len(source_rows), dtype=np.int16)
    cursor = 0
    for job in jobs:
        stop = cursor + len(job.source_rows)
        validation_order = balanced_indices(
            job.source_rows, sample_class, len(job.source_rows),
            coordinate_seed(
                args.seed, "validation", job.tech, job.device, job.variant,
                job.nfin, job.length, job.temperature,
            ),
        )
        roles[cursor:stop], ranks[cursor:stop] = assign_roles(
            job.source_rows, validation_order, scores[cursor:stop],
            validation_count=args.validation_per_bin,
            active_count=args.active_per_bin,
        )
        cursor = stop
    if int(np.sum(roles == ROLE_ACTIVE)) != plan["active_rows"]:
        raise RuntimeError("active-role count does not match the plan")
    if int(np.sum(roles == ROLE_VALIDATION)) != plan["validation_rows"]:
        raise RuntimeError("validation-role count does not match the plan")

    arrays_out = {
        name: _concatenate(parts, name)
        for name in (
            "inputs", "outputs", "jacobians", "tech_codes", "variant",
            "length", "nfin", "temperature", "source_rows", "sample_class",
            "bin_id",
        )
    }
    fd_errors = np.asarray([
        float(part["fd_error"]) for part in parts
        if np.isfinite(float(part["fd_error"]))
    ])
    bin_keys = [
        {
            "tech": job.tech, "device": job.device, "variant": job.variant,
            "NFIN": job.nfin, "L": job.length,
            "temperature_k": job.temperature,
        }
        for job in jobs
    ]
    artifact_arrays = {
        **arrays_out,
        "role": roles,
        "active_rank": ranks,
        "active_score": scores,
        "active_score_head_scale": score_head_scale,
        "parent_current_j_error": raw_scores,
        "parent_current_j_error_by_head": per_head_scores,
        "replay_source_rows": replay,
        "meta_output_columns": np.asarray(OUTPUT_COLUMNS),
        "meta_jacobian_voltage_columns": np.asarray(("Vd", "Vg", "Vb")),
        "meta_role_names": np.asarray(("unused", "active", "validation")),
        "meta_bin_keys_json": np.asarray(json.dumps(bin_keys, sort_keys=True)),
        "meta_parent_dataset": np.asarray(dataset.name),
        "meta_parent_dataset_sha256": np.asarray(dataset_hash),
        "meta_parent_dataset_marker_sha256": np.asarray(
            sha256_file(dataset_marker)),
        "meta_parent_checkpoint": np.asarray(checkpoint.name),
        "meta_parent_checkpoint_sha256": np.asarray(sha256_file(checkpoint)),
        "meta_parent_checkpoint_marker_sha256": np.asarray(
            sha256_file(checkpoint_marker)),
        "meta_parent_normalization": np.asarray(norm.name),
        "meta_parent_normalization_sha256": np.asarray(sha256_file(norm)),
        "meta_source_commit": np.asarray(source_commit),
        "meta_source_dirty": np.asarray(False),
        "meta_seed": np.asarray(args.seed),
        "meta_candidates_per_bin": np.asarray(args.candidates_per_bin),
        "meta_validation_per_bin": np.asarray(args.validation_per_bin),
        "meta_active_per_bin": np.asarray(args.active_per_bin),
        "meta_active_score": np.asarray(
            "mean(error_head / frozen_candidate_mean_head)"),
        "meta_bins": np.asarray(len(jobs)),
        "meta_candidate_queries": np.asarray(plan["queried_rows"]),
        "meta_fd_verification_bins": np.asarray(len(fd_errors)),
        "meta_fd_extra_queries": np.asarray(len(fd_errors) * 6),
        "meta_fd_max_tolerance_ratio": np.asarray(
            float(np.max(fd_errors)) if len(fd_errors) else np.nan),
        **{
            f"meta_parent_dataset_{name.removeprefix('meta_')}": value
            for name, value in arrays.items() if name.startswith("meta_")
        },
    }
    marker_fields = {
        **plan,
        "source_commit": source_commit,
        "dataset": dataset.name,
        "dataset_sha256": dataset_hash,
        "dataset_completion_marker_sha256": sha256_file(dataset_marker),
        "parent_checkpoint": checkpoint.name,
        "parent_checkpoint_sha256": sha256_file(checkpoint),
        "parent_checkpoint_marker_sha256": sha256_file(checkpoint_marker),
        "parent_normalization": norm.name,
        "parent_normalization_sha256": sha256_file(norm),
        "active_score": "mean(error_head / frozen_candidate_mean_head)",
        "active_score_head_scale": score_head_scale.tolist(),
        "fd_verification_bins": len(fd_errors),
        "fd_extra_queries": len(fd_errors) * 6,
        "fd_max_tolerance_ratio": (
            float(np.max(fd_errors)) if len(fd_errors) else None
        ),
    }
    return _publish_overlay(
        output, artifact_arrays, marker_fields, overwrite=overwrite)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("nmos", "pmos"), required=True)
    parser.add_argument("--seed", type=int, default=766)
    parser.add_argument("--expected-bins", type=int, default=840)
    parser.add_argument("--candidates-per-bin", type=int, default=64)
    parser.add_argument("--validation-per-bin", type=int, default=16)
    parser.add_argument("--active-per-bin", type=int, default=16)
    parser.add_argument("--replay-rows", type=int, default=262_144)
    parser.add_argument("--fd-bins", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--score-device", default="cpu")
    parser.add_argument("--score-batch-size", type=int, default=2048)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = generate(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
