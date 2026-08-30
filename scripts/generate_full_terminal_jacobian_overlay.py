#!/usr/bin/env python3
"""Build a generic OSDI Jacobian overlay from terminal-length DNF rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_compact_models"))
sys.path.insert(0, str(ROOT / "external_compact_models" / "bsim_cmg"))

from neural_network.config import (  # noqa: E402
    CODE_TO_TECH_VARIANT,
    local_variant_code,
)
from neural_network.eval.loo_labels import (  # noqa: E402
    get_or_build_tech_variant_labels,
)
from pycmg.nn_config import TECH_CONFIGS  # noqa: E402
from pycmg.nn_generate import _create_model_and_instance  # noqa: E402

OUTPUT_COLUMNS = ("i_d", "i_g", "i_b", "qd", "qg", "qb")
TERMINAL_ROWS = (0, 1, 3)
VOLTAGE_COLUMNS = (0, 1, 3)


@dataclass(frozen=True)
class _BinJob:
    tech: str
    device: str
    variant: str
    length: float
    nfin: float
    temperature: float
    inputs: np.ndarray
    expected_outputs: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _terminal_values(result: dict[str, float]) -> np.ndarray:
    return np.asarray([
        -result["id"], -result["ig"], -result["ie"],
        result["qd"], result["qg"], result["qb"],
    ], dtype=np.float64)


def _evaluate(
    instance: Any,
    point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    result = instance.eval_dc({
        "d": float(point[0]), "g": float(point[1]),
        "s": 0.0, "e": float(point[3]),
    })
    values = _terminal_values(result)
    current_jacobian = -instance.condense_last_jacobian()[
        np.ix_(TERMINAL_ROWS, VOLTAGE_COLUMNS)]
    charge_jacobian = instance.condense_last_react()[
        np.ix_(TERMINAL_ROWS, VOLTAGE_COLUMNS)]
    jacobian = np.concatenate(
        [current_jacobian, charge_jacobian], axis=0).astype(np.float64)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(jacobian)):
        raise RuntimeError("OSDI overlay evaluation produced NaN/Inf")
    if np.max(np.abs(values[:3])) > 1.0:
        raise RuntimeError("OSDI overlay terminal current exceeds 1 A")
    return values, jacobian


def _finite_difference_error(
    instance: Any,
    point: np.ndarray,
    analytical: np.ndarray,
) -> float:
    """Check the current-label orientation against central differences."""
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
            y_forward[:3] - y_backward[:3]) / (2.0 * step)
    scale = np.maximum.reduce([
        np.abs(numerical), np.abs(analytical[:3]),
        np.full_like(numerical, 1e-8),
    ])
    return float(np.max(np.abs(numerical - analytical[:3]) / scale))


def _run_bin(job: _BinJob) -> dict[str, Any]:
    tech = TECH_CONFIGS[job.tech]
    built = _create_model_and_instance(
        tech, job.device, job.variant, job.length, job.nfin,
        job.temperature,
    )
    if built is None:
        raise RuntimeError(f"failed to build overlay bin {job}")
    _model, instance, _process = built
    values = np.empty((len(job.inputs), 6), dtype=np.float64)
    jacobians = np.empty((len(job.inputs), 6, 3), dtype=np.float64)
    for index, point in enumerate(job.inputs):
        values[index], jacobians[index] = _evaluate(instance, point)

    # Warm-state iteration can perturb only solver-tolerance last bits. A
    # material mismatch means the overlay no longer describes its parent.
    if not np.allclose(
        values, job.expected_outputs, rtol=2e-6, atol=1e-15,
    ):
        worst = float(np.max(np.abs(values - job.expected_outputs)))
        raise RuntimeError(f"overlay values disagree with parent; max={worst:g}")
    strong_index = int(np.argmax(np.max(np.abs(values[:,:3]), axis=1)))
    fd_error = _finite_difference_error(
        instance, job.inputs[strong_index], jacobians[strong_index])
    if fd_error > 0.05:
        raise RuntimeError(
            f"current Jacobian sign/scale check failed: {fd_error:.3%}")
    code = local_variant_code(job.tech, job.tech, job.variant)
    return {
        "inputs": job.inputs,
        "outputs": values,
        "jacobians": jacobians,
        "tech_codes": np.full(len(values), code, dtype=np.int64),
        "variant": np.full(len(values), job.variant),
        "length": np.full(len(values), job.length, dtype=np.float64),
        "nfin": np.full(len(values), job.nfin, dtype=np.float64),
        "temperature": np.full(len(values), job.temperature, dtype=np.float64),
        "fd_error": fd_error,
    }


def _balanced_indices(
    indices: np.ndarray,
    sample_class: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    """Select deterministic per-class rows without replacement."""
    rng = np.random.default_rng(seed)
    groups = [indices[sample_class[indices] == code]
              for code in np.unique(sample_class[indices])]
    chosen: list[np.ndarray] = []
    remaining = count
    while remaining and any(len(group) for group in groups):
        active = [group for group in groups if len(group)]
        take_each = max(1, remaining // len(active))
        next_groups: list[np.ndarray] = []
        for group in groups:
            take = min(len(group), take_each, remaining)
            if take:
                pick = rng.choice(group, size=take, replace=False)
                chosen.append(np.asarray(pick, dtype=np.int64))
                remaining -= take
                keep = ~np.isin(group, pick, assume_unique=False)
                group = group[keep]
            next_groups.append(group)
        groups = next_groups
    return np.sort(np.concatenate(chosen))


def _jobs(
    dataset_path: Path,
    device: str,
    samples_per_bin: int,
    seed: int,
) -> list[_BinJob]:
    labels = get_or_build_tech_variant_labels(
        str(dataset_path), device, verbose=False)
    with np.load(dataset_path, allow_pickle=False) as data:
        inputs = np.asarray(data["inputs"])
        geometry = np.asarray(data["geometry"])
        outputs = np.asarray(data["outputs"])
        sample_class = np.asarray(data["sample_class"])
    terminal = geometry[:, 1] == np.max(geometry[:, 1])
    keys = np.column_stack([
        labels, geometry[:, 0], geometry[:, 1], geometry[:, 2],
    ])
    jobs: list[_BinJob] = []
    for ordinal, key in enumerate(np.unique(keys[terminal], axis=0)):
        code, nfin, length, temperature = key
        tech, variant = CODE_TO_TECH_VARIANT[int(code)]
        mask = terminal & np.all(keys == key, axis=1)
        candidates = np.flatnonzero(mask)
        selected = _balanced_indices(
            candidates, sample_class, min(samples_per_bin, len(candidates)),
            seed + ordinal,
        )
        jobs.append(_BinJob(
            tech=tech, device=device, variant=variant,
            length=float(length), nfin=float(nfin),
            temperature=float(temperature), inputs=inputs[selected],
            expected_outputs=outputs[selected],
        ))
    return jobs


def _concatenate(parts: Iterable[dict[str, Any]], name: str) -> np.ndarray:
    return np.concatenate([np.asarray(part[name]) for part in parts], axis=0)


def generate(
    dataset_path: Path,
    output_path: Path,
    device: str,
    samples_per_bin: int,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    source_commit, source_dirty = _source_identity()
    if source_dirty:
        raise RuntimeError("overlay generator source has tracked changes")
    jobs = _jobs(dataset_path, device, samples_per_bin, seed)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        parts = list(executor.map(_run_bin, jobs))
    arrays = {
        name: _concatenate(parts, name)
        for name in (
            "inputs", "outputs", "jacobians", "tech_codes", "variant",
            "length", "nfin", "temperature",
        )
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path, **arrays,
        meta_output_columns=np.asarray(OUTPUT_COLUMNS),
        meta_jacobian_voltage_columns=np.asarray(("Vd", "Vg", "Vb")),
        meta_parent_dataset=np.asarray(dataset_path.name),
        meta_parent_dataset_sha256=np.asarray(_sha256(dataset_path)),
        meta_source_commit=np.asarray(source_commit),
        meta_source_dirty=np.asarray(False),
        meta_seed=np.asarray(seed),
        meta_samples_per_bin=np.asarray(samples_per_bin),
        meta_bins=np.asarray(len(jobs)),
        meta_fd_max_scaled_error=np.asarray(
            max(float(part["fd_error"]) for part in parts)),
    )
    marker = {
        "artifact": output_path.name,
        "artifact_sha256": _sha256(output_path),
        "rows": len(arrays["outputs"]),
        "bins": len(jobs),
        "source_commit": source_commit,
        "parent_dataset_sha256": _sha256(dataset_path),
        "fd_max_scaled_error": max(
            float(part["fd_error"]) for part in parts),
    }
    output_path.with_suffix(output_path.suffix + ".complete").write_text(
        json.dumps(marker, sort_keys=True, indent=2) + "\n")
    return marker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("nmos", "pmos"), required=True)
    parser.add_argument("--samples-per-bin", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = generate(
        args.dataset.resolve(), args.output.resolve(), args.device,
        args.samples_per_bin, args.seed, args.workers,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
