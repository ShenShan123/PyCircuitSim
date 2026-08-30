#!/usr/bin/env python3
"""Fine-tune DirectNet-Full with an active Hermite overlay and hard gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "external_compact_models"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from neural_network.config import (  # noqa: E402
    CODE_TO_TECH_VARIANT,
    local_variant_code,
)
from neural_network.data.normalize import normalizer_from_stats  # noqa: E402
from neural_network.eval.loo_labels import (  # noqa: E402
    get_or_build_tech_variant_labels,
)
from pycircuitsim.models.mosfet_directnet_full import (  # noqa: E402
    _load_artifacts,
)
from scripts.generate_active_hermite_overlay import (  # noqa: E402
    OUTPUT_COLUMNS,
    ROLE_ACTIVE,
    ROLE_VALIDATION,
    VOLTAGE_COLUMNS,
    _normalization_target_jacobian,
    sha256_file,
)


TensorOrNone = torch.Tensor | None


def _seed(base_seed: int, domain: str, value: int) -> int:
    payload = f"{int(base_seed)}\x1f{domain}\x1f{int(value)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def epoch_order(length: int, *, seed: int, epoch: int) -> np.ndarray:
    """Return one deterministic permutation that exposes every row once."""
    if length < 0:
        raise ValueError("epoch length cannot be negative")
    return np.random.default_rng(
        _seed(seed, "epoch-order", epoch),
    ).permutation(length)


def project_jacobian_gradients(
    value_gradients: Sequence[TensorOrNone],
    jacobian_gradients: Sequence[TensorOrNone],
) -> tuple[list[TensorOrNone], bool]:
    """Project a globally conflicting Jacobian gradient off the value one."""
    if len(value_gradients) != len(jacobian_gradients):
        raise ValueError("gradient lists must align")
    dot = None
    value_norm = None
    for value, jacobian in zip(value_gradients, jacobian_gradients):
        if value is None or jacobian is None:
            continue
        term_dot = torch.sum(value * jacobian)
        term_norm = torch.sum(value * value)
        dot = term_dot if dot is None else dot + term_dot
        value_norm = term_norm if value_norm is None else value_norm + term_norm
    if dot is None or value_norm is None or float(dot.detach()) >= 0.0:
        return [
            None if gradient is None else gradient.clone()
            for gradient in jacobian_gradients
        ], False
    coefficient = dot / torch.clamp(value_norm, min=1e-30)
    projected: list[TensorOrNone] = []
    for value, jacobian in zip(value_gradients, jacobian_gradients):
        if jacobian is None:
            projected.append(None)
        elif value is None:
            projected.append(jacobian.clone())
        else:
            projected.append(jacobian - coefficient * value)
    return projected, True


def combine_gradients(
    value_gradients: Sequence[TensorOrNone],
    jacobian_gradients: Sequence[TensorOrNone],
    *,
    lambda_jacobian: float,
) -> tuple[list[TensorOrNone], bool]:
    """Combine value and projected Jacobian gradients at the declared weight."""
    if lambda_jacobian < 0.0:
        raise ValueError("lambda_jacobian must be non-negative")
    projected, conflicted = project_jacobian_gradients(
        value_gradients, jacobian_gradients,
    )
    combined: list[TensorOrNone] = []
    for value, jacobian in zip(value_gradients, projected):
        if value is None and jacobian is None:
            combined.append(None)
        elif value is None:
            assert jacobian is not None
            combined.append(lambda_jacobian * jacobian)
        elif jacobian is None or lambda_jacobian == 0.0:
            combined.append(value.clone())
        else:
            combined.append(value + lambda_jacobian * jacobian)
    return combined, conflicted


def _within_ratio(candidate: float, baseline: float, ratio: float) -> bool:
    limit = ratio * baseline
    tolerance = abs(limit) * 1e-12 + np.finfo(np.float64).tiny
    return candidate <= limit + tolerance


def evaluate_feasibility(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    require_jacobian_improvement: bool,
    value_mae_ratio: float = 1.02,
    value_max_ratio: float = 1.05,
    jacobian_ratio: float = 0.75,
) -> dict[str, object]:
    """Apply per-head value limits and the treatment-only Jacobian gate."""
    failures: list[str] = []
    ratios: dict[str, object] = {}
    for split in ("replay", "hermite"):
        baseline_split = baseline[split]
        candidate_split = candidate[split]
        assert isinstance(baseline_split, dict)
        assert isinstance(candidate_split, dict)
        for metric, limit in (
            ("normalized_mae", value_mae_ratio),
            ("physical_max_abs", value_max_ratio),
        ):
            baseline_values = np.asarray(baseline_split[metric], dtype=float)
            candidate_values = np.asarray(candidate_split[metric], dtype=float)
            if baseline_values.shape != (6,) or candidate_values.shape != (6,):
                raise ValueError(f"{split}.{metric} must contain six heads")
            metric_ratios: list[float] = []
            for head, (base, current) in enumerate(zip(
                baseline_values, candidate_values,
            )):
                metric_ratios.append(
                    float(current / base) if base != 0.0
                    else (1.0 if current == 0.0 else float("inf"))
                )
                if not _within_ratio(float(current), float(base), limit):
                    failures.append(f"{split}.{metric}[{head}]")
            ratios[f"{split}.{metric}"] = metric_ratios
    baseline_j = float(baseline["current_jacobian_mae"])
    candidate_j = float(candidate["current_jacobian_mae"])
    ratios["current_jacobian_mae"] = (
        candidate_j / baseline_j if baseline_j != 0.0
        else (1.0 if candidate_j == 0.0 else float("inf"))
    )
    if require_jacobian_improvement and not _within_ratio(
        candidate_j, baseline_j, jacobian_ratio,
    ):
        failures.append("current_jacobian_mae")
    return {
        "feasible": not failures,
        "failures": failures,
        "ratios": ratios,
        "limits": {
            "value_normalized_mae": value_mae_ratio,
            "value_physical_max_abs": value_max_ratio,
            "current_jacobian_mae": (
                jacobian_ratio if require_jacobian_improvement else None
            ),
        },
    }


def common_feasible_epochs(
    nmos_epochs: Iterable[int],
    pmos_epochs: Iterable[int],
) -> list[int]:
    """Pair polarities only at synchronized predeclared epoch numbers."""
    return sorted(
        set(int(value) for value in nmos_epochs).intersection(
            int(value) for value in pmos_epochs
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return result


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


def _normalization_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(
        checkpoint.name.replace("_best.pt", "_norm.npz"),
    )


def _raw_inputs(
    inputs: np.ndarray,
    nfin: np.ndarray,
    length: np.ndarray,
    temperature: np.ndarray,
) -> np.ndarray:
    return np.column_stack((
        inputs[:, 0], inputs[:, 1], np.zeros(len(inputs)), inputs[:, 3],
        np.log2(np.maximum(nfin, 1.0)), length, temperature,
    )).astype(np.float64)


def _normalize_inputs(raw: np.ndarray, stats: Any) -> np.ndarray:
    scale = np.asarray(stats.input_std, dtype=np.float64).copy()
    scale[scale < 1e-12] = 1.0
    return ((raw - np.asarray(stats.input_mean)) / scale).astype(np.float32)


def _normalize_outputs(outputs: np.ndarray, stats: Any) -> np.ndarray:
    normalizer = normalizer_from_stats(stats)
    return normalizer.normalize_outputs(outputs).astype(np.float32)


def _to_tensors(
    arrays: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, value in arrays.items():
        if name == "code":
            result[name] = torch.as_tensor(value, dtype=torch.long, device=device)
        else:
            result[name] = torch.as_tensor(
                value, dtype=torch.float32, device=device,
            )
    return result


def _verify_overlay(
    overlay_path: Path,
    checkpoint: Path,
    replay_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    marker_path = overlay_path.with_suffix(overlay_path.suffix + ".complete")
    if not marker_path.is_file():
        raise FileNotFoundError(marker_path)
    marker = _read_json(marker_path)
    if marker.get("artifact") != overlay_path.name:
        raise ValueError("Hermite marker names another artifact")
    if marker.get("artifact_sha256") != sha256_file(overlay_path):
        raise ValueError("Hermite artifact checksum mismatch")
    norm_path = _normalization_path(checkpoint)
    expected = {
        "parent_checkpoint_sha256": sha256_file(checkpoint),
        "parent_normalization_sha256": sha256_file(norm_path),
        "dataset_sha256": sha256_file(replay_path),
    }
    for name, value in expected.items():
        if marker.get(name) != value:
            raise ValueError(f"Hermite provenance mismatch for {name}")
    with np.load(overlay_path, allow_pickle=False) as overlay:
        arrays = {name: np.asarray(overlay[name]) for name in overlay.files}
    role = arrays["role"]
    bin_id = arrays["bin_id"]
    source_rows = arrays["source_rows"]
    replay_rows = arrays["replay_source_rows"]
    if len(role) != int(marker["queried_rows"]):
        raise ValueError("Hermite queried-row count mismatch")
    if len(np.unique(source_rows)) != len(source_rows):
        raise ValueError("Hermite source rows are not unique")
    if len(np.unique(replay_rows)) != len(replay_rows):
        raise ValueError("Hermite replay rows are not unique")
    if np.intersect1d(source_rows, replay_rows).size:
        raise ValueError("Hermite replay leaks queried candidates")
    for current_bin in np.unique(bin_id):
        mask = bin_id == current_bin
        if int(np.sum(role[mask] == ROLE_ACTIVE)) != 16:
            raise ValueError(f"bin {int(current_bin)} lacks 16 active rows")
        if int(np.sum(role[mask] == ROLE_VALIDATION)) != 16:
            raise ValueError(f"bin {int(current_bin)} lacks 16 validation rows")
    return marker, arrays


def _load_data(
    overlay_path: Path,
    replay_path: Path,
    checkpoint: Path,
    device_type: str,
    stats: Any,
    device: torch.device,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, Any],
]:
    marker, overlay = _verify_overlay(overlay_path, checkpoint, replay_path)
    role = overlay["role"]
    raw_overlay = _raw_inputs(
        overlay["inputs"], overlay["nfin"], overlay["length"],
        overlay["temperature"],
    )

    def overlay_split(selected_role: np.int8) -> dict[str, torch.Tensor]:
        mask = role == selected_role
        arrays = {
            "x": _normalize_inputs(raw_overlay[mask], stats),
            "y": _normalize_outputs(overlay["outputs"][mask], stats),
            "physical_y": overlay["outputs"][mask].astype(np.float32),
            "jacobian": overlay["jacobians"][mask].astype(np.float32),
            "code": overlay["tech_codes"][mask].astype(np.int64),
        }
        return _to_tensors(arrays, device)

    replay_rows = overlay["replay_source_rows"].astype(np.int64)
    labels = get_or_build_tech_variant_labels(
        str(replay_path), device_type, verbose=False,
    )
    with np.load(replay_path, allow_pickle=False) as replay_data:
        inputs = np.asarray(replay_data["inputs"])[replay_rows]
        geometry = np.asarray(replay_data["geometry"])[replay_rows, :3]
        outputs = np.asarray(replay_data["outputs"])[replay_rows]
    codes = np.asarray([
        local_variant_code("tsmc5", *CODE_TO_TECH_VARIANT[int(code)])
        for code in np.asarray(labels)[replay_rows]
    ], dtype=np.int64)
    replay = _to_tensors({
        "x": _normalize_inputs(
            _raw_inputs(
                inputs, geometry[:, 0], geometry[:, 1], geometry[:, 2],
            ),
            stats,
        ),
        "y": _normalize_outputs(outputs, stats),
        "physical_y": outputs.astype(np.float32),
        "code": codes,
    }, device)
    return (
        overlay_split(ROLE_ACTIVE), overlay_split(ROLE_VALIDATION),
        replay, marker,
    )


def _physical_predictions(
    normalized: torch.Tensor,
    stats: Any,
) -> np.ndarray:
    return normalizer_from_stats(stats).denormalize_outputs(
        normalized.detach().cpu().numpy(),
    )


def _value_metrics(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    stats: Any,
    batch_size: int,
) -> dict[str, list[float]]:
    model.eval()
    absolute_sum = np.zeros(6, dtype=np.float64)
    physical_max = np.zeros(6, dtype=np.float64)
    rows = 0
    with torch.no_grad():
        for start in range(0, len(data["x"]), batch_size):
            stop = min(start + batch_size, len(data["x"]))
            prediction = model(
                data["x"][start:stop], tech_codes=data["code"][start:stop],
            )
            error = torch.abs(prediction - data["y"][start:stop])
            absolute_sum += error.sum(dim=0).cpu().numpy()
            physical_prediction = _physical_predictions(prediction, stats)
            physical_true = data["physical_y"][start:stop].cpu().numpy()
            physical_max = np.maximum(
                physical_max,
                np.max(np.abs(physical_prediction - physical_true), axis=0),
            )
            rows += stop - start
    return {
        "normalized_mae": (absolute_sum / rows).tolist(),
        "physical_max_abs": physical_max.tolist(),
    }


def _jacobian_loss(
    prediction: torch.Tensor,
    x: torch.Tensor,
    physical_outputs: torch.Tensor,
    target: torch.Tensor,
    stats: Any,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted: list[torch.Tensor] = []
    for head in range(3):
        predicted.append(torch.autograd.grad(
            prediction[:, head].sum(), x,
            create_graph=create_graph,
            retain_graph=(create_graph or head < 2),
        )[0][:, list(VOLTAGE_COLUMNS)])
    predicted_tensor = torch.stack(predicted, dim=1)
    normalized_target = _normalization_target_jacobian(
        target, physical_outputs, stats, x.device,
    )
    per_head = torch.mean(
        torch.abs(predicted_tensor - normalized_target), dim=(0, 2),
    )
    return per_head.mean(), per_head


def _jacobian_metrics(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    stats: Any,
    batch_size: int,
) -> tuple[float, list[float]]:
    model.eval()
    total = np.zeros(3, dtype=np.float64)
    rows = 0
    for start in range(0, len(data["x"]), batch_size):
        stop = min(start + batch_size, len(data["x"]))
        x = data["x"][start:stop].detach().clone().requires_grad_(True)
        prediction = model(x, tech_codes=data["code"][start:stop])
        _loss, per_head = _jacobian_loss(
            prediction, x, data["physical_y"][start:stop],
            data["jacobian"][start:stop], stats, create_graph=False,
        )
        count = stop - start
        total += per_head.detach().cpu().numpy() * count
        rows += count
    means = total / rows
    return float(means.mean()), means.tolist()


def _model_metrics(
    model: torch.nn.Module,
    validation: dict[str, torch.Tensor],
    replay: dict[str, torch.Tensor],
    stats: Any,
    value_batch_size: int,
    jacobian_batch_size: int,
) -> dict[str, object]:
    jacobian, per_head = _jacobian_metrics(
        model, validation, stats, jacobian_batch_size,
    )
    return {
        "replay": _value_metrics(model, replay, stats, value_batch_size),
        "hermite": _value_metrics(
            model, validation, stats, value_batch_size,
        ),
        "current_jacobian_mae": jacobian,
        "current_jacobian_mae_by_head": per_head,
    }


def _batches(order: np.ndarray, size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(order), size):
        yield order[start:start + size]


def _save_feasible_checkpoint(
    model: torch.nn.Module,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, object],
    gate: dict[str, object],
    source_commit: str,
    overlay_marker: dict[str, Any],
    replay_marker: dict[str, Any],
) -> Path:
    epoch_dir = args.output_dir / f"epoch_{epoch:02d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = epoch_dir / f"{args.stem}_best.pt"
    norm_path = epoch_dir / f"{args.stem}_norm.npz"
    marker_path = checkpoint_path.with_suffix(".pt.complete")
    if checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(checkpoint_path)
    state = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    torch.save(state, checkpoint_path)
    parent_norm = _normalization_path(args.checkpoint)
    shutil.copy2(parent_norm, norm_path)
    if sha256_file(norm_path) != sha256_file(parent_norm):
        raise RuntimeError("candidate normalizer is not byte-identical to parent")
    replay_marker_path = args.replay_data.with_suffix(
        args.replay_data.suffix + ".complete",
    )
    marker = {
        "family": "directnet-full",
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "normalization": norm_path.name,
        "normalization_sha256": sha256_file(norm_path),
        "output_columns": list(OUTPUT_COLUMNS),
        "source_commit": source_commit,
        "dataset": args.replay_data.name,
        "dataset_sha256": sha256_file(args.replay_data),
        "dataset_completion_marker": replay_marker_path.name,
        "dataset_completion_marker_sha256": sha256_file(replay_marker_path),
        "dataset_source_commit": replay_marker.get("source_commit"),
        "parent_checkpoint": args.checkpoint.name,
        "parent_checkpoint_sha256": sha256_file(args.checkpoint),
        "parent_normalization_sha256": sha256_file(parent_norm),
        "overlay": args.overlay.name,
        "overlay_sha256": overlay_marker["artifact_sha256"],
        "overlay_completion_marker_sha256": sha256_file(
            args.overlay.with_suffix(args.overlay.suffix + ".complete")),
        "replay_rows": overlay_marker["replay_rows"],
        "epoch": epoch,
        "lambda_jacobian": args.lambda_jacobian,
        "lr": args.lr,
        "seed": args.seed,
        "metrics": metrics,
        "feasibility": gate,
    }
    marker_path.write_text(json.dumps(marker, sort_keys=True, indent=2) + "\n")
    return checkpoint_path


def finetune(args: argparse.Namespace) -> dict[str, object]:
    args.checkpoint = args.checkpoint.resolve()
    args.overlay = args.overlay.resolve()
    args.replay_data = args.replay_data.resolve()
    args.output_dir = args.output_dir.resolve()
    source_commit, source_dirty = _source_identity()
    if source_dirty:
        raise RuntimeError("active Hermite trainer source has tracked changes")
    replay_marker_path = args.replay_data.with_suffix(
        args.replay_data.suffix + ".complete",
    )
    if not replay_marker_path.is_file():
        raise FileNotFoundError(replay_marker_path)
    replay_marker = _read_json(replay_marker_path)
    if replay_marker.get("dataset_sha256") != sha256_file(args.replay_data):
        raise ValueError("replay dataset marker checksum mismatch")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model, stats, _num_codes, output_columns = _load_artifacts(args.checkpoint)
    if tuple(output_columns) != OUTPUT_COLUMNS:
        raise ValueError("parent is not the six-surface DirectNet contract")
    model = model.to(device)
    active, validation, replay, overlay_marker = _load_data(
        args.overlay, args.replay_data, args.checkpoint, args.device_type,
        stats, device,
    )
    baseline = _model_metrics(
        model, validation, replay, stats, args.value_batch_size,
        args.jacobian_batch_size,
    )
    print(f"baseline={json.dumps(baseline, sort_keys=True)}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.0,
        fused=device.type == "cuda",
    )
    parameters = [parameter for parameter in model.parameters()
                  if parameter.requires_grad]
    history: list[dict[str, object]] = []
    feasible_epochs: list[int] = []
    require_jacobian = args.lambda_jacobian > 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        replay_loss_sum = 0.0
        replay_seen = 0
        for indices_np in _batches(
            epoch_order(len(replay["x"]), seed=args.seed, epoch=epoch),
            args.replay_batch_size,
        ):
            indices = torch.as_tensor(indices_np, device=device)
            prediction = model(
                replay["x"][indices], tech_codes=replay["code"][indices],
            )
            loss = torch.mean(torch.abs(prediction - replay["y"][indices]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
            optimizer.step()
            replay_loss_sum += float(loss.detach()) * len(indices_np)
            replay_seen += len(indices_np)
        if replay_seen != len(replay["x"]):
            raise RuntimeError("epoch did not consume every fixed replay row")

        active_value_sum = 0.0
        active_j_sum = 0.0
        active_seen = 0
        conflicts = 0
        for indices_np in _batches(
            epoch_order(
                len(active["x"]), seed=args.seed + 1_000_003, epoch=epoch,
            ),
            args.hermite_batch_size,
        ):
            indices = torch.as_tensor(indices_np, device=device)
            x = active["x"][indices].detach().clone().requires_grad_(True)
            prediction = model(x, tech_codes=active["code"][indices])
            value_loss = torch.mean(torch.abs(
                prediction - active["y"][indices],
            ))
            optimizer.zero_grad(set_to_none=True)
            if require_jacobian:
                jacobian_loss, _per_head = _jacobian_loss(
                    prediction, x, active["physical_y"][indices],
                    active["jacobian"][indices], stats, create_graph=True,
                )
                value_gradients = torch.autograd.grad(
                    value_loss, parameters, retain_graph=True,
                    allow_unused=True,
                )
                jacobian_gradients = torch.autograd.grad(
                    jacobian_loss, parameters, allow_unused=True,
                )
                combined, conflicted = combine_gradients(
                    value_gradients, jacobian_gradients,
                    lambda_jacobian=args.lambda_jacobian,
                )
                conflicts += int(conflicted)
                for parameter, gradient in zip(parameters, combined):
                    parameter.grad = gradient
            else:
                jacobian_loss = torch.zeros((), device=device)
                value_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
            optimizer.step()
            count = len(indices_np)
            active_value_sum += float(value_loss.detach()) * count
            active_j_sum += float(jacobian_loss.detach()) * count
            active_seen += count
        metrics = _model_metrics(
            model, validation, replay, stats, args.value_batch_size,
            args.jacobian_batch_size,
        )
        gate = evaluate_feasibility(
            baseline, metrics,
            require_jacobian_improvement=require_jacobian,
            value_mae_ratio=args.value_mae_ratio,
            value_max_ratio=args.value_max_ratio,
            jacobian_ratio=args.jacobian_ratio,
        )
        row = {
            "epoch": epoch,
            "train_replay_mae": replay_loss_sum / replay_seen,
            "train_active_value_mae": active_value_sum / active_seen,
            "train_active_current_j_mae": active_j_sum / active_seen,
            "pcgrad_conflicts": conflicts,
            "pcgrad_batches": int(np.ceil(active_seen / args.hermite_batch_size)),
            "metrics": metrics,
            "feasibility": gate,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if bool(gate["feasible"]):
            feasible_epochs.append(epoch)
            _save_feasible_checkpoint(
                model, args, epoch, metrics, gate, source_commit,
                overlay_marker, replay_marker,
            )

    summary = {
        "source_commit": source_commit,
        "device_type": args.device_type,
        "stem": args.stem,
        "parent_checkpoint": args.checkpoint.name,
        "parent_checkpoint_sha256": sha256_file(args.checkpoint),
        "parent_normalization_sha256": sha256_file(
            _normalization_path(args.checkpoint)),
        "overlay": args.overlay.name,
        "overlay_sha256": sha256_file(args.overlay),
        "replay_dataset": args.replay_data.name,
        "replay_dataset_sha256": sha256_file(args.replay_data),
        "replay_rows": len(replay["x"]),
        "active_rows": len(active["x"]),
        "validation_rows": len(validation["x"]),
        "lambda_jacobian": args.lambda_jacobian,
        "lr": args.lr,
        "seed": args.seed,
        "baseline": baseline,
        "history": history,
        "feasible_epochs": feasible_epochs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = format(args.lambda_jacobian, ".8g").replace(".", "p")
    summary_path = args.output_dir / f"{args.stem}_lambdaJ_{tag}.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(summary_path)
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--replay-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--device-type", choices=("nmos", "pmos"), required=True)
    parser.add_argument("--lambda-jacobian", type=float, required=True)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--replay-batch-size", type=int, default=4096)
    parser.add_argument("--hermite-batch-size", type=int, default=1024)
    parser.add_argument("--value-batch-size", type=int, default=8192)
    parser.add_argument("--jacobian-batch-size", type=int, default=1024)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--value-mae-ratio", type=float, default=1.02)
    parser.add_argument("--value-max-ratio", type=float, default=1.05)
    parser.add_argument("--jacobian-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=766)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = finetune(args)
    print(json.dumps({
        "device_type": summary["device_type"],
        "lambda_jacobian": summary["lambda_jacobian"],
        "feasible_epochs": summary["feasible_epochs"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
