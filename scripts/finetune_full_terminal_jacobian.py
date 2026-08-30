#!/usr/bin/env python3
"""Fine-tune DirectNet-Full against exact OSDI current Jacobians."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external_compact_models"))

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

VOLTAGE_COLUMNS = (0, 1, 3)
CURRENT_HEADS = (0, 1, 2)


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


def _raw_inputs(
    inputs: np.ndarray,
    nfin: np.ndarray,
    length: np.ndarray,
    temperature: np.ndarray,
) -> np.ndarray:
    return np.column_stack([
        inputs[:, 0], inputs[:, 1], np.zeros(len(inputs)), inputs[:, 3],
        np.log2(np.maximum(nfin, 1.0)), length, temperature,
    ]).astype(np.float64)


def _normalize_inputs(raw: np.ndarray, stats: Any) -> np.ndarray:
    scale = np.asarray(stats.input_std, dtype=np.float64).copy()
    scale[scale < 1e-12] = 1.0
    return ((raw - np.asarray(stats.input_mean)) / scale).astype(np.float32)


def _normalize_outputs(outputs: np.ndarray, stats: Any) -> np.ndarray:
    if stats.mode != "asinh" or stats.asinh_scale is None:
        raise ValueError("full-terminal Jacobian fine-tune requires asinh norm")
    transformed = np.arcsinh(
        outputs / np.asarray(stats.asinh_scale, dtype=np.float64))
    return ((transformed - np.asarray(stats.output_mean))
            / np.asarray(stats.output_std)).astype(np.float32)


def _split_overlay(
    overlay: np.lib.npyio.NpzFile,
    stats: Any,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    raw = _raw_inputs(
        overlay["inputs"], overlay["nfin"], overlay["length"],
        overlay["temperature"],
    )
    arrays = {
        "x": _normalize_inputs(raw, stats),
        "y": _normalize_outputs(overlay["outputs"], stats),
        "jac": np.asarray(overlay["jacobians"], dtype=np.float32),
        "code": np.asarray(overlay["tech_codes"], dtype=np.int64),
    }
    group_columns = np.column_stack([
        overlay["variant"], overlay["length"].astype(str),
        overlay["nfin"].astype(str), overlay["temperature"].astype(str),
    ])
    group_keys = np.asarray(["|".join(row) for row in group_columns])
    train_indices: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for key in np.unique(group_keys):
        indices = np.flatnonzero(group_keys == key)
        indices = rng.permutation(indices)
        n_val = max(1, int(round(0.2 * len(indices))))
        val_indices.append(indices[:n_val])
        train_indices.append(indices[n_val:])
    train_idx = np.sort(np.concatenate(train_indices))
    val_idx = np.sort(np.concatenate(val_indices))
    return (
        {name: value[train_idx] for name, value in arrays.items()},
        {name: value[val_idx] for name, value in arrays.items()},
    )


def _replay_sample(
    path: Path,
    device_type: str,
    tech_scope: str,
    stats: Any,
    rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = get_or_build_tech_variant_labels(
        str(path), device_type, verbose=False)
    rng = np.random.default_rng(seed)
    with np.load(path, allow_pickle=False) as data:
        count = min(rows, len(data["outputs"]))
        indices = np.sort(rng.choice(
            len(data["outputs"]), size=count, replace=False))
        inputs = np.asarray(data["inputs"][indices])
        geometry = np.asarray(data["geometry"][indices, :3])
        outputs = np.asarray(data["outputs"][indices])
    raw = _raw_inputs(
        inputs, geometry[:, 0], geometry[:, 1], geometry[:, 2])
    codes = np.asarray([
        local_variant_code(tech_scope, *CODE_TO_TECH_VARIANT[int(code)])
        for code in labels[indices]
    ], dtype=np.int64)
    return (
        _normalize_inputs(raw, stats),
        _normalize_outputs(outputs, stats),
        codes,
    )


def _tensor(data: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(data, device=device)


def _jacobian_loss(
    prediction: torch.Tensor,
    x: torch.Tensor,
    target: torch.Tensor,
    stats_tensors: dict[str, torch.Tensor],
    create_graph: bool,
) -> torch.Tensor:
    u = (prediction * stats_tensors["output_std"]
         + stats_tensors["output_mean"])
    physical = stats_tensors["asinh_scale"] * torch.sinh(u)
    losses: list[torch.Tensor] = []
    for ordinal, head in enumerate(CURRENT_HEADS):
        gradient = torch.autograd.grad(
            prediction[:, head].sum(), x,
            create_graph=create_graph,
            retain_graph=(create_graph or ordinal + 1 < len(CURRENT_HEADS)),
        )[0][:, VOLTAGE_COLUMNS]
        factor = torch.sqrt(
            stats_tensors["asinh_scale"][head] ** 2
            + physical[:, head] ** 2 + 1e-30)
        target_normalized = (
            target[:, head, :]
            * stats_tensors["input_std"][list(VOLTAGE_COLUMNS)]
            / stats_tensors["output_std"][head]
            / factor[:, None]
        )
        losses.append(torch.mean(torch.abs(gradient - target_normalized)))
    return torch.stack(losses).mean()


def _metrics(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    stats_tensors: dict[str, torch.Tensor],
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    value_total = 0.0
    jac_total = 0.0
    rows = 0
    for start in range(0, len(data["x"]), batch_size):
        stop = min(start + batch_size, len(data["x"]))
        x = data["x"][start:stop].detach().clone().requires_grad_(True)
        prediction = model(x, tech_codes=data["code"][start:stop])
        value = torch.mean(torch.abs(prediction - data["y"][start:stop]))
        jacobian = _jacobian_loss(
            prediction, x, data["jac"][start:stop], stats_tensors,
            create_graph=False,
        )
        count = stop - start
        value_total += float(value.detach()) * count
        jac_total += float(jacobian.detach()) * count
        rows += count
    return {
        "value_mae": value_total / rows,
        "current_jacobian_mae": jac_total / rows,
    }


def _to_device(
    data: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: _tensor(value, device) for name, value in data.items()}


def finetune(args: argparse.Namespace) -> dict[str, Any]:
    source_commit, source_dirty = _source_identity()
    if source_dirty:
        raise RuntimeError("fine-tune source has tracked changes")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cuda" if args.cuda else "cpu")

    model, stats, _num_codes, output_columns = _load_artifacts(args.checkpoint)
    if tuple(output_columns) != (
        "i_d", "i_g", "i_b", "qd", "qg", "qb",
    ):
        raise ValueError("checkpoint is not the six-surface DNF contract")
    model = model.to(device)
    initial_state = copy.deepcopy(model.state_dict())
    with np.load(args.overlay, allow_pickle=False) as overlay:
        train_np, val_np = _split_overlay(overlay, stats, args.seed)
    replay_x, replay_y, replay_code = _replay_sample(
        args.replay_data, args.device_type, args.tech_scope, stats,
        args.replay_rows, args.seed,
    )
    train = _to_device(train_np, device)
    val = _to_device(val_np, device)
    replay = {
        "x": _tensor(replay_x, device),
        "y": _tensor(replay_y, device),
        "code": _tensor(replay_code, device),
    }
    stats_tensors = {
        "input_std": _tensor(
            np.asarray(stats.input_std, dtype=np.float32), device),
        "output_std": _tensor(
            np.asarray(stats.output_std, dtype=np.float32), device),
        "output_mean": _tensor(
            np.asarray(stats.output_mean, dtype=np.float32), device),
        "asinh_scale": _tensor(
            np.asarray(stats.asinh_scale, dtype=np.float32), device),
    }

    baseline = _metrics(model, val, stats_tensors, args.batch_size)
    baseline_score = (baseline["value_mae"]
                      + args.lambda_jacobian
                      * baseline["current_jacobian_mae"])
    print(f"baseline val={baseline}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.0,
        fused=(device.type == "cuda"),
    )
    best_score = baseline_score
    best_state = copy.deepcopy(initial_state)
    bad_epochs = 0
    rng = torch.Generator(device=device).manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(
            len(train["x"]), generator=rng, device=device)
        totals = np.zeros(4, dtype=np.float64)
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start:start + args.batch_size]
            replay_indices = torch.randint(
                len(replay["x"]), (len(indices),),
                generator=rng, device=device,
            )
            x = train["x"][indices].detach().clone().requires_grad_(True)
            prediction = model(x, tech_codes=train["code"][indices])
            value_loss = torch.mean(
                torch.abs(prediction - train["y"][indices]))
            jacobian_loss = _jacobian_loss(
                prediction, x, train["jac"][indices], stats_tensors,
                create_graph=True,
            )
            replay_prediction = model(
                replay["x"][replay_indices],
                tech_codes=replay["code"][replay_indices],
            )
            replay_loss = torch.mean(torch.abs(
                replay_prediction - replay["y"][replay_indices]))
            loss = (value_loss
                    + args.lambda_jacobian * jacobian_loss
                    + args.lambda_replay * replay_loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals += np.asarray([
                float(loss.detach()), float(value_loss.detach()),
                float(jacobian_loss.detach()), float(replay_loss.detach()),
            ]) * len(indices)
        totals /= len(train["x"])
        candidate = _metrics(model, val, stats_tensors, args.batch_size)
        score = (candidate["value_mae"]
                 + args.lambda_jacobian
                 * candidate["current_jacobian_mae"])
        print(
            f"epoch={epoch} train_total={totals[0]:.6f} "
            f"value={totals[1]:.6f} jac={totals[2]:.6f} "
            f"replay={totals[3]:.6f} val={candidate} score={score:.6f}",
            flush=True,
        )
        if score < best_score - 1e-6:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    model.load_state_dict(best_state)
    final = _metrics(model, val, stats_tensors, args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"{args.stem}_best.pt"
    norm_path = args.output_dir / f"{args.stem}_norm.npz"
    torch.save(model.state_dict(), model_path)
    source_norm = args.checkpoint.with_name(
        args.checkpoint.name.replace("_best.pt", "_norm.npz"))
    shutil.copy2(source_norm, norm_path)
    marker = {
        "family": "directnet-full",
        "checkpoint": model_path.name,
        "checkpoint_sha256": _sha256(model_path),
        "normalization": norm_path.name,
        "normalization_sha256": _sha256(norm_path),
        "output_columns": list(output_columns),
        "source_commit": source_commit,
        "parent_checkpoint": args.checkpoint.name,
        "parent_checkpoint_sha256": _sha256(args.checkpoint),
        "overlay": args.overlay.name,
        "overlay_sha256": _sha256(args.overlay),
        "replay_dataset": args.replay_data.name,
        "replay_dataset_sha256": _sha256(args.replay_data),
        "lambda_jacobian": args.lambda_jacobian,
        "lambda_replay": args.lambda_replay,
        "lr": args.lr,
        "seed": args.seed,
        "baseline_metrics": baseline,
        "selected_metrics": final,
        "selected_score": best_score,
    }
    model_path.with_suffix(".pt.complete").write_text(
        json.dumps(marker, sort_keys=True, indent=2) + "\n")
    return marker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--replay-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--device-type", choices=("nmos", "pmos"), required=True)
    parser.add_argument("--tech-scope", default="tsmc5")
    parser.add_argument("--lambda-jacobian", type=float, default=0.05)
    parser.add_argument("--lambda-replay", type=float, default=1.0)
    parser.add_argument("--replay-rows", type=int, default=65_536)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    for name in ("checkpoint", "overlay", "replay_data", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    marker = finetune(args)
    print(json.dumps(marker, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
