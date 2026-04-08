"""Unified training loops for DirectNet (baseline) and Transformer (BSIM-AR).

Two public entry points:

- `train_directnet`  — full training pipeline for `DirectNet` using `DirectLoss`
  or `ChargeConsistencyLoss`. Matches the old `nn_model.train.train`.

- `train_transformer` — full training pipeline for `TransformerEncoderModel`.
  Matches the old `external_compact_models.BSIMAR.script.main.main`, minus
  argument parsing (which lives in `bsimar.cli.train`).

Lower-level per-epoch helpers are exposed for tests or custom pipelines.
"""

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from bsimar.config import (
    DirectNetConfig, TransformerConfig,
    CHECKPOINT_DIR, RESULTS_DIR,
)
from bsimar.data.dataset import load_and_split, load_and_split_bsimar
from bsimar.data.normalize import (
    Normalizer, BSIMARNormalizer,
    reorder_outputs, unreorder_outputs, _UNREORDER_IDX,
)
from bsimar.models.direct_net import DirectNet
from bsimar.models.transformer import TransformerEncoderModel
from bsimar.losses.direct_loss import DirectLoss, ChargeConsistencyLoss
from bsimar.losses.bni_mae import (
    WeightedBNILoss, MAELoss, compute_lds_weights_per_target,
)
from bsimar.training.early_stopping import EarlyStopping


# ══════════════════════════════════════════════════════════════════════════════
# DirectNet (baseline MLP) per-epoch helpers
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch_direct_mlp(
    model: DirectNet,
    loader: DataLoader,
    criterion: DirectLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """One DirectNet training epoch (fast, no autograd derivatives)."""
    model.train()
    total_losses: Dict[str, float] = {}
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(x_batch)
        losses = criterion(pred, y_batch, x_batch)
        losses["total"].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def validate_epoch_direct_mlp(
    model: DirectNet,
    loader: DataLoader,
    criterion: DirectLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Validate DirectNet without gradients."""
    model.eval()
    total_losses: Dict[str, float] = {}
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model(x_batch)
        losses = criterion(pred, y_batch, x_batch)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


def train_epoch_consistency(
    model: DirectNet,
    loader: DataLoader,
    criterion: ChargeConsistencyLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train DirectNet one epoch with charge-cap autograd consistency."""
    model.train()
    total_losses: Dict[str, float] = {}
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        losses = criterion(model, x_batch, y_batch)
        losses["total"].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


def validate_epoch_consistency(
    model: DirectNet,
    loader: DataLoader,
    criterion: ChargeConsistencyLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Validate with charge-cap consistency loss (direct path for early stopping)."""
    model.eval()
    total_losses: Dict[str, float] = {}
    n_batches = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            losses = criterion.direct_loss(pred, y_batch, x_batch)

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.item()
            n_batches += 1

    avg_losses = {k: v / n_batches for k, v in total_losses.items()}

    x_sample, y_sample = next(iter(loader))
    x_sample, y_sample = x_sample.to(device), y_sample.to(device)
    consist_losses = criterion(model, x_sample, y_sample)
    avg_losses["cap_consist"] = consist_losses["cap_consist"].item()
    avg_losses["cond_consist"] = consist_losses["cond_consist"].item()

    return avg_losses


# ══════════════════════════════════════════════════════════════════════════════
# Transformer (BSIM-AR) per-epoch helpers
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch_direct_ar(
    model: nn.Module,
    loader: DataLoader,
    criterion: DirectLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    unreorder_fn=None,
) -> Dict[str, float]:
    """Train Transformer one epoch with DirectLoss + teacher forcing."""
    model.train()
    total_losses: Dict[str, float] = {}
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(x_batch, y_batch)

        pred_loss = unreorder_fn(pred) if unreorder_fn else pred
        y_loss = unreorder_fn(y_batch) if unreorder_fn else y_batch
        losses = criterion(pred_loss, y_loss, x_batch)
        losses["total"].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def validate_epoch_direct_ar(
    model: nn.Module,
    loader: DataLoader,
    criterion: DirectLoss,
    device: torch.device,
    unreorder_fn=None,
) -> Dict[str, float]:
    """Validate Transformer with autoregressive inference + DirectLoss."""
    model.eval()
    total_losses: Dict[str, float] = {}
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model(x_batch)

        pred_loss = unreorder_fn(pred) if unreorder_fn else pred
        y_loss = unreorder_fn(y_batch) if unreorder_fn else y_batch
        losses = criterion(pred_loss, y_loss, x_batch)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


def train_epoch_bni(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train Transformer one epoch with BNI/MAE + teacher forcing.

    LDS weights, if present, are carried in the loader as a 3rd batch element.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        if len(batch) == 3:
            x_batch, y_batch, w_batch = batch
            w_batch = w_batch.to(device)
        else:
            x_batch, y_batch = batch
            w_batch = None

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(x_batch, y_batch)

        if w_batch is not None:
            loss = criterion(pred, y_batch, weights=w_batch)
        else:
            loss = criterion(pred, y_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"total": total_loss / n_batches}


@torch.no_grad()
def validate_epoch_bni(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate Transformer with autoregressive inference + BNI/MAE loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        total_loss += loss.item()
        n_batches += 1

    return {"total": total_loss / n_batches}


@torch.no_grad()
def validate_epoch_tf(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    unreorder_fn=None,
    use_direct_loss: bool = False,
) -> Dict[str, float]:
    """Fast teacher-forced validation pass.

    Calls ``model(x, y)`` once instead of the AR loop's 13 sequential
    forwards. Empirically TF and AR val losses correlate tightly once
    the model clears R²_norm > 0.9 (the regime we care about), so we
    use TF for early-stopping and only run a full AR-val every N
    epochs as a sanity check.

    Used as the per-epoch validation hook in `train_transformer`. The
    full AR-val path is preserved as a periodic ground-truth check
    inside the training loop.
    """
    model.eval()
    total: Dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        pred = model(x, y)
        if use_direct_loss:
            pl = unreorder_fn(pred) if unreorder_fn else pred
            yl = unreorder_fn(y) if unreorder_fn else y
            losses = criterion(pl, yl, x)
        else:
            losses = {"total": criterion(pred, y)}
        for k, v in losses.items():
            total[k] = total.get(k, 0.0) + v.item()
        n_batches += 1
    return {k: v / n_batches for k, v in total.items()}


def train_epoch_scheduled(
    model: nn.Module,
    loader: DataLoader,
    criterion: DirectLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    ss_ratio: float = 0.0,
    unreorder_fn=None,
) -> Dict[str, float]:
    """Train Transformer one epoch with scheduled sampling."""
    model.train()
    total_losses: Dict[str, float] = {}
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model.forward_scheduled(x_batch, y_batch, ss_ratio=ss_ratio)

        pred_loss = unreorder_fn(pred) if unreorder_fn else pred
        y_loss = unreorder_fn(y_batch) if unreorder_fn else y_batch
        losses = criterion(pred_loss, y_loss, x_batch)
        losses["total"].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


def train_epoch_curriculum(
    model: nn.Module,
    loader: DataLoader,
    criterion: DirectLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    n_targets: int = 13,
    ss_ratio: float = 0.0,
    consistency_weight: float = 0.0,
    unreorder_fn=None,
) -> Dict[str, float]:
    """Train with curriculum on output length + optional scheduled sampling."""
    model.train()
    total_losses: Dict[str, float] = {}
    n_batches = 0
    target_dim = None

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if target_dim is None:
            target_dim = y_batch.shape[1]

        optimizer.zero_grad()

        pred = model.forward_curriculum(
            x_batch, y_batch, n_targets=n_targets, ss_ratio=ss_ratio)

        if n_targets < target_dim:
            pred_masked = pred.clone()
            pred_masked[:, n_targets:] = y_batch[:, n_targets:]
        else:
            pred_masked = pred

        pred_loss = unreorder_fn(pred_masked) if unreorder_fn else pred_masked
        y_loss = unreorder_fn(y_batch) if unreorder_fn else y_batch
        losses = criterion(pred_loss, y_loss, x_batch)

        total_loss = losses["total"]

        if consistency_weight > 0:
            pred_tf = model(x_batch, y_batch)
            with torch.no_grad():
                pred_ar = model(x_batch)
            loss_consist = torch.nn.functional.mse_loss(pred_tf, pred_ar)
            total_loss = total_loss + consistency_weight * loss_consist
            total_losses["consist"] = total_losses.get("consist", 0.0) + loss_consist.item()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def test_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on the test loader, return (pred_norm, true_norm) arrays."""
    model.eval()
    all_pred, all_true = [], []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        pred = model(x_batch)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y_batch.numpy())

    return np.concatenate(all_pred), np.concatenate(all_true)


# ══════════════════════════════════════════════════════════════════════════════
# High-level pipelines
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _collect_directnet_predictions(
    model: DirectNet,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model on `loader` and return (pred_norm, true_norm)."""
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            pred = model(x_batch)
            all_pred.append(pred.cpu().numpy())
            all_true.append(y_batch.numpy())
    return np.concatenate(all_pred), np.concatenate(all_true)


def train_directnet(
    data_path: str,
    config: DirectNetConfig = DirectNetConfig(),
    device_str: str = "cpu",
    save_prefix: str = "nmos",
    output_dim: int = 13,
    resume_from: Optional[str] = None,
    use_charge_consistency: bool = False,
    w_consistency: float = 1.0,
    w_cond_consistency: float = 0.0,
    norm_mode: str = "legacy",
    apply_filter: bool = False,
):
    """Full DirectNet training pipeline.

    Mirrors the legacy `nn_model.train.train` entry point, adapted to the new
    `bsimar` package paths. Saves `<save_prefix>_best.pt` and
    `<save_prefix>_norm.npz` under `bsimar.config.CHECKPOINT_DIR`.

    Args:
        norm_mode: 'legacy' uses the signed-log + z-score `Normalizer`
            (good for 14-decade dynamic range, but `inv_signed_log`
            amplifies physical-space errors). 'zscore' uses
            `BSIMARNormalizer(mode="zscore")` — same path as the
            BSIM-AR Transformer, makes physical-space metrics directly
            comparable across the two models.
        apply_filter: only honored when `norm_mode='zscore'`. Drops
            sub-floor cutoff samples (matches the BSIM-AR pipeline).
    """
    device = torch.device(device_str)
    print(f"Training DirectNet on {device}, output_dim={output_dim}, "
          f"norm_mode={norm_mode}")
    print(f"Loss weights: w_id={config.w_id}, w_charges={config.w_charges}, "
          f"w_caps={config.w_caps}")
    if use_charge_consistency:
        print(f"Charge consistency: w_consistency={w_consistency}, "
              f"w_cond_consistency={w_cond_consistency}")

    if norm_mode == "zscore":
        from bsimar.config import OUTPUT_COLUMNS
        train_ds, val_ds, test_ds, normalizer = load_and_split_bsimar(
            data_path,
            column_names=OUTPUT_COLUMNS,
            norm_mode="zscore",
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            apply_filter=apply_filter,
        )
    elif norm_mode == "legacy":
        train_ds, val_ds, test_ds, normalizer = load_and_split(
            data_path,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
        )
    else:
        raise ValueError(
            f"Unknown norm_mode '{norm_mode}'. Use 'legacy' or 'zscore'.")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    input_dim = train_ds.inputs.shape[1]
    dim_label = {
        6: "legacy", 7: "with PHIG",
        13: "universal (7 process params)",
        18: "universal (12 process params)",
        19: "universal (12 process params + L)",
    }
    print(f"Input dim: {input_dim} ({dim_label.get(input_dim, f'{input_dim}-dim')})")

    model = DirectNet(
        input_dim=input_dim,
        hidden_dim=config.trunk_hidden,
        n_layers=config.trunk_layers + 1,
        output_dim=output_dim,
    ).to(device)

    if resume_from is not None:
        print(f"Resuming from {resume_from}")
        state = torch.load(resume_from, weights_only=True, map_location=device)
        model_state = model.state_dict()
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
        model.load_state_dict(model_state)

    print(f"Model parameters: {model.count_parameters()}")

    if use_charge_consistency:
        criterion = ChargeConsistencyLoss(
            w_consistency=w_consistency,
            w_cond_consistency=w_cond_consistency,
            w_zero_bias=config.w_zero_bias,
            w_curr=config.w_id,
            w_cond=(config.w_gm + config.w_gds + config.w_gmb) / 3.0,
            w_charges=config.w_charges,
            w_caps=config.w_caps,
        )
        _train_fn = train_epoch_consistency
        _val_fn = validate_epoch_consistency
    else:
        criterion = DirectLoss(
            output_dim=output_dim,
            w_zero_bias=config.w_zero_bias,
            w_curr=config.w_id,
            w_cond=(config.w_gm + config.w_gds + config.w_gmb) / 3.0,
            w_charges=config.w_charges,
            w_caps=config.w_caps,
        )
        _train_fn = train_epoch_direct_mlp
        _val_fn = validate_epoch_direct_mlp

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    best_val_loss = float("inf")
    patience_counter = 0

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINT_DIR / f"{save_prefix}_best.pt"
    norm_path = CHECKPOINT_DIR / f"{save_prefix}_norm.npz"

    t_start = time.time()

    for epoch in range(1, config.max_epochs + 1):
        train_losses = _train_fn(model, train_loader, criterion, optimizer, device)
        val_losses = _val_fn(model, val_loader, criterion, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        status = ""
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            normalizer.stats.save(str(norm_path))
            status = " *best*"
        else:
            patience_counter += 1

        should_print = (epoch % 20 == 0 or epoch <= 5 or bool(status))
        if should_print:
            print(f"{epoch:4d} | train={train_losses['total']:.5f} "
                  f"val={val_losses['total']:.5f} lr={lr:.2e}{status}")

        if patience_counter >= config.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={config.patience})")
            break

    elapsed = time.time() - t_start
    print(f"\nTraining completed in {elapsed:.0f}s ({elapsed / epoch:.1f}s/epoch)")
    print(f"Best val loss: {best_val_loss:.6f}")

    model.load_state_dict(torch.load(best_path, weights_only=True))
    test_losses = _val_fn(model, test_loader, criterion, device)
    print(f"\nTest set losses:")
    for k, v in sorted(test_losses.items()):
        print(f"  {k:>10s}: {v:.6f}")

    # Use the shared metrics path so DirectNet reports the same
    # NRMSE / MRE / R² / R²_norm / MAE_n table as the BSIMAR Transformer.
    from bsimar.eval.metrics import compute_physical_metrics, print_metrics
    pred_norm, true_norm = _collect_directnet_predictions(
        model, test_loader, device)
    metrics = compute_physical_metrics(pred_norm, true_norm, normalizer)
    print(f"\nPhysical metrics (test set):")
    print_metrics(metrics)

    print(f"\nSaved: {best_path}")
    return model, normalizer


def train_transformer(
    data_path: str,
    save_prefix: str,
    device_type: str,
    loss_name: str = "mae",
    norm_mode: str = "zscore",
    apply_filter: bool = True,
    use_lds: bool = False,
    reorder: bool = True,
    scheduled_sampling: bool = False,
    ss_warmup: int = 100,
    ss_max_ratio: float = 0.5,
    consistency_weight: float = 0.0,
    curriculum: bool = False,
    curriculum_warmup: int = 50,
    config: TransformerConfig = TransformerConfig(),
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    patience: Optional[int] = None,
    lr: Optional[float] = None,
    device_str: str = "cpu",
    column_names: Optional[list] = None,
    overwrite: bool = False,
) -> Tuple[nn.Module, BSIMARNormalizer]:
    """Full BSIM-AR Transformer training pipeline.

    Mirrors the legacy `external_compact_models.BSIMAR.script.main.main`
    flow, adapted to the new `bsimar` package. Saves `<save_prefix>_best.pt`,
    `<save_prefix>_norm.npz`, and `<save_prefix>_config.npz`.
    """
    from bsimar.config import OUTPUT_COLUMNS
    from bsimar.eval.metrics import compute_physical_metrics, print_metrics
    from bsimar.eval.visualization import (
        plot_scatter_comparison, plot_loss_curves,
    )

    if column_names is None:
        column_names = OUTPUT_COLUMNS

    # A4: scheduled_sampling / curriculum / consistency_weight only have a
    # code path under `loss_name == "direct"`. Under mae/bni they were
    # silently ignored, which is a footgun (see CLAUDE.md memory note
    # `bsimar_loss_routing.md`). Fail loudly instead.
    if loss_name != "direct" and (
        scheduled_sampling or curriculum or consistency_weight > 0
    ):
        raise ValueError(
            "scheduled_sampling / curriculum / consistency_weight only "
            "work under --loss direct. Either switch to --loss direct, "
            "or remove these flags."
        )

    epochs = epochs if epochs is not None else config.max_epochs
    batch_size = batch_size if batch_size is not None else config.batch_size
    patience = patience if patience is not None else config.patience
    lr = lr if lr is not None else config.lr

    device = torch.device(device_str)
    print(f"Device: {device} | Norm: {norm_mode} | "
          f"Loss: {loss_name}{'+lds' if use_lds else ''} | "
          f"Filter: {apply_filter} | Data: {Path(data_path).name}")

    train_ds, val_ds, test_ds, normalizer = load_and_split_bsimar(
        str(data_path),
        column_names=column_names,
        norm_mode=norm_mode,
        apply_filter=apply_filter,
    )
    input_dim = train_ds.inputs.shape[1]
    output_dim = train_ds.outputs.shape[1]
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")

    _reorder_active = False
    if reorder:
        _reorder_active = True
        train_ds.outputs = torch.tensor(
            reorder_outputs(train_ds.outputs.numpy()), dtype=torch.float32)
        val_ds.outputs = torch.tensor(
            reorder_outputs(val_ds.outputs.numpy()), dtype=torch.float32)
        test_ds.outputs = torch.tensor(
            reorder_outputs(test_ds.outputs.numpy()), dtype=torch.float32)
        print("Output columns reordered: charges->caps->cond->id")

    if _reorder_active:
        _unreorder_idx_t = torch.tensor(_UNREORDER_IDX, dtype=torch.long)
        def _unreorder_tensor(t: torch.Tensor) -> torch.Tensor:
            return t[:, _unreorder_idx_t.to(t.device)]
        unreorder_fn = _unreorder_tensor
    else:
        unreorder_fn = None

    model = TransformerEncoderModel(
        input_dim=input_dim,
        target_dim=output_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        # P4 — parallel C-block experiment.
        parallel_caps=True,
        # A2 — grouped input tokens (voltages / geometry / process
        # params) collapse the 19 scalar context tokens into 3, dropping
        # sequence length from 28 to 12 under parallel_caps=True.
        grouped_inputs=True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    # Loss + training function selection
    if loss_name == "direct":
        criterion = DirectLoss(
            output_dim=output_dim,
            w_curr=config.w_curr, w_cond=config.w_cond,
            w_charges=config.w_charges, w_caps=config.w_caps,
            w_zero_bias=config.w_zero_bias,
        )
        train_fn = train_epoch_direct_ar
        val_fn = validate_epoch_direct_ar
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    elif loss_name == "mae":
        criterion = MAELoss()
        train_fn = train_epoch_bni
        val_fn = validate_epoch_bni

        if use_lds:
            print("Computing LDS weights...")
            lds_weights_np = compute_lds_weights_per_target(
                train_ds.outputs.numpy(), n_bins=100,
                lds_kernel="gaussian", lds_ks=5, lds_sigma=0.8,
            )
            train_ds_weighted = TensorDataset(
                train_ds.inputs, train_ds.outputs,
                torch.tensor(lds_weights_np, dtype=torch.float32),
            )
            train_loader = DataLoader(
                train_ds_weighted, batch_size=batch_size, shuffle=True)
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    else:  # bni
        criterion = WeightedBNILoss()
        train_fn = train_epoch_bni
        val_fn = validate_epoch_bni

        if use_lds:
            print("Computing LDS weights...")
            lds_weights_np = compute_lds_weights_per_target(
                train_ds.outputs.numpy(), n_bins=100,
                lds_kernel="gaussian", lds_ks=5, lds_sigma=0.8,
            )
            train_ds_weighted = TensorDataset(
                train_ds.inputs, train_ds.outputs,
                torch.tensor(lds_weights_np, dtype=torch.float32),
            )
            train_loader = DataLoader(
                train_ds_weighted, batch_size=batch_size, shuffle=True)
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=config.weight_decay)

    # C2: Cosine LR decay. The DirectNet path already uses
    # CosineAnnealingLR (line ~557); copy-paste it here so the
    # transformer path stops training at flat AdamW.
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINT_DIR / f"{save_prefix}_best.pt"
    norm_path = CHECKPOINT_DIR / f"{save_prefix}_norm.npz"
    results_subdir = str(RESULTS_DIR / save_prefix)

    # A6: refuse to silently overwrite an existing checkpoint. Concurrent
    # runs sharing --exp-name otherwise clobber each other (see CLAUDE.md
    # smoke-test bug note). Pass --overwrite to allow.
    if best_path.exists() and not overwrite:
        raise SystemExit(
            f"Refusing to overwrite {best_path}. Pass --overwrite or "
            "choose a unique --exp-name."
        )

    early_stopping = EarlyStopping(
        patience=patience, min_delta=1e-5, save_path=str(best_path))

    print(f"\nTraining {save_prefix} for {epochs} epochs (patience={patience})")
    train_history, val_history = [], []
    best_val_loss = float("inf")
    best_ar_val_loss = float("inf")
    ar_best_path = best_path.with_suffix(".ar.pt")
    # T1: physical-space early-stopping tracker. Runs alongside the TF
    # early-stopping on the same `ar_check_every` schedule. We keep a
    # separate `*_best.phys.pt` checkpoint so the final test load can
    # prefer the best physical-space model instead of the best TF-val.
    best_phys_score = float("inf")
    best_phys_nrmse = float("nan")
    best_phys_r2 = float("nan")
    phys_best_path = best_path.with_suffix(".phys.pt")
    # C1: how often to run a full AR-validation as a ground-truth sanity
    # check on the TF-driven early stopping. Cheap because val set is
    # only swept slowly.
    ar_check_every = 10
    t_start = time.time()
    epoch = 0

    for epoch in range(1, epochs + 1):
        ss_ratio = (min(epoch / ss_warmup, ss_max_ratio)
                    if scheduled_sampling else 0.0)
        n_targets = (max(1, int(output_dim * min(
            epoch / curriculum_warmup, 1.0)))
            if curriculum else output_dim)

        if loss_name == "direct":
            if curriculum or consistency_weight > 0 or scheduled_sampling:
                t_losses = train_epoch_curriculum(
                    model, train_loader, criterion, optimizer, device,
                    n_targets=n_targets, ss_ratio=ss_ratio,
                    consistency_weight=consistency_weight,
                    unreorder_fn=unreorder_fn)
            else:
                t_losses = train_fn(
                    model, train_loader, criterion, optimizer, device,
                    unreorder_fn=unreorder_fn)
        else:
            t_losses = train_fn(model, train_loader, criterion, optimizer, device)

        # C1: Fast TF-based validation drives early stopping every epoch.
        # The AR validation is ~10x slower (sequential 13-step decode on
        # the full val set) and is empirically tightly correlated with
        # TF-val once the model has cleared R²_norm > 0.9. Run the full
        # AR pass only every ar_check_every epochs as a ground-truth
        # check, and keep a separate "AR best" checkpoint so we have an
        # honest reference at the end of training.
        v_losses = validate_epoch_tf(
            model, val_loader, criterion, device,
            unreorder_fn=unreorder_fn,
            use_direct_loss=(loss_name == "direct"),
        )

        run_ar_check = (epoch % ar_check_every == 0) or (epoch == epochs)
        ar_status = ""
        if run_ar_check:
            if loss_name == "direct":
                ar_v = val_fn(
                    model, val_loader, criterion, device,
                    unreorder_fn=unreorder_fn)
            else:
                ar_v = val_fn(model, val_loader, criterion, device)
            ar_loss = ar_v["total"]
            if ar_loss < best_ar_val_loss:
                best_ar_val_loss = ar_loss
                torch.save(model.state_dict(), str(ar_best_path))
                ar_status = " *ar-best*"
            print(f"  AR-val check @ epoch {epoch}: "
                  f"ar={ar_loss:.5f} (tf={v_losses['total']:.5f})"
                  f"{ar_status}")

            # T1: physical-space early-stopping probe. Reuse the existing
            # AR-check cadence so we don't add extra cost beyond one extra
            # teacher-forced pass over the val set. The score blends
            # physical-space NRMSE and (1 - R2) so that a collapsed
            # normalizer (R2 << 0) can never beat a healthy one even if
            # NRMSE happens to look small.
            pred_val_norm, true_val_norm = test_model(
                model, val_loader, device)
            if _reorder_active:
                pred_val_norm = unreorder_outputs(pred_val_norm)
                true_val_norm = unreorder_outputs(true_val_norm)
            phys_metrics = compute_physical_metrics(
                pred_val_norm, true_val_norm, normalizer)
            nrmse_arr = np.array(
                [m["NRMSE(%)"] for m in phys_metrics.values()],
                dtype=np.float64,
            )
            r2_arr = np.array(
                [m["R2"] for m in phys_metrics.values()],
                dtype=np.float64,
            )
            nrmse_avg = float(np.nanmean(nrmse_arr))
            r2_avg = float(np.nanmean(r2_arr))
            # NaN guard: if every target is masked out we cannot score.
            if np.isnan(nrmse_avg) or np.isnan(r2_avg):
                phys_score = float("inf")
            else:
                phys_score = nrmse_avg + 0.1 * (1.0 - r2_avg)
            phys_status = ""
            if phys_score < best_phys_score:
                best_phys_score = phys_score
                best_phys_nrmse = nrmse_avg
                best_phys_r2 = r2_avg
                torch.save(model.state_dict(), str(phys_best_path))
                phys_status = " *phys-best*"
            print(f"  PHYS-val @ epoch {epoch}: "
                  f"nrmse_avg={nrmse_avg:.3f} r2_avg={r2_avg:.4f} "
                  f"score={phys_score:.3f}{phys_status}")

        train_loss = t_losses["total"]
        val_loss = v_losses["total"]
        train_history.append(train_loss)
        val_history.append(val_loss)

        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            normalizer.stats.save(str(norm_path))
            status = " *best*"

        if early_stopping(val_loss, model):
            print(f"Early stopping at epoch {epoch}")
            break

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        if epoch % 20 == 0 or epoch <= 5 or status:
            if loss_name == "direct":
                print(f"  {epoch:4d} | train={train_loss:.5f} "
                      f"val={val_loss:.5f} lr={lr_now:.2e} | "
                      f"id={v_losses.get('id', 0):.5f} "
                      f"gm={v_losses.get('gm', 0):.5f} "
                      f"q={v_losses.get('charges', 0):.5f} "
                      f"cap={v_losses.get('caps', 0):.5f}{status}")
            else:
                print(f"  {epoch:4d} | train={train_loss:.5f} "
                      f"val={val_loss:.5f} lr={lr_now:.2e}{status}")

    elapsed = time.time() - t_start
    epochs_run = max(epoch, 1)
    print(f"\nDone in {elapsed:.0f}s ({elapsed / epochs_run:.1f}s/epoch)")
    print(f"Best val loss (TF):  {best_val_loss:.6f}")
    if best_ar_val_loss < float("inf"):
        print(f"Best val loss (AR):  {best_ar_val_loss:.6f}  -> {ar_best_path}")
    if best_phys_score < float("inf"):
        print(f"Best phys score:   {best_phys_score:.6f}  "
              f"(NRMSE={best_phys_nrmse:.3f}%, R2={best_phys_r2:.4f})  "
              f"-> {phys_best_path}")

    arch_config = {
        "input_dim": input_dim, "target_dim": output_dim,
        "d_model": config.d_model, "nhead": config.nhead,
        "num_layers": config.num_layers,
        "dim_feedforward": config.dim_feedforward,
        "dropout": config.dropout,
        "parallel_caps": True,
        # A2 — grouped input tokens. Persisted so checkpoint loaders
        # can rebuild the architecture.
        "grouped_inputs": True,
    }
    config_path = CHECKPOINT_DIR / f"{save_prefix}_config.npz"
    np.savez(str(config_path),
             **{k: np.array(v) for k, v in arch_config.items()})
    print(f"Arch config: {config_path}")

    # T1: prefer the phys-space-best checkpoint for the final test if it
    # exists. Falls back to the TF-val-best checkpoint otherwise, which
    # matches the pre-T1 behaviour.
    if phys_best_path.exists():
        load_path = phys_best_path
        print(f"Loading phys-best checkpoint for final test: {load_path}")
    else:
        load_path = best_path
        print(f"Loading TF-val-best checkpoint for final test: {load_path}")
    model.load_state_dict(torch.load(str(load_path), weights_only=True))
    pred_norm, true_norm = test_model(model, test_loader, device)

    if _reorder_active:
        pred_norm = unreorder_outputs(pred_norm)
        true_norm = unreorder_outputs(true_norm)

    metrics = compute_physical_metrics(pred_norm, true_norm, normalizer)
    print_metrics(metrics)

    pred_phys = normalizer.denormalize_outputs(pred_norm)
    true_phys = normalizer.denormalize_outputs(true_norm)
    plot_scatter_comparison(true_phys, pred_phys, results_subdir)
    plot_loss_curves(train_history, val_history, results_subdir,
                     title_prefix=f"BSIM-AR {save_prefix} ")

    print(f"\nCheckpoint: {best_path}")
    print(f"Norm stats: {norm_path}")
    print(f"Results:    {results_subdir}/")

    return model, normalizer
