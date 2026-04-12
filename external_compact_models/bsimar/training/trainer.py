"""Training pipelines for DirectNet (baseline), BSIMAR Transformer (v3), and
BSIMAR v4 (tech-code embedding).

Three public entry points:

- ``train_directnet``  — DirectNet MLP baseline.
- ``train_transformer`` — BSIMAR v3 Transformer (19-dim input with process params).
- ``train_transformer_v4`` — BSIMAR v4 Transformer (7-dim input + discrete tech codes).

Lower-level per-epoch helpers are exposed for tests and custom pipelines.
"""

import time
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from bsimar.config import (
    DirectNetConfig, TransformerConfig,
    CHECKPOINT_DIR, RESULTS_DIR,
    NUM_TSMC_CODES_WITH_UNKNOWN,
)
from bsimar.data.dataset import load_and_split_bsimar
from bsimar.data.normalize import (
    BSIMARNormalizer,
    reorder_outputs, unreorder_outputs, _UNREORDER_IDX,
)
from bsimar.models.direct_net import DirectNet
from bsimar.models.transformer import TransformerEncoderModel
from bsimar.losses.direct_loss import DirectLoss, ChargeConsistencyLoss
from bsimar.losses.bni_mae import MAELoss, compute_lds_weights_per_target
from bsimar.training.early_stopping import EarlyStopping


# ══════════════════════════════════════════════════════════════════════════════
# DirectNet per-epoch helpers
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
    """Validate with charge-cap consistency loss."""
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
# BSIMAR Transformer per-epoch helpers
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch_mae(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Teacher-forced training epoch with MAE (+ optional LDS weights).

    LDS weights, if present, are carried in the loader as a 3rd batch
    element of shape (B, 13). The criterion must accept a ``weights``
    keyword argument.
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
def validate_epoch_ar(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Autoregressive validation: ``model(x)`` only (no teacher forcing)."""
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
) -> Dict[str, float]:
    """Teacher-forced validation — one forward per batch, not 8.

    Empirically TF and AR val losses correlate tightly once the model
    clears R²_norm > 0.9, so TF drives early-stopping and we run the
    full AR pass every ``ar_check_every`` epochs as a ground-truth check.
    """
    model.eval()
    total = 0.0
    n_batches = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        pred = model(x, y)
        total += criterion(pred, y).item()
        n_batches += 1
    return {"total": total / n_batches}


def train_epoch_scheduled_mae(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    ss_ratio: float = 1.0,
) -> Dict[str, float]:
    """N3 fine-tune helper: scheduled-sampling AR step with MAE.

    Identical to ``train_epoch_mae`` except the forward pass uses
    ``model.forward_scheduled(x, y, ss_ratio=ss_ratio)`` so the model
    is rolled out autoregressively (feeding its own detached
    predictions instead of ground truth) for the AR target block.
    Caps still emit in parallel.
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
        pred = model.forward_scheduled(x_batch, y_batch, ss_ratio=ss_ratio)

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
def test_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run AR inference on the loader, return (pred_norm, true_norm)."""
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
    apply_filter: bool = False,
):
    """DirectNet baseline training pipeline.

    Uses ``BSIMARNormalizer(mode='zscore')`` for normalisation — the
    legacy signed-log path was removed in the v3 sprint. Saves
    ``<save_prefix>_best.pt`` and ``<save_prefix>_norm.npz`` under
    ``bsimar.config.CHECKPOINT_DIR``.

    Args:
        apply_filter: Drop sub-floor cutoff samples (matches the BSIMAR
            data path). Default False for DirectNet to preserve
            baseline behaviour.
    """
    from bsimar.config import OUTPUT_COLUMNS

    device = torch.device(device_str)
    print(f"Training DirectNet on {device}, output_dim={output_dim}")
    print(f"Loss weights: w_id={config.w_id}, w_charges={config.w_charges}, "
          f"w_caps={config.w_caps}")
    if use_charge_consistency:
        print(f"Charge consistency: w_consistency={w_consistency}, "
              f"w_cond_consistency={w_cond_consistency}")

    train_ds, val_ds, test_ds, normalizer = load_and_split_bsimar(
        data_path,
        column_names=OUTPUT_COLUMNS,
        norm_mode="zscore",
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        apply_filter=apply_filter,
    )

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
    epoch = 0

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
    epochs_run = max(epoch, 1)
    print(f"\nTraining completed in {elapsed:.0f}s ({elapsed / epochs_run:.1f}s/epoch)")
    print(f"Best val loss: {best_val_loss:.6f}")

    model.load_state_dict(torch.load(best_path, weights_only=True))
    test_losses = _val_fn(model, test_loader, criterion, device)
    print(f"\nTest set losses:")
    for k, v in sorted(test_losses.items()):
        print(f"  {k:>10s}: {v:.6f}")

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
    config: TransformerConfig = TransformerConfig(),
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    patience: Optional[int] = None,
    lr: Optional[float] = None,
    device_str: str = "cpu",
    ar_finetune_epochs: int = 5,
    overwrite: bool = False,
) -> Tuple[nn.Module, BSIMARNormalizer]:
    """BSIMAR v3 Transformer training pipeline.

    Hard-wires the winning recipe from the 2026-04-08 improvement
    sprint. Caller-visible knobs are **architecture** (via ``config``)
    and **schedule** (``epochs`` / ``batch_size`` / ``patience`` /
    ``lr`` / ``ar_finetune_epochs``). The loss, normalisation, data
    filtering, LDS weighting, Vov-LDS weighting, output reorder, and
    phys-best checkpoint tracker are all fixed and not user-tunable.

    Produces three files under ``CHECKPOINT_DIR``:

    - ``<save_prefix>_best.pt``       — TF-val-best checkpoint
    - ``<save_prefix>_best.ar.pt``    — AR-val-best checkpoint
    - ``<save_prefix>_best.phys.pt``  — phys-space-best checkpoint
                                        (loaded for the final test)
    - ``<save_prefix>_norm.npz``      — ``BSIMARNormStats`` (asinh mode)
    - ``<save_prefix>_config.npz``    — architecture config
    """
    from bsimar.config import OUTPUT_COLUMNS
    from bsimar.eval.metrics import compute_physical_metrics, print_metrics
    from bsimar.eval.visualization import (
        plot_scatter_comparison, plot_loss_curves,
    )

    epochs = epochs if epochs is not None else config.max_epochs
    batch_size = batch_size if batch_size is not None else config.batch_size
    patience = patience if patience is not None else config.patience
    lr = lr if lr is not None else config.lr

    device = torch.device(device_str)
    print(f"Device: {device} | Recipe: BSIMAR v3 "
          f"(asinh+zscore, MAE+LDS+VovLDS, parallel_caps, grouped_inputs) | "
          f"Data: {Path(data_path).name}")

    train_ds, val_ds, test_ds, normalizer = load_and_split_bsimar(
        str(data_path),
        column_names=OUTPUT_COLUMNS,
        norm_mode="asinh",
        apply_filter=True,
    )
    input_dim = train_ds.inputs.shape[1]
    output_dim = train_ds.outputs.shape[1]
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")

    # Reorder outputs to the BSIMAR paper AR order (charges → I-V → caps).
    train_ds.outputs = torch.tensor(
        reorder_outputs(train_ds.outputs.numpy()), dtype=torch.float32)
    val_ds.outputs = torch.tensor(
        reorder_outputs(val_ds.outputs.numpy()), dtype=torch.float32)
    test_ds.outputs = torch.tensor(
        reorder_outputs(test_ds.outputs.numpy()), dtype=torch.float32)
    print("Output columns reordered: charges->caps->cond->id")

    _unreorder_idx_t = torch.tensor(_UNREORDER_IDX, dtype=torch.long)

    model = TransformerEncoderModel(
        input_dim=input_dim,
        target_dim=output_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    # MAE loss + per-target LDS + Vov(=Vg) LDS (N7).
    criterion = MAELoss()
    print("Computing LDS weights (per-target + Vov)...")
    lds_weights_np = compute_lds_weights_per_target(
        train_ds.outputs.numpy(), n_bins=100,
        lds_kernel="gaussian", lds_ks=5, lds_sigma=0.8,
    )
    vg_col = train_ds.inputs[:, 1:2].numpy()
    vg_weights_np = compute_lds_weights_per_target(
        vg_col, n_bins=50,
        lds_kernel="gaussian", lds_ks=5, lds_sigma=1.0,
    )  # (N, 1)
    lds_weights_np = lds_weights_np * vg_weights_np
    col_means = lds_weights_np.mean(axis=0, keepdims=True)
    col_means[col_means < 1e-12] = 1.0
    lds_weights_np = lds_weights_np / col_means

    train_ds_weighted = TensorDataset(
        train_ds.inputs, train_ds.outputs,
        torch.tensor(lds_weights_np, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_ds_weighted, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINT_DIR / f"{save_prefix}_best.pt"
    norm_path = CHECKPOINT_DIR / f"{save_prefix}_norm.npz"
    results_subdir = str(RESULTS_DIR / save_prefix)

    # Refuse to silently overwrite an existing checkpoint. Concurrent
    # runs sharing --exp-name otherwise clobber each other.
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
    # T1: physical-space early-stopping tracker. Blends NRMSE + (1-R²)
    # so that a collapsed denorm can never beat a healthy one even if
    # NRMSE happens to look small.
    best_phys_score = float("inf")
    best_phys_nrmse = float("nan")
    best_phys_r2 = float("nan")
    phys_best_path = best_path.with_suffix(".phys.pt")
    # How often to run a full AR-validation as a ground-truth sanity
    # check on the TF-driven early stopping.
    ar_check_every = 10
    t_start = time.time()
    epoch = 0

    for epoch in range(1, epochs + 1):
        t_losses = train_epoch_mae(
            model, train_loader, criterion, optimizer, device)
        # Fast TF-based validation drives early stopping every epoch.
        v_losses = validate_epoch_tf(
            model, val_loader, criterion, device)

        run_ar_check = (epoch % ar_check_every == 0) or (epoch == epochs)
        ar_status = ""
        if run_ar_check:
            ar_v = validate_epoch_ar(
                model, val_loader, criterion, device)
            ar_loss = ar_v["total"]
            if ar_loss < best_ar_val_loss:
                best_ar_val_loss = ar_loss
                torch.save(model.state_dict(), str(ar_best_path))
                ar_status = " *ar-best*"
            print(f"  AR-val check @ epoch {epoch}: "
                  f"ar={ar_loss:.5f} (tf={v_losses['total']:.5f})"
                  f"{ar_status}")

            pred_val_norm, true_val_norm = test_model(
                model, val_loader, device)
            pred_val_norm = unreorder_outputs(pred_val_norm)
            true_val_norm = unreorder_outputs(true_val_norm)
            phys_metrics = compute_physical_metrics(
                pred_val_norm, true_val_norm, normalizer)
            nrmse_arr = np.array(
                [m["NRMSE(%)"] for m in phys_metrics.values()],
                dtype=np.float64)
            r2_arr = np.array(
                [m["R2"] for m in phys_metrics.values()],
                dtype=np.float64)
            nrmse_avg = float(np.nanmean(nrmse_arr))
            r2_avg = float(np.nanmean(r2_arr))
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
    }
    config_path = CHECKPOINT_DIR / f"{save_prefix}_config.npz"
    np.savez(str(config_path),
             **{k: np.array(v) for k, v in arch_config.items()})
    print(f"Arch config: {config_path}")

    # ── N3 — AR fine-tune phase ──────────────────────────────────────────
    # After the cosine TF schedule completes, run a short pure-AR
    # fine-tune phase (ss_ratio = 1.0) so the model trains on its own
    # decoded sequence and closes the residual TF↔AR gap. Loads the
    # phys-best checkpoint as the starting point, uses a fixed-low LR
    # (10× the final cosine LR, floored at 1e-5), and plain MAE (no
    # LDS during finetune).
    if ar_finetune_epochs > 0:
        if phys_best_path.exists():
            print(f"\n[N3] Loading phys-best checkpoint for AR finetune: "
                  f"{phys_best_path}")
            model.load_state_dict(
                torch.load(str(phys_best_path), weights_only=True))
        finetune_lr = max(scheduler.get_last_lr()[0] * 10, 1e-5)
        print(f"[N3] AR finetune for {ar_finetune_epochs} epochs at "
              f"lr={finetune_lr:.2e}, ss_ratio=1.0, criterion=MAELoss "
              f"(no LDS during finetune)")
        ft_optimizer = torch.optim.AdamW(
            model.parameters(), lr=finetune_lr,
            weight_decay=config.weight_decay)
        ft_criterion = MAELoss()
        ft_train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True)
        for ft_epoch in range(1, ar_finetune_epochs + 1):
            t_losses = train_epoch_scheduled_mae(
                model, ft_train_loader, ft_criterion, ft_optimizer,
                device, ss_ratio=1.0,
            )
            v_losses = validate_epoch_ar(
                model, val_loader, ft_criterion, device)
            train_loss = t_losses["total"]
            val_loss = v_losses["total"]
            train_history.append(train_loss)
            val_history.append(val_loss)

            pred_val_norm, true_val_norm = test_model(
                model, val_loader, device)
            pred_val_norm = unreorder_outputs(pred_val_norm)
            true_val_norm = unreorder_outputs(true_val_norm)
            phys_metrics = compute_physical_metrics(
                pred_val_norm, true_val_norm, normalizer)
            nrmse_arr = np.array(
                [m["NRMSE(%)"] for m in phys_metrics.values()],
                dtype=np.float64)
            r2_arr = np.array(
                [m["R2"] for m in phys_metrics.values()],
                dtype=np.float64)
            nrmse_avg = float(np.nanmean(nrmse_arr))
            r2_avg = float(np.nanmean(r2_arr))
            phys_score = (
                float("inf") if (np.isnan(nrmse_avg) or np.isnan(r2_avg))
                else nrmse_avg + 0.1 * (1.0 - r2_avg))
            phys_status = ""
            if phys_score < best_phys_score:
                best_phys_score = phys_score
                best_phys_nrmse = nrmse_avg
                best_phys_r2 = r2_avg
                torch.save(model.state_dict(), str(phys_best_path))
                phys_status = " *phys-best*"
            print(f"  [FT {ft_epoch:3d}] train={train_loss:.5f} "
                  f"val={val_loss:.5f} | nrmse={nrmse_avg:.3f}% "
                  f"r2={r2_avg:.4f}{phys_status}")

    # ── Final test ────────────────────────────────────────────────────────
    # Prefer the phys-space-best checkpoint for the final test if it
    # exists. Falls back to the TF-val-best checkpoint otherwise.
    if phys_best_path.exists():
        load_path = phys_best_path
        print(f"Loading phys-best checkpoint for final test: {load_path}")
    else:
        load_path = best_path
        print(f"Loading TF-val-best checkpoint for final test: {load_path}")
    model.load_state_dict(torch.load(str(load_path), weights_only=True))
    pred_norm, true_norm = test_model(model, test_loader, device)

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


# ══════════════════════════════════════════════════════════════════════════════
# BSIMAR v4 per-epoch helpers (tech-code-aware batches)
# ══════════════════════════════════════════════════════════════════════════════

def _train_epoch_mae_v4(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Teacher-forced training epoch for v4 (batches carry tech codes).

    Batch layout: (x, y, tech_codes) or (x, y, tech_codes, lds_weights).
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        if len(batch) == 4:
            x, y, tc, w = batch
            w = w.to(device)
        else:
            x, y, tc = batch
            w = None

        x = x.to(device)
        y = y.to(device)
        tc = tc.to(device)

        optimizer.zero_grad()
        pred = model(x, y, tech_codes=tc)

        loss = criterion(pred, y, weights=w) if w is not None else criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"total": total_loss / n_batches}


@torch.no_grad()
def _validate_epoch_tf_v4(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Teacher-forced validation for v4."""
    model.eval()
    total = 0.0
    n = 0
    for x, y, tc in loader:
        x, y, tc = x.to(device), y.to(device), tc.to(device)
        pred = model(x, y, tech_codes=tc)
        total += criterion(pred, y).item()
        n += 1
    return {"total": total / n}


@torch.no_grad()
def _validate_epoch_ar_v4(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Autoregressive validation for v4."""
    model.eval()
    total = 0.0
    n = 0
    for x, y, tc in loader:
        x, y, tc = x.to(device), y.to(device), tc.to(device)
        pred = model(x, tech_codes=tc)
        total += criterion(pred, y).item()
        n += 1
    return {"total": total / n}


def _train_epoch_scheduled_mae_v4(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    ss_ratio: float = 1.0,
) -> Dict[str, float]:
    """N3 AR fine-tune helper for v4."""
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        if len(batch) == 4:
            x, y, tc, w = batch
            w = w.to(device)
        else:
            x, y, tc = batch
            w = None

        x, y, tc = x.to(device), y.to(device), tc.to(device)

        optimizer.zero_grad()
        pred = model.forward_scheduled(x, y, ss_ratio=ss_ratio, tech_codes=tc)
        loss = criterion(pred, y, weights=w) if w is not None else criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return {"total": total_loss / n}


def _print_per_tech_metrics(
    pred_norm: np.ndarray,
    true_norm: np.ndarray,
    tech_codes: np.ndarray,
    normalizer,
) -> None:
    """Print per-tech-variant physical metrics (NRMSE, R²) on test set."""
    from bsimar.config import CODE_TO_TECH_VARIANT
    from bsimar.eval.metrics import compute_physical_metrics

    unique_codes = np.unique(tech_codes)
    print(f"\n{'Tech':>15s} | {'n_test':>6s} | {'NRMSE%':>8s} | {'R2':>8s}")
    print("-" * 50)
    all_nrmse, all_r2 = [], []
    for code in sorted(unique_codes):
        mask = tech_codes == code
        tech_name, variant = CODE_TO_TECH_VARIANT.get(
            int(code), ("unk", "unk"))
        label = f"{tech_name}:{variant}"
        m = compute_physical_metrics(
            pred_norm[mask], true_norm[mask], normalizer)
        nrmse_vals = [v["NRMSE(%)"] for v in m.values()
                      if not np.isnan(v["NRMSE(%)"])]
        r2_vals = [v["R2"] for v in m.values()
                   if not np.isnan(v["R2"])]
        avg_nrmse = float(np.mean(nrmse_vals)) if nrmse_vals else float("nan")
        avg_r2 = float(np.mean(r2_vals)) if r2_vals else float("nan")
        print(f"{label:>15s} | {mask.sum():6d} | {avg_nrmse:8.3f} | {avg_r2:8.4f}")
        if not np.isnan(avg_nrmse):
            all_nrmse.append(avg_nrmse)
        if not np.isnan(avg_r2):
            all_r2.append(avg_r2)
    print("-" * 50)
    print(f"{'OVERALL':>15s} | {len(tech_codes):6d} | "
          f"{np.mean(all_nrmse):8.3f} | {np.mean(all_r2):8.4f}")


@torch.no_grad()
def _test_model_v4(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run AR inference for v4, return (pred_norm, true_norm)."""
    model.eval()
    all_pred, all_true = [], []
    for x, y, tc in loader:
        x, tc = x.to(device), tc.to(device)
        pred = model(x, tech_codes=tc)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.numpy())
    return np.concatenate(all_pred), np.concatenate(all_true)


# ══════════════════════════════════════════════════════════════════════════════
# v4 High-level pipeline
# ══════════════════════════════════════════════════════════════════════════════

def train_transformer_v4(
    data_path: str,
    save_prefix: str,
    device_type: str = "nmos",
    config: TransformerConfig = TransformerConfig(),
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    patience: Optional[int] = None,
    lr: Optional[float] = None,
    device_str: str = "cpu",
    ar_finetune_epochs: int = 5,
    overwrite: bool = False,
    exclude_techs: Optional[Set[str]] = None,
    num_tech_codes: int = NUM_TSMC_CODES_WITH_UNKNOWN,
    p_unknown: float = 0.1,
) -> Tuple[nn.Module, BSIMARNormalizer]:
    """BSIMAR v4 Transformer training pipeline.

    Same v3 recipe (MAE+LDS+VovLDS, asinh, reorder, AR finetune,
    phys-best ckpt), but input is 7-dim continuous + discrete tech codes.

    Args:
        data_path: Path to universal .npz dataset.
        save_prefix: Prefix for checkpoint files.
        device_type: "nmos" or "pmos" (for tech labeling).
        exclude_techs: Tech names to exclude entirely (e.g., {"asap7"}).
        num_tech_codes: Embedding vocabulary size.
        p_unknown: Prob of replacing tech code with UNKNOWN during training.
    """
    from bsimar.config import OUTPUT_COLUMNS, INPUT_DIM_V4
    from bsimar.data.dataset import load_and_split_bsimar_v4
    from bsimar.eval.metrics import compute_physical_metrics, print_metrics
    from bsimar.eval.visualization import (
        plot_scatter_comparison, plot_loss_curves,
    )

    epochs = epochs if epochs is not None else config.max_epochs
    batch_size = batch_size if batch_size is not None else config.batch_size
    patience = patience if patience is not None else config.patience
    lr = lr if lr is not None else config.lr

    device = torch.device(device_str)
    print(f"Device: {device} | Recipe: BSIMAR v4 "
          f"(tech-code embedding, {num_tech_codes} codes, "
          f"p_unknown={p_unknown}) | Data: {Path(data_path).name}")
    if exclude_techs:
        print(f"Excluded techs: {exclude_techs}")

    train_ds, val_ds, test_ds, normalizer = load_and_split_bsimar_v4(
        str(data_path),
        column_names=OUTPUT_COLUMNS,
        device_type=device_type,
        apply_filter=True,
        exclude_techs=exclude_techs,
    )
    input_dim = train_ds.inputs.shape[1]
    output_dim = train_ds.outputs.shape[1]
    assert input_dim == INPUT_DIM_V4, (
        f"Expected v4 input dim {INPUT_DIM_V4}, got {input_dim}")
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")

    # Reorder outputs to BSIMAR paper AR order.
    train_ds.outputs = torch.tensor(
        reorder_outputs(train_ds.outputs.numpy()), dtype=torch.float32)
    val_ds.outputs = torch.tensor(
        reorder_outputs(val_ds.outputs.numpy()), dtype=torch.float32)
    test_ds.outputs = torch.tensor(
        reorder_outputs(test_ds.outputs.numpy()), dtype=torch.float32)
    print("Output columns reordered: charges->caps->cond->id")

    model = TransformerEncoderModel(
        input_dim=input_dim,
        target_dim=output_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        use_tech_codes=True,
        num_tech_codes=num_tech_codes,
        tech_embed_dropout=p_unknown,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    # MAE loss + per-target LDS + Vov(=Vg) LDS.
    criterion = MAELoss()
    print("Computing LDS weights (per-target + Vov)...")
    lds_weights_np = compute_lds_weights_per_target(
        train_ds.outputs.numpy(), n_bins=100,
        lds_kernel="gaussian", lds_ks=5, lds_sigma=0.8,
    )
    vg_col = train_ds.inputs[:, 1:2].numpy()
    vg_weights_np = compute_lds_weights_per_target(
        vg_col, n_bins=50,
        lds_kernel="gaussian", lds_ks=5, lds_sigma=1.0,
    )
    lds_weights_np = lds_weights_np * vg_weights_np
    col_means = lds_weights_np.mean(axis=0, keepdims=True)
    col_means[col_means < 1e-12] = 1.0
    lds_weights_np = lds_weights_np / col_means

    # Build weighted TensorDataset: (x, y, tech_codes, lds_weights).
    train_ds_weighted = TensorDataset(
        train_ds.inputs, train_ds.outputs, train_ds.tech_codes,
        torch.tensor(lds_weights_np, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_ds_weighted, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINT_DIR / f"{save_prefix}_best.pt"
    norm_path = CHECKPOINT_DIR / f"{save_prefix}_norm.npz"
    results_subdir = str(RESULTS_DIR / save_prefix)

    if best_path.exists() and not overwrite:
        raise SystemExit(
            f"Refusing to overwrite {best_path}. Pass --overwrite or "
            "choose a unique --exp-name.")

    early_stopping = EarlyStopping(
        patience=patience, min_delta=1e-5, save_path=str(best_path))

    print(f"\nTraining {save_prefix} for {epochs} epochs (patience={patience})")
    train_history, val_history = [], []
    best_val_loss = float("inf")
    best_ar_val_loss = float("inf")
    ar_best_path = best_path.with_suffix(".ar.pt")
    best_phys_score = float("inf")
    best_phys_nrmse = float("nan")
    best_phys_r2 = float("nan")
    phys_best_path = best_path.with_suffix(".phys.pt")
    ar_check_every = 10
    t_start = time.time()
    epoch = 0

    for epoch in range(1, epochs + 1):
        t_losses = _train_epoch_mae_v4(
            model, train_loader, criterion, optimizer, device)
        v_losses = _validate_epoch_tf_v4(
            model, val_loader, criterion, device)

        run_ar_check = (epoch % ar_check_every == 0) or (epoch == epochs)
        ar_status = ""
        if run_ar_check:
            ar_v = _validate_epoch_ar_v4(
                model, val_loader, criterion, device)
            ar_loss = ar_v["total"]
            if ar_loss < best_ar_val_loss:
                best_ar_val_loss = ar_loss
                torch.save(model.state_dict(), str(ar_best_path))
                ar_status = " *ar-best*"
            print(f"  AR-val check @ epoch {epoch}: "
                  f"ar={ar_loss:.5f} (tf={v_losses['total']:.5f})"
                  f"{ar_status}")

            pred_val_norm, true_val_norm = _test_model_v4(
                model, val_loader, device)
            pred_val_norm = unreorder_outputs(pred_val_norm)
            true_val_norm = unreorder_outputs(true_val_norm)
            phys_metrics = compute_physical_metrics(
                pred_val_norm, true_val_norm, normalizer)
            nrmse_arr = np.array(
                [m["NRMSE(%)"] for m in phys_metrics.values()],
                dtype=np.float64)
            r2_arr = np.array(
                [m["R2"] for m in phys_metrics.values()],
                dtype=np.float64)
            nrmse_avg = float(np.nanmean(nrmse_arr))
            r2_avg = float(np.nanmean(r2_arr))
            phys_score = (
                float("inf") if (np.isnan(nrmse_avg) or np.isnan(r2_avg))
                else nrmse_avg + 0.1 * (1.0 - r2_avg))
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

    # Save arch config with v4 metadata.
    arch_config = {
        "input_dim": input_dim, "target_dim": output_dim,
        "d_model": config.d_model, "nhead": config.nhead,
        "num_layers": config.num_layers,
        "dim_feedforward": config.dim_feedforward,
        "dropout": config.dropout,
        "use_tech_codes": True,
        "num_tech_codes": num_tech_codes,
    }
    config_path = CHECKPOINT_DIR / f"{save_prefix}_config.npz"
    np.savez(str(config_path),
             **{k: np.array(v) for k, v in arch_config.items()})
    print(f"Arch config: {config_path}")

    # ── N3 — AR fine-tune phase ──────────────────────────────────────────
    if ar_finetune_epochs > 0:
        if phys_best_path.exists():
            print(f"\n[N3] Loading phys-best checkpoint for AR finetune: "
                  f"{phys_best_path}")
            model.load_state_dict(
                torch.load(str(phys_best_path), weights_only=True))
        finetune_lr = max(scheduler.get_last_lr()[0] * 10, 1e-5)
        print(f"[N3] AR finetune for {ar_finetune_epochs} epochs at "
              f"lr={finetune_lr:.2e}, ss_ratio=1.0")
        ft_optimizer = torch.optim.AdamW(
            model.parameters(), lr=finetune_lr,
            weight_decay=config.weight_decay)
        ft_criterion = MAELoss()
        ft_train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True)
        for ft_epoch in range(1, ar_finetune_epochs + 1):
            t_losses = _train_epoch_scheduled_mae_v4(
                model, ft_train_loader, ft_criterion, ft_optimizer,
                device, ss_ratio=1.0,
            )
            v_losses = _validate_epoch_ar_v4(
                model, val_loader, ft_criterion, device)
            train_loss = t_losses["total"]
            val_loss = v_losses["total"]
            train_history.append(train_loss)
            val_history.append(val_loss)

            pred_val_norm, true_val_norm = _test_model_v4(
                model, val_loader, device)
            pred_val_norm = unreorder_outputs(pred_val_norm)
            true_val_norm = unreorder_outputs(true_val_norm)
            phys_metrics = compute_physical_metrics(
                pred_val_norm, true_val_norm, normalizer)
            nrmse_arr = np.array(
                [m["NRMSE(%)"] for m in phys_metrics.values()],
                dtype=np.float64)
            r2_arr = np.array(
                [m["R2"] for m in phys_metrics.values()],
                dtype=np.float64)
            nrmse_avg = float(np.nanmean(nrmse_arr))
            r2_avg = float(np.nanmean(r2_arr))
            phys_score = (
                float("inf") if (np.isnan(nrmse_avg) or np.isnan(r2_avg))
                else nrmse_avg + 0.1 * (1.0 - r2_avg))
            phys_status = ""
            if phys_score < best_phys_score:
                best_phys_score = phys_score
                best_phys_nrmse = nrmse_avg
                best_phys_r2 = r2_avg
                torch.save(model.state_dict(), str(phys_best_path))
                phys_status = " *phys-best*"
            print(f"  [FT {ft_epoch:3d}] train={train_loss:.5f} "
                  f"val={val_loss:.5f} | nrmse={nrmse_avg:.3f}% "
                  f"r2={r2_avg:.4f}{phys_status}")

    # ── Final test ────────────────────────────────────────────────────────
    if phys_best_path.exists():
        load_path = phys_best_path
        print(f"Loading phys-best checkpoint for final test: {load_path}")
    else:
        load_path = best_path
        print(f"Loading TF-val-best checkpoint for final test: {load_path}")
    model.load_state_dict(torch.load(str(load_path), weights_only=True))
    pred_norm, true_norm = _test_model_v4(model, test_loader, device)

    pred_norm = unreorder_outputs(pred_norm)
    true_norm = unreorder_outputs(true_norm)

    metrics = compute_physical_metrics(pred_norm, true_norm, normalizer)
    print_metrics(metrics)

    # Per-tech breakdown
    test_tc = test_ds.tech_codes.numpy()
    _print_per_tech_metrics(pred_norm, true_norm, test_tc, normalizer)

    pred_phys = normalizer.denormalize_outputs(pred_norm)
    true_phys = normalizer.denormalize_outputs(true_norm)
    plot_scatter_comparison(true_phys, pred_phys, results_subdir)
    plot_loss_curves(train_history, val_history, results_subdir,
                     title_prefix=f"BSIM-AR v4 {save_prefix} ")

    print(f"\nCheckpoint: {best_path}")
    print(f"Norm stats: {norm_path}")
    print(f"Results:    {results_subdir}/")

    return model, normalizer
