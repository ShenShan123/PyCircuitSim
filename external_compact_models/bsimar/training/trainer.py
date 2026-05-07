"""Training pipelines for DirectNet and BSIMAR Transformer.

Two public entry points:

- ``train_directnet``    — DirectNet MLP with tech-code embedding.
- ``train_transformer``  — BSIMAR Transformer (7-dim input + discrete tech codes).

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
from bsimar.data.normalize import (
    BSIMARNormalizer,
    reorder_outputs, unreorder_outputs,
)
from bsimar.models.transformer import TransformerEncoderModel
from bsimar.losses.bni_mae import (
    MAELoss, JacobianConsistencyLoss, compute_lds_weights_per_target,
)
from bsimar.training.early_stopping import EarlyStopping


# ══════════════════════════════════════════════════════════════════════════════
# DirectNet per-epoch helpers
# ══════════════════════════════════════════════════════════════════════════════

def _train_epoch_direct(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    jac_loss: JacobianConsistencyLoss | None = None,
    jac_norm_tensors: dict | None = None,
) -> Dict[str, float]:
    """Training epoch for DirectNet (MAE + LDS weights).

    Batch layout: (x, y, tech_codes) or (x, y, tech_codes, lds_weights).
    When ``jac_loss`` is provided, ``λ_jac · L_jac`` is added to the
    main MAE term; ``jac_norm_tensors`` must contain ``in_std``,
    ``out_std``, ``out_mean`` (and optionally ``asinh_scale``) for
    the chain-rule conversion.
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_jac = 0.0
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
        if jac_loss is not None:
            x = x.requires_grad_(True)
        pred = model(x, tech_codes=tc)
        mae = criterion(pred, y, weights=w) if w is not None else criterion(pred, y)
        if jac_loss is not None:
            jac_term = jac_loss(
                x_norm=x, y_pred_norm=pred, y_true_norm=y,
                in_std=jac_norm_tensors["in_std"],
                out_std=jac_norm_tensors["out_std"],
                out_mean=jac_norm_tensors["out_mean"],
                asinh_scale=jac_norm_tensors.get("asinh_scale"),
                weights=w,
            )
            loss = mae + jac_term
            total_jac += jac_term.item()
        else:
            loss = mae
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_mae += mae.item()
        n += 1
    return {
        "total": total_loss / n,
        "mae": total_mae / n,
        "jac": total_jac / n,
    }


@torch.no_grad()
def _validate_epoch_direct(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validation epoch for DirectNet (unweighted MAE)."""
    model.eval()
    total = 0.0
    n = 0
    for _b in loader:
        x, y, tc = _b[0], _b[1], _b[2]
        x, y, tc = x.to(device), y.to(device), tc.to(device)
        pred = model(x, tech_codes=tc)
        total += criterion(pred, y).item()
        n += 1
    return {"total": total / n}


@torch.no_grad()
def _collect_directnet_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run DirectNet on loader, return (pred_norm, true_norm, tech_codes)."""
    model.eval()
    all_pred, all_true, all_tc = [], [], []
    for _b in loader:
        x, y, tc = _b[0], _b[1], _b[2]
        x, tc = x.to(device), tc.to(device)
        pred = model(x, tech_codes=tc)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.numpy())
        all_tc.append(tc.cpu().numpy())
    return (np.concatenate(all_pred),
            np.concatenate(all_true),
            np.concatenate(all_tc))


# ══════════════════════════════════════════════════════════════════════════════
# BSIMAR Transformer per-epoch helpers (tech-code-aware batches)
# ══════════════════════════════════════════════════════════════════════════════

def _train_epoch_mae(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    jac_loss: JacobianConsistencyLoss | None = None,
    jac_norm_tensors: dict | None = None,
) -> Dict[str, float]:
    """Teacher-forced training epoch (batches carry tech codes).

    Batch layout: (x, y, tech_codes) or (x, y, tech_codes, lds_weights).
    When ``jac_loss`` is provided, ``λ_jac · L_jac`` is added to the
    main MAE term; targets are in BSIMAR_COLUMN_ORDER (the JAC loss is
    constructed with that column order so indexing is correct).

    Note: when JAC is on, we wrap the body in a context that disables
    flash and memory-efficient SDPA attention, since neither's backward
    supports second-order derivatives (PyTorch 2.x).  Falls back to the
    math-kernel path which supports double-backward at the cost of
    ~30 % slower attention.
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_jac = 0.0
    n_batches = 0

    # Disable flash + memory-efficient attention for the JAC path; both
    # break under double-backward.
    if jac_loss is not None:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        sdp_ctx = sdpa_kernel([SDPBackend.MATH])
    else:
        from contextlib import nullcontext
        sdp_ctx = nullcontext()

    with sdp_ctx:
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
            if jac_loss is not None:
                x = x.requires_grad_(True)
            pred = model(x, y, tech_codes=tc)

            mae = (criterion(pred, y, weights=w)
                   if w is not None else criterion(pred, y))
            if jac_loss is not None:
                jac_term = jac_loss(
                    x_norm=x, y_pred_norm=pred, y_true_norm=y,
                    in_std=jac_norm_tensors["in_std"],
                    out_std=jac_norm_tensors["out_std"],
                    out_mean=jac_norm_tensors["out_mean"],
                    asinh_scale=jac_norm_tensors.get("asinh_scale"),
                    weights=w,
                )
                loss = mae + jac_term
                total_jac += jac_term.item()
            else:
                loss = mae
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_mae += mae.item()
            n_batches += 1

    return {
        "total": total_loss / n_batches,
        "mae": total_mae / n_batches,
        "jac": total_jac / n_batches,
    }


@torch.no_grad()
def _validate_epoch_tf(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Teacher-forced validation."""
    model.eval()
    total = 0.0
    n = 0
    for _b in loader:
        x, y, tc = _b[0], _b[1], _b[2]
        x, y, tc = x.to(device), y.to(device), tc.to(device)
        pred = model(x, y, tech_codes=tc)
        total += criterion(pred, y).item()
        n += 1
    return {"total": total / n}


@torch.no_grad()
def _validate_epoch_ar(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Autoregressive validation."""
    model.eval()
    total = 0.0
    n = 0
    for _b in loader:
        x, y, tc = _b[0], _b[1], _b[2]
        x, y, tc = x.to(device), y.to(device), tc.to(device)
        pred = model(x, tech_codes=tc)
        total += criterion(pred, y).item()
        n += 1
    return {"total": total / n}


@torch.no_grad()
def test_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run AR inference, return (pred_norm, true_norm)."""
    model.eval()
    all_pred, all_true = [], []
    for _b in loader:
        x, y, tc = _b[0], _b[1], _b[2]
        x, tc = x.to(device), tc.to(device)
        pred = model(x, tech_codes=tc)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.numpy())
    return np.concatenate(all_pred), np.concatenate(all_true)


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# High-level pipelines
# ══════════════════════════════════════════════════════════════════════════════

def train_directnet(
    data_path: str,
    device_type: str = "nmos",
    config: DirectNetConfig = DirectNetConfig(),
    device_str: str = "cpu",
    save_prefix: str = "v4_re_dn_universal_nmos",
    exclude_techs: Optional[Set[str]] = None,
    num_tech_codes: int = NUM_TSMC_CODES_WITH_UNKNOWN,
    p_unknown: float = 0.1,
    jacobian_consistency: bool = False,
    lam_jac: float = 0.1,
) -> Tuple[nn.Module, BSIMARNormalizer]:
    """DirectNet training pipeline (7-dim input + tech-code embedding).

    Uses the data loader (``load_and_split_bsimar``) and
    ``DirectNet`` model with discrete tech-code embedding, making it
    directly comparable to the BSIMAR Transformer.
    """
    from bsimar.config import OUTPUT_COLUMNS
    from bsimar.data.dataset import load_and_split_bsimar
    from bsimar.models.direct_net import DirectNet

    device = torch.device(device_str)
    print(f"Training DirectNet on {device}")
    print(f"Tech codes: {num_tech_codes} codes, p_unknown={p_unknown}")
    if exclude_techs:
        print(f"Excluding techs: {exclude_techs}")

    train_ds, val_ds, test_ds, normalizer = load_and_split_bsimar(
        data_path,
        column_names=OUTPUT_COLUMNS,
        device_type=device_type,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        apply_filter=True,
        exclude_techs=exclude_techs,
        norm_mode="zscore",
    )

    input_dim = train_ds.inputs.shape[1]
    output_dim = train_ds.outputs.shape[1]
    print(f"Input dim: {input_dim} (7-dim + tech code), Output dim: {output_dim}")

    # MAE loss + per-target LDS (paper recipe).
    criterion = MAELoss()
    print("Computing LDS weights (per-target)...")
    lds_weights_np = compute_lds_weights_per_target(
        train_ds.outputs.numpy(), n_bins=100,
        lds_kernel="gaussian", lds_ks=5, lds_sigma=0.8,
    )
    col_means = lds_weights_np.mean(axis=0, keepdims=True)
    col_means[col_means < 1e-12] = 1.0
    lds_weights_np = lds_weights_np / col_means

    train_ds_weighted = TensorDataset(
        train_ds.inputs, train_ds.outputs, train_ds.tech_codes,
        torch.tensor(lds_weights_np, dtype=torch.float32),
    )
    # 8 worker threads + pinned memory: prevents the dataloader from
    # being the GPU bottleneck on 10M+ row datasets (V5 Phase C).
    _NW = 8
    train_loader = DataLoader(
        train_ds_weighted, batch_size=config.batch_size, shuffle=True,
        num_workers=_NW, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=_NW, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=_NW, pin_memory=True, persistent_workers=True)

    model = DirectNet(
        input_dim=input_dim,
        hidden_dim=config.trunk_hidden,
        n_layers=config.trunk_layers + 1,
        output_dim=output_dim,
        num_tech_codes=num_tech_codes,
        tech_embed_dim=32,
        tech_embed_dropout=p_unknown,
    ).to(device)
    print(f"Model parameters: {model.count_parameters()}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    best_val_loss = float("inf")
    patience_counter = 0

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINT_DIR / f"{save_prefix}_best.pt"
    norm_path = CHECKPOINT_DIR / f"{save_prefix}_norm.npz"

    # ── Optional Jacobian-consistency aux loss (V5 Phase C) ─────────────
    jac_loss: JacobianConsistencyLoss | None = None
    jac_norm_tensors: dict | None = None
    if jacobian_consistency:
        jac_loss = JacobianConsistencyLoss(
            lam=lam_jac, norm_mode=normalizer.stats.mode,
            column_order=OUTPUT_COLUMNS,
        )
        in_std_t = torch.tensor(
            normalizer.stats.input_std, dtype=torch.float32, device=device)
        out_std_t = torch.tensor(
            normalizer.stats.output_std, dtype=torch.float32, device=device)
        out_mean_t = torch.tensor(
            normalizer.stats.output_mean, dtype=torch.float32, device=device)
        # zscore for DirectNet — no asinh_scale needed
        jac_norm_tensors = {
            "in_std": in_std_t, "out_std": out_std_t, "out_mean": out_mean_t,
        }
        if normalizer.stats.mode == "asinh":
            jac_norm_tensors["asinh_scale"] = torch.tensor(
                normalizer.stats.asinh_scale, dtype=torch.float32,
                device=device)
        print(f"Jacobian-consistency loss enabled (λ_jac={lam_jac})")

    t_start = time.time()
    epoch = 0

    for epoch in range(1, config.max_epochs + 1):
        train_losses = _train_epoch_direct(
            model, train_loader, criterion, optimizer, device,
            jac_loss=jac_loss, jac_norm_tensors=jac_norm_tensors)
        val_losses = _validate_epoch_direct(
            model, val_loader, criterion, device)
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
            extra = ""
            if jac_loss is not None:
                extra = (f" mae={train_losses['mae']:.5f} "
                         f"jac={train_losses['jac']:.5f}")
            print(f"{epoch:4d} | train={train_losses['total']:.5f}{extra} "
                  f"val={val_losses['total']:.5f} lr={lr:.2e}{status}")

        if patience_counter >= config.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(patience={config.patience})")
            break

    elapsed = time.time() - t_start
    epochs_run = max(epoch, 1)
    print(f"\nTraining completed in {elapsed:.0f}s "
          f"({elapsed / epochs_run:.1f}s/epoch)")
    print(f"Best val loss: {best_val_loss:.6f}")

    # Test evaluation.
    model.load_state_dict(torch.load(best_path, weights_only=True))
    test_losses = _validate_epoch_direct(
        model, test_loader, criterion, device)
    print(f"\nTest set losses:")
    for k, v in sorted(test_losses.items()):
        print(f"  {k:>10s}: {v:.6f}")

    from bsimar.eval.metrics import compute_physical_metrics, print_metrics
    pred_norm, true_norm, test_tech_codes = _collect_directnet_predictions(
        model, test_loader, device)
    metrics = compute_physical_metrics(pred_norm, true_norm, normalizer)
    print(f"\nPhysical metrics (test set):")
    print_metrics(metrics)

    _print_per_tech_metrics(
        pred_norm, true_norm, test_tech_codes, normalizer)

    print(f"\nSaved: {best_path}")
    return model, normalizer


def train_transformer(
    data_path: str,
    save_prefix: str,
    device_type: str = "nmos",
    config: TransformerConfig = TransformerConfig(),
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    patience: Optional[int] = None,
    lr: Optional[float] = None,
    device_str: str = "cpu",
    overwrite: bool = False,
    exclude_techs: Optional[Set[str]] = None,
    num_tech_codes: int = NUM_TSMC_CODES_WITH_UNKNOWN,
    p_unknown: float = 0.1,
    prebuilt_data: Optional[Tuple[object, object, object, BSIMARNormalizer]] = None,
    jacobian_consistency: bool = False,
    lam_jac: float = 0.1,
) -> Tuple[nn.Module, BSIMARNormalizer]:
    """BSIMAR Transformer training pipeline.

    Uses the recipe (MAE+LDS, asinh, reorder, phys-best ckpt) with
    7-dim continuous input + discrete tech codes.

    Args:
        data_path: Path to universal .npz dataset.
        save_prefix: Prefix for checkpoint files.
        device_type: "nmos" or "pmos" (for tech labeling).
        exclude_techs: Tech names to exclude entirely (e.g., {"asap7"}).
        num_tech_codes: Embedding vocabulary size.
        p_unknown: Prob of replacing tech code with UNKNOWN during training.
    """
    from bsimar.config import OUTPUT_COLUMNS, INPUT_DIM
    from bsimar.data.dataset import load_and_split_bsimar
    from bsimar.eval.metrics import compute_physical_metrics, print_metrics
    from bsimar.eval.visualization import (
        plot_scatter_comparison, plot_loss_curves,
    )

    epochs = epochs if epochs is not None else config.max_epochs
    batch_size = batch_size if batch_size is not None else config.batch_size
    patience = patience if patience is not None else config.patience
    lr = lr if lr is not None else config.lr

    device = torch.device(device_str)
    print(f"Device: {device} | Recipe: BSIMAR "
          f"(tech-code embedding, {num_tech_codes} codes, "
          f"p_unknown={p_unknown}) | Data: {Path(data_path).name}")
    if exclude_techs:
        print(f"Excluded techs: {exclude_techs}")

    if prebuilt_data is not None:
        train_ds, val_ds, test_ds, normalizer = prebuilt_data
        print(f"[prebuilt] Using caller-supplied datasets: "
              f"train={len(train_ds)}, val={len(val_ds)}, "
              f"test={len(test_ds)} "
              f"(load_and_split_bsimar bypassed)")
    else:
        train_ds, val_ds, test_ds, normalizer = load_and_split_bsimar(
            str(data_path),
            column_names=OUTPUT_COLUMNS,
            device_type=device_type,
            apply_filter=True,
            exclude_techs=exclude_techs,
        )
    input_dim = train_ds.inputs.shape[1]
    output_dim = train_ds.outputs.shape[1]
    assert input_dim == INPUT_DIM, (
        f"Expected input dim {INPUT_DIM}, got {input_dim}")
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
        num_tech_codes=num_tech_codes,
        tech_embed_dropout=p_unknown,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    # MAE loss + per-target LDS (paper recipe).
    criterion = MAELoss()
    print("Computing LDS weights (per-target)...")
    lds_weights_np = compute_lds_weights_per_target(
        train_ds.outputs.numpy(), n_bins=100,
        lds_kernel="gaussian", lds_ks=5, lds_sigma=0.8,
    )
    col_means = lds_weights_np.mean(axis=0, keepdims=True)
    col_means[col_means < 1e-12] = 1.0
    lds_weights_np = lds_weights_np / col_means

    train_ds_weighted = TensorDataset(
        train_ds.inputs, train_ds.outputs, train_ds.tech_codes,
        torch.tensor(lds_weights_np, dtype=torch.float32),
    )
    # Use multiple worker threads + pinned memory to prevent the dataloader
    # from being the GPU bottleneck. Without these, the trainer's per-epoch
    # time on A100/Blackwell drops by ~10x for large models.
    _NW = 8
    train_loader = DataLoader(
        train_ds_weighted, batch_size=batch_size, shuffle=True,
        num_workers=_NW, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=_NW, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=_NW, pin_memory=True, persistent_workers=True)

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

    # Median-based phys-score (post-fix); flags _best.phys.pt as trustworthy
    # so the simulator loader will prefer it over _best.pt.
    normalizer.stats.phys_best_metric = "median"

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

    # ── Optional Jacobian-consistency aux loss (V5 Phase C) ─────────────
    # Targets are in BSIMAR_COLUMN_ORDER (post reorder_outputs), so the
    # JAC loss must use that column order for indexing.
    jac_loss: JacobianConsistencyLoss | None = None
    jac_norm_tensors: dict | None = None
    if jacobian_consistency:
        from bsimar.data.normalize import BSIMAR_COLUMN_ORDER
        # The norm stats live in OUTPUT_COLUMN_ORDER; we need them
        # permuted to BSIMAR_COLUMN_ORDER to match the target tensor.
        from bsimar.data.normalize import OUTPUT_COLUMN_ORDER
        perm = [OUTPUT_COLUMN_ORDER.index(c) for c in BSIMAR_COLUMN_ORDER]
        jac_loss = JacobianConsistencyLoss(
            lam=lam_jac, norm_mode=normalizer.stats.mode,
            column_order=BSIMAR_COLUMN_ORDER,
        )
        in_std_t = torch.tensor(
            normalizer.stats.input_std, dtype=torch.float32, device=device)
        out_std_t = torch.tensor(
            normalizer.stats.output_std[perm], dtype=torch.float32, device=device)
        out_mean_t = torch.tensor(
            normalizer.stats.output_mean[perm], dtype=torch.float32, device=device)
        jac_norm_tensors = {
            "in_std": in_std_t, "out_std": out_std_t, "out_mean": out_mean_t,
        }
        if normalizer.stats.mode == "asinh":
            jac_norm_tensors["asinh_scale"] = torch.tensor(
                normalizer.stats.asinh_scale[perm], dtype=torch.float32,
                device=device)
        print(f"Jacobian-consistency loss enabled (λ_jac={lam_jac})")
    t_start = time.time()
    epoch = 0

    for epoch in range(1, epochs + 1):
        t_losses = _train_epoch_mae(
            model, train_loader, criterion, optimizer, device,
            jac_loss=jac_loss, jac_norm_tensors=jac_norm_tensors)
        v_losses = _validate_epoch_tf(
            model, val_loader, criterion, device)

        run_ar_check = (epoch % ar_check_every == 0) or (epoch == epochs)
        ar_status = ""
        if run_ar_check:
            ar_v = _validate_epoch_ar(
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
            # Median over outputs is robust to a single-column blowup
            # (e.g. id under AR-rollout sinh overflow, see plan §2B).
            nrmse_med = float(np.nanmedian(nrmse_arr))
            r2_med = float(np.nanmedian(r2_arr))
            phys_score = (
                float("inf") if (np.isnan(nrmse_med) or np.isnan(r2_med))
                else nrmse_med + 0.1 * (1.0 - r2_med))
            phys_status = ""
            if phys_score < best_phys_score:
                best_phys_score = phys_score
                best_phys_nrmse = nrmse_med
                best_phys_r2 = r2_med
                torch.save(model.state_dict(), str(phys_best_path))
                phys_status = " *phys-best*"
            print(f"  PHYS-val @ epoch {epoch}: "
                  f"nrmse_med={nrmse_med:.3f} r2_med={r2_med:.4f} "
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
            extra = ""
            if jac_loss is not None:
                extra = (f" mae={t_losses['mae']:.5f} "
                         f"jac={t_losses['jac']:.5f}")
            print(f"  {epoch:4d} | train={train_loss:.5f}{extra} "
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

    # Save arch config with metadata.
    arch_config = {
        "input_dim": input_dim, "target_dim": output_dim,
        "d_model": config.d_model, "nhead": config.nhead,
        "num_layers": config.num_layers,
        "dim_feedforward": config.dim_feedforward,
        "dropout": config.dropout,
        "num_tech_codes": num_tech_codes,
    }
    config_path = CHECKPOINT_DIR / f"{save_prefix}_config.npz"
    np.savez(str(config_path),
             **{k: np.array(v) for k, v in arch_config.items()})
    print(f"Arch config: {config_path}")

    # ── Final test ────────────────────────────────────────────────────────
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

    # Per-tech breakdown
    test_tc = test_ds.tech_codes.numpy()
    _print_per_tech_metrics(pred_norm, true_norm, test_tc, normalizer)

    pred_phys = normalizer.denormalize_outputs(pred_norm)
    true_phys = normalizer.denormalize_outputs(true_norm)
    plot_scatter_comparison(true_phys, pred_phys, results_subdir)
    plot_loss_curves(train_history, val_history, results_subdir,
                     title_prefix=f"BSIM-AR {save_prefix} ")

    print(f"\nCheckpoint: {best_path}")
    print(f"Norm stats: {norm_path}")
    print(f"Results:    {results_subdir}/")

    return model, normalizer
