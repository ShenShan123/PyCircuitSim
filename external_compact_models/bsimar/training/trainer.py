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
    reorder_outputs, unreorder_outputs, _UNREORDER_IDX,
)
from bsimar.models.transformer import TransformerEncoderModel
from bsimar.models.id_gate import apply_id_gate
from bsimar.losses.bni_mae import (
    MAELoss, compute_lds_weights_per_target,
)
from bsimar.losses.slope_loss import SlopeMatchLoss
from bsimar.training.early_stopping import EarlyStopping


# Column indices for the structural Vds gate (B3). DirectNet outputs in
# OUTPUT_COLUMN_ORDER (id at 0); BSIMAR Transformer outputs in
# BSIMAR_COLUMN_ORDER (id at 4).
_DN_ID_IDX = 0
_TF_ID_IDX = 4
_VT_ARCH = 0.04  # gate width in volts (plan §4 B3)


# ══════════════════════════════════════════════════════════════════════════════
# DirectNet per-epoch helpers
# ══════════════════════════════════════════════════════════════════════════════

def _train_epoch_direct(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    slope_loss: Optional[nn.Module] = None,
    slope_weight: float = 0.0,
    *,
    id_gate: bool = False,
    normalizer=None,
) -> Dict[str, float]:
    """Training epoch for DirectNet (MAE + LDS weights).

    Batch layout: (x, y, tech_codes, lds_weights) or
    (x, y, tech_codes, lds_weights, sample_class) when slope loss is
    active.

    When ``id_gate=True``, the model output's id column is replaced by
    the gated value (B3) before going into the loss/slope-loss. The
    model is unchanged; this is a post-hoc wrap of the forward pass.
    ``normalizer`` must be provided when the gate is active.
    """
    model.train()
    total_loss = 0.0
    total_slope = 0.0
    n = 0
    use_slope = slope_loss is not None and slope_weight > 0.0
    if id_gate and normalizer is None:
        raise ValueError("id_gate=True requires a fitted normalizer")
    for batch in loader:
        sc = None
        if len(batch) == 5:
            x, y, tc, w, sc = batch
            w = w.to(device)
            sc = sc.to(device)
        elif len(batch) == 4:
            x, y, tc, w = batch
            w = w.to(device)
        else:
            x, y, tc = batch
            w = None
        x, y, tc = x.to(device), y.to(device), tc.to(device)
        if use_slope or id_gate:
            # Slope loss needs requires_grad=True; the gate uses x_norm
            # for Vds derivation but does not require it.  Setting it
            # unconditionally when either path is active keeps the
            # control flow simple.
            x = x.detach().clone().requires_grad_(True)
        optimizer.zero_grad()
        pred = model(x, tech_codes=tc)
        if id_gate:
            pred = apply_id_gate(
                x, pred, normalizer,
                id_idx_in_output=_DN_ID_IDX, vt_arch=_VT_ARCH,
            )
        loss = criterion(pred, y, weights=w) if w is not None else criterion(pred, y)
        if use_slope and sc is not None:
            sl = slope_loss(x, pred, y, sc)
            loss = loss + slope_weight * sl
            total_slope += sl.item()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    out = {"total": total_loss / n}
    if use_slope:
        out["slope"] = total_slope / n
    return out


@torch.no_grad()
def _validate_epoch_direct(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    id_gate: bool = False,
    normalizer=None,
) -> Dict[str, float]:
    """Validation epoch for DirectNet (unweighted MAE).

    Honours the structural Vds gate (B3) when ``id_gate=True`` so the
    val-best checkpoint tracker compares pred-vs-target consistently
    with the training loss.
    """
    model.eval()
    total = 0.0
    n = 0
    for _b in loader:
        x, y, tc = _b[0], _b[1], _b[2]
        x, y, tc = x.to(device), y.to(device), tc.to(device)
        pred = model(x, tech_codes=tc)
        if id_gate:
            pred = apply_id_gate(
                x, pred, normalizer,
                id_idx_in_output=_DN_ID_IDX, vt_arch=_VT_ARCH,
            )
        total += criterion(pred, y).item()
        n += 1
    return {"total": total / n}


@torch.no_grad()
def _collect_directnet_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    id_gate: bool = False,
    normalizer=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run DirectNet on loader, return (pred_norm, true_norm, tech_codes).

    Honours the gate when ``id_gate=True`` so downstream metrics see
    the same id slot the simulator will consume.
    """
    model.eval()
    all_pred, all_true, all_tc = [], [], []
    for _b in loader:
        x, y, tc = _b[0], _b[1], _b[2]
        x, tc = x.to(device), tc.to(device)
        pred = model(x, tech_codes=tc)
        if id_gate:
            pred = apply_id_gate(
                x, pred, normalizer,
                id_idx_in_output=_DN_ID_IDX, vt_arch=_VT_ARCH,
            )
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
    slope_loss: Optional[nn.Module] = None,
    slope_weight: float = 0.0,
    *,
    id_gate: bool = False,
    normalizer=None,
) -> Dict[str, float]:
    """Teacher-forced training epoch (batches carry tech codes).

    Batch layout: (x, y, tech_codes), (x, y, tech_codes, lds_weights),
    or (x, y, tech_codes, lds_weights, sample_class) when slope loss
    is active.

    When ``id_gate=True``, the post-forward output's id column is
    replaced by the gated value (B3) before the loss/slope-loss. AR
    conditioning inside ``model.forward`` is unchanged — the gate is a
    *post-hoc* wrap, exactly the dual-head pattern called for in the
    plan.
    """
    model.train()
    total_loss = 0.0
    total_slope = 0.0
    n_batches = 0
    use_slope = slope_loss is not None and slope_weight > 0.0
    if id_gate and normalizer is None:
        raise ValueError("id_gate=True requires a fitted normalizer")

    for batch in loader:
        sc = None
        if len(batch) == 5:
            x, y, tc, w, sc = batch
            w = w.to(device)
            sc = sc.to(device)
        elif len(batch) == 4:
            x, y, tc, w = batch
            w = w.to(device)
        else:
            x, y, tc = batch
            w = None

        x = x.to(device)
        y = y.to(device)
        tc = tc.to(device)
        if use_slope or id_gate:
            x = x.detach().clone().requires_grad_(True)

        optimizer.zero_grad()
        pred = model(x, y, tech_codes=tc)
        if id_gate:
            pred = apply_id_gate(
                x, pred, normalizer,
                id_idx_in_output=_TF_ID_IDX, id_idx_in_stats=0,
                vt_arch=_VT_ARCH,
            )

        loss = criterion(pred, y, weights=w) if w is not None else criterion(pred, y)
        if use_slope and sc is not None:
            sl = slope_loss(x, pred, y, sc)
            loss = loss + slope_weight * sl
            total_slope += sl.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    out = {"total": total_loss / n_batches}
    if use_slope:
        out["slope"] = total_slope / n_batches
    return out


@torch.no_grad()
def _validate_epoch_tf(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    id_gate: bool = False,
    normalizer=None,
) -> Dict[str, float]:
    """Teacher-forced validation."""
    model.eval()
    total = 0.0
    n = 0
    for _b in loader:
        x, y, tc = _b[0], _b[1], _b[2]
        x, y, tc = x.to(device), y.to(device), tc.to(device)
        pred = model(x, y, tech_codes=tc)
        if id_gate:
            pred = apply_id_gate(
                x, pred, normalizer,
                id_idx_in_output=_TF_ID_IDX, id_idx_in_stats=0,
                vt_arch=_VT_ARCH,
            )
        total += criterion(pred, y).item()
        n += 1
    return {"total": total / n}


@torch.no_grad()
def _validate_epoch_ar(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    id_gate: bool = False,
    normalizer=None,
) -> Dict[str, float]:
    """Autoregressive validation."""
    model.eval()
    total = 0.0
    n = 0
    for _b in loader:
        x, y, tc = _b[0], _b[1], _b[2]
        x, y, tc = x.to(device), y.to(device), tc.to(device)
        pred = model(x, tech_codes=tc)
        if id_gate:
            pred = apply_id_gate(
                x, pred, normalizer,
                id_idx_in_output=_TF_ID_IDX, id_idx_in_stats=0,
                vt_arch=_VT_ARCH,
            )
        total += criterion(pred, y).item()
        n += 1
    return {"total": total / n}


def _train_epoch_scheduled_mae(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    ss_ratio: float = 1.0,
    *,
    id_gate: bool = False,
    normalizer=None,
) -> Dict[str, float]:
    """N3 AR fine-tune helper.

    The gate wraps ``forward_scheduled`` exactly like the TF helper
    wraps ``forward``: the AR rollout still conditions on raw next-
    token output (id_raw_norm), but the loss target is the gated id.
    """
    model.train()
    total_loss = 0.0
    n = 0
    if id_gate and normalizer is None:
        raise ValueError("id_gate=True requires a fitted normalizer")

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
        if id_gate:
            pred = apply_id_gate(
                x, pred, normalizer,
                id_idx_in_output=_TF_ID_IDX, id_idx_in_stats=0,
                vt_arch=_VT_ARCH,
            )
        loss = criterion(pred, y, weights=w) if w is not None else criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return {"total": total_loss / n}


@torch.no_grad()
def test_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    id_gate: bool = False,
    normalizer=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run AR inference, return (pred_norm, true_norm).

    Honours the gate so phys-best tracking and final-test metrics see
    the same id slot the simulator consumes.
    """
    model.eval()
    all_pred, all_true = [], []
    for _b in loader:
        x, y, tc = _b[0], _b[1], _b[2]
        x, tc = x.to(device), tc.to(device)
        pred = model(x, tech_codes=tc)
        if id_gate:
            pred = apply_id_gate(
                x, pred, normalizer,
                id_idx_in_output=_TF_ID_IDX, id_idx_in_stats=0,
                vt_arch=_VT_ARCH,
            )
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
    save_prefix: str = "v4_dn_universal_nmos",
    exclude_techs: Optional[Set[str]] = None,
    num_tech_codes: int = NUM_TSMC_CODES_WITH_UNKNOWN,
    p_unknown: float = 0.1,
    slope_weight: float = 0.0,
    slope_warmup_frac: float = 0.7,
    id_gate: bool = True,
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
    print(f"[B3] id_gate: {'ON (vt_arch=0.04 V)' if id_gate else 'OFF (legacy)'}")
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

    # When slope loss is active and the dataset carries sample_class,
    # the training loader emits (x, y, tc, w, sample_class) batches.
    use_slope = slope_weight > 0.0
    if use_slope and train_ds.sample_class is None:
        raise RuntimeError(
            "slope_weight > 0 but the loaded dataset has no sample_class "
            "column. Regenerate the dataset (B1 sprint) before enabling "
            "slope loss.")
    if use_slope:
        train_ds_weighted = TensorDataset(
            train_ds.inputs, train_ds.outputs, train_ds.tech_codes,
            torch.tensor(lds_weights_np, dtype=torch.float32),
            train_ds.sample_class,
        )
    else:
        train_ds_weighted = TensorDataset(
            train_ds.inputs, train_ds.outputs, train_ds.tech_codes,
            torch.tensor(lds_weights_np, dtype=torch.float32),
        )
    train_loader = DataLoader(train_ds_weighted,
                              batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                            shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size,
                             shuffle=False)

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

    # Slope-loss setup (B2). DirectNet output is in OUTPUT_COLUMN_ORDER
    # so id is at index 0.
    slope_loss_module: Optional[nn.Module] = None
    if use_slope:
        slope_loss_module = SlopeMatchLoss(
            normalizer=normalizer,
            mode=normalizer.stats.mode,
            id_idx_in_output=0,
            vg_idx_in_input=1,
            max_samples=256,
        ).to(device)
        warmup_epoch = max(1, int(slope_warmup_frac * config.max_epochs))
        print(f"[B2] SlopeMatchLoss enabled: weight={slope_weight}, "
              f"warmup_frac={slope_warmup_frac} -> active from "
              f"epoch {warmup_epoch}")
    else:
        warmup_epoch = config.max_epochs + 1  # never active

    t_start = time.time()
    epoch = 0

    # Stash the id_gate flag onto the normalizer stats so it survives a
    # save/load round-trip and the simulator knows whether to apply the
    # gate at inference time.
    normalizer.stats.id_gate = bool(id_gate)

    for epoch in range(1, config.max_epochs + 1):
        slope_active = use_slope and epoch >= warmup_epoch
        train_losses = _train_epoch_direct(
            model, train_loader, criterion, optimizer, device,
            slope_loss=slope_loss_module if slope_active else None,
            slope_weight=slope_weight if slope_active else 0.0,
            id_gate=id_gate, normalizer=normalizer,
        )
        val_losses = _validate_epoch_direct(
            model, val_loader, criterion, device,
            id_gate=id_gate, normalizer=normalizer)
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
            if "slope" in train_losses:
                extra = f" slope={train_losses['slope']:.5f}"
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
        model, test_loader, criterion, device,
        id_gate=id_gate, normalizer=normalizer)
    print(f"\nTest set losses:")
    for k, v in sorted(test_losses.items()):
        print(f"  {k:>10s}: {v:.6f}")

    from bsimar.eval.metrics import compute_physical_metrics, print_metrics
    pred_norm, true_norm, test_tech_codes = _collect_directnet_predictions(
        model, test_loader, device,
        id_gate=id_gate, normalizer=normalizer)
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
    ar_finetune_epochs: int = 5,
    overwrite: bool = False,
    exclude_techs: Optional[Set[str]] = None,
    num_tech_codes: int = NUM_TSMC_CODES_WITH_UNKNOWN,
    p_unknown: float = 0.1,
    slope_weight: float = 0.0,
    slope_warmup_frac: float = 0.7,
    id_gate: bool = True,
) -> Tuple[nn.Module, BSIMARNormalizer]:
    """BSIMAR Transformer training pipeline.

    Uses the recipe (MAE+LDS+VovLDS, asinh, reorder, AR finetune,
    phys-best ckpt) with 7-dim continuous input + discrete tech codes.

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
    print(f"[B3] id_gate: {'ON (vt_arch=0.04 V)' if id_gate else 'OFF (legacy)'}")
    if exclude_techs:
        print(f"Excluded techs: {exclude_techs}")

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

    # Build weighted TensorDataset. When slope loss is active and the
    # dataset carries sample_class, append it as a 5th column so the
    # train epoch can mask to grid rows.
    use_slope = slope_weight > 0.0
    if use_slope and train_ds.sample_class is None:
        raise RuntimeError(
            "slope_weight > 0 but the loaded dataset has no sample_class "
            "column. Regenerate the dataset (B1 sprint) before enabling "
            "slope loss.")
    if use_slope:
        train_ds_weighted = TensorDataset(
            train_ds.inputs, train_ds.outputs, train_ds.tech_codes,
            torch.tensor(lds_weights_np, dtype=torch.float32),
            train_ds.sample_class,
        )
    else:
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

    # Stash the id_gate flag onto normalizer.stats so save() persists it.
    normalizer.stats.id_gate = bool(id_gate)
    # Median-based phys-score (post-fix); flags _best.phys.pt as trustworthy
    # so the simulator loader will prefer it over _best.pt.
    normalizer.stats.phys_best_metric = "median"

    early_stopping = EarlyStopping(
        patience=patience, min_delta=1e-5, save_path=str(best_path))

    # Slope-loss setup (B2). Transformer output is in BSIMAR_COLUMN_ORDER
    # (qg, qb, qd, qs, id, gm, ...) so id is at index 4.
    slope_loss_module: Optional[nn.Module] = None
    if use_slope:
        slope_loss_module = SlopeMatchLoss(
            normalizer=normalizer,
            mode=normalizer.stats.mode,
            id_idx_in_output=4,
            vg_idx_in_input=1,
            max_samples=256,
        ).to(device)
        warmup_epoch = max(1, int(slope_warmup_frac * epochs))
        print(f"[B2] SlopeMatchLoss enabled: weight={slope_weight}, "
              f"warmup_frac={slope_warmup_frac} -> active from "
              f"epoch {warmup_epoch}")
        # NOTE: B2 needs double-backward through attention, which the
        # fused flash/efficient SDP kernels do not support. The toggle
        # to math SDP is **deferred** until slope_active fires (see the
        # per-epoch loop below), so Blackwell/A100 keep flash/efficient
        # SDP for the warmup phase. Some GPUs (Blackwell) have very
        # slow math kernel paths, hence we delay as long as possible.
    else:
        warmup_epoch = epochs + 1  # never active
    _sdp_math_active = False  # state for deferred toggle

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
        slope_active = use_slope and epoch >= warmup_epoch
        if slope_active and not _sdp_math_active:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            _sdp_math_active = True
            print(f"[B2] Switching SDP kernel to math at epoch {epoch} "
                  "(double-backward enabled)")
        t_losses = _train_epoch_mae(
            model, train_loader, criterion, optimizer, device,
            slope_loss=slope_loss_module if slope_active else None,
            slope_weight=slope_weight if slope_active else 0.0,
            id_gate=id_gate, normalizer=normalizer,
        )
        v_losses = _validate_epoch_tf(
            model, val_loader, criterion, device,
            id_gate=id_gate, normalizer=normalizer)

        run_ar_check = (epoch % ar_check_every == 0) or (epoch == epochs)
        ar_status = ""
        if run_ar_check:
            ar_v = _validate_epoch_ar(
                model, val_loader, criterion, device,
                id_gate=id_gate, normalizer=normalizer)
            ar_loss = ar_v["total"]
            if ar_loss < best_ar_val_loss:
                best_ar_val_loss = ar_loss
                torch.save(model.state_dict(), str(ar_best_path))
                ar_status = " *ar-best*"
            print(f"  AR-val check @ epoch {epoch}: "
                  f"ar={ar_loss:.5f} (tf={v_losses['total']:.5f})"
                  f"{ar_status}")

            pred_val_norm, true_val_norm = test_model(
                model, val_loader, device,
                id_gate=id_gate, normalizer=normalizer)
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
            if "slope" in t_losses:
                extra = f" slope={t_losses['slope']:.5f}"
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
            t_losses = _train_epoch_scheduled_mae(
                model, ft_train_loader, ft_criterion, ft_optimizer,
                device, ss_ratio=1.0,
                id_gate=id_gate, normalizer=normalizer,
            )
            v_losses = _validate_epoch_ar(
                model, val_loader, ft_criterion, device,
                id_gate=id_gate, normalizer=normalizer)
            train_loss = t_losses["total"]
            val_loss = v_losses["total"]
            train_history.append(train_loss)
            val_history.append(val_loss)

            pred_val_norm, true_val_norm = test_model(
                model, val_loader, device,
                id_gate=id_gate, normalizer=normalizer)
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
            print(f"  [FT {ft_epoch:3d}] train={train_loss:.5f} "
                  f"val={val_loss:.5f} | nrmse={nrmse_med:.3f}% "
                  f"r2={r2_med:.4f}{phys_status}")

    # ── Final test ────────────────────────────────────────────────────────
    if phys_best_path.exists():
        load_path = phys_best_path
        print(f"Loading phys-best checkpoint for final test: {load_path}")
    else:
        load_path = best_path
        print(f"Loading TF-val-best checkpoint for final test: {load_path}")
    model.load_state_dict(torch.load(str(load_path), weights_only=True))
    pred_norm, true_norm = test_model(
        model, test_loader, device,
        id_gate=id_gate, normalizer=normalizer)

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
