"""Single training loop for DirectNet (MLP) and BSIMAR (Transformer).

Both architectures share the same data, normaliser, LDS-MAE loss, cosine
schedule, and early-stop pattern. The only differences:

* the Transformer's ``forward`` takes a teacher-forced ``y`` argument
  during training and runs autoregressively at eval time;
* the Transformer trains in ``BSIMAR_COLUMN_ORDER`` and saves an
  architecture sidecar so the simulator can rebuild the model.

Both differences are gated on a single ``is_transformer`` flag inside
``_train_loop``. Public entry points: ``train_directnet`` and
``train_transformer``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from bsimar.config import (
    DirectNetConfig, TransformerConfig,
    CHECKPOINT_DIR, RESULTS_DIR,
    NUM_TSMC_CODES_WITH_UNKNOWN,
)
from bsimar.data.dataset import load_and_split_bsimar
from bsimar.data.normalize import (
    OUTPUT_COLUMN_ORDER, reorder_outputs, unreorder_outputs,
    _NormalizerBase,
)
from bsimar.losses.bni_mae import MAELoss, compute_lds_weights_per_target

# V6 Tier 2 (2026-05-09): both DirectNet and the Transformer train with
# asinh + z-score outputs. Concentrates loss on the small-Id band that
# dominates inverter trip-point NRMSE.
_NORM_MODE = "asinh"
_NUM_WORKERS = 8


# ── Per-epoch helpers ──────────────────────────────────────────────────────

def _epoch_train(
    model: nn.Module, loader: DataLoader,
    criterion: MAELoss, optimizer: optim.Optimizer,
    device: torch.device, is_transformer: bool,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, y, tc, w in loader:
        x, y, tc, w = (x.to(device), y.to(device),
                       tc.to(device), w.to(device))
        optimizer.zero_grad()
        pred = (model(x, y, tech_codes=tc) if is_transformer
                else model(x, tech_codes=tc))
        loss = criterion(pred, y, weights=w)
        loss.backward()
        if is_transformer:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def _epoch_eval(
    model: nn.Module, loader: DataLoader, criterion: MAELoss,
    device: torch.device, is_transformer: bool,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x, y, tc in loader:
        x, y, tc = x.to(device), y.to(device), tc.to(device)
        # Teacher-forced eval for the Transformer (val loss aligned with train).
        pred = (model(x, y, tech_codes=tc) if is_transformer
                else model(x, tech_codes=tc))
        total += criterion(pred, y).item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def _collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device,
    is_transformer: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect (pred_norm, true_norm, tech_codes) on a loader (AR for TF)."""
    model.eval()
    all_pred, all_true, all_tc = [], [], []
    for x, y, tc in loader:
        x, tc = x.to(device), tc.to(device)
        pred = model(x, tech_codes=tc)  # AR inference for the Transformer
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.numpy())
        all_tc.append(tc.cpu().numpy())
    return (np.concatenate(all_pred),
            np.concatenate(all_true),
            np.concatenate(all_tc))


def _per_tech_report(
    pred_norm: np.ndarray, true_norm: np.ndarray,
    tech_codes: np.ndarray, normalizer: _NormalizerBase,
) -> None:
    from bsimar.config import CODE_TO_TECH_VARIANT
    from bsimar.eval.metrics import compute_physical_metrics

    print(f"\n{'Tech':>15s} | {'n_test':>6s} | "
          f"{'NRMSE%':>8s} | {'R2':>8s}")
    print("-" * 50)
    for code in sorted(np.unique(tech_codes)):
        mask = tech_codes == code
        tech, variant = CODE_TO_TECH_VARIANT.get(int(code), ("unk", "unk"))
        m = compute_physical_metrics(
            pred_norm[mask], true_norm[mask], normalizer)
        nr = [v["NRMSE(%)"] for v in m.values()
              if not np.isnan(v["NRMSE(%)"])]
        r2 = [v["R2"] for v in m.values() if not np.isnan(v["R2"])]
        print(f"{tech}:{variant:>9s} | {mask.sum():6d} | "
              f"{(np.mean(nr) if nr else float('nan')):8.3f} | "
              f"{(np.mean(r2) if r2 else float('nan')):8.4f}")


# ── Generic train loop ─────────────────────────────────────────────────────

def _train_loop(
    *,
    model: nn.Module,
    train_ds, val_ds, test_ds,
    normalizer: _NormalizerBase,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    save_prefix: str,
    device: torch.device,
    overwrite: bool,
    is_transformer: bool,
    arch_config: Optional[dict] = None,
    column_weights: Optional[np.ndarray] = None,
) -> Tuple[nn.Module, _NormalizerBase]:
    if is_transformer:
        for ds in (train_ds, val_ds, test_ds):
            ds.outputs = torch.tensor(
                reorder_outputs(ds.outputs.numpy()), dtype=torch.float32)
        print("  Outputs reordered to BSIMAR_COLUMN_ORDER")

    print("  Computing LDS weights …")
    lds = compute_lds_weights_per_target(
        train_ds.outputs.numpy(), n_bins=100,
        lds_kernel="gaussian", lds_ks=5, lds_sigma=0.8)
    means = lds.mean(axis=0, keepdims=True)
    means[means < 1e-12] = 1.0
    lds = lds / means

    if column_weights is not None:
        cw = np.asarray(column_weights, dtype=np.float32)
        if cw.shape != (lds.shape[1],):
            raise ValueError(
                f"column_weights shape {cw.shape} does not match "
                f"output dim {lds.shape[1]}")
        lds = lds * cw[None, :]
        print(f"  Column-weight preset: {cw.tolist()}")

    train_w = TensorDataset(
        train_ds.inputs, train_ds.outputs, train_ds.tech_codes,
        torch.tensor(lds, dtype=torch.float32))
    train_loader = DataLoader(
        train_w, batch_size=batch_size, shuffle=True,
        num_workers=_NUM_WORKERS, pin_memory=True,
        persistent_workers=True)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=_NUM_WORKERS, pin_memory=True,
        persistent_workers=True)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=_NUM_WORKERS, pin_memory=True,
        persistent_workers=True)

    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = MAELoss()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINT_DIR / f"{save_prefix}_best.pt"
    norm_path = CHECKPOINT_DIR / f"{save_prefix}_norm.npz"

    if best_path.exists() and not overwrite:
        raise SystemExit(
            f"Refusing to overwrite {best_path}. "
            "Pass --overwrite or pick a unique --exp-name.")

    # Trustworthy phys-best aggregator (post-fix; see plan §2B).
    normalizer.stats.phys_best_metric = "median"

    best_val = float("inf")
    bad = 0
    print(f"  Training {save_prefix} for {epochs} epochs "
          f"(patience={patience})")
    t0 = time.time()
    epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss = _epoch_train(
            model, train_loader, criterion, optimizer, device, is_transformer)
        val_loss = _epoch_eval(
            model, val_loader, criterion, device, is_transformer)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        marker = ""
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            bad = 0
            torch.save(model.state_dict(), str(best_path))
            normalizer.stats.save(str(norm_path))
            marker = " *best*"
        else:
            bad += 1

        if epoch <= 5 or epoch % 10 == 0 or marker:
            print(f"  {epoch:4d} | train={train_loss:.5f} "
                  f"val={val_loss:.5f} lr={lr_now:.2e}{marker}")

        if bad >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s "
          f"({elapsed / max(epoch, 1):.1f}s/epoch). Best val={best_val:.6f}")

    if is_transformer and arch_config is not None:
        np.savez(
            str(CHECKPOINT_DIR / f"{save_prefix}_config.npz"),
            **{k: np.array(v) for k, v in arch_config.items()})

    # Final test eval — use AR inference for the Transformer.
    model.load_state_dict(torch.load(str(best_path), weights_only=True))
    pred_norm, true_norm, test_tc = _collect_predictions(
        model, test_loader, device, is_transformer)
    if is_transformer:
        pred_norm = unreorder_outputs(pred_norm)
        true_norm = unreorder_outputs(true_norm)

    from bsimar.eval.metrics import compute_physical_metrics, print_metrics
    metrics = compute_physical_metrics(pred_norm, true_norm, normalizer)
    print("\nPhysical metrics (test set):")
    print_metrics(metrics)
    _per_tech_report(pred_norm, true_norm, test_tc, normalizer)
    print(f"\nSaved checkpoint: {best_path}")
    print(f"Saved norm stats: {norm_path}")
    return model, normalizer


# ── Public entry points ────────────────────────────────────────────────────

def train_directnet(
    data_path: str,
    device_type: str = "nmos",
    config: DirectNetConfig = DirectNetConfig(),
    device_str: str = "cpu",
    save_prefix: str = "v4_re_dn_universal_nmos",
    exclude_techs: Optional[Set[str]] = None,
    num_tech_codes: int = NUM_TSMC_CODES_WITH_UNKNOWN,
    p_unknown: float = 0.1,
    max_rows: Optional[int] = None,
    overwrite: bool = False,
    column_weights: Optional[np.ndarray] = None,
    output_subset: Optional[list[str]] = None,
    tech_scope: str = "universal",
    keep_offstate: bool = False,
    **_: object,  # swallow legacy kwargs
) -> Tuple[nn.Module, _NormalizerBase]:
    """DirectNet MLP training pipeline.

    ``keep_offstate`` (plan §4e): when True, the ``id>1e-15`` off-state
    ingestion filter is disabled so sub-threshold / hold-leakage rows
    (switched-cap, SRAM hold) survive into training.

    ``column_weights`` (length = output dim): per-target multiplier on
    the loss (combined with LDS). Use to down-weight or zero out targets
    the simulator does not consume — e.g. ``qs`` (always replaced by KCL).

    ``output_subset`` (list of column names): if given, train only on
    this subset of the 13 outputs (E2 4-output head). The model's
    ``output_dim`` becomes ``len(output_subset)`` and the saved norm
    stats record which columns were kept.
    """
    from bsimar.models.direct_net import DirectNet

    device = torch.device(device_str)
    print(f"DirectNet on {device}; tech codes={num_tech_codes}, "
          f"p_unknown={p_unknown}")
    if exclude_techs:
        print(f"  Excluding techs: {exclude_techs}")

    if keep_offstate:
        print("  [keep-offstate] Id>1e-15 filter DISABLED — "
              "off-state rows retained")
    train_ds, val_ds, test_ds, normalizer = load_and_split_bsimar(
        data_path, OUTPUT_COLUMN_ORDER, device_type=device_type,
        train_ratio=config.train_ratio, val_ratio=config.val_ratio,
        apply_filter=not keep_offstate, exclude_techs=exclude_techs,
        norm_mode=_NORM_MODE, max_rows=max_rows,
        output_subset=output_subset,
        tech_scope=tech_scope,
    )
    in_dim = train_ds.inputs.shape[1]
    out_dim = train_ds.outputs.shape[1]
    if output_subset is not None:
        print(f"  Output subset: {output_subset} (output_dim={out_dim})")

    # Place UNKNOWN at the last slot of whatever vocab we have. Universal
    # vocab=18 keeps unknown=17 (existing convention); per-tech vocab=5
    # (TSMC5) → unknown=4 / vocab=4 (TSMC7) → unknown=3. Without this,
    # `p_unknown` training-time dropout would write code 17 into a 5-row
    # embedding and trigger a CUDA assert.
    model = DirectNet(
        input_dim=in_dim, hidden_dim=config.trunk_hidden,
        n_layers=config.trunk_layers + 1, output_dim=out_dim,
        num_tech_codes=num_tech_codes,
        tech_embed_dim=32, tech_embed_dropout=p_unknown,
        unknown_code_id=num_tech_codes - 1,
    ).to(device)
    print(f"  Params: {model.count_parameters():,}")

    return _train_loop(
        model=model, is_transformer=False,
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        normalizer=normalizer,
        epochs=config.max_epochs, batch_size=config.batch_size,
        lr=config.lr, weight_decay=config.weight_decay,
        patience=config.patience, save_prefix=save_prefix,
        device=device, overwrite=overwrite,
        column_weights=column_weights,
    )


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
    max_rows: Optional[int] = None,
    tech_scope: str = "universal",
    **_: object,  # swallow legacy kwargs
) -> Tuple[nn.Module, _NormalizerBase]:
    """BSIMAR Transformer training pipeline."""
    from bsimar.models.transformer import TransformerEncoderModel

    epochs = epochs if epochs is not None else config.max_epochs
    batch_size = batch_size if batch_size is not None else config.batch_size
    patience = patience if patience is not None else config.patience
    lr = lr if lr is not None else config.lr

    device = torch.device(device_str)
    print(f"BSIMAR Transformer on {device}; tech codes={num_tech_codes}, "
          f"p_unknown={p_unknown}")
    if exclude_techs:
        print(f"  Excluding techs: {exclude_techs}")

    train_ds, val_ds, test_ds, normalizer = load_and_split_bsimar(
        data_path, OUTPUT_COLUMN_ORDER, device_type=device_type,
        apply_filter=True, exclude_techs=exclude_techs,
        norm_mode=_NORM_MODE, max_rows=max_rows,
        tech_scope=tech_scope,
    )
    in_dim = train_ds.inputs.shape[1]
    out_dim = train_ds.outputs.shape[1]

    model = TransformerEncoderModel(
        input_dim=in_dim, target_dim=out_dim,
        d_model=config.d_model, nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        num_tech_codes=num_tech_codes,
        tech_embed_dropout=p_unknown,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")

    arch_config = {
        "input_dim": in_dim, "target_dim": out_dim,
        "d_model": config.d_model, "nhead": config.nhead,
        "num_layers": config.num_layers,
        "dim_feedforward": config.dim_feedforward,
        "dropout": config.dropout,
        "num_tech_codes": num_tech_codes,
    }
    return _train_loop(
        model=model, is_transformer=True,
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        normalizer=normalizer,
        epochs=epochs, batch_size=batch_size,
        lr=lr, weight_decay=config.weight_decay,
        patience=patience, save_prefix=save_prefix,
        device=device, overwrite=overwrite,
        arch_config=arch_config,
    )
