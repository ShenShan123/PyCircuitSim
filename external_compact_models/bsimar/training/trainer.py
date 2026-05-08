"""Single training loop for DirectNet (MLP) and BSIMAR (Transformer).

Both architectures share 95% of the scaffolding (load → compute LDS →
build dataloaders → cosine + early-stop → eval per-tech). The only
differences are:

* forward signature (Transformer takes teacher-forced ``y`` during train)
* AR validation (Transformer only)
* output column reorder (Transformer trains in BSIMAR_COLUMN_ORDER)

Those go through a small ``ArchAdapter`` so the rest of the loop is
arch-agnostic.

Public entry points (back-compat with the old CLI):

    train_directnet(data_path, ...)      → DirectNet
    train_transformer(data_path, ...)    → BSIMAR
"""

from __future__ import annotations

import time
from dataclasses import dataclass
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
    OUTPUT_COLUMN_ORDER, BSIMAR_COLUMN_ORDER,
    reorder_outputs, unreorder_outputs,
    _NormalizerBase,
)
from bsimar.losses.bni_mae import MAELoss, compute_lds_weights_per_target


# ── Arch adapters ──────────────────────────────────────────────────────────

@dataclass
class ArchAdapter:
    """Bridge between the generic loop and the model-specific forward."""
    name: str                    # "direct" | "transformer"
    norm_mode: str               # "zscore" | "asinh"
    reorder_outputs: bool        # True for transformer
    save_arch_config: bool       # True for transformer
    needs_ar_eval: bool          # True for transformer

    def forward_train(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor,
        tc: torch.Tensor,
    ) -> torch.Tensor:
        if self.name == "transformer":
            return model(x, y, tech_codes=tc)
        return model(x, tech_codes=tc)

    def forward_eval(
        self, model: nn.Module, x: torch.Tensor, tc: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # AR Transformer inference is y=None; teacher-forced eval passes y.
        if self.name == "transformer":
            return model(x, y, tech_codes=tc) if y is not None else \
                   model(x, tech_codes=tc)
        return model(x, tech_codes=tc)


_ADAPTERS = {
    # V6 Tier 2 (2026-05-09): DirectNet flipped from "zscore" to "asinh"
    # outputs. Concentrates loss on the small-Id band that dominates
    # inverter trip-point NRMSE; matches the Transformer's normaliser.
    # Chain rule already correct via AsinhNormalizer.denormalize_derivative.
    "direct": ArchAdapter(
        name="direct", norm_mode="asinh",
        reorder_outputs=False, save_arch_config=False, needs_ar_eval=False),
    "transformer": ArchAdapter(
        name="transformer", norm_mode="asinh",
        reorder_outputs=True, save_arch_config=True, needs_ar_eval=True),
}


# ── Per-epoch helpers ──────────────────────────────────────────────────────

def _epoch_train(
    model: nn.Module, loader: DataLoader,
    criterion: MAELoss, optimizer: optim.Optimizer,
    device: torch.device, adapter: ArchAdapter,
    clip_grad: bool,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        x, y, tc, w = batch
        x, y, tc, w = (
            x.to(device), y.to(device), tc.to(device), w.to(device))
        optimizer.zero_grad()
        pred = adapter.forward_train(model, x, y, tc)
        loss = criterion(pred, y, weights=w)
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def _epoch_eval(
    model: nn.Module, loader: DataLoader, criterion: MAELoss,
    device: torch.device, adapter: ArchAdapter,
    teacher_forced: bool,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x, y, tc in loader:
        x, y, tc = x.to(device), y.to(device), tc.to(device)
        pred = adapter.forward_eval(
            model, x, tc, y if teacher_forced else None)
        total += criterion(pred, y).item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def _collect(
    model: nn.Module, loader: DataLoader, device: torch.device,
    adapter: ArchAdapter,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect (pred_norm, true_norm, tech_codes) on a loader."""
    model.eval()
    all_pred, all_true, all_tc = [], [], []
    for x, y, tc in loader:
        x, tc = x.to(device), tc.to(device)
        pred = adapter.forward_eval(model, x, tc)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.numpy())
        all_tc.append(tc.cpu().numpy())
    return (np.concatenate(all_pred),
            np.concatenate(all_true),
            np.concatenate(all_tc))


# ── Reporting helpers ──────────────────────────────────────────────────────

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
    adapter: ArchAdapter,
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
    arch_config: Optional[dict] = None,
    column_weights: Optional[np.ndarray] = None,
) -> Tuple[nn.Module, _NormalizerBase]:
    if adapter.reorder_outputs:
        train_ds.outputs = torch.tensor(
            reorder_outputs(train_ds.outputs.numpy()), dtype=torch.float32)
        val_ds.outputs = torch.tensor(
            reorder_outputs(val_ds.outputs.numpy()), dtype=torch.float32)
        test_ds.outputs = torch.tensor(
            reorder_outputs(test_ds.outputs.numpy()), dtype=torch.float32)
        print("  Outputs reordered to BSIMAR_COLUMN_ORDER")

    print(f"  Computing LDS weights …")
    lds = compute_lds_weights_per_target(
        train_ds.outputs.numpy(), n_bins=100,
        lds_kernel="gaussian", lds_ks=5, lds_sigma=0.8)
    means = lds.mean(axis=0, keepdims=True)
    means[means < 1e-12] = 1.0
    lds = lds / means

    # Multiply by per-column loss preset (rule 1: simulator only reads
    # id/qg/qd/qb at inference; everything else is a smoothness prior).
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
    nw = 8
    train_loader = DataLoader(
        train_w, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=True)

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
    history: list[Tuple[float, float]] = []
    epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss = _epoch_train(
            model, train_loader, criterion, optimizer, device,
            adapter, clip_grad=(adapter.name == "transformer"))
        val_loss = _epoch_eval(
            model, val_loader, criterion, device, adapter,
            teacher_forced=(adapter.name == "transformer"))
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        history.append((train_loss, val_loss))

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

    if adapter.save_arch_config and arch_config is not None:
        np.savez(
            str(CHECKPOINT_DIR / f"{save_prefix}_config.npz"),
            **{k: np.array(v) for k, v in arch_config.items()})

    # Final test eval
    model.load_state_dict(
        torch.load(str(best_path), weights_only=True))

    pred_norm, true_norm, test_tc = _collect(
        model, test_loader, device, adapter)
    if adapter.reorder_outputs:
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


# ── Public entry points (kept for back-compat with old CLI) ────────────────

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
    **_: object,  # swallow legacy kwargs
) -> Tuple[nn.Module, _NormalizerBase]:
    """DirectNet MLP training pipeline.

    ``column_weights`` (length = output dim): per-target multiplier on the
    loss (combined with LDS). Use to down-weight or zero out targets the
    simulator does not consume — e.g. ``qs`` (always replaced by KCL),
    ``gm/gds/gmb``, ``c*`` (all autograd-derived at inference).

    ``output_subset`` (list of column names): if given, train only on this
    subset of the 13 outputs (E2 4-output head). The model's ``output_dim``
    becomes ``len(output_subset)`` and the saved norm stats record which
    columns were kept so the simulator can rebuild the column-name map.
    """
    from bsimar.models.direct_net import DirectNet

    adapter = _ADAPTERS["direct"]
    device = torch.device(device_str)
    print(f"DirectNet on {device}; tech codes={num_tech_codes}, "
          f"p_unknown={p_unknown}")
    if exclude_techs:
        print(f"  Excluding techs: {exclude_techs}")

    train_ds, val_ds, test_ds, normalizer = load_and_split_bsimar(
        data_path, OUTPUT_COLUMN_ORDER, device_type=device_type,
        train_ratio=config.train_ratio, val_ratio=config.val_ratio,
        apply_filter=True, exclude_techs=exclude_techs,
        norm_mode=adapter.norm_mode, max_rows=max_rows,
        output_subset=output_subset,
    )
    in_dim = train_ds.inputs.shape[1]
    out_dim = train_ds.outputs.shape[1]
    if output_subset is not None:
        print(f"  Output subset: {output_subset} (output_dim={out_dim})")

    model = DirectNet(
        input_dim=in_dim, hidden_dim=config.trunk_hidden,
        n_layers=config.trunk_layers + 1, output_dim=out_dim,
        num_tech_codes=num_tech_codes,
        tech_embed_dim=32, tech_embed_dropout=p_unknown,
    ).to(device)
    print(f"  Params: {model.count_parameters():,}")

    return _train_loop(
        model=model, adapter=adapter,
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
    **_: object,  # swallow legacy kwargs
) -> Tuple[nn.Module, _NormalizerBase]:
    """BSIMAR Transformer training pipeline."""
    from bsimar.models.transformer import TransformerEncoderModel

    adapter = _ADAPTERS["transformer"]
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
        norm_mode=adapter.norm_mode, max_rows=max_rows,
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
        model=model, adapter=adapter,
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        normalizer=normalizer,
        epochs=epochs, batch_size=batch_size,
        lr=lr, weight_decay=config.weight_decay,
        patience=patience, save_prefix=save_prefix,
        device=device, overwrite=overwrite,
        arch_config=arch_config,
    )
