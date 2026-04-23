"""Fine-tuning harness for BSIMAR v4 on unseen technologies.

Loads a TSMC-only pretrained v4 model, expands the tech embedding table
to include new tech codes, and fine-tunes on the new tech's data.

Usage:
    from bsimar.training.finetune import finetune_v4

    finetune_v4(
        pretrained_path="checkpoints/v4_universal_nmos_best.phys.pt",
        data_path="data/datasets/universal_nmos.npz",
        save_prefix="v4_ft_asap7_nmos",
        device_type="nmos",
        finetune_techs={"asap7"},
        new_num_tech_codes=22,  # expanded from 18 -> 22
    )
"""

from pathlib import Path
from typing import Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from bsimar.config import (
    TransformerConfig,
    CHECKPOINT_DIR, RESULTS_DIR,
    UNKNOWN_CODE_ID, NUM_TOTAL_CODES,
    TECH_VARIANT_CODES,
)
from bsimar.data.normalize import (
    BSIMARNormStats, BSIMARNormalizer,
    reorder_outputs, unreorder_outputs,
)
from bsimar.models.transformer import TransformerEncoderModel
from bsimar.losses.bni_mae import MAELoss


def finetune_v4(
    pretrained_path: str,
    data_path: str,
    save_prefix: str,
    device_type: str = "nmos",
    finetune_techs: Optional[Set[str]] = None,
    new_num_tech_codes: int = NUM_TOTAL_CODES,
    epochs: int = 50,
    batch_size: int = 1024,
    lr: float = 1e-4,
    ar_finetune_epochs: int = 3,
    device_str: str = "cpu",
    overwrite: bool = False,
) -> Tuple[nn.Module, BSIMARNormalizer]:
    """Fine-tune a pretrained v4 model on new technology data.

    Steps:
    1. Load pretrained model and its config/norm.
    2. Expand tech embedding to accommodate new tech codes.
    3. Load and filter data for the finetune techs.
    4. Fine-tune with low LR.
    5. Save expanded checkpoint.

    Args:
        pretrained_path: Path to pretrained .pt checkpoint (e.g., _best.phys.pt).
        data_path: Path to universal .npz dataset.
        save_prefix: Prefix for output checkpoint files.
        finetune_techs: Tech names to fine-tune on (e.g., {"asap7"}).
        new_num_tech_codes: Expanded embedding size (default 22).
        epochs: Fine-tuning epochs.
        lr: Learning rate for fine-tuning.
    """
    import time
    from bsimar.config import OUTPUT_COLUMNS
    from bsimar.data.dataset import load_and_split_bsimar
    from bsimar.eval.metrics import compute_physical_metrics, print_metrics
    from bsimar.training.trainer import (
        _train_epoch_mae, _validate_epoch_ar,
        _train_epoch_scheduled_mae, test_model,
    )

    device = torch.device(device_str)
    pretrained_path = Path(pretrained_path)

    # Load pretrained config
    config_path = pretrained_path.parent / (
        pretrained_path.stem.replace("_best", "_config")
        .replace(".phys", "").replace(".ar", "") + ".npz")
    norm_path = pretrained_path.parent / (
        pretrained_path.stem.replace("_best", "_norm")
        .replace(".phys", "").replace(".ar", "") + ".npz")

    print(f"Pretrained model: {pretrained_path}")
    print(f"Config: {config_path}")
    print(f"Norm: {norm_path}")

    cfg = np.load(str(config_path))
    old_num_codes = int(cfg["num_tech_codes"])
    d_model = int(cfg["d_model"])
    nhead = int(cfg["nhead"])
    num_layers = int(cfg["num_layers"])
    dim_feedforward = int(cfg["dim_feedforward"])
    dropout = float(cfg["dropout"])
    input_dim = int(cfg["input_dim"])
    target_dim = int(cfg["target_dim"])

    print(f"Old embedding size: {old_num_codes}, expanding to: {new_num_tech_codes}")

    # Build model with OLD embedding size, load weights
    model = TransformerEncoderModel(
        input_dim=input_dim,
        target_dim=target_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_tech_codes=old_num_codes,
        tech_embed_dropout=0.0,  # no dropout during fine-tuning
    )
    state = torch.load(str(pretrained_path), weights_only=True, map_location="cpu")
    model.load_state_dict(state)

    # Expand tech embedding
    if new_num_tech_codes > old_num_codes:
        old_weight = model.tech_embedding.weight.data  # (old, d_model)
        new_embedding = nn.Embedding(new_num_tech_codes, d_model)
        # Copy old weights
        new_embedding.weight.data[:old_num_codes] = old_weight
        # Initialize new slots from UNKNOWN embedding (warm start)
        unknown_emb = old_weight[UNKNOWN_CODE_ID]
        for i in range(old_num_codes, new_num_tech_codes):
            new_embedding.weight.data[i] = unknown_emb
        model.tech_embedding = new_embedding
        model.num_tech_codes = new_num_tech_codes
        print(f"Expanded embedding: {old_num_codes} -> {new_num_tech_codes} "
              f"(new slots initialized from UNKNOWN)")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    # Load normalizer from pretrained (reuse TSMC normalizer)
    normalizer = BSIMARNormalizer(
        mode="asinh",
        stats=BSIMARNormStats.load(str(norm_path)),
    )

    # Load finetune data — only the finetune techs
    # We use a trick: load all data, but put only finetune_techs in train/val
    # and the original train techs in test (for regression check).
    if finetune_techs is None:
        finetune_techs = {"asap7"}

    # Determine which tech codes belong to finetune_techs
    ft_code_set = {
        code for (tech, _), code in TECH_VARIANT_CODES.items()
        if tech in finetune_techs
    }
    print(f"Fine-tuning on techs: {finetune_techs}, codes: {sorted(ft_code_set)}")

    # Load full dataset
    from bsimar.eval.loo_labels import get_or_build_tech_variant_labels
    from bsimar.data.dataset import filter_small_targets, MOSFETDataset

    raw = np.load(data_path, allow_pickle=True)
    inputs = raw["inputs"]
    geometry = raw["geometry"]
    outputs = raw["outputs"]

    # Filter small targets
    keep = filter_small_targets(outputs, OUTPUT_COLUMNS)
    inputs, geometry, outputs = inputs[keep], geometry[keep], outputs[keep]

    tech_codes = get_or_build_tech_variant_labels(
        data_path, device_type, verbose=True)
    tech_codes = tech_codes[keep]

    # Split: finetune techs -> train/val, everything else -> test
    is_ft = np.array([int(c) in ft_code_set for c in tech_codes], dtype=bool)
    ft_idx = np.nonzero(is_ft)[0]
    other_idx = np.nonzero(~is_ft)[0]

    rng = np.random.default_rng(42)
    rng.shuffle(ft_idx)
    n_val = max(1, int(len(ft_idx) * 0.1))
    val_idx = ft_idx[:n_val]
    train_idx = ft_idx[n_val:]
    test_idx = other_idx  # for regression check

    print(f"Fine-tune split: train={len(train_idx)}, val={len(val_idx)}, "
          f"test(regression)={len(test_idx)}")

    # Normalize using the pretrained normalizer
    def _make_ds(idx: np.ndarray) -> MOSFETDataset:
        x = normalizer.normalize_inputs(inputs[idx], geometry[idx])
        y = normalizer.normalize_outputs(outputs[idx])
        return MOSFETDataset(x, y, tech_codes[idx])

    train_ds = _make_ds(train_idx)
    val_ds = _make_ds(val_idx)

    # Reorder outputs
    train_ds.outputs = torch.tensor(
        reorder_outputs(train_ds.outputs.numpy()), dtype=torch.float32)
    val_ds.outputs = torch.tensor(
        reorder_outputs(val_ds.outputs.numpy()), dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Regression-check set only built if there's non-finetune data available
    if len(test_idx) > 0:
        test_ds = _make_ds(test_idx)
        test_ds.outputs = torch.tensor(
            reorder_outputs(test_ds.outputs.numpy()), dtype=torch.float32)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False)
    else:
        test_ds = None
        test_loader = None
        print("No non-finetune samples in this dataset — skipping "
              "regression-check split (test_loader=None).")

    criterion = MAELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINT_DIR / f"{save_prefix}_best.pt"
    phys_best_path = best_path.with_suffix(".phys.pt")

    if best_path.exists() and not overwrite:
        raise SystemExit(f"Refusing to overwrite {best_path}. Pass overwrite=True.")

    best_phys_score = float("inf")
    best_val_loss = float("inf")
    t_start = time.time()

    print(f"\nFine-tuning {save_prefix} for {epochs} epochs (lr={lr:.1e})")
    for epoch in range(1, epochs + 1):
        t_losses = _train_epoch_mae(
            model, train_loader, criterion, optimizer, device)
        v_losses = _validate_epoch_ar(
            model, val_loader, criterion, device)

        train_loss = t_losses["total"]
        val_loss = v_losses["total"]

        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(best_path))
            status = " *best*"

        # Phys-space check every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            pred_val, true_val = test_model(model, val_loader, device)
            pred_val = unreorder_outputs(pred_val)
            true_val = unreorder_outputs(true_val)
            phys_m = compute_physical_metrics(pred_val, true_val, normalizer)
            nrmse_avg = float(np.nanmean([m["NRMSE(%)"] for m in phys_m.values()]))
            r2_avg = float(np.nanmean([m["R2"] for m in phys_m.values()]))
            phys_score = nrmse_avg + 0.1 * (1.0 - r2_avg)
            if phys_score < best_phys_score:
                best_phys_score = phys_score
                torch.save(model.state_dict(), str(phys_best_path))
                status += " *phys-best*"
            print(f"  {epoch:4d} | train={train_loss:.5f} val={val_loss:.5f} "
                  f"nrmse={nrmse_avg:.3f}% r2={r2_avg:.4f}{status}")
        elif epoch % 10 == 0 or epoch <= 3 or status:
            print(f"  {epoch:4d} | train={train_loss:.5f} "
                  f"val={val_loss:.5f}{status}")

        scheduler.step()

    # AR fine-tune
    if ar_finetune_epochs > 0 and phys_best_path.exists():
        model.load_state_dict(
            torch.load(str(phys_best_path), weights_only=True))
        ft_lr = max(lr * 0.1, 1e-5)
        print(f"\n[FT-AR] {ar_finetune_epochs} epochs at lr={ft_lr:.1e}")
        ft_opt = optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=1e-4)
        for ft_ep in range(1, ar_finetune_epochs + 1):
            t = _train_epoch_scheduled_mae(
                model, train_loader, criterion, ft_opt, device, ss_ratio=1.0)
            v = _validate_epoch_ar(model, val_loader, criterion, device)
            pred_val, true_val = test_model(model, val_loader, device)
            pred_val = unreorder_outputs(pred_val)
            true_val = unreorder_outputs(true_val)
            phys_m = compute_physical_metrics(pred_val, true_val, normalizer)
            nrmse_avg = float(np.nanmean([m["NRMSE(%)"] for m in phys_m.values()]))
            r2_avg = float(np.nanmean([m["R2"] for m in phys_m.values()]))
            phys_score = nrmse_avg + 0.1 * (1.0 - r2_avg)
            ps = ""
            if phys_score < best_phys_score:
                best_phys_score = phys_score
                torch.save(model.state_dict(), str(phys_best_path))
                ps = " *phys-best*"
            print(f"  [FT-AR {ft_ep}] train={t['total']:.5f} "
                  f"val={v['total']:.5f} nrmse={nrmse_avg:.3f}%{ps}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s")

    # Save expanded config
    config_path = CHECKPOINT_DIR / f"{save_prefix}_config.npz"
    np.savez(str(config_path),
             input_dim=np.array(input_dim),
             target_dim=np.array(target_dim),
             d_model=np.array(d_model),
             nhead=np.array(nhead),
             num_layers=np.array(num_layers),
             dim_feedforward=np.array(dim_feedforward),
             dropout=np.array(dropout),
             num_tech_codes=np.array(new_num_tech_codes))

    # Copy normalizer
    norm_out = CHECKPOINT_DIR / f"{save_prefix}_norm.npz"
    normalizer.stats.save(str(norm_out))

    # Final test on finetune tech
    if phys_best_path.exists():
        model.load_state_dict(
            torch.load(str(phys_best_path), weights_only=True))
    pred_norm, true_norm = test_model(model, val_loader, device)
    pred_norm = unreorder_outputs(pred_norm)
    true_norm = unreorder_outputs(true_norm)
    metrics = compute_physical_metrics(pred_norm, true_norm, normalizer)
    print(f"\nFinal metrics on {finetune_techs} val set:")
    print_metrics(metrics)

    print(f"\nCheckpoint: {best_path}")
    print(f"Phys-best: {phys_best_path}")
    print(f"Config: {config_path}")

    return model, normalizer
