"""Training loop for NN-based MOSFET compact model.

Two training modes:
  --mode direct13: Predict all 13 outputs (id, gm, gds, gmb, charges, caps)
                   directly. Fast (~2s/epoch). Default for Phase 1.
  --mode finetune: Fine-tune with autograd derivative supervision via
                   PhysicsLoss. Slow (~100s/epoch). Phase 2 after direct13.

Usage:
    conda run -n pycircuitsim python -u -m nn_model.train --device-type nmos --mode direct13
    conda run -n pycircuitsim python -u -m nn_model.train --device-type nmos --mode finetune
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nn_model.config import TrainConfig, CHECKPOINT_DIR, DATA_DIR, TECH_CONFIGS
from nn_model.data.dataset import load_and_split
from nn_model.data.normalize import Normalizer, inv_signed_log
from nn_model.architecture.direct_loss import DirectNet, DirectLoss


def train_epoch(
    model: DirectNet,
    loader: DataLoader,
    criterion: DirectLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch."""
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def validate_epoch(
    model: DirectNet,
    loader: DataLoader,
    criterion: DirectLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Validate without gradients."""
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


@torch.no_grad()
def compute_physical_metrics(
    model: DirectNet,
    loader: DataLoader,
    normalizer: Normalizer,
    device: torch.device,
) -> Dict[str, float]:
    """Compute per-output metrics in physical units after denormalization.

    Reports:
    - id NRMSE (normalized to peak-to-peak range)
    - gm, gds mean absolute error in log-space (meaningful for multi-decade data)
    """
    model.eval()
    all_pred = []
    all_true = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        pred = model(x_batch)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y_batch.numpy())

    pred_norm = np.concatenate(all_pred)   # (N, output_dim)
    true_norm = np.concatenate(all_true)   # (N, 13)

    stats = normalizer.stats
    metrics: Dict[str, float] = {}

    # id (column 0) — denormalize and compute NRMSE
    id_pred_log = pred_norm[:, 0] * stats.output_std[0] + stats.output_mean[0]
    id_true_log = true_norm[:, 0] * stats.output_std[0] + stats.output_mean[0]
    id_pred_phys = inv_signed_log(id_pred_log, floor=stats.output_log_floors[0])
    id_true_phys = inv_signed_log(id_true_log, floor=stats.output_log_floors[0])

    id_range = id_true_phys.max() - id_true_phys.min()
    if id_range > 0:
        rmse = np.sqrt(np.mean((id_pred_phys - id_true_phys) ** 2))
        metrics["id_nrmse_pct"] = rmse / id_range * 100
    else:
        metrics["id_nrmse_pct"] = 0.0

    # Log-space MAE for all columns (more meaningful for multi-decade data)
    output_dim = pred_norm.shape[1]
    col_names_13 = ["id", "gm", "gds", "gmb", "qg", "qd", "qs", "qb",
                    "cgg", "cgd", "cgs", "cdg", "cdd"]
    for i in range(min(output_dim, 13)):
        mae_norm = np.mean(np.abs(pred_norm[:, i] - true_norm[:, i]))
        metrics[f"{col_names_13[i]}_mae_norm"] = mae_norm

    return metrics


def train(
    data_path: str,
    config: TrainConfig = TrainConfig(),
    device_str: str = "cpu",
    save_prefix: str = "nmos",
    output_dim: int = 13,
    resume_from: str = None,
) -> Tuple[DirectNet, Normalizer]:
    """Full training pipeline.

    Args:
        data_path: Path to .npz dataset.
        config: Training hyperparameters.
        device_str: 'cpu' or 'cuda'.
        save_prefix: Prefix for saved model files.
        output_dim: 4 (id,qg,qd,qb only) or 13 (all outputs).
        resume_from: Path to checkpoint to resume/fine-tune from.
    """
    device = torch.device(device_str)
    print(f"Training on {device}, output_dim={output_dim}")
    print(f"Loss weights: w_id={config.w_id}, w_charges={config.w_charges}, w_caps={config.w_caps}")

    # Load and split
    train_ds, val_ds, test_ds, normalizer = load_and_split(
        data_path,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, num_workers=0
    )

    # Auto-detect input_dim from dataset (6 for legacy, 7 with PHIG)
    input_dim = train_ds.inputs.shape[1]
    print(f"Input dim: {input_dim} ({'with PHIG' if input_dim == 7 else 'legacy'})")

    # Model
    model = DirectNet(
        input_dim=input_dim,
        hidden_dim=config.trunk_hidden,
        n_layers=config.trunk_layers + 1,
        output_dim=output_dim,
    ).to(device)

    if resume_from is not None:
        print(f"Resuming from {resume_from}")
        state = torch.load(resume_from, weights_only=True, map_location=device)
        # Handle output_dim mismatch: load matching layers only
        model_state = model.state_dict()
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
        model.load_state_dict(model_state)

    print(f"Model parameters: {model.count_parameters()}")

    criterion = DirectLoss(
        output_dim=output_dim,
        w_zero_bias=config.w_zero_bias,
        w_curr=config.w_id,
        w_cond=(config.w_gm + config.w_gds + config.w_gmb) / 3.0,
        w_charges=config.w_charges,
        w_caps=config.w_caps,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs)

    best_val_loss = float("inf")
    patience_counter = 0

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINT_DIR / f"{save_prefix}_best.pt"
    norm_path = CHECKPOINT_DIR / f"{save_prefix}_norm.npz"

    if output_dim == 13:
        header = (f"{'Ep':>4s} | {'Train':>8s} | {'Val':>8s} | {'LR':>9s} | "
                  f"{'id':>7s} {'gm':>7s} {'gds':>7s} {'q':>7s} {'cap':>7s} | {'Status'}")
    else:
        header = (f"{'Ep':>4s} | {'Train':>8s} | {'Val':>8s} | {'LR':>9s} | "
                  f"{'id':>7s} | {'Status'}")
    print(f"\n{header}")
    print("-" * len(header) + "-" * 10)

    t_start = time.time()

    for epoch in range(1, config.max_epochs + 1):
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device)
        val_losses = validate_epoch(model, val_loader, criterion, device)
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
            if output_dim == 13:
                print(f"{epoch:4d} | {train_losses['total']:8.5f} | "
                      f"{val_losses['total']:8.5f} | {lr:9.2e} | "
                      f"{val_losses['id']:7.5f} {val_losses['gm']:7.5f} "
                      f"{val_losses['gds']:7.5f} {val_losses['charges']:7.5f} "
                      f"{val_losses['caps']:7.5f} |{status}")
            else:
                print(f"{epoch:4d} | {train_losses['total']:8.5f} | "
                      f"{val_losses['total']:8.5f} | {lr:9.2e} | "
                      f"{val_losses['id']:7.5f} |{status}")

        if patience_counter >= config.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={config.patience})")
            break

    elapsed = time.time() - t_start
    print(f"\nTraining completed in {elapsed:.0f}s ({elapsed / epoch:.1f}s/epoch)")
    print(f"Best val loss: {best_val_loss:.6f}")

    # Load best and evaluate
    model.load_state_dict(torch.load(best_path, weights_only=True))
    test_losses = validate_epoch(model, test_loader, criterion, device)
    print(f"\nTest set losses:")
    for k, v in sorted(test_losses.items()):
        print(f"  {k:>10s}: {v:.6f}")

    phys = compute_physical_metrics(model, test_loader, normalizer, device)
    print(f"\nPhysical metrics (test set):")
    print(f"  id NRMSE: {phys['id_nrmse_pct']:.2f}%")
    for k, v in sorted(phys.items()):
        if k.endswith("_mae_norm"):
            name = k.replace("_mae_norm", "")
            print(f"  {name:>6s} MAE(norm): {v:.4f}")

    print(f"\nSaved: {best_path}")
    return model, normalizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NN-based MOSFET model")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--device-type", choices=["nmos", "pmos"], default="nmos")
    parser.add_argument("--mode", choices=["direct4", "direct13", "finetune"],
                        default="direct13")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256,
                        help="Hidden layer dimension")
    parser.add_argument("--layers", type=int, default=5,
                        help="Number of hidden layers")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--tech", choices=list(TECH_CONFIGS.keys()),
                        default="asap7",
                        help="Technology (default: asap7)")
    parser.add_argument("--w-charges", type=float, default=None,
                        help="Loss weight for charges (default: 0.5, finetune: 1.5)")
    parser.add_argument("--w-caps", type=float, default=None,
                        help="Loss weight for capacitances (default: 0.3, finetune: 1.0)")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    tech_name = args.tech.lower()

    if args.data is None:
        data_path = DATA_DIR / f"{tech_name}_{args.device_type}.npz"
        if not data_path.exists():
            print(f"Dataset not found: {data_path}")
            print(f"Run: python -m nn_model.data.generate --device {args.device_type} --tech {tech_name}")
            sys.exit(1)
    else:
        data_path = Path(args.data)

    # Save prefix includes tech name for non-ASAP7 techs
    if tech_name == "asap7":
        save_prefix = args.device_type
    else:
        save_prefix = f"{tech_name}_{args.device_type}"

    output_dim = 13 if args.mode in ("direct13", "finetune") else 4

    config = TrainConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        trunk_hidden=args.hidden,
        trunk_layers=args.layers,
        patience=args.patience,
    )

    # For finetune mode, lower LR and resume from existing
    if args.mode == "finetune":
        config.lr = args.lr if args.lr != 1e-3 else 1e-4  # Default to lower LR
        # Boost charge/cap weights for transient accuracy (3x default)
        config.w_charges = args.w_charges if args.w_charges is not None else 1.5
        config.w_caps = args.w_caps if args.w_caps is not None else 1.0
        if args.resume is None:
            resume = str(CHECKPOINT_DIR / f"{save_prefix}_best.pt")
        else:
            resume = args.resume
    else:
        if args.w_charges is not None:
            config.w_charges = args.w_charges
        if args.w_caps is not None:
            config.w_caps = args.w_caps
        resume = args.resume

    device_str = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    train(str(data_path), config, device_str,
          save_prefix=save_prefix,
          output_dim=output_dim,
          resume_from=resume)


if __name__ == "__main__":
    main()
