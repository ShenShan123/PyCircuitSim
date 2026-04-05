"""BSIM-AR: Autoregressive Transformer training for MOSFET compact modeling.

Supports two normalization modes:
  --norm-mode zscore   : plain z-score (paper's approach, default)
  --norm-mode signedlog: signed_log + z-score (nn_model compat)

Supports multiple loss functions:
  --loss direct : DirectLoss (group-weighted MSE, same as DirectNet)
  --loss mae    : MAE (simple or composed with --lds)
  --loss bni    : WeightedBNILoss (batch-normalized interpolation)

Usage:
    # Paper's recommended setup (zscore + MAE + LDS + filtering)
    conda run -n pycircuitsim python -m external_compact_models.BSIMAR.script.main \
        --device-type nmos --universal --loss mae --lds --cuda

    # Backward compat (signedlog + DirectLoss, no filter)
    conda run -n pycircuitsim python -m external_compact_models.BSIMAR.script.main \
        --device-type nmos --universal --norm-mode signedlog --loss direct --no-filter --cuda
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nn_model.config import TECH_CONFIGS, OUTPUT_COLUMNS
from nn_model.data.normalize import (
    OUTPUT_COLUMN_ORDER, reorder_outputs, unreorder_outputs, _UNREORDER_IDX,
)
from nn_model.architecture.direct_loss import DirectLoss

from external_compact_models.BSIMAR.script.model import TransformerEncoderModel
from external_compact_models.BSIMAR.script.config import (
    BSIMARConfig, CHECKPOINT_DIR, RESULTS_DIR, DATA_DIR, TARGETS,
)
from external_compact_models.BSIMAR.script.normalize import BSIMARNormalizer
from external_compact_models.BSIMAR.script.data import load_and_split_bsimar
from external_compact_models.BSIMAR.script.train import (
    train_epoch_direct, validate_epoch_direct,
    train_epoch_bni, validate_epoch_bni,
    train_epoch_scheduled, train_epoch_hybrid, train_epoch_curriculum,
    test_model, EarlyStopping,
)
from external_compact_models.BSIMAR.script.losses import (
    WeightedBNILoss, MAELoss, compute_lds_weights_per_target,
)
from external_compact_models.BSIMAR.script.metrics import (
    compute_physical_metrics, print_metrics,
)
from external_compact_models.BSIMAR.script.visualization import (
    plot_scatter_comparison, plot_loss_curves,
)
from external_compact_models.BSIMAR.script.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BSIM-AR: Autoregressive Transformer for MOSFET modeling")
    # Data
    parser.add_argument("--device-type", choices=["nmos", "pmos"], default="nmos")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to .npz dataset (auto-resolved if omitted)")
    parser.add_argument("--tech", choices=list(TECH_CONFIGS.keys()), default="asap7")
    parser.add_argument("--universal", action="store_true",
                        help="Train universal model across all techs/variants")
    # Normalization
    parser.add_argument("--norm-mode", choices=["zscore", "signedlog"],
                        default="zscore",
                        help="Normalization mode (default: zscore)")
    # Data filtering
    parser.add_argument("--no-filter", action="store_true",
                        help="Skip small-value data filtering")
    # Loss
    parser.add_argument("--loss", choices=["direct", "mae", "bni"], default="mae",
                        help="Loss function (default: mae)")
    parser.add_argument("--lds", action="store_true",
                        help="Enable LDS reweighting (works with mae/bni)")
    # Training
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=8e-4)
    # Transformer architecture
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    # DirectLoss weights (only used with --loss direct)
    parser.add_argument("--w-curr", type=float, default=1.0)
    parser.add_argument("--w-cond", type=float, default=1.0)
    parser.add_argument("--w-charges", type=float, default=0.5)
    parser.add_argument("--w-caps", type=float, default=0.3)
    parser.add_argument("--w-zero-bias", type=float, default=5.0)
    # Hardware
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    # Output reordering
    parser.add_argument("--reorder", action="store_true",
                        help="Reorder outputs for autoregressive "
                             "(charges->caps->cond->id)")
    # Scheduled sampling
    parser.add_argument("--scheduled-sampling", action="store_true")
    parser.add_argument("--ss-warmup", type=int, default=100)
    parser.add_argument("--ss-max-ratio", type=float, default=0.5)
    # Consistency loss
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    # Curriculum
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--curriculum-warmup", type=int, default=50)
    parser.add_argument("--exp-name", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    # -- Resolve data path --
    if args.universal:
        tech_label = "universal"
        data_path = (Path(args.data) if args.data
                     else DATA_DIR / f"universal_{args.device_type}.npz")
        save_prefix = f"ar_universal_{args.device_type}"
    else:
        tech_label = args.tech.lower()
        data_path = (Path(args.data) if args.data
                     else DATA_DIR / f"{tech_label}_{args.device_type}.npz")
        save_prefix = f"ar_{tech_label}_{args.device_type}"

    if args.exp_name:
        save_prefix = f"{args.exp_name}_{args.device_type}"

    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        sys.exit(1)

    # -- Device --
    device = torch.device(
        "cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Norm: {args.norm_mode} | "
          f"Loss: {args.loss}{'+lds' if args.lds else ''} | "
          f"Filter: {not args.no_filter} | Data: {data_path.name}")

    # -- Load data (unified loader, handles both norm modes) --
    train_ds, val_ds, test_ds, normalizer = load_and_split_bsimar(
        str(data_path),
        column_names=OUTPUT_COLUMNS,
        norm_mode=args.norm_mode,
        apply_filter=not args.no_filter,
    )
    input_dim = train_ds.inputs.shape[1]
    output_dim = train_ds.outputs.shape[1]
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")

    # -- Optional output reordering --
    _reorder_active = False
    if args.reorder:
        _reorder_active = True
        train_ds.outputs = torch.tensor(
            reorder_outputs(train_ds.outputs.numpy()), dtype=torch.float32)
        val_ds.outputs = torch.tensor(
            reorder_outputs(val_ds.outputs.numpy()), dtype=torch.float32)
        test_ds.outputs = torch.tensor(
            reorder_outputs(test_ds.outputs.numpy()), dtype=torch.float32)
        print("Output columns reordered: charges->caps->cond->id")

    # Unreorder function for DirectLoss (expects original column order)
    if _reorder_active:
        _unreorder_idx_t = torch.tensor(_UNREORDER_IDX, dtype=torch.long)
        def _unreorder_tensor(t: torch.Tensor) -> torch.Tensor:
            return t[:, _unreorder_idx_t.to(t.device)]
        unreorder_fn = _unreorder_tensor
    else:
        unreorder_fn = None

    # -- Model (must be created before optimizer) --
    model = TransformerEncoderModel(
        input_dim=input_dim,
        target_dim=output_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    # -- Loss + training function selection --
    if args.loss == "direct":
        criterion = DirectLoss(
            output_dim=output_dim,
            w_curr=args.w_curr, w_cond=args.w_cond,
            w_charges=args.w_charges, w_caps=args.w_caps,
            w_zero_bias=args.w_zero_bias,
        )
        train_fn = train_epoch_direct
        val_fn = validate_epoch_direct
        # DirectLoss doesn't use LDS weights in the loader
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True)

    elif args.loss == "mae":
        criterion = MAELoss()
        train_fn = train_epoch_bni   # handles (x, y) and (x, y, w) batches
        val_fn = validate_epoch_bni

        if args.lds:
            print("Computing LDS weights...")
            lds_weights_np = compute_lds_weights_per_target(
                train_ds.outputs.numpy(), n_bins=100,
                lds_kernel="gaussian", lds_ks=5, lds_sigma=2.0,
            )
            train_ds_weighted = TensorDataset(
                train_ds.inputs, train_ds.outputs,
                torch.tensor(lds_weights_np, dtype=torch.float32),
            )
            train_loader = DataLoader(
                train_ds_weighted, batch_size=args.batch_size, shuffle=True)
        else:
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True)

    else:  # bni
        criterion = WeightedBNILoss()
        train_fn = train_epoch_bni
        val_fn = validate_epoch_bni

        if args.lds:
            print("Computing LDS weights...")
            lds_weights_np = compute_lds_weights_per_target(
                train_ds.outputs.numpy(), n_bins=100,
                lds_kernel="gaussian", lds_ks=5, lds_sigma=2.0,
            )
            train_ds_weighted = TensorDataset(
                train_ds.inputs, train_ds.outputs,
                torch.tensor(lds_weights_np, dtype=torch.float32),
            )
            train_loader = DataLoader(
                train_ds_weighted, batch_size=args.batch_size, shuffle=True)
        else:
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4)

    # -- Directories --
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINT_DIR / f"{save_prefix}_best.pt"
    norm_path = CHECKPOINT_DIR / f"{save_prefix}_norm.npz"
    results_subdir = str(RESULTS_DIR / save_prefix)

    early_stopping = EarlyStopping(
        patience=args.patience, min_delta=1e-5, save_path=str(best_path))

    # -- Training loop --
    print(f"\nTraining {save_prefix} for {args.epochs} epochs "
          f"(patience={args.patience})")
    train_history, val_history = [], []
    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        ss_ratio = (min(epoch / args.ss_warmup, args.ss_max_ratio)
                    if args.scheduled_sampling else 0.0)
        n_targets = (max(1, int(output_dim * min(
            epoch / args.curriculum_warmup, 1.0)))
            if args.curriculum else output_dim)

        # Select training path
        if args.loss == "direct":
            if args.curriculum or args.consistency_weight > 0 or args.scheduled_sampling:
                t_losses = train_epoch_curriculum(
                    model, train_loader, criterion, optimizer, device,
                    n_targets=n_targets, ss_ratio=ss_ratio,
                    consistency_weight=args.consistency_weight,
                    unreorder_fn=unreorder_fn)
            else:
                t_losses = train_fn(
                    model, train_loader, criterion, optimizer, device,
                    unreorder_fn=unreorder_fn)
            v_losses = val_fn(
                model, val_loader, criterion, device,
                unreorder_fn=unreorder_fn)
        else:
            # mae / bni: no unreorder_fn needed
            t_losses = train_fn(
                model, train_loader, criterion, optimizer, device)
            v_losses = val_fn(model, val_loader, criterion, device)

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

        if epoch % 20 == 0 or epoch <= 5 or status:
            if args.loss == "direct":
                print(f"  {epoch:4d} | train={train_loss:.5f} "
                      f"val={val_loss:.5f} | "
                      f"id={v_losses.get('id', 0):.5f} "
                      f"gm={v_losses.get('gm', 0):.5f} "
                      f"q={v_losses.get('charges', 0):.5f} "
                      f"cap={v_losses.get('caps', 0):.5f}{status}")
            else:
                print(f"  {epoch:4d} | train={train_loss:.5f} "
                      f"val={val_loss:.5f}{status}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed / epoch:.1f}s/epoch)")
    print(f"Best val loss: {best_val_loss:.6f}")

    # -- Save architecture config --
    arch_config = {
        "input_dim": input_dim, "target_dim": output_dim,
        "d_model": args.d_model, "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }
    config_path = CHECKPOINT_DIR / f"{save_prefix}_config.npz"
    np.savez(str(config_path),
             **{k: np.array(v) for k, v in arch_config.items()})
    print(f"Arch config: {config_path}")

    # -- Load best and test --
    model.load_state_dict(torch.load(str(best_path), weights_only=True))
    pred_norm, true_norm = test_model(model, test_loader, device)

    # Unreorder back to standard column order for metrics
    if _reorder_active:
        pred_norm = unreorder_outputs(pred_norm)
        true_norm = unreorder_outputs(true_norm)

    # -- Metrics --
    metrics = compute_physical_metrics(pred_norm, true_norm, normalizer)
    print_metrics(metrics)

    # -- Visualization --
    pred_phys = normalizer.denormalize_outputs(pred_norm)
    true_phys = normalizer.denormalize_outputs(true_norm)
    plot_scatter_comparison(true_phys, pred_phys, results_subdir)
    plot_loss_curves(train_history, val_history, results_subdir,
                     title_prefix=f"BSIM-AR {save_prefix} ")

    print(f"\nCheckpoint: {best_path}")
    print(f"Norm stats: {norm_path}")
    print(f"Results:    {results_subdir}/")


if __name__ == "__main__":
    main()
