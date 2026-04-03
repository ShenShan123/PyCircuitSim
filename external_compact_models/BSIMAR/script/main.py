"""BSIM-AR: Autoregressive Transformer training for MOSFET compact modeling.

Uses the same .npz datasets and 18-in/13-out format as DirectNet (LEVEL=73).
Default loss is DirectLoss with signed_log normalization. BNI loss is optional.

Usage:
    # Universal model (all techs, DirectLoss)
    conda run -n pycircuitsim python -m external_compact_models.BSIMAR.script.main \
        --device-type nmos --universal --cuda

    # Single tech with BNI loss
    conda run -n pycircuitsim python -m external_compact_models.BSIMAR.script.main \
        --device-type nmos --tech asap7 --loss bni --cuda

    # Custom architecture
    conda run -n pycircuitsim python -m external_compact_models.BSIMAR.script.main \
        --device-type pmos --universal --d-model 128 --nhead 4 --num-layers 4 --cuda
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
from nn_model.data.dataset import load_and_split
from nn_model.data.normalize import Normalizer
from nn_model.architecture.direct_loss import DirectLoss

from external_compact_models.BSIMAR.script.model import TransformerEncoderModel
from external_compact_models.BSIMAR.script.config import (
    BSIMARConfig, CHECKPOINT_DIR, RESULTS_DIR, DATA_DIR, TARGETS,
)
from external_compact_models.BSIMAR.script.train import (
    train_epoch_direct, validate_epoch_direct,
    train_epoch_bni, validate_epoch_bni,
    train_epoch_scheduled, train_epoch_hybrid, train_epoch_curriculum,
    test_model, EarlyStopping,
)
from external_compact_models.BSIMAR.script.losses import (
    WeightedBNILoss, compute_lds_weights_per_target,
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
    parser.add_argument("--device-type", choices=["nmos", "pmos"], default="nmos")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to .npz dataset (auto-resolved if omitted)")
    parser.add_argument("--tech", choices=list(TECH_CONFIGS.keys()), default="asap7")
    parser.add_argument("--universal", action="store_true",
                        help="Train universal model across all techs/variants")
    parser.add_argument("--loss", choices=["direct", "bni"], default="direct",
                        help="Loss function: direct (DirectLoss, default) or bni (WeightedBNILoss)")
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
    # DirectLoss weights
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
                        help="Reorder outputs for autoregressive (charges->caps->cond->id)")
    # Scheduled sampling
    parser.add_argument("--scheduled-sampling", action="store_true",
                        help="Use scheduled sampling (gradual teacher forcing -> autoregressive)")
    parser.add_argument("--ss-warmup", type=int, default=100,
                        help="Epochs to ramp scheduled sampling ratio (default: 100)")
    parser.add_argument("--ss-max-ratio", type=float, default=0.5,
                        help="Max autoregressive ratio (default: 0.5)")
    # Consistency loss
    parser.add_argument("--consistency-weight", type=float, default=0.0,
                        help="Weight for TF-vs-AR consistency loss (0=off, default: 0)")
    # Curriculum on output length
    parser.add_argument("--curriculum", action="store_true",
                        help="Use curriculum on output length (ramp from 1 to 13 targets)")
    parser.add_argument("--curriculum-warmup", type=int, default=50,
                        help="Epochs to ramp curriculum from 1 to all targets (default: 50)")
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

    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        print(f"Generate with: python -m nn_model.data.generate "
              f"--device {args.device_type} {'--universal' if args.universal else f'--tech {tech_label}'}")
        sys.exit(1)

    # -- Device --
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Loss: {args.loss} | Data: {data_path.name}")

    # -- Load data --
    train_ds, val_ds, test_ds, normalizer = load_and_split(str(data_path))
    input_dim = train_ds.inputs.shape[1]
    output_dim = train_ds.outputs.shape[1]
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")

    # -- Optional output reordering for autoregressive --
    _reorder_active = False
    if args.reorder:
        from nn_model.data.normalize import reorder_outputs, unreorder_outputs, _UNREORDER_IDX
        _reorder_active = True
        train_ds.outputs = torch.tensor(
            reorder_outputs(train_ds.outputs.numpy()), dtype=torch.float32)
        val_ds.outputs = torch.tensor(
            reorder_outputs(val_ds.outputs.numpy()), dtype=torch.float32)
        test_ds.outputs = torch.tensor(
            reorder_outputs(test_ds.outputs.numpy()), dtype=torch.float32)
        print("Output columns reordered: charges->caps->cond->id")

    # Create unreorder function for loss computation
    if _reorder_active:
        _unreorder_idx_t = torch.tensor(_UNREORDER_IDX, dtype=torch.long)
        def _unreorder_tensor(t: torch.Tensor) -> torch.Tensor:
            return t[:, _unreorder_idx_t.to(t.device)]
        unreorder_fn = _unreorder_tensor
    else:
        unreorder_fn = None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # -- Model --
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

    # -- Loss + training functions --
    if args.loss == "direct":
        criterion = DirectLoss(
            output_dim=output_dim,
            w_curr=args.w_curr,
            w_cond=args.w_cond,
            w_charges=args.w_charges,
            w_caps=args.w_caps,
            w_zero_bias=args.w_zero_bias,
        )
        train_fn = train_epoch_direct
        val_fn = validate_epoch_direct
    else:
        criterion = WeightedBNILoss()
        # Compute LDS weights and wrap into a weighted Dataset so weights
        # stay associated with samples through DataLoader shuffle.
        print("Computing LDS weights...")
        lds_weights_np = compute_lds_weights_per_target(
            train_ds.outputs.numpy(), n_bins=100,
            lds_kernel="gaussian", lds_ks=5, lds_sigma=2.0,
        )
        # Create a new Dataset that yields (x, y, w) tuples
        train_ds = TensorDataset(
            train_ds.inputs,
            train_ds.outputs,
            torch.tensor(lds_weights_np, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        train_fn = train_epoch_bni
        val_fn = validate_epoch_bni

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
    print(f"\nTraining {save_prefix} for {args.epochs} epochs (patience={args.patience})")
    train_history, val_history = [], []
    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # Compute per-epoch scheduled sampling and curriculum parameters
        ss_ratio = min(epoch / args.ss_warmup, args.ss_max_ratio) if args.scheduled_sampling else 0.0
        n_targets = max(1, int(output_dim * min(epoch / args.curriculum_warmup, 1.0))) if args.curriculum else output_dim

        if args.loss == "direct" and (args.curriculum or args.consistency_weight > 0 or args.scheduled_sampling):
            t_losses = train_epoch_curriculum(
                model, train_loader, criterion, optimizer, device,
                n_targets=n_targets, ss_ratio=ss_ratio,
                consistency_weight=args.consistency_weight,
                unreorder_fn=unreorder_fn)
        elif args.loss == "direct":
            t_losses = train_fn(model, train_loader, criterion, optimizer, device,
                                unreorder_fn=unreorder_fn)
        else:
            # BNI loss: no unreorder_fn (doesn't use DirectLoss column indices)
            t_losses = train_fn(model, train_loader, criterion, optimizer, device)

        if args.loss == "direct":
            v_losses = val_fn(model, val_loader, criterion, device,
                              unreorder_fn=unreorder_fn)
        else:
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
                print(f"  {epoch:4d} | train={train_loss:.5f} val={val_loss:.5f} | "
                      f"id={v_losses.get('id', 0):.5f} "
                      f"gm={v_losses.get('gm', 0):.5f} "
                      f"q={v_losses.get('charges', 0):.5f} "
                      f"cap={v_losses.get('caps', 0):.5f}{status}")
            else:
                print(f"  {epoch:4d} | train={train_loss:.5f} val={val_loss:.5f}{status}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed / epoch:.1f}s/epoch)")
    print(f"Best val loss: {best_val_loss:.6f}")

    # -- Save architecture config for inference reconstruction --
    arch_config = {
        "input_dim": input_dim,
        "target_dim": output_dim,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
    }
    config_path = CHECKPOINT_DIR / f"{save_prefix}_config.npz"
    np.savez(str(config_path), **{k: np.array(v) for k, v in arch_config.items()})
    print(f"Arch config: {config_path}")

    # -- Load best and test --
    model.load_state_dict(torch.load(str(best_path), weights_only=True))
    pred_norm, true_norm = test_model(model, test_loader, device)

    # Unreorder predictions and targets back to original column order for metrics
    if _reorder_active:
        from nn_model.data.normalize import unreorder_outputs
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
