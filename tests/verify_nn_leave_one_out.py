#!/usr/bin/env python3
"""Leave-one-out transferability experiment for universal NN compact model.

Holds out 1 variant per technology for zero-shot testing:
  - ASAP7: slvt
  - TSMC5: elvt
  - TSMC7: ulvt
  - TSMC12: hvt
  - TSMC16: lnvt

For each device type (nmos, pmos):
  1. Load universal dataset, identify held-out variant data by process params
  2. If held-out variant not in dataset, generate ground truth via PyCMG
  3. Train on remaining (seen) variants
  4. Evaluate zero-shot accuracy on held-out variants
  5. Compare against in-distribution accuracy on seen variants

Usage:
    conda run -n pycircuitsim python tests/verify_nn_leave_one_out.py --device both
    conda run -n pycircuitsim python tests/verify_nn_leave_one_out.py --device nmos --epochs 400
"""
from __future__ import annotations

import sys
import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, "/home/shenshan/pycmg-wrapper")

from nn_model.config import (
    TECH_CONFIGS, TechConfig, CHECKPOINT_DIR, DATA_DIR,
    OSDI_PATH, PROCESS_PARAM_NAMES, TrainConfig,
)
from nn_model.data.dataset import MOSFETDataset
from nn_model.data.normalize import Normalizer, inv_signed_log
from nn_model.architecture.direct_loss import DirectNet, DirectLoss
from nn_model.train import train_epoch, validate_epoch

# ---------------------------------------------------------------------------
# Held-out variant definition
# ---------------------------------------------------------------------------
HELD_OUT: Dict[str, str] = {
    "asap7": "slvt",
    "tsmc5": "elvt",
    "tsmc7": "ulvt",
    "tsmc12": "hvt",
    "tsmc16": "lnvt",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def get_variant_process_array(
    tech_key: str, variant_name: str, device_type: str,
) -> np.ndarray:
    """Return the 7-element process param array for a given variant."""
    tc = TECH_CONFIGS[tech_key]
    vc = tc.variants[variant_name]
    pp = vc.get_process_params(device_type)
    return np.array(pp.as_array())


def match_variant_mask(
    geometry: np.ndarray, process_array: np.ndarray,
) -> np.ndarray:
    """Return boolean mask for rows whose process params match the target.

    Args:
        geometry: (N, 9) array with columns [NFIN, T, PHIG, U0, VSAT, EOT, ETA0, CIT, RDSW].
        process_array: (7,) target process params [PHIG, U0, VSAT, EOT, ETA0, CIT, RDSW].

    Returns:
        (N,) boolean mask.
    """
    proc_cols = geometry[:, 2:]  # (N, 7)
    return np.all(np.isclose(proc_cols, process_array, rtol=1e-4, atol=1e-10), axis=1)


def generate_variant_ground_truth(
    tech_key: str, variant_name: str, device_type: str,
    nfin_values: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    """Generate PyCMG ground truth for a variant not in the universal dataset.

    Uses the same bias sweep as nn_model.data.generate but for a single variant.

    Returns:
        Dict with 'inputs' (N, 4), 'geometry' (N, 9), 'outputs' (N, 13).
    """
    from nn_model.data.generate import (
        generate_dataset, create_pycmg_instance,
    )

    tc = TECH_CONFIGS[tech_key]
    if nfin_values is None:
        nfin_values = tc.nfin_values

    data = generate_dataset(
        tc, device_type,
        variant_names=[variant_name],
        verbose=False,
    )
    return data


def build_held_out_test_data(
    device_type: str,
    inputs: np.ndarray,
    geometry: np.ndarray,
    outputs: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Split universal data into train (seen) and test (held-out) subsets.

    For held-out variants present in the dataset, extract them.
    For held-out variants absent from the dataset, generate via PyCMG.

    Args:
        device_type: 'nmos' or 'pmos'.
        inputs: (N, 4) universal dataset inputs.
        geometry: (N, 9) universal dataset geometry.
        outputs: (N, 13) universal dataset outputs.

    Returns:
        Tuple of:
          - held_out_data: dict mapping 'tech/variant' -> {'inputs', 'geometry', 'outputs'}
          - train_mask: (N,) boolean mask for training data (all non-held-out rows)
    """
    N = inputs.shape[0]
    train_mask = np.ones(N, dtype=bool)
    held_out_data: Dict[str, Dict[str, np.ndarray]] = {}

    for tech_key, variant_name in HELD_OUT.items():
        label = f"{tech_key}/{variant_name}"
        proc_arr = get_variant_process_array(tech_key, variant_name, device_type)
        mask = match_variant_mask(geometry, proc_arr)
        n_matched = mask.sum()

        if n_matched > 0:
            # Variant data exists in dataset: extract it
            print(f"  {label}/{device_type}: found {n_matched} pts in dataset (will remove from training)")
            held_out_data[label] = {
                "inputs": inputs[mask],
                "geometry": geometry[mask],
                "outputs": outputs[mask],
            }
            train_mask &= ~mask
        else:
            # Variant not in dataset: generate via PyCMG
            print(f"  {label}/{device_type}: NOT in dataset, generating via PyCMG...")
            data = generate_variant_ground_truth(tech_key, variant_name, device_type)
            held_out_data[label] = {
                "inputs": data["inputs"],
                "geometry": data["geometry"],
                "outputs": data["outputs"],
            }

    n_train = train_mask.sum()
    print(f"  Training data: {n_train}/{N} pts ({n_train/N*100:.1f}% of universal dataset)")
    return held_out_data, train_mask


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_loo_model(
    device_type: str,
    train_inputs: np.ndarray,
    train_geometry: np.ndarray,
    train_outputs: np.ndarray,
    epochs: int = 400,
    patience: int = 100,
    hidden: int = 384,
    layers: int = 6,
    batch_size: int = 2048,
    use_cuda: bool = False,
) -> Tuple[DirectNet, Normalizer]:
    """Train a model on the reduced (leave-out) training set.

    Uses 85/15 train/val split from the remaining data.

    Returns:
        Tuple of (trained model, normalizer).
    """
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    N = train_inputs.shape[0]

    # Shuffle and split: 85% train, 15% val
    rng = np.random.default_rng(42)
    indices = rng.permutation(N)
    n_train = int(N * 0.85)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # Fit normalizer on training portion
    normalizer = Normalizer()
    normalizer.fit(
        train_inputs[train_idx],
        train_geometry[train_idx],
        train_outputs[train_idx],
    )

    # Normalize
    train_in_norm = normalizer.normalize_inputs(train_inputs[train_idx], train_geometry[train_idx])
    val_in_norm = normalizer.normalize_inputs(train_inputs[val_idx], train_geometry[val_idx])
    train_out_norm = normalizer.normalize_outputs(train_outputs[train_idx])
    val_out_norm = normalizer.normalize_outputs(train_outputs[val_idx])

    train_ds = MOSFETDataset(train_in_norm, train_out_norm)
    val_ds = MOSFETDataset(val_in_norm, val_out_norm)

    input_dim = train_in_norm.shape[1]
    print(f"  Input dim: {input_dim}, Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    output_dim = 13
    model = DirectNet(
        input_dim=input_dim,
        hidden_dim=hidden,
        n_layers=layers + 1,
        output_dim=output_dim,
    ).to(device)
    print(f"  Model parameters: {model.count_parameters()}")

    criterion = DirectLoss(output_dim=output_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    patience_counter = 0

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINT_DIR / f"loo_{device_type}_best.pt"
    norm_path = CHECKPOINT_DIR / f"loo_{device_type}_norm.npz"

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device)
        val_losses = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        status = ""
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            normalizer.stats.save(str(norm_path))
            status = " *best*"
        else:
            patience_counter += 1

        if epoch % 50 == 0 or epoch <= 3 or bool(status):
            lr = scheduler.get_last_lr()[0]
            print(f"    Ep {epoch:4d} | train={train_losses['total']:.5f} "
                  f"val={val_losses['total']:.5f} lr={lr:.2e}{status}")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - t_start
    print(f"  Training: {elapsed:.0f}s ({elapsed/epoch:.1f}s/ep), best_val={best_val_loss:.6f}")

    # Load best checkpoint
    model.load_state_dict(torch.load(best_path, weights_only=True))
    return model, normalizer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_on_subset(
    model: DirectNet,
    normalizer: Normalizer,
    inputs: np.ndarray,
    geometry: np.ndarray,
    outputs: np.ndarray,
    device_type: str,
) -> Dict[str, float]:
    """Evaluate model on a data subset, return id NRMSE in physical units.

    Args:
        model: Trained DirectNet.
        normalizer: Fitted Normalizer.
        inputs: (N, 4) raw voltages.
        geometry: (N, 9) raw geometry + process params.
        outputs: (N, 13) raw outputs (ground truth).
        device_type: 'nmos' or 'pmos'.

    Returns:
        Dict with 'id_nrmse_pct' and 'n_points'.
    """
    device = next(model.parameters()).device
    model.eval()

    # Normalize inputs
    in_norm = normalizer.normalize_inputs(inputs, geometry)
    out_norm = normalizer.normalize_outputs(outputs)

    x_tensor = torch.tensor(in_norm, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_norm = model(x_tensor).cpu().numpy()

    # Denormalize predictions for id (column 0)
    stats = normalizer.stats
    id_pred_log = pred_norm[:, 0] * stats.output_std[0] + stats.output_mean[0]
    id_true_log = out_norm[:, 0] * stats.output_std[0] + stats.output_mean[0]
    id_pred_phys = inv_signed_log(id_pred_log, floor=stats.output_log_floors[0])
    id_true_phys = inv_signed_log(id_true_log, floor=stats.output_log_floors[0])

    # NRMSE
    ptp = id_true_phys.max() - id_true_phys.min()
    if ptp < 1e-30:
        nrmse_pct = 0.0
    else:
        rmse = np.sqrt(np.mean((id_pred_phys - id_true_phys) ** 2))
        nrmse_pct = rmse / ptp * 100.0

    return {
        "id_nrmse_pct": nrmse_pct,
        "n_points": len(inputs),
    }


def evaluate_in_distribution(
    model: DirectNet,
    normalizer: Normalizer,
    inputs: np.ndarray,
    geometry: np.ndarray,
    outputs: np.ndarray,
    device_type: str,
) -> Dict[str, Dict[str, float]]:
    """Evaluate on each seen (in-distribution) variant separately.

    Returns:
        Dict mapping 'tech/variant' -> evaluation metrics.
    """
    results: Dict[str, Dict[str, float]] = {}

    for tech_key, tc in TECH_CONFIGS.items():
        for vname in tc.variants:
            if vname == HELD_OUT.get(tech_key):
                continue  # Skip held-out

            label = f"{tech_key}/{vname}"
            proc_arr = get_variant_process_array(tech_key, vname, device_type)
            mask = match_variant_mask(geometry, proc_arr)
            n_matched = mask.sum()

            if n_matched == 0:
                continue

            metrics = evaluate_on_subset(
                model, normalizer,
                inputs[mask], geometry[mask], outputs[mask],
                device_type,
            )
            results[label] = metrics

    return results


# ---------------------------------------------------------------------------
# Result reporting
# ---------------------------------------------------------------------------
@dataclass
class LOOResult:
    tech: str
    variant: str
    device: str
    in_dist_nrmse: float
    zero_shot_nrmse: float
    gap: float
    n_zero_shot_pts: int


def run_single_device(
    device_type: str,
    epochs: int,
    patience: int,
    hidden: int,
    layers: int,
    batch_size: int,
    use_cuda: bool = False,
) -> List[LOOResult]:
    """Run the full leave-one-out experiment for one device type."""
    print(f"\n{'='*70}")
    print(f"  Leave-One-Out Experiment: {device_type.upper()}")
    print(f"{'='*70}")

    # 1. Load universal dataset
    data_path = DATA_DIR / f"universal_{device_type}.npz"
    if not data_path.exists():
        print(f"  ERROR: Universal dataset not found: {data_path}")
        print(f"  Run: python -m nn_model.data.generate --device {device_type} --universal")
        return []

    data = np.load(str(data_path), allow_pickle=True)
    inputs = data["inputs"]
    geometry = data["geometry"]
    outputs = data["outputs"]
    print(f"  Loaded universal dataset: {inputs.shape[0]} points, "
          f"geometry={geometry.shape}")

    # 2. Build held-out test sets and training mask
    print(f"\n  --- Identifying held-out variants ---")
    held_out_data, train_mask = build_held_out_test_data(
        device_type, inputs, geometry, outputs,
    )

    # 3. Train model on seen variants
    print(f"\n  --- Training LOO model ---")
    train_inputs = inputs[train_mask]
    train_geometry = geometry[train_mask]
    train_outputs = outputs[train_mask]

    model, normalizer = train_loo_model(
        device_type,
        train_inputs, train_geometry, train_outputs,
        epochs=epochs,
        patience=patience,
        hidden=hidden,
        layers=layers,
        batch_size=batch_size,
        use_cuda=use_cuda,
    )

    # 4. Evaluate on held-out (zero-shot) variants
    print(f"\n  --- Evaluating zero-shot accuracy ---")
    zero_shot_results: Dict[str, Dict[str, float]] = {}
    for label, ho_data in held_out_data.items():
        metrics = evaluate_on_subset(
            model, normalizer,
            ho_data["inputs"], ho_data["geometry"], ho_data["outputs"],
            device_type,
        )
        zero_shot_results[label] = metrics
        print(f"    {label}: NRMSE={metrics['id_nrmse_pct']:.2f}% "
              f"({metrics['n_points']} pts)")

    # 5. Evaluate on seen (in-distribution) variants
    print(f"\n  --- Evaluating in-distribution accuracy ---")
    in_dist_results = evaluate_in_distribution(
        model, normalizer,
        inputs, geometry, outputs,  # Use full dataset (includes held-out rows for lookup)
        device_type,
    )

    # Compute per-tech average in-distribution NRMSE
    tech_in_dist_avg: Dict[str, float] = {}
    for label, metrics in in_dist_results.items():
        tech_key = label.split("/")[0]
        if tech_key not in tech_in_dist_avg:
            tech_in_dist_avg[tech_key] = []
        tech_in_dist_avg[tech_key].append(metrics["id_nrmse_pct"])

    for tk in tech_in_dist_avg:
        vals = tech_in_dist_avg[tk]
        tech_in_dist_avg[tk] = float(np.mean(vals))

    # Print per-variant in-distribution results
    for label, metrics in sorted(in_dist_results.items()):
        print(f"    {label}: NRMSE={metrics['id_nrmse_pct']:.2f}% "
              f"({metrics['n_points']} pts)")

    # 6. Assemble results table
    loo_results: List[LOOResult] = []
    for tech_key, variant_name in HELD_OUT.items():
        label = f"{tech_key}/{variant_name}"
        zs = zero_shot_results.get(label, {})
        zs_nrmse = zs.get("id_nrmse_pct", float("nan"))
        n_pts = zs.get("n_points", 0)

        # In-distribution: average of seen variants from same tech
        in_dist = tech_in_dist_avg.get(tech_key, float("nan"))

        gap = zs_nrmse - in_dist
        loo_results.append(LOOResult(
            tech=tech_key,
            variant=variant_name,
            device=device_type,
            in_dist_nrmse=in_dist,
            zero_shot_nrmse=zs_nrmse,
            gap=gap,
            n_zero_shot_pts=n_pts,
        ))

    return loo_results


def print_results_table(results: List[LOOResult]) -> None:
    """Print formatted results table."""
    print(f"\n{'='*90}")
    print(f"  Leave-One-Out Transferability Results")
    print(f"{'='*90}")
    print(f"  {'HeldOut Variant':<20s} {'Device':<8s} {'In-Dist(%)':<12s} "
          f"{'Zero-Shot(%)':<14s} {'Gap(%)':<10s} {'ZS Points':<10s}")
    print(f"  {'-'*20} {'-'*8} {'-'*12} {'-'*14} {'-'*10} {'-'*10}")

    for r in results:
        label = f"{r.tech}/{r.variant}"
        print(f"  {label:<20s} {r.device:<8s} {r.in_dist_nrmse:<12.2f} "
              f"{r.zero_shot_nrmse:<14.2f} {r.gap:<10.2f} {r.n_zero_shot_pts:<10d}")

    # Summary statistics
    gaps = [r.gap for r in results if not np.isnan(r.gap)]
    zs_nrmses = [r.zero_shot_nrmse for r in results if not np.isnan(r.zero_shot_nrmse)]
    id_nrmses = [r.in_dist_nrmse for r in results if not np.isnan(r.in_dist_nrmse)]

    if gaps:
        print(f"\n  Summary:")
        print(f"    In-distribution avg NRMSE:  {np.mean(id_nrmses):.2f}%")
        print(f"    Zero-shot avg NRMSE:        {np.mean(zs_nrmses):.2f}%")
        print(f"    Average transferability gap: {np.mean(gaps):.2f}%")
        print(f"    Max transferability gap:     {np.max(gaps):.2f}% "
              f"({results[np.argmax(gaps)].tech}/{results[np.argmax(gaps)].variant})")
        print(f"    Min transferability gap:     {np.min(gaps):.2f}% "
              f"({results[np.argmin(gaps)].tech}/{results[np.argmin(gaps)].variant})")

        # Pass/fail: gap < 5% considered good transferability
        n_good = sum(1 for g in gaps if g < 5.0)
        print(f"    Good transfer (gap < 5%):   {n_good}/{len(gaps)}")


def export_csv(results: List[LOOResult], output_path: Path) -> None:
    """Export results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Tech", "Variant", "Device", "InDist_NRMSE_pct",
            "ZeroShot_NRMSE_pct", "Gap_pct", "ZeroShot_Points",
        ])
        for r in results:
            writer.writerow([
                r.tech, r.variant, r.device,
                f"{r.in_dist_nrmse:.4f}",
                f"{r.zero_shot_nrmse:.4f}",
                f"{r.gap:.4f}",
                r.n_zero_shot_pts,
            ])
    print(f"\n  CSV exported to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-one-out transferability experiment for universal NN model",
    )
    parser.add_argument(
        "--device", choices=["nmos", "pmos", "both"], default="both",
        help="Device type(s) to evaluate (default: both)",
    )
    parser.add_argument(
        "--epochs", type=int, default=400,
        help="Max training epochs (default: 400)",
    )
    parser.add_argument(
        "--patience", type=int, default=100,
        help="Early stopping patience (default: 100)",
    )
    parser.add_argument(
        "--hidden", type=int, default=384,
        help="Hidden layer dimension (default: 384)",
    )
    parser.add_argument(
        "--layers", type=int, default=6,
        help="Number of hidden layers (default: 6)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048,
        help="Training batch size (default: 2048)",
    )
    parser.add_argument(
        "--cuda", action="store_true",
        help="Use CUDA GPU for training",
    )
    args = parser.parse_args()

    devices = ["nmos", "pmos"] if args.device == "both" else [args.device]

    print("Leave-One-Out Transferability Experiment")
    print(f"Held-out variants: {HELD_OUT}")
    print(f"Devices: {devices}")
    print(f"Training: epochs={args.epochs}, patience={args.patience}, "
          f"hidden={args.hidden}, layers={args.layers}, batch_size={args.batch_size}")

    all_results: List[LOOResult] = []

    for device_type in devices:
        results = run_single_device(
            device_type,
            epochs=args.epochs,
            patience=args.patience,
            hidden=args.hidden,
            layers=args.layers,
            batch_size=args.batch_size,
            use_cuda=args.cuda,
        )
        all_results.extend(results)

    if all_results:
        print_results_table(all_results)

        # Export CSV
        results_dir = PROJECT_ROOT / "tests" / "verify_nn_loo_results"
        csv_path = results_dir / "leave_one_out_results.csv"
        export_csv(all_results, csv_path)
    else:
        print("\nNo results produced. Check that universal datasets exist.")
        print("Generate: python -m nn_model.data.generate --device both --universal")


if __name__ == "__main__":
    main()
