# BSIM-AR Alignment with DirectNet I/O — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify BSIM-AR's Transformer-based autoregressive model to use the same 18-input / 13-output format as DirectNet, reuse our existing `.npz` data pipeline, and default to `DirectLoss` with signed_log normalization while preserving `WeightedBNILoss` as an alternative.

**Architecture:** BSIM-AR remains under `external_compact_models/BSIMAR/script/` as a standalone training pipeline that **imports** from `nn_model/` for data loading, normalization, and loss. No code duplication. The Transformer decoder-only architecture is kept; only I/O dimensions and data plumbing change.

**Tech Stack:** PyTorch, numpy, scipy, matplotlib. Imports from `nn_model.config`, `nn_model.data.normalize`, `nn_model.data.dataset`, `nn_model.architecture.direct_loss`.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| **Modify** | `external_compact_models/BSIMAR/script/model.py` | Update default `input_dim=18, target_dim=13` |
| **Rewrite** | `external_compact_models/BSIMAR/script/config.py` | Import from `nn_model.config`, CLI-friendly device selection, remove hardcoded paths |
| **Keep** | `external_compact_models/BSIMAR/script/losses.py` | WeightedBNILoss + LDS (no changes) |
| **Rewrite** | `external_compact_models/BSIMAR/script/train.py` | Support DirectLoss (default) + BNI, use DataLoader, save `_norm.npz` |
| **Rewrite** | `external_compact_models/BSIMAR/script/main.py` | New CLI entry point using `.npz` data, argparse matching `nn_model/train.py` style |
| **Rewrite** | `external_compact_models/BSIMAR/script/metrics.py` | Align target names to 13 OUTPUT_COLUMNS |
| **Rewrite** | `external_compact_models/BSIMAR/script/visualization.py` | 13 targets, remove hardcoded config import, use passed args |
| **Keep** | `external_compact_models/BSIMAR/script/utils.py` | No changes |
| **Delete** | `external_compact_models/BSIMAR/script/read_csv.py` | Replaced by `nn_model.data.dataset` |
| **Delete** | `external_compact_models/BSIMAR/script/data_processing.py` | Replaced by `nn_model.data.normalize` |
| **Create** | `external_compact_models/BSIMAR/script/__init__.py` | Make importable as a package |
| **Create** | `external_compact_models/__init__.py` | Enable `-m` package execution |
| **Create** | `external_compact_models/BSIMAR/__init__.py` | Enable `-m` package execution |
| **Delete** | `external_compact_models/BSIMAR/models/*.pth` | Legacy 9-dim checkpoints (incompatible) |

---

### Task 1: Update model.py — Change Default Dimensions

**Files:**
- Modify: `external_compact_models/BSIMAR/script/model.py:25` (constructor defaults)

- [ ] **Step 1: Update default dimensions**

Change the `TransformerEncoderModel` constructor defaults from `input_dim=15, target_dim=9` to `input_dim=18, target_dim=13`:

```python
class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim=18, target_dim=13, d_model=32, nhead=4, 
                 num_layers=2, dim_feedforward=64, dropout=0.1):
```

No other changes to `model.py` — the architecture is dimension-agnostic.

- [ ] **Step 2: Verify import works**

```bash
cd /home/shenshan/NN_SPICE && conda run -n pycircuitsim python -c "
import sys; sys.path.insert(0, 'external_compact_models/BSIMAR')
from script.model import TransformerEncoderModel
m = TransformerEncoderModel()
print(f'input_dim={m.input_dim}, target_dim={m.target_dim}')
assert m.input_dim == 18 and m.target_dim == 13
print('OK')
"
```

Expected: `input_dim=18, target_dim=13` and `OK`.

- [ ] **Step 3: Commit**

```bash
git add external_compact_models/BSIMAR/script/model.py
git commit -m "feat(bsimar): update default dims to 18-in/13-out matching DirectNet"
```

---

### Task 2: Rewrite config.py — Import from nn_model, Remove Hardcoded Paths

**Files:**
- Rewrite: `external_compact_models/BSIMAR/script/config.py`

- [ ] **Step 1: Write new config.py**

Replace the entire file. The new config imports from `nn_model.config` and defines BSIM-AR-specific hyperparameters (Transformer arch, training schedule). No hardcoded data paths — those come from CLI args in `main.py`.

```python
"""BSIM-AR configuration — imports shared infra from nn_model.config."""

import sys
from pathlib import Path
from dataclasses import dataclass

import torch

# Resolve project root and ensure nn_model is importable
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # BSIMAR/script/ -> NN_SPICE/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nn_model.config import (
    TECH_CONFIGS, NNTechConfig, OUTPUT_COLUMNS, INPUT_COLUMNS,
    CHECKPOINT_DIR as NN_CHECKPOINT_DIR,
    DATA_DIR as NN_DATA_DIR,
)

# BSIM-AR checkpoint and results directories
BSIMAR_DIR = Path(__file__).resolve().parent.parent  # external_compact_models/BSIMAR/
CHECKPOINT_DIR = BSIMAR_DIR / "checkpoints"
RESULTS_DIR = BSIMAR_DIR / "results"

# Shared data directory — reuse nn_model's generated .npz files
DATA_DIR = NN_DATA_DIR

# Output columns (same 13 as DirectNet)
TARGETS = OUTPUT_COLUMNS


@dataclass
class BSIMARConfig:
    """BSIM-AR Transformer training hyperparameters."""
    # Architecture
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.2

    # Training schedule
    batch_size: int = 1024
    max_epochs: int = 500
    lr: float = 8e-4
    weight_decay: float = 1e-4

    # Early stopping
    patience: int = 30
    delta: float = 1e-5

    # Loss weights (only used with DirectLoss)
    w_curr: float = 1.0
    w_cond: float = 1.0
    w_charges: float = 0.5
    w_caps: float = 0.3
    w_zero_bias: float = 5.0
```

- [ ] **Step 2: Verify imports resolve**

```bash
cd /home/shenshan/NN_SPICE && conda run -n pycircuitsim python -c "
import sys; sys.path.insert(0, 'external_compact_models/BSIMAR')
from script.config import TARGETS, DATA_DIR, BSIMARConfig, TECH_CONFIGS
print(f'TARGETS ({len(TARGETS)}): {TARGETS}')
print(f'DATA_DIR: {DATA_DIR}')
print(f'Techs: {list(TECH_CONFIGS.keys())}')
cfg = BSIMARConfig()
print(f'd_model={cfg.d_model}, nhead={cfg.nhead}, num_layers={cfg.num_layers}')
print('OK')
"
```

Expected: 13 targets matching `OUTPUT_COLUMNS`, correct paths, all techs listed.

- [ ] **Step 3: Commit**

```bash
git add external_compact_models/BSIMAR/script/config.py
git commit -m "feat(bsimar): rewrite config to import from nn_model, remove hardcoded paths"
```

---

### Task 3: Create __init__.py Files + Delete Obsolete Files

**Files:**
- Create: `external_compact_models/__init__.py`
- Create: `external_compact_models/BSIMAR/__init__.py`
- Create: `external_compact_models/BSIMAR/script/__init__.py`
- Delete: `external_compact_models/BSIMAR/script/read_csv.py`
- Delete: `external_compact_models/BSIMAR/script/data_processing.py`

All three `__init__.py` files are needed for `python -m external_compact_models.BSIMAR.script.main` to work.

- [ ] **Step 1: Create __init__.py files**

`external_compact_models/__init__.py`:
```python
"""External compact model wrappers (PyCMG, BSIMAR)."""
```

`external_compact_models/BSIMAR/__init__.py`:
```python
"""BSIM-AR: Autoregressive Transformer for MOSFET compact modeling."""
```

`external_compact_models/BSIMAR/script/__init__.py`:
```python
"""BSIM-AR training scripts."""
```

- [ ] **Step 2: Delete obsolete data files**

```bash
git rm external_compact_models/BSIMAR/script/read_csv.py
git rm external_compact_models/BSIMAR/script/data_processing.py
```

- [ ] **Step 3: Note about legacy .pth files**

The `external_compact_models/BSIMAR/models/` directory contains pre-trained weights from the old 9-output pipeline. These are incompatible with the new 13-output model. Leave them in place (they are part of the upstream BSIMAR repo) but do NOT load them.

- [ ] **Step 4: Commit**

```bash
git add external_compact_models/__init__.py external_compact_models/BSIMAR/__init__.py external_compact_models/BSIMAR/script/__init__.py
git commit -m "refactor(bsimar): add __init__.py for -m execution, remove CSV data pipeline"
```

---

### Task 4: Rewrite train.py — Support DirectLoss (default) + BNI

**Files:**
- Rewrite: `external_compact_models/BSIMAR/script/train.py`

This is the core change. The new `train.py` must:
1. Use `DataLoader` from `nn_model.data.dataset` (not manual batch slicing)
2. Support two loss modes: `DirectLoss` (default, signed_log) and `WeightedBNILoss` (optional, BNI)
3. Pass teacher-forcing targets during training (`model(inputs, targets)`)
4. Use autoregressive inference during validation/test (`model(inputs)` — no targets)
5. Save `_norm.npz` alongside `.pt` checkpoints
6. Support `EarlyStopping` based on validation loss (not training loss as before)

- [ ] **Step 1: Write new train.py**

```python
"""Training loop for BSIM-AR Transformer model.

Supports two loss functions:
  --loss direct: DirectLoss with signed_log normalization (default)
  --loss bni:    WeightedBNILoss with LDS sample weights

Usage:
    python -m external_compact_models.BSIMAR.script.main --device-type nmos --universal
"""

import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nn_model.architecture.direct_loss import DirectLoss


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 30, min_delta: float = 1e-5,
                 save_path: Optional[str] = None):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_epoch_direct(
    model: nn.Module,
    loader: DataLoader,
    criterion: DirectLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train one epoch with DirectLoss + teacher forcing."""
    model.train()
    total_losses: Dict[str, float] = {}
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(x_batch, y_batch)  # teacher forcing
        losses = criterion(pred, y_batch, x_batch)
        losses["total"].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def validate_epoch_direct(
    model: nn.Module,
    loader: DataLoader,
    criterion: DirectLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Validate with autoregressive inference (no teacher forcing)."""
    model.eval()
    total_losses: Dict[str, float] = {}
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model(x_batch)  # autoregressive — no targets
        losses = criterion(pred, y_batch, x_batch)

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


def train_epoch_bni(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train one epoch with WeightedBNILoss + teacher forcing.

    LDS weights are stored per-sample in the Dataset (third element of
    each __getitem__ return), so they stay correctly associated with
    samples regardless of DataLoader shuffle order.
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
        pred = model(x_batch, y_batch)  # teacher forcing

        if w_batch is not None:
            loss = criterion(pred, y_batch, weights=w_batch)
        else:
            loss = criterion(pred, y_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"total": total_loss / n_batches}


@torch.no_grad()
def validate_epoch_bni(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate with autoregressive inference + BNI loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model(x_batch)  # autoregressive
        loss = criterion(pred, y_batch)
        total_loss += loss.item()
        n_batches += 1

    return {"total": total_loss / n_batches}


@torch.no_grad()
def test_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """Run autoregressive inference, return (pred_norm, true_norm) arrays."""
    model.eval()
    all_pred, all_true = [], []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        pred = model(x_batch)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y_batch.numpy())

    return np.concatenate(all_pred), np.concatenate(all_true)
```

- [ ] **Step 2: Verify import**

```bash
cd /home/shenshan/NN_SPICE && conda run -n pycircuitsim python -c "
import sys; sys.path.insert(0, 'external_compact_models/BSIMAR')
from script.train import train_epoch_direct, validate_epoch_direct, EarlyStopping
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add external_compact_models/BSIMAR/script/train.py
git commit -m "feat(bsimar): rewrite train.py with DirectLoss default + BNI option"
```

---

### Task 5: Rewrite metrics.py — Align to 13 OUTPUT_COLUMNS

**Files:**
- Rewrite: `external_compact_models/BSIMAR/script/metrics.py`

- [ ] **Step 1: Write new metrics.py**

Reuse the physical metrics logic from `nn_model/train.py:162-212` but adapted for BSIM-AR (Normalizer-based denormalization).

```python
"""Evaluation metrics for BSIM-AR predictions."""

import numpy as np
from typing import Dict, List

from nn_model.data.normalize import Normalizer, inv_signed_log, OUTPUT_COLUMN_ORDER


def compute_physical_metrics(
    pred_norm: np.ndarray,
    true_norm: np.ndarray,
    normalizer: Normalizer,
) -> Dict[str, Dict[str, float]]:
    """Compute per-output metrics after denormalization.

    Args:
        pred_norm: (N, 13) normalized predictions.
        true_norm: (N, 13) normalized ground truth.
        normalizer: Fitted normalizer with stats.

    Returns:
        Dict mapping output name to {NRMSE_pct, MAE_norm, R2}.
    """
    stats = normalizer.stats
    metrics: Dict[str, Dict[str, float]] = {}

    # Denormalize both
    pred_phys = normalizer.denormalize_outputs(pred_norm)
    true_phys = normalizer.denormalize_outputs(true_norm)

    for i, name in enumerate(OUTPUT_COLUMN_ORDER):
        y_t = true_phys[:, i]
        y_p = pred_phys[:, i]

        # Filter near-zero values for MRE (avoid division by tiny numbers)
        floor = stats.output_log_floors[i]
        valid = np.abs(y_t) > floor * 100  # 2 decades above floor

        # NRMSE (normalized to peak-to-peak range)
        data_range = y_t.max() - y_t.min()
        if data_range > 0:
            rmse = np.sqrt(np.mean((y_p - y_t) ** 2))
            nrmse_pct = rmse / data_range * 100
        else:
            nrmse_pct = 0.0

        # MRE on valid samples
        if valid.sum() > 0:
            mre_pct = np.mean(np.abs((y_t[valid] - y_p[valid]) / y_t[valid])) * 100
        else:
            mre_pct = float("nan")

        # R2
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - y_t.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # MAE in normalized space
        mae_norm = np.mean(np.abs(pred_norm[:, i] - true_norm[:, i]))

        metrics[name] = {
            "NRMSE(%)": nrmse_pct,
            "MRE(%)": mre_pct,
            "R2": r2,
            "MAE_norm": mae_norm,
        }

    return metrics


def print_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print metrics table."""
    print(f"\n{'Target':>8s} | {'NRMSE%':>8s} | {'MRE%':>8s} | {'R2':>8s} | {'MAE_n':>8s}")
    print("-" * 52)
    for name in OUTPUT_COLUMN_ORDER:
        m = metrics[name]
        mre_str = f"{m['MRE(%)']:8.2f}" if not np.isnan(m["MRE(%)"]) else "     N/A"
        print(f"{name:>8s} | {m['NRMSE(%)']:8.3f} | {mre_str} | {m['R2']:8.4f} | {m['MAE_norm']:8.4f}")

    # Averages
    avg_nrmse = np.mean([m["NRMSE(%)"] for m in metrics.values()])
    valid_mre = [m["MRE(%)"] for m in metrics.values() if not np.isnan(m["MRE(%)"])]
    avg_mre = np.mean(valid_mre) if valid_mre else float("nan")
    avg_r2 = np.mean([m["R2"] for m in metrics.values()])
    print("-" * 52)
    mre_avg_str = f"{avg_mre:8.2f}" if not np.isnan(avg_mre) else "     N/A"
    print(f"{'AVG':>8s} | {avg_nrmse:8.3f} | {mre_avg_str} | {avg_r2:8.4f} |")
```

- [ ] **Step 2: Verify**

```bash
cd /home/shenshan/NN_SPICE && conda run -n pycircuitsim python -c "
import sys; sys.path.insert(0, 'external_compact_models/BSIMAR')
from script.metrics import compute_physical_metrics, print_metrics
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add external_compact_models/BSIMAR/script/metrics.py
git commit -m "feat(bsimar): rewrite metrics.py with 13-output NRMSE/MRE/R2"
```

---

### Task 6: Rewrite visualization.py — 13 Targets, No Hardcoded Imports

**Files:**
- Rewrite: `external_compact_models/BSIMAR/script/visualization.py`

- [ ] **Step 1: Write new visualization.py**

Remove the `from config import BASE_PLOT_PATH` dependency. All paths passed as arguments. Support 13-output layout (4x4 grid).

```python
"""Visualization for BSIM-AR training results."""

import os
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

from nn_model.data.normalize import OUTPUT_COLUMN_ORDER


def plot_scatter_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: str,
    target_names: List[str] = OUTPUT_COLUMN_ORDER,
) -> None:
    """Scatter plots of predicted vs true values for each output."""
    n_targets = len(target_names)
    ncols = 4
    nrows = (n_targets + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), dpi=100)
    axes = axes.flatten()

    for i, target in enumerate(target_names):
        ax = axes[i]
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        # Filter zeros for cleaner plot
        non_zero = y_t != 0
        if non_zero.sum() > 0:
            y_t_nz, y_p_nz = y_t[non_zero], y_p[non_zero]
        else:
            y_t_nz, y_p_nz = y_t, y_p

        r2 = r2_score(y_t_nz, y_p_nz) if len(y_t_nz) > 1 else 0.0
        ax.scatter(y_t_nz, y_p_nz, alpha=0.3, s=2)
        lims = [min(y_t_nz.min(), y_p_nz.min()), max(y_t_nz.max(), y_p_nz.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_title(f"{target} (R2={r2:.4f})", fontsize=11)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(n_targets, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "scatter_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Scatter plot saved: {path}")


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_dir: str,
    title_prefix: str = "",
) -> None:
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(train_losses, "b-", linewidth=1.5, label="Train")
    ax.plot(val_losses, "r-", linewidth=1.5, label="Validation")
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.set_title(f"{title_prefix}Loss Curves", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "loss_curves.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Loss curves saved: {path}")
```

- [ ] **Step 2: Verify**

```bash
cd /home/shenshan/NN_SPICE && conda run -n pycircuitsim python -c "
import sys; sys.path.insert(0, 'external_compact_models/BSIMAR')
from script.visualization import plot_scatter_comparison, plot_loss_curves
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add external_compact_models/BSIMAR/script/visualization.py
git commit -m "feat(bsimar): rewrite visualization for 13 targets, no hardcoded paths"
```

---

### Task 7: Rewrite main.py — New CLI Entry Point

**Files:**
- Rewrite: `external_compact_models/BSIMAR/script/main.py`

This is the orchestration script. CLI mirrors `nn_model/train.py` style: `--device-type`, `--universal`, `--loss`, `--epochs`, `--cuda`, etc.

- [ ] **Step 1: Write new main.py**

```python
"""BSIM-AR: Autoregressive Transformer training for MOSFET compact modeling.

Uses the same .npz datasets and 13-in/13-out format as DirectNet (LEVEL=73).
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
from torch.utils.data import DataLoader

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nn_model.config import TECH_CONFIGS, OUTPUT_COLUMNS
from nn_model.data.dataset import load_and_split, MOSFETDataset
from nn_model.data.normalize import Normalizer
from nn_model.architecture.direct_loss import DirectLoss

from external_compact_models.BSIMAR.script.model import TransformerEncoderModel
from external_compact_models.BSIMAR.script.config import (
    BSIMARConfig, CHECKPOINT_DIR, RESULTS_DIR, DATA_DIR, TARGETS,
)
from external_compact_models.BSIMAR.script.train import (
    train_epoch_direct, validate_epoch_direct,
    train_epoch_bni, validate_epoch_bni,
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
    args = parser.parse_args()

    set_seed(args.seed)

    # ── Resolve data path ──
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

    # ── Device ──
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Loss: {args.loss} | Data: {data_path.name}")

    # ── Load data ──
    train_ds, val_ds, test_ds, normalizer = load_and_split(str(data_path))
    input_dim = train_ds.inputs.shape[1]
    output_dim = train_ds.outputs.shape[1]
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ── Model ──
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

    # ── Loss + training functions ──
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
        lds_weights = None
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
        from torch.utils.data import TensorDataset
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

    # ── Directories ──
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINT_DIR / f"{save_prefix}_best.pt"
    norm_path = CHECKPOINT_DIR / f"{save_prefix}_norm.npz"
    results_subdir = str(RESULTS_DIR / save_prefix)

    early_stopping = EarlyStopping(
        patience=args.patience, min_delta=1e-5, save_path=str(best_path))

    # ── Training loop ──
    print(f"\nTraining {save_prefix} for {args.epochs} epochs (patience={args.patience})")
    train_history, val_history = [], []
    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_losses = train_fn(model, train_loader, criterion, optimizer, device)
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

    # ── Load best and test ──
    model.load_state_dict(torch.load(str(best_path), weights_only=True))
    pred_norm, true_norm = test_model(model, test_loader, device)

    # ── Metrics ──
    metrics = compute_physical_metrics(pred_norm, true_norm, normalizer)
    print_metrics(metrics)

    # ── Visualization ──
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
```

- [ ] **Step 2: Smoke test (dry run, no GPU needed)**

```bash
cd /home/shenshan/NN_SPICE && conda run -n pycircuitsim python -m external_compact_models.BSIMAR.script.main --device-type nmos --universal --epochs 2
```

Expected: Loads `universal_nmos.npz`, trains 2 epochs on CPU, prints metrics, saves checkpoint to `external_compact_models/BSIMAR/checkpoints/ar_universal_nmos_best.pt`.

- [ ] **Step 3: Verify BNI mode**

```bash
cd /home/shenshan/NN_SPICE && conda run -n pycircuitsim python -m external_compact_models.BSIMAR.script.main --device-type nmos --tech asap7 --loss bni --epochs 2
```

Expected: Computes LDS weights, trains 2 epochs with BNI loss, no errors.

- [ ] **Step 4: Commit**

```bash
git add external_compact_models/BSIMAR/script/main.py
git commit -m "feat(bsimar): new CLI entry point with DirectLoss default + BNI option"
```

---

### Task 8: End-to-End Validation — Full Training Run

**Files:** No code changes — validation only.

- [ ] **Step 1: Short training run (50 epochs, CPU) to verify convergence**

```bash
cd /home/shenshan/NN_SPICE && conda run -n pycircuitsim python -m external_compact_models.BSIMAR.script.main \
    --device-type nmos --tech asap7 --epochs 50 --patience 40 \
    --d-model 64 --nhead 4 --num-layers 2 --dim-feedforward 128
```

Expected: Loss decreases, metrics printed, scatter plot saved. Confirm `id` NRMSE is not NaN/inf.

- [ ] **Step 2: Verify checkpoint compatibility**

The saved `_best.pt` + `_norm.npz` should be loadable by the existing pipeline:

```bash
cd /home/shenshan/NN_SPICE && conda run -n pycircuitsim python -c "
from nn_model.data.normalize import NormStats
from pathlib import Path
norm = NormStats.load(str(Path('external_compact_models/BSIMAR/checkpoints/ar_asap7_nmos_norm.npz')))
print(f'input_min shape: {norm.input_min.shape}')
print(f'output_mean shape: {norm.output_mean.shape}')
assert norm.output_mean.shape[0] == 13, 'Expected 13 output columns'
print('Norm stats compatible')
"
```

- [ ] **Step 3: Clean up test artifacts**

```bash
rm -rf external_compact_models/BSIMAR/checkpoints/ar_asap7_nmos*
rm -rf external_compact_models/BSIMAR/results/ar_asap7_nmos/
```

- [ ] **Step 4: Final commit with .gitignore**

Add `.gitignore` to exclude generated files:

```bash
cat > external_compact_models/BSIMAR/checkpoints/.gitignore << 'EOF'
*
!.gitignore
EOF

cat > external_compact_models/BSIMAR/results/.gitignore << 'EOF'
*
!.gitignore
EOF

git add external_compact_models/BSIMAR/checkpoints/.gitignore external_compact_models/BSIMAR/results/.gitignore
git commit -m "chore(bsimar): add .gitignore for checkpoints and results"
```
