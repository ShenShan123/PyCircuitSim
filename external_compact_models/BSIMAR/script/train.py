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

from nn_model.architecture.direct_loss import DirectLoss


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(
        self,
        patience: int = 30,
        min_delta: float = 1e-5,
        save_path: Optional[str] = None,
    ):
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
    unreorder_fn=None,
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

        # Unreorder pred and targets for DirectLoss (expects original column order)
        pred_loss = unreorder_fn(pred) if unreorder_fn else pred
        y_loss = unreorder_fn(y_batch) if unreorder_fn else y_batch
        losses = criterion(pred_loss, y_loss, x_batch)
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
    unreorder_fn=None,
) -> Dict[str, float]:
    """Validate with autoregressive inference (no teacher forcing)."""
    model.eval()
    total_losses: Dict[str, float] = {}
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model(x_batch)  # autoregressive — no targets

        # Unreorder pred and targets for DirectLoss (expects original column order)
        pred_loss = unreorder_fn(pred) if unreorder_fn else pred
        y_loss = unreorder_fn(y_batch) if unreorder_fn else y_batch
        losses = criterion(pred_loss, y_loss, x_batch)

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


def train_epoch_scheduled(
    model: nn.Module,
    loader: DataLoader,
    criterion: DirectLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    ss_ratio: float = 0.0,
    unreorder_fn=None,
) -> Dict[str, float]:
    """Train one epoch with scheduled sampling."""
    model.train()
    total_losses: Dict[str, float] = {}
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model.forward_scheduled(x_batch, y_batch, ss_ratio=ss_ratio)

        # Unreorder for DirectLoss (expects original column order)
        pred_loss = unreorder_fn(pred) if unreorder_fn else pred
        y_loss = unreorder_fn(y_batch) if unreorder_fn else y_batch
        losses = criterion(pred_loss, y_loss, x_batch)
        losses["total"].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


def train_epoch_hybrid(
    model: nn.Module,
    loader: DataLoader,
    criterion: DirectLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    ss_ratio: float = 0.0,
    consistency_weight: float = 0.1,
    unreorder_fn=None,
) -> Dict[str, float]:
    """Train with scheduled sampling + consistency loss."""
    model.train()
    total_losses: Dict[str, float] = {}
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        # Scheduled-sampling prediction (main supervised loss)
        pred_ss = model.forward_scheduled(x_batch, y_batch, ss_ratio=ss_ratio)

        # Unreorder for DirectLoss (expects original column order)
        pred_loss = unreorder_fn(pred_ss) if unreorder_fn else pred_ss
        y_loss = unreorder_fn(y_batch) if unreorder_fn else y_batch
        losses = criterion(pred_loss, y_loss, x_batch)

        # Consistency: compare pure teacher-forcing vs pure autoregressive
        pred_tf = model(x_batch, y_batch)
        with torch.no_grad():
            pred_ar = model(x_batch)
        loss_consistency = torch.nn.functional.mse_loss(pred_tf, pred_ar)

        total_loss = losses["total"] + consistency_weight * loss_consistency
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        total_losses["consist"] = total_losses.get("consist", 0.0) + loss_consistency.item()
        # Track the actual total used for backprop
        total_losses["total_combined"] = total_losses.get("total_combined", 0.0) + total_loss.item()
        n_batches += 1

    avg = {k: v / n_batches for k, v in total_losses.items()}
    avg["total"] = avg.pop("total_combined", avg["total"])
    return avg


def train_epoch_curriculum(
    model: nn.Module,
    loader: DataLoader,
    criterion: DirectLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    n_targets: int = 13,
    ss_ratio: float = 0.0,
    consistency_weight: float = 0.0,
    unreorder_fn=None,
) -> Dict[str, float]:
    """Train with curriculum on output length + optional scheduled sampling + consistency.

    Only the first n_targets positions contribute to the loss.
    Positions >= n_targets are masked out to avoid diluting the gradient signal.
    """
    model.train()
    total_losses: Dict[str, float] = {}
    n_batches = 0
    target_dim = None

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if target_dim is None:
            target_dim = y_batch.shape[1]

        optimizer.zero_grad()

        pred = model.forward_curriculum(
            x_batch, y_batch, n_targets=n_targets, ss_ratio=ss_ratio)

        # Mask: only compute loss on active (first n_targets) positions.
        # Replace inactive positions in pred with targets (zero loss).
        if n_targets < target_dim:
            pred_masked = pred.clone()
            pred_masked[:, n_targets:] = y_batch[:, n_targets:]
        else:
            pred_masked = pred

        # Unreorder for DirectLoss
        pred_loss = unreorder_fn(pred_masked) if unreorder_fn else pred_masked
        y_loss = unreorder_fn(y_batch) if unreorder_fn else y_batch
        losses = criterion(pred_loss, y_loss, x_batch)

        total_loss = losses["total"]

        if consistency_weight > 0:
            pred_tf = model(x_batch, y_batch)
            with torch.no_grad():
                pred_ar = model(x_batch)
            loss_consist = torch.nn.functional.mse_loss(pred_tf, pred_ar)
            total_loss = total_loss + consistency_weight * loss_consist
            total_losses["consist"] = total_losses.get("consist", 0.0) + loss_consist.item()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


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
