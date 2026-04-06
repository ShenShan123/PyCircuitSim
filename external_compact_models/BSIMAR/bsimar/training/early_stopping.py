"""Early-stopping helper shared by both training paths."""

from typing import Optional

import torch
import torch.nn as nn


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
