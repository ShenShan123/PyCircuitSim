"""Small utilities shared across BSIMAR training scripts."""

import os
from typing import List

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(paths: List[str]) -> None:
    """Create directories if they don't exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")
