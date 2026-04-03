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

    # Scheduled sampling
    ss_warmup_epochs: int = 100   # epochs to ramp from 0 to ss_max_ratio
    ss_max_ratio: float = 0.5     # max fraction of autoregressive tokens

    # Hybrid consistency loss
    consistency_weight: float = 0.1  # weight of consistency term

    # Curriculum on output length
    curriculum_warmup: int = 50  # epochs to ramp from 1 target to target_dim
