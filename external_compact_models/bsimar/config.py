"""Configuration for BSIMAR training and inference.

- Re-exports PyCMG's `nn_config` (tech registry, process params, columns).
- Defines project paths: checkpoints, results, data.
- Defines training hyperparameter dataclasses for both architectures.

Replaces the old `nn_model.config` + `external_compact_models.BSIMAR.script.config`
split. Downstream consumers (pycircuitsim parser, mosfet_nn, mosfet_bsimar,
tests) should import from here.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

# ── Project paths ────────────────────────────────────────────────────────────
# Path hierarchy (after path-depth collapse):
#   parents[0] = bsimar/                          (BSIMAR_ROOT — package lives at the top of its dir)
#   parents[1] = external_compact_models/
#   parents[2] = <project root>                   (PROJECT_ROOT)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BSIMAR_ROOT = Path(__file__).resolve().parents[0]
CHECKPOINT_DIR = BSIMAR_ROOT / "checkpoints"
RESULTS_DIR = BSIMAR_ROOT / "results"
DATA_DIR = BSIMAR_ROOT / "data" / "datasets"

# ── Make pycmg importable ────────────────────────────────────────────────────
PYCMG_DIR = PROJECT_ROOT / "external_compact_models" / "PyCMG"
_PYCMG_PYPATH = str(PYCMG_DIR)
if _PYCMG_PYPATH not in sys.path:
    sys.path.insert(0, _PYCMG_PYPATH)

# ── Re-export PyCMG's NN config (single source of truth) ─────────────────────
from pycmg.nn_config import (  # noqa: E402
    OSDI_PATH,
    DEFAULT_TEMPERATURE,
    PROCESS_PARAM_NAMES,
    ProcessParams,
    NNTechConfig,
    TECH_CONFIGS,
    OUTPUT_COLUMNS,
    INPUT_COLUMNS,
    extract_process_params,
    DEFAULT_NFIN_VALUES,
)

# Backward-compat alias retained for existing tests/netlist parser
TechConfig = NNTechConfig


# ── Training hyperparameters ─────────────────────────────────────────────────
@dataclass
class DirectNetConfig:
    """Training hyperparameters for the DirectNet (MLP baseline)."""
    # Data
    batch_size: int = 1024
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    # Architecture
    trunk_hidden: int = 128
    trunk_layers: int = 3
    # Optimization
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 500
    patience: int = 50
    # DirectLoss group weights
    w_id: float = 1.0
    w_gm: float = 0.5
    w_gds: float = 0.5
    w_gmb: float = 0.3
    w_charges: float = 0.5
    w_caps: float = 0.3
    w_zero_bias: float = 5.0


@dataclass
class TransformerConfig:
    """Training hyperparameters for the BSIM-AR Transformer model."""
    # Architecture
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.2
    # Optimization
    batch_size: int = 1024
    max_epochs: int = 500
    lr: float = 8e-4
    weight_decay: float = 1e-4
    # Early stopping
    patience: int = 30
    delta: float = 1e-5
    # DirectLoss group weights (only used with --loss direct)
    w_curr: float = 1.0
    w_cond: float = 1.0
    w_charges: float = 0.5
    w_caps: float = 0.3
    w_zero_bias: float = 5.0
    # Scheduled sampling
    ss_warmup_epochs: int = 100
    ss_max_ratio: float = 0.5
    # Hybrid consistency
    consistency_weight: float = 0.1
    # Curriculum
    curriculum_warmup: int = 50


# Legacy alias — some downstream code still references TrainConfig
TrainConfig = DirectNetConfig
BSIMARConfig = TransformerConfig
