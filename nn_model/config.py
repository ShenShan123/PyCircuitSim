"""NN compact model configuration.

Process params, tech configs, output/input columns, and OSDI path are now
owned by PyCMG (pycmg/nn_config.py) and re-exported here for backward
compatibility. Path constants and training hyperparameters remain here.

Note: INPUT_COLUMNS now includes "L" (19 features total, up from 18).
Existing checkpoints trained on 18-feature data are incompatible and must
be retrained after regenerating datasets with the new format.

Note: ProcessParams are no longer hardcoded per variant. They are extracted
on-the-fly from modelcards via extract_process_params(). NNVariantConfig
has been removed — variants are identified by name strings.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NN_MODEL_DIR = PROJECT_ROOT / "nn_model"
CHECKPOINT_DIR = NN_MODEL_DIR / "checkpoints"
DATA_DIR = NN_MODEL_DIR / "data" / "datasets"

# ── Make pycmg importable ─────────────────────────────────────────────────────
PYCMG_DIR = PROJECT_ROOT / "external_compact_models" / "PyCMG"
_PYCMG_PYPATH = str(PYCMG_DIR)
if _PYCMG_PYPATH not in sys.path:
    sys.path.insert(0, _PYCMG_PYPATH)

# ── Re-export everything from pycmg.nn_config ────────────────────────────────
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

# Backward compat alias used by test files
TechConfig = NNTechConfig


# ── Training hyperparameters (NN-project-specific) ───────────────────────────
@dataclass
class TrainConfig:
    """Training hyperparameters — not part of data generation."""
    batch_size: int = 1024
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    trunk_hidden: int = 128
    trunk_layers: int = 3
    head_hidden: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 500
    patience: int = 50
    w_id: float = 1.0
    w_gm: float = 0.5
    w_gds: float = 0.5
    w_gmb: float = 0.3
    w_charges: float = 0.5
    w_caps: float = 0.3
    w_zero_bias: float = 5.0
