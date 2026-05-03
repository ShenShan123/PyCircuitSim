"""Configuration for BSIMAR training and inference.

- Re-exports PyCMG's `nn_config` (tech registry, output columns).
- Defines project paths: checkpoints, results, data.
- Defines training hyperparameter dataclasses for both architectures.
- Defines the tech-variant code registry (discrete tech embedding).

Downstream consumers (pycircuitsim parser, mosfet_directnet, mosfet_bsimar,
tests) should import from here.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
    NNTechConfig,
    TECH_CONFIGS,
    OUTPUT_COLUMNS,
    DEFAULT_NFIN_VALUES,
)

# Backward-compat alias retained for existing tests/netlist parser
TechConfig = NNTechConfig


# ── Tech-Variant Code Registry ──────────────────────────────────────────────
# Each (tech, variant) pair gets a stable integer ID for the tech embedding.
# TSMC codes occupy 0-16, UNKNOWN is 17, ASAP7 codes are 18-21.
# Total vocabulary size: 22.
#
# During TSMC-only pre-training the model sees codes 0-16 + 17 (UNKNOWN).
# ASAP7 codes (18-21) are added when fine-tuning on ASAP7 data.

def _build_tech_variant_codes() -> Tuple[
    Dict[Tuple[str, str], int],
    Dict[int, Tuple[str, str]],
    List[Tuple[str, str]],
]:
    """Build the canonical (tech, variant) -> code mapping.

    Returns (forward_map, reverse_map, ordered_list).
    """
    # Order: TSMC techs first (sorted by node), then ASAP7.
    # Within each tech, variants follow the order in TECH_CONFIGS.
    ordered: List[Tuple[str, str]] = []
    for tech_name in ("tsmc5", "tsmc7", "tsmc12", "tsmc16"):
        cfg = TECH_CONFIGS[tech_name]
        for variant in cfg.variant_names:
            ordered.append((tech_name, variant))
    # slot 17 = UNKNOWN (reserved, not in the list)
    for variant in TECH_CONFIGS["asap7"].variant_names:
        ordered.append(("asap7", variant))

    forward: Dict[Tuple[str, str], int] = {}
    reverse: Dict[int, Tuple[str, str]] = {}
    code = 0
    for tv in ordered:
        if code == 17:
            code = 18  # skip the UNKNOWN slot
        forward[tv] = code
        reverse[code] = tv
        code += 1
    return forward, reverse, ordered


TECH_VARIANT_CODES: Dict[Tuple[str, str], int]
CODE_TO_TECH_VARIANT: Dict[int, Tuple[str, str]]
_TECH_VARIANT_ORDER: List[Tuple[str, str]]
TECH_VARIANT_CODES, CODE_TO_TECH_VARIANT, _TECH_VARIANT_ORDER = (
    _build_tech_variant_codes()
)

UNKNOWN_CODE_ID: int = 17
NUM_TSMC_CODES: int = 17           # codes 0-16
NUM_TSMC_CODES_WITH_UNKNOWN: int = 18  # codes 0-17 (pre-train vocab)
NUM_TOTAL_CODES: int = 22          # codes 0-21 (full vocab after fine-tune)

# Input layout: 7 continuous features (no process params)
INPUT_COLUMNS: List[str] = [
    "Vd", "Vg", "Vs", "Vb",   # 4 terminal voltages
    "NFIN", "L", "T",          # 3 geometry / operating-condition scalars
]
INPUT_DIM: int = 7


def tech_variant_to_code(tech: str, variant: str) -> int:
    """Look up the integer code for a (tech, variant) pair.

    Returns UNKNOWN_CODE_ID if the pair is not in the registry.
    """
    return TECH_VARIANT_CODES.get(
        (tech.lower(), variant.lower()), UNKNOWN_CODE_ID
    )


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
