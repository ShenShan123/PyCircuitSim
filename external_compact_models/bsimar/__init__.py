"""BSIMAR: NN-based MOSFET compact modeling.

Unified training + inference package for two complementary architectures:

- **DirectNet** (baseline) — `bsimar.models.direct_net.DirectNet`
    Fast MLP with tech-code embedding that predicts all 13 I-V / Q-V / C-V
    outputs in one forward pass. Used as the reference model for comparison.

- **BSIM-AR Transformer** — `bsimar.models.transformer.TransformerEncoderModel`
    Autoregressive Transformer encoder that generates outputs one-by-one with
    teacher forcing during training. Higher accuracy at higher inference cost.

Both models share:
- 7-feature continuous input [V(4), NFIN_log, L, T] + discrete tech-variant code
- 13-column output (id, gm, gds, gmb, qg, qd, qs, qb, cgg, cgd, cgs, cdg, cdd)
- Normalization pipeline (asinh + z-score, or plain z-score)
- Dataset loading and splits

Data generation is handled externally by PyCMG
(`external_compact_models/PyCMG/scripts/generate_nn_data.py`).
"""

from bsimar.config import (
    OSDI_PATH, DEFAULT_TEMPERATURE,
    NNTechConfig, TECH_CONFIGS,
    OUTPUT_COLUMNS,
    DEFAULT_NFIN_VALUES,
    BSIMAR_ROOT, CHECKPOINT_DIR, RESULTS_DIR, DATA_DIR,
    DirectNetConfig, TransformerConfig,
    TechConfig,  # backward-compat alias for NNTechConfig
    TECH_VARIANT_CODES, CODE_TO_TECH_VARIANT,
    tech_variant_to_code, UNKNOWN_CODE_ID,
    INPUT_COLUMNS, INPUT_DIM,
    NUM_TSMC_CODES, NUM_TSMC_CODES_WITH_UNKNOWN, NUM_TOTAL_CODES,
)

__all__ = [
    "OSDI_PATH", "DEFAULT_TEMPERATURE",
    "NNTechConfig", "TECH_CONFIGS", "TechConfig",
    "OUTPUT_COLUMNS", "INPUT_COLUMNS", "INPUT_DIM",
    "DEFAULT_NFIN_VALUES",
    "BSIMAR_ROOT", "CHECKPOINT_DIR", "RESULTS_DIR", "DATA_DIR",
    "DirectNetConfig", "TransformerConfig",
    "TECH_VARIANT_CODES", "CODE_TO_TECH_VARIANT",
    "tech_variant_to_code", "UNKNOWN_CODE_ID",
    "NUM_TSMC_CODES", "NUM_TSMC_CODES_WITH_UNKNOWN", "NUM_TOTAL_CODES",
]
