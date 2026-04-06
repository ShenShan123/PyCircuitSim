"""BSIMAR: NN-based MOSFET compact modeling.

Unified training + inference package for two complementary architectures:

- **DirectNet** (baseline) — `bsimar.models.direct_net.DirectNet`
    Fast MLP that predicts all 13 I-V / Q-V / C-V outputs in one forward pass.
    Used as the reference model for comparison.

- **BSIM-AR Transformer** — `bsimar.models.transformer.TransformerEncoderModel`
    Autoregressive Transformer encoder that generates outputs one-by-one with
    teacher forcing during training. Higher accuracy at higher inference cost.

Both models share:
- 19-feature input (4 voltages + log2(NFIN) + L + T + 12 process params)
- 13-column output (id, gm, gds, gmb, qg, qd, qs, qb, cgg, cgd, cgs, cdg, cdd)
- Normalization pipeline (signed-log or z-score)
- Dataset loading and splits
- Physical-units evaluation metrics

Data generation is handled externally by PyCMG
(`external_compact_models/PyCMG/scripts/generate_nn_data.py`).
"""

from bsimar.config import (
    OSDI_PATH, DEFAULT_TEMPERATURE,
    PROCESS_PARAM_NAMES, ProcessParams,
    NNTechConfig, TECH_CONFIGS,
    OUTPUT_COLUMNS, INPUT_COLUMNS,
    extract_process_params, DEFAULT_NFIN_VALUES,
    BSIMAR_ROOT, CHECKPOINT_DIR, RESULTS_DIR, DATA_DIR,
    DirectNetConfig, TransformerConfig,
    TechConfig,  # backward-compat alias for NNTechConfig
)

__all__ = [
    "OSDI_PATH", "DEFAULT_TEMPERATURE",
    "PROCESS_PARAM_NAMES", "ProcessParams",
    "NNTechConfig", "TECH_CONFIGS", "TechConfig",
    "OUTPUT_COLUMNS", "INPUT_COLUMNS",
    "extract_process_params", "DEFAULT_NFIN_VALUES",
    "BSIMAR_ROOT", "CHECKPOINT_DIR", "RESULTS_DIR", "DATA_DIR",
    "DirectNetConfig", "TransformerConfig",
]
