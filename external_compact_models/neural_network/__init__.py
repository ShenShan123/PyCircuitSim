"""Full-terminal NN-based MOSFET compact modeling.

Unified training + inference package for two complementary architectures:

- **DirectNet-Full** — `neural_network.models.direct_net.DirectNet`
    Fast MLP with a tech-code embedding and six terminal surfaces.

- **BSIM-AR-Full** — `neural_network.models.transformer.TransformerEncoderModel`
    Autoregressive Transformer over the same six terminal surfaces.

Both models share:
- 7-feature continuous input [V(4), NFIN_log, L, T] + discrete tech-variant code
- six independent current/charge surfaces; source values follow by closure
- Normalization pipeline (asinh + z-score, or plain z-score)
- Dataset loading and splits

Data generation is handled externally by PyCMG
(`external_compact_models/bsim_cmg/scripts/generate_nn_data.py`).
"""

from neural_network.config import (
    OSDI_PATH, DEFAULT_TEMPERATURE,
    NNTechConfig, TECH_CONFIGS,
    DEFAULT_NFIN_VALUES,
    NN_ROOT, CHECKPOINT_DIR, RESULTS_DIR, DATA_DIR,
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
    "INPUT_COLUMNS", "INPUT_DIM",
    "DEFAULT_NFIN_VALUES",
    "NN_ROOT", "CHECKPOINT_DIR", "RESULTS_DIR", "DATA_DIR",
    "DirectNetConfig", "TransformerConfig",
    "TECH_VARIANT_CODES", "CODE_TO_TECH_VARIANT",
    "tech_variant_to_code", "UNKNOWN_CODE_ID",
    "NUM_TSMC_CODES", "NUM_TSMC_CODES_WITH_UNKNOWN", "NUM_TOTAL_CODES",
]
