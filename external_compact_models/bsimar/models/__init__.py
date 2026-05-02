"""Model architectures for BSIMAR.

- `DirectNet` — Fast MLP baseline predicting all 13 outputs in one shot.
- `TransformerEncoderModel` — Autoregressive Transformer (primary model).
"""

from bsimar.models.direct_net import DirectNet
from bsimar.models.transformer import TransformerEncoderModel
from bsimar.models.id_gate import apply_id_gate

__all__ = ["DirectNet", "TransformerEncoderModel", "apply_id_gate"]
