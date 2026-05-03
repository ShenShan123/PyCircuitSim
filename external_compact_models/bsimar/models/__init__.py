"""Model architectures for BSIMAR.

- `DirectNet` — Fast MLP baseline predicting all 13 outputs in one shot.
- `TransformerEncoderModel` — Autoregressive Transformer (primary model).
"""

from bsimar.models.direct_net import DirectNet
from bsimar.models.transformer import TransformerEncoderModel

__all__ = ["DirectNet", "TransformerEncoderModel"]
