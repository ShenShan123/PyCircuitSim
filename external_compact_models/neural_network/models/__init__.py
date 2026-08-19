"""Model architectures for BSIMAR.

- `DirectNet` — Fast MLP baseline predicting all 13 outputs in one shot.
- `TransformerEncoderModel` — Autoregressive Transformer (primary model).
"""

from neural_network.models.direct_net import DirectNet
from neural_network.models.transformer import TransformerEncoderModel

__all__ = ["DirectNet", "TransformerEncoderModel"]
