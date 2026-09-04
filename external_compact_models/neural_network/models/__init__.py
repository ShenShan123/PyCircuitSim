"""Full-terminal compact-model architectures.

- `DirectNet` — MLP predicting six independent terminal surfaces.
- `TransformerEncoderModel` — autoregressive six-surface Transformer.
"""

from neural_network.models.direct_net import DirectNet
from neural_network.models.transformer import TransformerEncoderModel

__all__ = ["DirectNet", "TransformerEncoderModel"]
