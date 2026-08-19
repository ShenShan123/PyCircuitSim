"""Loss functions for BSIMAR training."""

from neural_network.losses.bni_mae import (
    MAELoss,
    SobolevIdLoss,
    SOBOLEV_ID_CHANNELS,
    compute_lds_weights_per_target,
)

__all__ = [
    "MAELoss", "SobolevIdLoss", "SOBOLEV_ID_CHANNELS",
    "compute_lds_weights_per_target",
]
