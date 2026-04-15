"""DirectLoss and ChargeConsistencyLoss for direct-output MOSFET NN training.

DirectLoss trains on all 13 output columns directly. Conductances and
capacitances from PyCMG are used as direct supervision targets — the NN
learns to predict them as additional outputs rather than deriving them
via autograd. This is ~50x faster than PhysicsLoss since it avoids
create_graph=True.

Output column layout (13 columns):
    [0]  id   — drain current
    [1]  gm   — transconductance
    [2]  gds  — output conductance
    [3]  gmb  — bulk transconductance
    [4]  qg   — gate charge
    [5]  qd   — drain charge
    [6]  qs   — source charge
    [7]  qb   — bulk charge
    [8]  cgg  — dqg/dVg
    [9]  cgd  — dqg/dVd
    [10] cgs  — dqg/dVs
    [11] cdg  — dqd/dVg
    [12] cdd  — dqd/dVd

ChargeConsistencyLoss extends DirectLoss with autograd-enforced
charge-capacitance consistency: dq/dV from autograd must match the
capacitance targets from PyCMG. This is slower (~5-10x) due to
create_graph=True but produces smoother charge surfaces that improve
transient simulation accuracy.
"""

from typing import Dict

import torch
import torch.nn as nn


class DirectLoss(nn.Module):
    """Weighted MSE loss on direct outputs.

    Supports both 4-output (id, qg, qd, qb) and 13-output (all) modes.
    """

    def __init__(
        self,
        output_dim: int = 4,
        w_zero_bias: float = 5.0,
        w_curr: float = 1.0,
        w_cond: float = 1.0,
        w_charges: float = 0.5,
        w_caps: float = 0.3,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.w_zero_bias = w_zero_bias
        self.w_curr = w_curr
        self.w_cond = w_cond
        self.w_charges = w_charges
        self.w_caps = w_caps
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        targets: torch.Tensor,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.output_dim == 13:
            return self._forward_13(pred, targets, x)
        return self._forward_4(pred, targets, x)

    def _forward_13(
        self,
        pred: torch.Tensor,
        targets: torch.Tensor,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Loss on all 13 outputs with per-group weighting."""
        loss_id = self.mse(pred[:, 0:1], targets[:, 0:1])
        loss_gm = self.mse(pred[:, 1:2], targets[:, 1:2])
        loss_gds = self.mse(pred[:, 2:3], targets[:, 2:3])
        loss_gmb = self.mse(pred[:, 3:4], targets[:, 3:4])
        loss_qg = self.mse(pred[:, 4:5], targets[:, 4:5])
        loss_qd = self.mse(pred[:, 5:6], targets[:, 5:6])
        loss_qs = self.mse(pred[:, 6:7], targets[:, 6:7])
        loss_qb = self.mse(pred[:, 7:8], targets[:, 7:8])
        loss_cgg = self.mse(pred[:, 8:9], targets[:, 8:9])
        loss_cgd = self.mse(pred[:, 9:10], targets[:, 9:10])
        loss_cgs = self.mse(pred[:, 10:11], targets[:, 10:11])
        loss_cdg = self.mse(pred[:, 11:12], targets[:, 11:12])
        loss_cdd = self.mse(pred[:, 12:13], targets[:, 12:13])

        loss_curr = loss_id
        loss_cond = (loss_gm + loss_gds + loss_gmb) / 3.0
        loss_charges = (loss_qg + loss_qd + loss_qs + loss_qb) / 4.0
        loss_caps = (loss_cgg + loss_cgd + loss_cgs + loss_cdg + loss_cdd) / 5.0

        total = (self.w_curr * loss_curr
                 + self.w_cond * loss_cond
                 + self.w_charges * loss_charges
                 + self.w_caps * loss_caps)

        zero_mask = (x[:, 0] < 0.15) & (x[:, 1] < 0.15)
        if zero_mask.sum() > 0:
            loss_zero = self.mse(pred[zero_mask, 0:1], targets[zero_mask, 0:1])
        else:
            loss_zero = torch.tensor(0.0, device=pred.device)

        total = total + self.w_zero_bias * loss_zero

        return {
            "total": total,
            "id": loss_id.detach(),
            "gm": loss_gm.detach(),
            "gds": loss_gds.detach(),
            "gmb": loss_gmb.detach(),
            "charges": loss_charges.detach(),
            "caps": loss_caps.detach(),
            "zero_bias": loss_zero.detach(),
        }

    def _forward_4(
        self,
        pred: torch.Tensor,
        targets: torch.Tensor,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Legacy 4-output loss (id, qg, qd, qb)."""
        t_id = targets[:, 0:1]
        t_qg = targets[:, 4:5]
        t_qd = targets[:, 5:6]
        t_qb = targets[:, 7:8]
        t_direct = torch.cat([t_id, t_qg, t_qd, t_qb], dim=1)

        loss_main = self.mse(pred, t_direct)
        loss_id = self.mse(pred[:, 0:1], t_id)

        zero_mask = (x[:, 0] < 0.15) & (x[:, 1] < 0.15)
        if zero_mask.sum() > 0:
            loss_zero = self.mse(pred[zero_mask, 0:1], t_id[zero_mask])
        else:
            loss_zero = torch.tensor(0.0, device=pred.device)

        total = loss_main + self.w_zero_bias * loss_zero

        return {
            "total": total,
            "id": loss_id.detach(),
            "gm": torch.tensor(0.0),
            "gds": torch.tensor(0.0),
            "gmb": torch.tensor(0.0),
            "charges": torch.tensor(0.0),
            "caps": torch.tensor(0.0),
            "zero_bias": loss_zero.detach(),
        }
