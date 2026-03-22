"""Physics-informed loss function for MOSFET NN training.

Combines direct output loss (id, charges) with derivative-supervised loss
(conductances, capacitances) computed via autograd. This ensures the NN
Jacobian is consistent with its predictions, which is critical for
Newton-Raphson convergence in circuit simulation.

Output column mapping (13 columns):
    [0]  id   — drain current
    [1]  gm   — transconductance (did/dVg)
    [2]  gds  — output conductance (did/dVd)
    [3]  gmb  — bulk transconductance (did/dVb)
    [4]  qg   — gate charge
    [5]  qd   — drain charge
    [6]  qs   — source charge (not predicted, derived from conservation)
    [7]  qb   — bulk charge
    [8]  cgg  — dqg/dVg
    [9]  cgd  — dqg/dVd
    [10] cgs  — dqg/dVs
    [11] cdg  — dqd/dVg
    [12] cdd  — dqd/dVd
"""

import torch
import torch.nn as nn
from typing import Dict

from nn_model.architecture.mosfet_net import MOSFETNet, compute_derivatives
from nn_model.config import TrainConfig


class PhysicsLoss(nn.Module):
    """Multi-component loss with autograd-derived derivative supervision."""

    def __init__(self, config: TrainConfig = TrainConfig()):
        super().__init__()
        self.config = config
        self.mse = nn.MSELoss()

    def forward(
        self,
        model: MOSFETNet,
        x: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted multi-component loss.

        Args:
            model: MOSFETNet instance.
            x: (B, 6) normalized inputs.
            targets: (B, 13) normalized target outputs.

        Returns:
            Dict with 'total' loss and individual component losses.
        """
        cfg = self.config

        # Forward + autograd derivatives
        id_pred, q_pred, cond_pred, cap_gate_pred, cap_drain_pred = \
            compute_derivatives(model, x)

        # Target columns
        t_id = targets[:, 0:1]                # (B, 1)
        t_cond = targets[:, 1:4]              # (B, 3) [gm, gds, gmb]
        t_q = targets[:, 4:7]                 # (B, 3) [qg, qd, qs]
        t_qb = targets[:, 7:8]               # (B, 1) [qb]
        t_cap_gate = targets[:, 8:11]         # (B, 3) [cgg, cgd, cgs]
        t_cap_drain = targets[:, 11:13]       # (B, 2) [cdg, cdd]

        # --- Component losses ---

        # L_id: drain current
        loss_id = self.mse(id_pred, t_id)

        # L_cond: conductances (autograd-derived vs PyCMG targets)
        loss_gm = self.mse(cond_pred[:, 0:1], t_cond[:, 0:1])
        loss_gds = self.mse(cond_pred[:, 1:2], t_cond[:, 1:2])
        loss_gmb = self.mse(cond_pred[:, 2:3], t_cond[:, 2:3])

        # L_charges: qg, qd, qb (qs derived from conservation, not trained)
        # q_pred is [qg, qd, qb], targets [qg, qd, qs]
        # We train on qg (col 0), qd (col 1), qb (separate)
        loss_qg = self.mse(q_pred[:, 0:1], t_q[:, 0:1])
        loss_qd = self.mse(q_pred[:, 1:2], t_q[:, 1:2])
        loss_qb = self.mse(q_pred[:, 2:3], t_qb)
        loss_charges = loss_qg + loss_qd + loss_qb

        # L_caps: capacitances (autograd-derived vs PyCMG targets)
        loss_cap_gate = self.mse(cap_gate_pred, t_cap_gate)
        loss_cap_drain = self.mse(cap_drain_pred, t_cap_drain)
        loss_caps = loss_cap_gate + loss_cap_drain

        # L_zero_bias: extra penalty for zero-bias samples
        # Zero-bias = both Vd and Vg near 0 (in normalized space, near input_min)
        v_norm = x[:, :2]  # Vd_norm, Vg_norm
        zero_mask = (v_norm[:, 0] < 0.15) & (v_norm[:, 1] < 0.15)
        if zero_mask.sum() > 0:
            loss_zero = self.mse(id_pred[zero_mask], t_id[zero_mask])
        else:
            loss_zero = torch.tensor(0.0, device=x.device)

        # --- Weighted total ---
        total = (
            cfg.w_id * loss_id
            + cfg.w_gm * loss_gm
            + cfg.w_gds * loss_gds
            + cfg.w_gmb * loss_gmb
            + cfg.w_charges * loss_charges
            + cfg.w_caps * loss_caps
            + cfg.w_zero_bias * loss_zero
        )

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


if __name__ == "__main__":
    # Quick test: loss computes without error
    model = MOSFETNet()
    criterion = PhysicsLoss()

    x = torch.randn(8, 6)
    targets = torch.randn(8, 13)

    losses = criterion(model, x, targets)
    print("Loss components:")
    for name, val in losses.items():
        print(f"  {name:>10s}: {val.item():.4f}")
    print("Physics loss test PASSED!")
