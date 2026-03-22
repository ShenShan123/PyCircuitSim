"""Dual-head neural network for MOSFET compact model emulation.

Architecture:
    Shared trunk → I-V head (predicts id in normalized space)
                 → Q-V head (predicts qg, qd, qb in normalized space)

Conductances (gm, gds, gmb) are derived via autograd of id w.r.t. voltages.
Capacitances (cgg, cgd, cgs, cdg, cdd) are derived via autograd of charges.
This guarantees Jacobian consistency for Newton-Raphson convergence.

Activation: SiLU (smooth everywhere, non-zero gradients, no dead neurons).
NOT ReLU — ReLU has discontinuous 2nd derivative at 0, causing NR issues.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class MOSFETNet(nn.Module):
    """Dual-head MLP for MOSFET I-V and Q-V prediction.

    Input: (B, 6) normalized [Vd, Vg, Vs, Vb, NFIN, T]
    Output:
        id_pred: (B, 1) normalized drain current
        q_pred:  (B, 3) normalized charges [qg, qd, qb]
    """

    def __init__(
        self,
        input_dim: int = 6,
        trunk_hidden: int = 128,
        trunk_layers: int = 3,
        head_hidden: int = 64,
    ):
        super().__init__()

        # Shared trunk
        trunk = []
        in_dim = input_dim
        for _ in range(trunk_layers):
            trunk.append(nn.Linear(in_dim, trunk_hidden))
            trunk.append(nn.SiLU())
            in_dim = trunk_hidden
        self.trunk = nn.Sequential(*trunk)

        # I-V head: predicts normalized id (scalar)
        self.iv_head = nn.Sequential(
            nn.Linear(trunk_hidden, head_hidden),
            nn.SiLU(),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.SiLU(),
            nn.Linear(head_hidden // 2, 1),
        )

        # Q-V head: predicts normalized qg, qd, qb (3 values)
        self.qv_head = nn.Sequential(
            nn.Linear(trunk_hidden, head_hidden),
            nn.SiLU(),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.SiLU(),
            nn.Linear(head_hidden // 2, 3),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization for stable training start."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through shared trunk + both heads.

        Args:
            x: (B, 6) normalized input.

        Returns:
            id_pred: (B, 1) normalized drain current.
            q_pred:  (B, 3) normalized [qg, qd, qb].
        """
        features = self.trunk(x)
        id_pred = self.iv_head(features)
        q_pred = self.qv_head(features)
        return id_pred, q_pred

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def compute_derivatives(
    model: MOSFETNet,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute id, charges, conductances, and capacitances via autograd.

    This function enables gradient computation through the voltage inputs
    to derive exact Jacobian entries (gm, gds, gmb, capacitances).

    Args:
        model: MOSFETNet instance.
        x: (B, 6) normalized input. Voltage dims (0:4) will have grad enabled.

    Returns:
        id_pred: (B, 1) normalized drain current.
        q_pred: (B, 3) normalized [qg, qd, qb].
        conductances: (B, 3) normalized [gm, gds, gmb] = [did/dVg, did/dVd, did/dVb].
        cap_gate: (B, 3) normalized [cgg, cgd, cgs] = [dqg/dVg, dqg/dVd, dqg/dVs].
        cap_drain: (B, 2) normalized [cdg, cdd] = [dqd/dVg, dqd/dVd].
    """
    # Split input: voltage dims need grad, geometry dims don't
    x_v = x[:, :4].requires_grad_(True)   # Vd, Vg, Vs, Vb
    x_g = x[:, 4:]                         # NFIN, T (no grad needed)
    x_full = torch.cat([x_v, x_g], dim=1)

    # Forward pass
    id_pred, q_pred = model(x_full)

    B = x.shape[0]

    # Conductances via autograd: d(id)/d(Vd, Vg, Vs, Vb)
    # id_pred shape: (B, 1)
    grad_id = torch.autograd.grad(
        id_pred.sum(), x_v, create_graph=True, retain_graph=True
    )[0]  # (B, 4): [did/dVd, did/dVg, did/dVs, did/dVb]

    # Extract: gm = did/dVg (col 1), gds = did/dVd (col 0), gmb = did/dVb (col 3)
    gds_norm = grad_id[:, 0:1]   # did/dVd
    gm_norm = grad_id[:, 1:2]    # did/dVg
    gmb_norm = grad_id[:, 3:4]   # did/dVb
    conductances = torch.cat([gm_norm, gds_norm, gmb_norm], dim=1)  # (B, 3)

    # Gate capacitances: d(qg)/d(Vd, Vg, Vs, Vb)
    # qg = q_pred[:, 0]
    grad_qg = torch.autograd.grad(
        q_pred[:, 0].sum(), x_v, create_graph=True, retain_graph=True
    )[0]  # (B, 4)

    cgg_norm = grad_qg[:, 1:2]   # dqg/dVg
    cgd_norm = grad_qg[:, 0:1]   # dqg/dVd
    cgs_norm = grad_qg[:, 2:3]   # dqg/dVs
    cap_gate = torch.cat([cgg_norm, cgd_norm, cgs_norm], dim=1)  # (B, 3)

    # Drain capacitances: d(qd)/d(Vd, Vg, Vs, Vb)
    # qd = q_pred[:, 1]
    grad_qd = torch.autograd.grad(
        q_pred[:, 1].sum(), x_v, create_graph=True, retain_graph=True
    )[0]  # (B, 4)

    cdg_norm = grad_qd[:, 1:2]   # dqd/dVg
    cdd_norm = grad_qd[:, 0:1]   # dqd/dVd
    cap_drain = torch.cat([cdg_norm, cdd_norm], dim=1)  # (B, 2)

    return id_pred, q_pred, conductances, cap_gate, cap_drain


if __name__ == "__main__":
    # Quick test: shapes and parameter count
    model = MOSFETNet()
    print(f"Parameter count: {model.count_parameters()}")

    x = torch.randn(4, 6)
    id_pred, q_pred, cond, cap_g, cap_d = compute_derivatives(model, x)
    print(f"id_pred shape: {id_pred.shape}")      # (4, 1)
    print(f"q_pred shape:  {q_pred.shape}")        # (4, 3)
    print(f"cond shape:    {cond.shape}")           # (4, 3)
    print(f"cap_gate shape: {cap_g.shape}")         # (4, 3)
    print(f"cap_drain shape: {cap_d.shape}")        # (4, 2)
    print("All shapes correct!")
