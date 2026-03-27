"""Direct output loss for fast MOSFET NN training (no autograd derivatives).

Trains on all 13 output columns directly. Conductances and capacitances
from PyCMG are used as direct supervision targets — the NN learns to
predict them as additional outputs rather than deriving them via autograd.

This is ~50x faster than PhysicsLoss since it avoids create_graph=True.
Jacobian consistency at inference time is still provided by autograd
in mosfet_nn.py (single-sample, fast).

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
charge-capacitance consistency: dq/dV from autograd must match
the capacitance targets from PyCMG. This is slower (~5-10x) due to
create_graph=True but produces smoother charge surfaces that improve
transient simulation accuracy.
"""

import torch
import torch.nn as nn
from typing import Dict


class DirectNet(nn.Module):
    """MLP predicting MOSFET outputs directly.

    Supports two modes:
    - output_dim=4:  predict [id, qg, qd, qb] only (Phase 1)
    - output_dim=13: predict all 13 outputs including derivatives (Phase 2)

    Input: (B, 6) normalized [Vd, Vg, Vs, Vb, NFIN, T]
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        n_layers: int = 4,
        output_dim: int = 4,
    ):
        super().__init__()
        self.output_dim = output_dim

        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
        targets: torch.Tensor,  # (B, 13) full targets
        x: torch.Tensor,        # (B, 6) inputs
    ) -> Dict[str, torch.Tensor]:
        if self.output_dim == 13:
            return self._forward_13(pred, targets, x)
        else:
            return self._forward_4(pred, targets, x)

    def _forward_13(
        self,
        pred: torch.Tensor,    # (B, 13)
        targets: torch.Tensor,  # (B, 13)
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Loss on all 13 outputs with per-group weighting."""
        # Per-group MSE
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

        # Group losses
        loss_curr = loss_id
        loss_cond = (loss_gm + loss_gds + loss_gmb) / 3.0
        loss_charges = (loss_qg + loss_qd + loss_qs + loss_qb) / 4.0
        loss_caps = (loss_cgg + loss_cgd + loss_cgs + loss_cdg + loss_cdd) / 5.0

        # Weighted total (configurable via constructor)
        total = (self.w_curr * loss_curr
                 + self.w_cond * loss_cond
                 + self.w_charges * loss_charges
                 + self.w_caps * loss_caps)

        # Zero-bias penalty
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
        pred: torch.Tensor,    # (B, 4)
        targets: torch.Tensor,  # (B, 13)
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Legacy 4-output loss."""
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


class ChargeConsistencyLoss(nn.Module):
    """DirectLoss + autograd-enforced charge-capacitance consistency.

    Adds a consistency term: dq/dV (via autograd) must match capacitance
    targets from PyCMG. This forces the charge surface to be smooth and
    differentially consistent, which directly improves transient simulation
    accuracy by reducing timing errors at switching edges.

    The consistency loss is computed on the charge outputs (cols 4-5: qg, qd)
    by backpropagating through the network with create_graph=True. This is
    ~5-10x slower than pure DirectLoss but much faster than full PhysicsLoss
    (which also supervises conductances via autograd).

    Args:
        w_consistency: Weight for the autograd charge-cap consistency term.
        w_cond_consistency: Weight for autograd conductance consistency (did/dV).
            Set > 0 to also enforce gm/gds consistency. Default 0 (off).
        **kwargs: Passed to DirectLoss.
    """

    def __init__(
        self,
        w_consistency: float = 1.0,
        w_cond_consistency: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.direct_loss = DirectLoss(output_dim=13, **kwargs)
        self.w_consistency = w_consistency
        self.w_cond_consistency = w_cond_consistency
        self.mse = nn.MSELoss()

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,       # (B, input_dim) normalized
        targets: torch.Tensor,  # (B, 13) normalized
    ) -> Dict[str, torch.Tensor]:
        """Compute combined direct + consistency loss.

        Unlike DirectLoss.forward(pred, targets, x), this takes the model
        itself so it can run autograd through the forward pass.
        """
        # Split: voltage dims need grad, geometry dims don't
        x_v = x[:, :4].requires_grad_(True)
        x_g = x[:, 4:]
        x_full = torch.cat([x_v, x_g], dim=1)

        # Forward pass through model
        pred = model(x_full)  # (B, 13)

        # Standard direct loss (no autograd, fast)
        direct_losses = self.direct_loss(pred, targets, x)

        # --- Charge-capacitance consistency via autograd ---
        # Compute dqg/dV and dqd/dV from the charge predictions
        # qg = pred[:, 4], qd = pred[:, 5]

        grad_qg = torch.autograd.grad(
            pred[:, 4].sum(), x_v, create_graph=True, retain_graph=True,
        )[0]  # (B, 4): dqg/d[Vd, Vg, Vs, Vb]

        grad_qd = torch.autograd.grad(
            pred[:, 5].sum(), x_v, create_graph=True, retain_graph=True,
        )[0]  # (B, 4): dqd/d[Vd, Vg, Vs, Vb]

        # Autograd caps: [cgg, cgd, cgs, cdg, cdd]
        cgg_ag = grad_qg[:, 1:2]   # dqg/dVg
        cgd_ag = grad_qg[:, 0:1]   # dqg/dVd
        cgs_ag = grad_qg[:, 2:3]   # dqg/dVs
        cdg_ag = grad_qd[:, 1:2]   # dqd/dVg
        cdd_ag = grad_qd[:, 0:1]   # dqd/dVd

        # Target caps from PyCMG (cols 8-12)
        t_cgg = targets[:, 8:9]
        t_cgd = targets[:, 9:10]
        t_cgs = targets[:, 10:11]
        t_cdg = targets[:, 11:12]
        t_cdd = targets[:, 12:13]

        # Consistency: autograd caps vs PyCMG targets
        loss_cap_consistency = (
            self.mse(cgg_ag, t_cgg)
            + self.mse(cgd_ag, t_cgd)
            + self.mse(cgs_ag, t_cgs)
            + self.mse(cdg_ag, t_cdg)
            + self.mse(cdd_ag, t_cdd)
        ) / 5.0

        # Optional: conductance consistency (did/dV vs targets)
        loss_cond_consistency = torch.tensor(0.0, device=x.device)
        if self.w_cond_consistency > 0:
            grad_id = torch.autograd.grad(
                pred[:, 0].sum(), x_v, create_graph=True, retain_graph=True,
            )[0]  # (B, 4)
            gm_ag = grad_id[:, 1:2]    # did/dVg
            gds_ag = grad_id[:, 0:1]   # did/dVd
            gmb_ag = grad_id[:, 3:4]   # did/dVb
            loss_cond_consistency = (
                self.mse(gm_ag, targets[:, 1:2])
                + self.mse(gds_ag, targets[:, 2:3])
                + self.mse(gmb_ag, targets[:, 3:4])
            ) / 3.0

        total = (direct_losses["total"]
                 + self.w_consistency * loss_cap_consistency
                 + self.w_cond_consistency * loss_cond_consistency)

        return {
            "total": total,
            "id": direct_losses["id"],
            "gm": direct_losses["gm"],
            "gds": direct_losses["gds"],
            "gmb": direct_losses["gmb"],
            "charges": direct_losses["charges"],
            "caps": direct_losses["caps"],
            "zero_bias": direct_losses["zero_bias"],
            "cap_consist": loss_cap_consistency.detach(),
            "cond_consist": loss_cond_consistency.detach(),
        }
