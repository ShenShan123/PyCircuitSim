"""MAE + LDS loss for BSIMAR / DirectNet training.

The production loss is plain MAE multiplied by per-target Label
Distribution Smoothing weights (Yang et al., ICML 2021). Computed
once on the train split and broadcast over batches.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.signal.windows import triang


def _kernel(kernel: str, ks: int, sigma: float) -> np.ndarray:
    assert kernel in ("gaussian", "triang", "laplace")
    half = (ks - 1) // 2
    if kernel == "gaussian":
        base = [0.0] * half + [1.0] + [0.0] * half
        w = gaussian_filter1d(base, sigma=sigma)
    elif kernel == "triang":
        w = triang(ks)
    else:
        f = lambda x: np.exp(-abs(x) / sigma) / (2.0 * sigma)
        w = np.array([f(x) for x in range(-half, half + 1)])
    return w / w.max()


def compute_lds_weights_per_target(
    y_train: np.ndarray,
    n_bins: int = 100,
    lds_kernel: str = "gaussian",
    lds_ks: int = 5,
    lds_sigma: float = 0.8,
    strategy: str = "uniform",
) -> np.ndarray:
    """Per-sample LDS weight, one column per target. Mean-normalised."""
    from sklearn.preprocessing import KBinsDiscretizer

    n, d = y_train.shape
    weights = np.ones((n, d), dtype=np.float32)
    kernel = _kernel(lds_kernel, lds_ks, lds_sigma)

    for k in range(d):
        col = y_train[:, k:k + 1]
        if col.max() == col.min():
            continue
        try:
            disc = KBinsDiscretizer(
                n_bins=n_bins, encode="ordinal", strategy=strategy)
            bin_idx = disc.fit_transform(col).flatten().astype(int)
        except Exception:
            continue
        counts = np.clip(
            np.bincount(bin_idx, minlength=n_bins).astype(np.float32),
            1e-8, None)
        smoothed = np.clip(
            convolve1d(counts, weights=kernel, mode="constant"), 1e-8, None)
        eff = np.clip(smoothed[bin_idx], 1e-4, None)
        w = np.clip(1.0 / eff, 0.01, 100.0)
        weights[:, k] = w / w.mean()
    return weights


class MAELoss(nn.Module):
    """MAE with optional per-sample-per-target weights."""

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ae = torch.abs(y_pred - y_true)
        if weights is not None:
            if weights.dim() == 1:
                weights = weights.unsqueeze(1)
            ae = ae * weights
        return ae.mean()


# ── Sobolev id-derivative consistency loss (V6.4.7 P4 / S10) ─────────────────
#
# Couples the autograd ∂id/∂V the SOLVER consumes (NN Rule 1) to the OSDI
# gm/gds/gmb columns the 13-head loss already supervises. The predicted gm/gds
# /gmb *heads* are accurate (~1 % NRMSE), but the autograd slope of the id
# head — the only Jacobian NR uses — drifts (S10 deriv-fidelity ref: NMOS gds
# 6-13 %, PMOS gds 20-70 %). This term supervises that slope.
#
# id-CHANNELS ONLY. The V5 Phase-C 8-channel form (id + 5 cap channels) was
# net-detrimental at S-scale (results/v5_phase_c_jac_loss_ab_2026_05_07.md);
# the chain-rule transform is recovered from `git show 930c274`, but restricted
# to the three id channels per the P4 design.
#
# SAME SPACE AS THE GATE: the comparison happens in the normalized-asinh
# derivative space the deriv-fidelity scorer measures
# (scripts/v6_4_7_deriv_fidelity.py), so the term supervises exactly the
# quantity the P4 promotion gate scores.
#
# SIGN CONVENTION (the P0-I §2 trap, settled empirically — see
# `_verify_sobolev_target` in scripts/v6_4_7_s10_sign_check.py and the scorer
# float64-FD selfcheck): the stored OSDI opvars are positive-magnitude
# conductances while the stored `id` keeps the PyCMG terminal sign, so
#     stored gm  = -d(id_stored)/dVg
#     stored gds = -d(id_stored)/dVd
#     stored gmb = -d(id_stored)/dVb
# i.e. a UNIFORM negation of all three channels maps autograd ∂id/∂V to the
# OSDI columns — for BOTH device types. This is NOT the 930c274 "gds is the
# diagonal so no flip" rule: that comment is wrong for this stored convention
# (it would compare ∂id/∂Vd against +gds, doubling the gds residual). The
# scorer's float64 central-FD selfcheck (<0.5 % median rel err) and a
# well-trained control-v2 net (autograd ∂id/∂Vd already ≈ -gds to ~7 % for
# NMOS) both confirm the uniform negation.

# (target_name, voltage-input column index) — ∂id/∂V_col vs the stored target.
SOBOLEV_ID_CHANNELS: list[tuple[str, int]] = [
    ("gm", 1),    # ∂id/∂Vg  vs gm
    ("gds", 0),   # ∂id/∂Vd  vs gds
    ("gmb", 3),   # ∂id/∂Vb  vs gmb
]


class SobolevIdLoss(nn.Module):
    """λ · mean_chan MAE(autograd ∂id/∂V, -target) in normalized-asinh space.

    DirectNet-only (asinh output norm). For each id channel the autograd
    derivative ``∂(id_norm)/∂(V_norm)`` is compared against the supervised
    gm/gds/gmb target transformed into the same normalized-derivative units
    via the asinh chain rule:

        d(id_norm)/d(V_norm) = (in_std_V / out_std_id)
            · d(id_phys)/d(V) / sqrt(s_id² + id_phys²)
        target_in_norm = -target_phys · in_std_V / out_std_id
                          / sqrt(s_id² + id_phys_pred²)

    Rows with ``|id_true| <= id_floor`` are masked out (their OSDI gm/gds/gmb
    are solve-tolerance noise with random sign — supervising slopes on noise
    is the V5 Phase-C failure mode; P4 rev-3 (iii) trust-floor mask). Rows
    with ``|id_true| > strong_floor`` may be up-weighted by ``strong_boost``
    to emphasise the conducting / opamp-gain corridor (gain ∝ gm/gds at
    strong inversion; P4 rev-2 corridor importance without P5's harvested
    classes).
    """

    def __init__(
        self,
        lam: float = 0.1,
        column_order: list[str] | None = None,
        id_floor: float = 1e-12,
        strong_boost: float = 1.0,
        strong_floor: float = 1e-6,
        corridor_only: bool = False,
    ) -> None:
        super().__init__()
        self.lam = float(lam)
        self.id_floor = float(id_floor)
        self.strong_boost = float(strong_boost)
        self.strong_floor = float(strong_floor)
        # corridor_only: hard-mask to the conducting / opamp-gain corridor
        # (|id_true| > strong_floor) instead of the noise trust-floor. Focuses
        # the term on the op-point slope rather than diluting it across the
        # mass of weak-inversion rows (the plan's predicted failure mode of a
        # globally-uniform λ).
        self.corridor_only = bool(corridor_only)
        from bsimar.data.normalize import OUTPUT_COLUMN_ORDER
        self.column_order = column_order or OUTPUT_COLUMN_ORDER
        col = {c: i for i, c in enumerate(self.column_order)}
        if "id" not in col:
            raise ValueError("SobolevIdLoss requires an 'id' output column")
        self.id_col = col["id"]
        # (target_col, voltage_input_col) for channels present in this order.
        self.channels: list[tuple[int, int]] = [
            (col[t], vidx) for (t, vidx) in SOBOLEV_ID_CHANNELS if t in col]

    def forward(
        self,
        x_norm: torch.Tensor,           # (B, in_dim) requires_grad=True
        y_pred_norm: torch.Tensor,      # (B, out_dim)
        y_true_norm: torch.Tensor,      # (B, out_dim)
        in_std: torch.Tensor,           # (in_dim,)
        out_std: torch.Tensor,          # (out_dim,)
        out_mean: torch.Tensor,         # (out_dim,)
        asinh_scale: torch.Tensor,      # (out_dim,)
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Caller must set ``x_norm.requires_grad_(True)`` before the forward
        pass that produced ``y_pred_norm``."""
        idc = self.id_col
        s_id = asinh_scale[idc]
        out_std_id = out_std[idc]

        # Predicted id physical (chain-rule denominator factor).
        u_id = y_pred_norm[:, idc] * out_std_id + out_mean[idc]
        id_phys_pred = s_id * torch.sinh(u_id)
        factor = torch.sqrt(s_id * s_id + id_phys_pred * id_phys_pred + 1e-30)

        # True id physical → trust-floor / corridor mask + strong boost.
        with torch.no_grad():
            u_id_t = y_true_norm[:, idc] * out_std_id + out_mean[idc]
            abs_id_true = torch.abs(s_id * torch.sinh(u_id_t))
            hard_floor = (self.strong_floor if self.corridor_only
                          else self.id_floor)
            row_w = (abs_id_true > hard_floor).to(y_pred_norm.dtype)
            if self.strong_boost != 1.0 and not self.corridor_only:
                row_w = row_w * torch.where(
                    abs_id_true > self.strong_floor,
                    torch.as_tensor(self.strong_boost, dtype=row_w.dtype,
                                    device=row_w.device),
                    torch.ones((), dtype=row_w.dtype, device=row_w.device))

        # One autograd.grad: all 3 channels differentiate the SAME id head,
        # and DirectNet is per-row independent, so grad of the column-sum is
        # the per-row gradient (same trick as the scorer / _MOSFETNNBase).
        g = torch.autograd.grad(
            y_pred_norm[:, idc].sum(), x_norm,
            create_graph=True, retain_graph=True)[0]   # (B, in_dim)

        norm_rows = row_w.sum().clamp_min(1.0)
        total = torch.zeros((), device=y_pred_norm.device,
                            dtype=y_pred_norm.dtype)
        n_chan = 0
        for tgt_col, vcol in self.channels:
            u_tgt = y_true_norm[:, tgt_col] * out_std[tgt_col] + out_mean[tgt_col]
            tgt_phys = asinh_scale[tgt_col] * torch.sinh(u_tgt)
            # UNIFORM negation (all three id channels) — see class docstring.
            tgt_in_norm = -tgt_phys * in_std[vcol] / out_std_id / factor
            ae = torch.abs(g[:, vcol] - tgt_in_norm) * row_w
            if weights is not None:
                w = weights[:, tgt_col] if weights.dim() == 2 else weights
                ae = ae * w
            total = total + ae.sum() / norm_rows
            n_chan += 1
        if n_chan == 0:
            return total
        return self.lam * total / n_chan


# ── Charge-derivative (cap) Sobolev consistency loss (V6.7 / charge channels) ─
#
# The TRANSIENT and AC solvers consume the small-signal capacitances as the
# AUTOGRAD derivatives of the predicted terminal charges qg/qd
# (mosfet_nn._unpack_eval / solver._stamp_cap_ac / _stamp_mosfet_transient):
#
#     cgg_sim = +∂qg/∂Vg     cgd_sim = +∂qg/∂Vd     (raw autograd, NO flip)
#     cdg_sim = +∂qd/∂Vg     cdd_sim = +∂qd/∂Vd
#
# The directly-predicted cgg..cdd OUTPUT COLUMNS are supervised in training but
# NEVER read at inference. So — exactly as SobolevIdLoss couples the autograd
# ∂id/∂V to the supervised gm/gds/gmb — this term couples the autograd ∂q/∂V to
# the supervised cap columns, the quantity the AC pole / switchcap charge / RO
# timing actually depend on. The autograd cap can drift from the (accurate)
# supervised cap column (V6.7 diag: ~1% on the grid average but up to ~25% on
# mid-trajectory corners), the cap analogue of the S10 id-slope drift.
#
# SIGN CONVENTION (V6.7 diag, confirmed empirically by
# tests/diag_charge_cap_fidelity.py and rooted in pycmg/model._condense_caps):
# OSDI stores the SPICE condensed caps with the OFF-DIAGONALS NEGATED
# (cgd_data = −∂Qg/∂Vd, cdg_data = −∂Qd/∂Vg) while the diagonals are unflipped
# (cgg_data = +∂Qg/∂Vg, cdd_data = +∂Qd/∂Vd). The autograd ∂q/∂V is the raw
# derivative, so the per-channel sign that maps autograd→OSDI is +,−,−,+:
#
#     autograd ∂qg/∂Vg  vs  +cgg      autograd ∂qg/∂Vd  vs  −cgd
#     autograd ∂qd/∂Vg  vs  −cdg      autograd ∂qd/∂Vd  vs  +cdd
#
# (cgs = ∂qg/∂Vs is degenerate — Vs≡0 in training — and the AC stamp does not
# consume it, so it is deliberately excluded.) The asinh chain rule and masking
# mirror SobolevIdLoss; the differentiated head is qg/qd (its own asinh scale).

# (target_cap, charge_head, voltage_input_col, sign)
CHARGE_SOBOLEV_CHANNELS: list[tuple[str, str, int, float]] = [
    ("cgg", "qg", 1, +1.0),   # ∂qg/∂Vg  vs  +cgg
    ("cgd", "qg", 0, -1.0),   # ∂qg/∂Vd  vs  −cgd
    ("cdg", "qd", 1, -1.0),   # ∂qd/∂Vg  vs  −cdg
    ("cdd", "qd", 0, +1.0),   # ∂qd/∂Vd  vs  +cdd
]


class ChargeSobolevLoss(nn.Module):
    """λ · mean_chan MAE(autograd ∂q/∂V, sign·cap_target) in normalized-asinh space.

    DirectNet-only (asinh output norm). For each cap channel the autograd
    derivative ``∂(q_norm)/∂(V_norm)`` of the relevant charge head is compared
    against the supervised cap column transformed into the same
    normalized-derivative units via the asinh chain rule on the CHARGE head:

        d(q_norm)/d(V_norm) = (in_std_V / out_std_q)
            · d(q_phys)/dV / sqrt(s_q² + q_phys_pred²)
        target_in_norm = sign · cap_phys · in_std_V / out_std_q
                          / sqrt(s_q² + q_phys_pred²)

    Two ``autograd.grad`` calls (qg, qd column-sums; DirectNet is per-row
    independent so the column-sum gradient is the per-row gradient). Rows whose
    cap target is below ``cap_floor`` (noise) are masked per channel. Honours
    the per-target LDS / class ``weights`` (so ``--class-weights`` can steer the
    term toward the reverse_vds / vbs corridors where the autograd cap drifts).
    """

    def __init__(
        self,
        lam: float = 0.05,
        column_order: list[str] | None = None,
        cap_floor: float = 1e-19,
    ) -> None:
        super().__init__()
        self.lam = float(lam)
        self.cap_floor = float(cap_floor)
        from bsimar.data.normalize import OUTPUT_COLUMN_ORDER
        self.column_order = column_order or OUTPUT_COLUMN_ORDER
        col = {c: i for i, c in enumerate(self.column_order)}
        for q in ("qg", "qd"):
            if q not in col:
                raise ValueError(f"ChargeSobolevLoss requires a '{q}' column")
        # Group channels by charge head so each head differentiates once.
        # head -> list of (cap_col, vcol, sign)
        self.heads: dict[str, list[tuple[int, int, float]]] = {}
        self.head_col: dict[str, int] = {}
        for cap, head, vcol, sign in CHARGE_SOBOLEV_CHANNELS:
            if cap not in col or head not in col:
                continue
            self.heads.setdefault(head, []).append((col[cap], vcol, sign))
            self.head_col[head] = col[head]

    def forward(
        self,
        x_norm: torch.Tensor,           # (B, in_dim) requires_grad=True
        y_pred_norm: torch.Tensor,      # (B, out_dim)
        y_true_norm: torch.Tensor,      # (B, out_dim)
        in_std: torch.Tensor,           # (in_dim,)
        out_std: torch.Tensor,          # (out_dim,)
        out_mean: torch.Tensor,         # (out_dim,)
        asinh_scale: torch.Tensor,      # (out_dim,)
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Caller must set ``x_norm.requires_grad_(True)`` before the forward."""
        total = torch.zeros((), device=y_pred_norm.device,
                            dtype=y_pred_norm.dtype)
        n_chan = 0
        n_heads = len(self.heads)
        for h, (head, chans) in enumerate(self.heads.items()):
            qc = self.head_col[head]
            s_q = asinh_scale[qc]
            out_std_q = out_std[qc]
            # Predicted physical charge → asinh chain-rule denominator factor.
            u_q = y_pred_norm[:, qc] * out_std_q + out_mean[qc]
            q_phys_pred = s_q * torch.sinh(u_q)
            factor = torch.sqrt(s_q * s_q + q_phys_pred * q_phys_pred + 1e-40)
            # One autograd.grad per charge head (column-sum trick).
            retain = not (h == n_heads - 1)  # last head can free the graph
            g = torch.autograd.grad(
                y_pred_norm[:, qc].sum(), x_norm,
                create_graph=True, retain_graph=True)[0]    # (B, in_dim)
            for cap_col, vcol, sign in chans:
                u_t = (y_true_norm[:, cap_col] * out_std[cap_col]
                       + out_mean[cap_col])
                cap_phys = asinh_scale[cap_col] * torch.sinh(u_t)
                with torch.no_grad():
                    row_w = (cap_phys.abs() > self.cap_floor).to(
                        y_pred_norm.dtype)
                tgt_in_norm = (sign * cap_phys * in_std[vcol]
                               / out_std_q / factor)
                ae = torch.abs(g[:, vcol] - tgt_in_norm) * row_w
                if weights is not None:
                    w = (weights[:, cap_col] if weights.dim() == 2 else weights)
                    ae = ae * w
                total = total + ae.sum() / row_w.sum().clamp_min(1.0)
                n_chan += 1
        if n_chan == 0:
            return total
        return self.lam * total / n_chan


# ── Subthreshold id value loss (V6.4.7 P3 / S11) ────────────────────────────
#
# Target: the SRAM force_ic inboard attractor (0/8). At the stuck fixed point
# the pinning NMOS over-predicts its weak-inversion current ~7.5x (NN 6.36 vs
# OSDI 0.84 uA at Vov ~ +45 mV) and the hard-OFF PMOS predicts +0.50 uA where
# OSDI is ~0 (P0-D, results/v6_4_6/phase0_D_sram_attractor.md). The released
# cell therefore cannot suppress the OFF branch enough to rail.
#
# WHY THE STANDARD LOSS MISSES IT: the global asinh scale s_id ~ 2.6e-5 maps
# the whole sub-uA roll-off to ~0.01 % of normalized range — a 1 nA error is
# asinh(1e-9/2.6e-5) ~ 4e-5, ~zero loss mass. The regen-v2 data HAS the rows
# (S9b: ~15 % of each cell below 1 uA, ~6.7 % below 1 nA) but the asinh+LDS
# transform gives them no signal. This term re-scales the subthreshold band
# with a SMALL asinh scale s2 ~ 1e-9 so each weak-inversion decade carries
# unit-order loss (asinh ~ log above s2 ⇒ decade-balanced by construction).
#
# TWO BANDS (plan P3 Stage-1 (i),(ii),(vi)):
#   1. VALUE (Huber, sign-aware): id_floor < |id_true| < upper — teaches the
#      true sub-uA value, directly suppressing the 7.5x pinning over-prediction.
#      asinh preserves sign ⇒ reverse_vds rows with genuine negative id are
#      supervised correctly (plan (iii) needs no special-case).
#   2. CEILING HINGE (sign-AGNOSTIC): |id_true| <= off_floor — penalize only
#      relu(|id_pred| above k*NFIN*1nA). Suppresses the hard-OFF over-prediction
#      WITHOUT ever injecting current (never a floor — NOT the D4 Ioff_rail
#      dead end). Sign-agnostic ⇒ the random-sign sub-floor OSDI noise (the
#      v5 §4-B4 filter rationale) is irrelevant, and a reverse_vds row that
#      genuinely conducts has |id_true| > off_floor so it is auto-exempt.
#
# The term is ADDED to the MAE+LDS loss (does not replace it) so strong
# inversion / the ~20x VTC trip gain band is untouched (upper mask = 1e-6).


class SubthresholdIdLoss(nn.Module):
    """λ · (Huber value term + ceiling_w · ceiling hinge) in asinh(id/s2) space.

    DirectNet-only (asinh output norm). ``s2`` (default 1e-9) sets the
    subthreshold resolution; ``upper`` (1e-6) caps the value band below the
    trip gain; ``id_floor`` (data trust floor, 1e-12 post-regen) masks the
    value MAE; ``off_floor`` (1e-10) selects the hard-OFF rows the ceiling
    hinge suppresses; ``ceiling_k`` sets the per-fin OFF ceiling k·NFIN·1nA.
    """

    def __init__(
        self,
        lam: float = 0.05,
        column_order: list[str] | None = None,
        s2: float = 1e-9,
        upper: float = 1e-6,
        id_floor: float = 1e-12,
        off_floor: float = 1e-10,
        ceiling_k: float = 1.0,
        ceiling_w: float = 1.0,
        huber_delta: float = 1.0,
        nfin_col: int = 4,
    ) -> None:
        super().__init__()
        self.lam = float(lam)
        self.s2 = float(s2)
        self.upper = float(upper)
        self.id_floor = float(id_floor)
        self.off_floor = float(off_floor)
        self.ceiling_k = float(ceiling_k)
        self.ceiling_w = float(ceiling_w)
        self.huber_delta = float(huber_delta)
        self.nfin_col = int(nfin_col)
        from bsimar.data.normalize import OUTPUT_COLUMN_ORDER
        self.column_order = column_order or OUTPUT_COLUMN_ORDER
        col = {c: i for i, c in enumerate(self.column_order)}
        if "id" not in col:
            raise ValueError("SubthresholdIdLoss requires an 'id' output column")
        self.id_col = col["id"]

    @staticmethod
    def _huber(r: torch.Tensor, delta: float) -> torch.Tensor:
        a = r.abs()
        return torch.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))

    def forward(
        self,
        x_norm: torch.Tensor,           # (B, in_dim)
        y_pred_norm: torch.Tensor,      # (B, out_dim)
        y_true_norm: torch.Tensor,      # (B, out_dim)
        in_mean: torch.Tensor,          # (in_dim,)
        in_std: torch.Tensor,           # (in_dim,)
        out_std: torch.Tensor,          # (out_dim,)
        out_mean: torch.Tensor,         # (out_dim,)
        asinh_scale: torch.Tensor,      # (out_dim,)
    ) -> torch.Tensor:
        idc = self.id_col
        s_id = asinh_scale[idc]
        out_std_id = out_std[idc]
        s2 = self.s2

        # Predicted physical id (clamp the asinh argument for sinh safety).
        u_pred = (y_pred_norm[:, idc] * out_std_id + out_mean[idc]).clamp(-30.0, 30.0)
        id_phys_pred = s_id * torch.sinh(u_pred)

        with torch.no_grad():
            u_true = (y_true_norm[:, idc] * out_std_id + out_mean[idc]).clamp(-30.0, 30.0)
            id_phys_true = s_id * torch.sinh(u_true)
            abs_id_true = id_phys_true.abs()
            value_mask = ((abs_id_true > self.id_floor)
                          & (abs_id_true < self.upper)).to(y_pred_norm.dtype)
            off_mask = (abs_id_true <= self.off_floor).to(y_pred_norm.dtype)
            # Per-fin OFF ceiling: recover physical NFIN from the normalized
            # log2(NFIN) input column. NFIN never carries gradient here.
            nfin_log2 = (x_norm[:, self.nfin_col] * in_std[self.nfin_col]
                         + in_mean[self.nfin_col])
            nfin = torch.pow(torch.as_tensor(2.0, device=x_norm.device,
                                             dtype=x_norm.dtype), nfin_log2)
            ceiling = (self.ceiling_k * nfin * 1e-9).clamp_min(1e-30)
            a_true = torch.asinh(id_phys_true / s2)
            a_ceil = torch.asinh(ceiling / s2)

        # 1. VALUE term (sign-aware, Huber in asinh-s2 space).
        a_pred = torch.asinh(id_phys_pred / s2)
        val = self._huber(a_pred - a_true, self.huber_delta) * value_mask
        val_term = val.sum() / value_mask.sum().clamp_min(1.0)

        # 2. CEILING hinge (sign-agnostic magnitude suppression).
        a_absp = torch.asinh(id_phys_pred.abs() / s2)
        hinge = torch.relu(a_absp - a_ceil) * off_mask
        ceil_term = hinge.sum() / off_mask.sum().clamp_min(1.0)

        return self.lam * (val_term + self.ceiling_w * ceil_term)
