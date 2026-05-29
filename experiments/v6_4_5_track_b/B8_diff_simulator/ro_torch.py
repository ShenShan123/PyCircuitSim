"""B8 — Differentiable 5-stage ring-oscillator simulator (torch).

Track-B Tier-3 lever for the TSMC7 ring-oscillator gate (V6.4.5).

Thesis: the RO period is an *integrated* metric that pointwise Id-MAE
training cannot see. This module builds a fully differentiable transient
simulation of the actual TSMC7 5-stage ring oscillator with the DirectNet
model weights in the loop, so test-time fine-tuning (TTFT) can minimise the
period error *directly*.

Design — kept SELF-CONTAINED; the production scipy solver is never touched.

The differentiable transient replicates the PRODUCTION transient exactly at
the *fixed point* (the only thing that determines the waveform):

  * DirectNet forward → id and the four terminal charges (qg, qd, qs, qb),
    differentiable w.r.t. the terminal node voltages AND the model weights.
  * PMOS source-relative frame; softplus voltage clamp + z-score input prep;
    asinh output denorm — all bit-for-bit the same maths as
    ``_MOSFETNNBase`` / ``normalize.py``.
  * The inference-time Vds correction (Rule 15 parts a/b/d) on ``id``.
    (Part c only shapes ``gds`` — the Jacobian — which does not move the
    converged node voltages, so it is irrelevant to the waveform and is
    omitted here.)
  * NMOS ``id ≤ 0`` / PMOS ``id ≥ 0`` sign enforcement (smooth, so the
    autograd path to the weights survives).
  * Charge-based companion integration: Backward-Euler on step 1,
    Trapezoidal on step 2+ (the production schedule). KCL per node:

        Σ transistor terminal currents (leaving node)
        + capacitor charge-rate currents  =  0.

  * Each output timestep is solved with a few Newton iterations done in
    torch (dense 5×5 system, autograd Jacobian, unrolled → differentiable).

The node order is n1..n5 with vdd held at 0.75 V and a 0.5 fF lumped load on
every node. The ring feeds back n5 → n1. .ic = [0, VDD, 0, VDD, 0].

The intrinsic device charges already provide the dynamic storage; the
explicit 0.5 fF load caps are added on top, exactly as in the netlist.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[3]
if str(PROJECT_ROOT / "external_compact_models") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from bsimar.data.normalize import NormStats, OUTPUT_COLUMN_ORDER  # noqa: E402
from bsimar.models.direct_net import DirectNet  # noqa: E402

_OC = {n: i for i, n in enumerate(OUTPUT_COLUMN_ORDER)}


# ---------------------------------------------------------------------------
# A torch DirectNet device — weights trainable, forward differentiable
# ---------------------------------------------------------------------------
class TorchDirectNetDevice:
    """Differentiable DirectNet MOSFET (id + charges) in torch.

    Mirrors ``_MOSFETNNBase`` voltage-prep / denorm / Rule-15 maths, but
    keeps the whole forward as a torch graph so gradients flow to the model
    weights. NFIN / L / T geometry is constant per device.
    """

    def __init__(
        self,
        model: DirectNet,
        stats: NormStats,
        *,
        is_pmos: bool,
        L: float,
        NFIN: float,
        tech_code: int,
        temperature: float = 300.15,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.model = model
        self.stats = stats
        self.is_pmos = is_pmos
        self.dev = device
        self.dtype = dtype
        self.tech_code = torch.tensor([tech_code], dtype=torch.long, device=device)

        # — geometry (constant) normalised exactly like _MOSFETNNBase —
        nfin_log = float(np.log2(max(NFIN, 1.0)))
        geo_raw = np.array([nfin_log, float(L), float(temperature)], dtype=np.float64)
        geo_std = stats.input_std[4:7].copy()
        geo_std[geo_std < 1e-12] = 1.0
        geo_norm = (geo_raw - stats.input_mean[4:7]) / geo_std
        self.geo_norm = torch.tensor(geo_norm, dtype=dtype, device=device)

        # — voltage prep params —
        v_std = stats.input_std[:4].copy()
        v_std[v_std < 1e-12] = 1.0
        self.v_mean = torch.tensor(stats.input_mean[:4], dtype=dtype, device=device)
        self.v_std = torch.tensor(v_std, dtype=dtype, device=device)
        self.v_min = torch.tensor(stats.input_min[:4], dtype=dtype, device=device)
        self.v_max = torch.tensor(stats.input_max[:4], dtype=dtype, device=device)
        v_range = torch.clamp(self.v_max - self.v_min, min=0.01)
        self.clamp_beta = 1.0 / (0.05 * v_range)

        # — output denorm params (asinh or zscore) —
        self.mode = stats.mode
        cols = stats.output_columns or OUTPUT_COLUMN_ORDER
        self.col_idx = {n: cols.index(n) for n in cols}
        self.out_mean = torch.tensor(stats.output_mean, dtype=dtype, device=device)
        self.out_std = torch.tensor(stats.output_std, dtype=dtype, device=device)
        if stats.asinh_scale is not None:
            self.asinh_scale = torch.tensor(
                stats.asinh_scale, dtype=dtype, device=device)
        else:
            self.asinh_scale = None

        # — VDD estimate for Rule-15 —
        vd_range = max(abs(float(stats.input_max[0])), abs(float(stats.input_min[0])))
        self.vdd_train = vd_range / 2.0

    # -- voltage prep (matches _clamp_norm_voltages, _raw_voltages) --
    def _raw_voltages(self, vd, vg, vs, vb):
        if self.is_pmos:
            return vd - vs, vg - vs, vs * 0.0, vb - vs
        return vd, vg, vs, vb

    def _clamp_norm(self, v_raw: torch.Tensor) -> torch.Tensor:
        beta = self.clamp_beta
        bx_lo = beta * (v_raw - self.v_min)
        v = self.v_min + torch.where(
            bx_lo > 20.0, v_raw - self.v_min,
            torch.log1p(torch.exp(bx_lo)) / beta)
        bx_hi = beta * (self.v_max - v)
        v = self.v_max - torch.where(
            bx_hi > 20.0, self.v_max - v,
            torch.log1p(torch.exp(bx_hi)) / beta)
        return (v - self.v_mean) / self.v_std

    def _denorm_col(self, out_row: torch.Tensor, name: str) -> torch.Tensor:
        i = self.col_idx[name]
        u = out_row[..., i] * self.out_std[i] + self.out_mean[i]
        if self.mode == "asinh":
            return self.asinh_scale[i] * torch.sinh(u)
        return u

    def forward(
        self, vd: torch.Tensor, vg: torch.Tensor, vs: torch.Tensor, vb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (id, qg, qd, qs, qb) — physical, differentiable.

        Scalar-per-call vd/vg/vs/vb (0-dim tensors). Returns 0-dim tensors.
        ``id`` carries the Rule-15 Vds correction + sign enforcement; charges
        carry charge conservation (qs = -(qg+qd+qb)).
        """
        vd_nn, vg_nn, vs_nn, vb_nn = self._raw_voltages(vd, vg, vs, vb)
        v_raw = torch.stack([vd_nn, vg_nn, vs_nn, vb_nn])
        v_norm = self._clamp_norm(v_raw)
        x = torch.cat([v_norm, self.geo_norm]).unsqueeze(0)
        out = self.model(x, tech_codes=self.tech_code)[0]

        id_phys = self._denorm_col(out, "id")
        qg = self._denorm_col(out, "qg")
        qd = self._denorm_col(out, "qd")
        qb = self._denorm_col(out, "qb")
        qs = -(qg + qd + qb)

        vds = vd_nn - vs_nn
        id_phys = self._vds_correct(id_phys, vds)
        return id_phys, qg, qd, qs, qb

    def _vds_correct(self, id_raw: torch.Tensor, vds: torch.Tensor) -> torch.Tensor:
        """Rule-15 parts (a), (b), (d) on ``id`` — differentiable.

        (a) rail-restoring extrapolation past |Vds| > VDD_train,
        (b) one-sided (1-exp(-|Vds|/VT)) factor in the normal direction,
        (d) wrong-sign clamp (smooth so gradients survive).

        Part (c) only touches gds (Jacobian) → no effect on the fixed point,
        omitted. Implemented with smooth masks (sigmoid) so the period
        gradient w.r.t. the weights is well-defined.
        """
        VDD = self.vdd_train
        VT = max(0.06 * VDD, 0.026)
        abs_vds = torch.abs(vds)
        # normal_dir: NMOS vds>0, PMOS vds<0
        if self.is_pmos:
            normal = (vds < 0.0)
        else:
            normal = (vds > 0.0)
        normal_f = normal.to(self.dtype)

        idv = id_raw

        # (a) rail-restoring extrapolation (only the id part; matches the
        # production sign: NMOS id -= id_extra, PMOS id += id_extra, only in
        # the normal direction).
        over = abs_vds - VDD
        g_max = 1.0e-3
        x_ref = 0.5 * VDD
        x_cap = 5.0 * x_ref
        over_pos = torch.clamp(over, min=0.0)
        # quadratic then linear past x_cap
        quad = 0.5 * g_max * over_pos * over_pos / x_ref
        id_at_cap = 0.5 * g_max * x_cap * x_cap / x_ref
        g_at_cap = g_max * x_cap / x_ref
        lin = id_at_cap + g_at_cap * (over_pos - x_cap)
        id_extra = torch.where(over_pos <= x_cap, quad, lin)
        id_extra = id_extra * (over > 0.0).to(self.dtype) * normal_f
        if self.is_pmos:
            idv = idv + id_extra
        else:
            idv = idv - id_extra

        # (b) one-sided turn-on factor.
        # fast path in production returns early when normal & abs_vds>20VT
        # (f_id == 1 there); the formula below already gives f_id≈1, so a
        # single expression is correct.
        exp_sym = torch.exp(-torch.clamp(abs_vds, max=20.0 * VT) / VT)
        exp_sym = torch.where(abs_vds <= 20.0 * VT, exp_sym,
                              torch.zeros_like(exp_sym))
        f_sym = 1.0 - exp_sym
        f_id = f_sym * normal_f  # 0 in reverse direction

        # Apply (b) only outside the fast-path region; inside fast-path
        # f_id≈1 so multiplying is a no-op. Keep it simple: always multiply.
        idv = idv * f_id

        # (d) wrong-sign clamp (smooth). NMOS wants id<=0, PMOS id>=0.
        if self.is_pmos:
            # zero out negative id
            idv = torch.nn.functional.relu(idv)
        else:
            idv = -torch.nn.functional.relu(-idv)
        return idv


# ---------------------------------------------------------------------------
# The 5-stage ring oscillator, fully in torch
# ---------------------------------------------------------------------------
class RingOscTorch:
    """Differentiable 5-stage CMOS ring oscillator (TSMC7 DirectNet)."""

    def __init__(
        self,
        nmos: TorchDirectNetDevice,
        pmos: TorchDirectNetDevice,
        *,
        vdd: float = 0.75,
        cload: float = 0.5e-15,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.nmos = nmos
        self.pmos = pmos
        self.vdd = vdd
        self.cload = cload
        self.dev = device
        self.dtype = dtype
        self.n_stages = 5

    def _stage_currents_charges(
        self, vn: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """For node voltages vn (5,), return (i_node, q_node) tensors (5,).

        i_node[k] = net current LEAVING node k through the two transistors of
        the inverter driving node k (terminal-drain currents), and
        q_node[k] = total charge stored at node k from the drain terminals of
        the two transistors driving it. Source/bulk of each transistor sit on
        rails (vdd or gnd), so only their drain terminals couple to the ring
        nodes; the gate of stage k is node k-1 (an input, no DC gate current,
        and its gate charge is sourced from the *previous* node — handled via
        that previous node's charge bookkeeping below).

        We track per-node the sum of: drain charge of the local inverter pair
        + gate charge contributions of the *next* stage (whose gate is this
        node). This matches the production node-charge KCL where every
        terminal connected to a node contributes its terminal current.
        """
        vdd_t = torch.tensor(self.vdd, dtype=self.dtype, device=self.dev)
        zero = torch.zeros((), dtype=self.dtype, device=self.dev)

        # First pass: per-stage device evals. Stage k drives node k; its gate
        # is node k-1 (mod 5).
        id_n = [None] * self.n_stages
        qd_n = [None] * self.n_stages
        qd_p = [None] * self.n_stages
        id_p = [None] * self.n_stages
        qg_n = [None] * self.n_stages
        qg_p = [None] * self.n_stages
        for k in range(self.n_stages):
            vout = vn[k]
            vin = vn[(k - 1) % self.n_stages]
            # NMOS: drain=vout, gate=vin, source=0, bulk=0
            idn, qgn, qdn, qsn, qbn = self.nmos.forward(vout, vin, zero, zero)
            # PMOS: drain=vout, gate=vin, source=vdd, bulk=vdd
            idp, qgp, qdp, qsp, qbp = self.pmos.forward(vout, vin, vdd_t, vdd_t)
            id_n[k] = idn
            id_p[k] = idp
            qd_n[k] = qdn
            qd_p[k] = qdp
            qg_n[k] = qgn
            qg_p[k] = qgp

        i_node = []
        q_node = []
        for k in range(self.n_stages):
            # Current leaving node k (drain node), production frame
            # (_stamp_mosfet_dc): i_leaving = -calculate_current for PMOS,
            # +calculate_current for NMOS. With forward()→id_phys (PyCMG sign,
            # NMOS id<0 / PMOS id>0):
            #   NMOS calculate_current = -id_phys → i_leaving = +(-id_phys)
            #                                              = -id_n
            #   PMOS calculate_current = +id_phys → i_leaving = -(+id_phys)
            #                                              = -id_p
            i_leave_n = -id_n[k]   # NMOS contribution leaving drain
            i_leave_p = -id_p[k]   # PMOS contribution leaving drain
            i_node.append(i_leave_n + i_leave_p)

            # charge at node k: drain charge of the local pair PLUS gate charge
            # of the next stage (whose gate is node k).
            nxt = (k + 1) % self.n_stages
            q_k = qd_n[k] + qd_p[k] + qg_n[nxt] + qg_p[nxt]
            q_node.append(q_k)

        return torch.stack(i_node), torch.stack(q_node)

    def _node_residual(
        self,
        vn: torch.Tensor,
        q_hist: torch.Tensor,
        coeff: float,
        i_hist: torch.Tensor,
        cload: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """KCL residual at all 5 nodes for the charge-based companion model.

        At the solution: i_dc(node) + i_cap(node) = 0, where
          i_cap(node) = coeff*Q(node) - h(node),
          h(node)     = coeff*Q_prev(node) + i_prev(node)   (Trap)
                      = coeff*Q_prev(node)                   (BE, i_hist=0).
        The lumped load cap adds C*coeff*(V - V_prev_companion) too; we fold
        the load cap into the per-node charge as Q_load = C*V (linear), so the
        same companion expression covers it. i.e. Q_total = Q_dev + C*V.

        Returns (residual(5,), i_dc(5,), q_total(5,)).
        """
        i_dc, q_dev = self._stage_currents_charges(vn)
        q_total = q_dev + cload * vn
        i_cap = coeff * q_total - (coeff * q_hist + i_hist)
        # KCL: sum of currents leaving node = 0  →  i_dc + i_cap = 0
        resid = i_dc + i_cap
        return resid, i_dc, q_total

    def _jacobian(
        self, v_req: torch.Tensor, q_hist: torch.Tensor, coeff: float,
        i_hist: torch.Tensor, cload: float, *, create_graph: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Residual + dense 5×5 Jacobian via autograd at ``v_req``."""
        resid, _, _ = self._node_residual(v_req, q_hist, coeff, i_hist, cload)
        rows = []
        for r in range(5):
            g = torch.autograd.grad(
                resid[r], v_req, retain_graph=True,
                create_graph=create_graph)[0]
            rows.append(g)
        J = torch.stack(rows)
        return resid, J

    def step(
        self,
        vn: torch.Tensor,
        q_hist: torch.Tensor,
        coeff: float,
        i_hist: torch.Tensor,
        *,
        n_newton: int = 6,
        damping: float = 1.0,
        keep_graph: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One companion-model timestep solved by unrolled Newton.

        Returns (vn_new, q_total_new, i_cap_new). When ``keep_graph`` the
        final implicit-function refinement keeps the autograd graph so
        gradients flow to the model weights through the converged voltages.
        The inner Newton iterations always run under ``enable_grad`` (the
        local Jacobian needs ∂resid/∂V), but each iterate is detached so the
        graph does not blow up.
        """
        cload = self.cload
        v = vn.detach()
        # Residual-accepted, best-iterate, adaptively-damped Newton. The NN
        # id surface has softplus/Rule-15 kinks → the Jacobian is locally
        # inexact and a full Newton step oscillates around the solution.
        # Track the best (lowest |residual|_inf) iterate and damp on stall;
        # the cap-dominated system (G_eq=coeff≈1e12) makes the true solution
        # very well-conditioned, so a damped Newton homes in cleanly.
        best_v = v
        best_r = float("inf")
        lam = 1.0  # damping
        with torch.enable_grad():
            for _ in range(n_newton):
                v_req = v.detach().requires_grad_(True)
                resid, J = self._jacobian(
                    v_req, q_hist, coeff, i_hist, cload, create_graph=False)
                r_val = resid.detach()
                r_inf = float(torch.max(torch.abs(r_val)))
                if r_inf < best_r:
                    best_r = r_inf
                    best_v = v.detach()
                if r_inf < 1e-12:
                    break
                try:
                    dv = torch.linalg.solve(J.detach(), -r_val)
                except Exception:  # noqa: BLE001
                    dv = torch.linalg.lstsq(
                        J.detach(), -r_val.unsqueeze(-1)).solution.squeeze(-1)
                dv = torch.clamp(dv, min=-self.vdd, max=self.vdd)
                # Damped line search: take lam·dv; if residual worsens, halve
                # lam (up to a few tries) so the iterate cannot run away.
                accepted = False
                for _bt in range(6):
                    v_try = (v + lam * dv).detach()
                    r_try, _, _ = self._node_residual(
                        v_try, q_hist, coeff, i_hist, cload)
                    r_try_inf = float(torch.max(torch.abs(r_try.detach())))
                    if r_try_inf < r_inf or _bt == 5:
                        v = v_try
                        if r_try_inf < r_inf:
                            lam = min(1.0, lam * 1.5)
                        accepted = True
                        break
                    lam *= 0.5
                if not accepted:
                    break
        v = best_v

        if not keep_graph:
            with torch.no_grad():
                _, q_dev = self._stage_currents_charges(v)
                q_total_new = q_dev + cload * v
                i_cap_new = coeff * q_total_new - (coeff * q_hist + i_hist)
            return v.detach(), q_total_new.detach(), i_cap_new.detach()

        # Final implicit-function refinement keeping the graph:
        #   v* = v_conv - J^{-1} resid     (one Newton step, differentiable).
        # q_hist / i_hist carry the graph from prior steps; v_conv is detached
        # so only the model-weight dependence (through resid and J) survives,
        # which is exactly the converged-fixed-point sensitivity we want.
        v_in = v.detach().requires_grad_(True)
        with torch.enable_grad():
            resid, J = self._jacobian(
                v_in, q_hist, coeff, i_hist, cload, create_graph=True)
            dv = torch.linalg.solve(J, -resid)
            v_star = v_in + dv
            i_dc, q_dev = self._stage_currents_charges(v_star)
            q_total_new = q_dev + cload * v_star
            i_cap_new = coeff * q_total_new - (coeff * q_hist + i_hist)
        return v_star, q_total_new, i_cap_new

    def simulate(
        self,
        *,
        tstep: float = 2e-12,
        tstop: float = 1.2e-9,
        n_newton: int = 6,
        keep_graph_from: float = 0.0,
        probe_node: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the transient. Returns (time(T,), v_probe(T,)).

        ``probe_node`` is the 0-indexed node to report (4 → n5, matching the
        harness' v(n5)). ``keep_graph_from``: only keep the autograd graph for
        timesteps at/after this time (saves memory; the period/waveform loss
        is measured post-settle, so early steps can run no-grad).
        """
        device, dtype = self.dev, self.dtype
        vdd = self.vdd
        # .ic = [0, VDD, 0, VDD, 0]
        v = torch.tensor([0.0, vdd, 0.0, vdd, 0.0], dtype=dtype, device=device)

        # init charge history at the .ic point
        with torch.no_grad():
            _, q0 = self._stage_currents_charges(v)
            q_hist = (q0 + self.cload * v).detach()
        i_hist = torch.zeros(5, dtype=dtype, device=device)

        n_steps = int(np.ceil(tstop / tstep)) + 1
        times = [0.0]
        vprobe = [v[probe_node]]
        for step_i in range(1, n_steps):
            t = step_i * tstep
            # BE on step 1, Trap thereafter (production schedule).
            if step_i == 1:
                coeff = 1.0 / tstep
                i_hist_use = torch.zeros(5, dtype=dtype, device=device)
            else:
                coeff = 2.0 / tstep
                i_hist_use = i_hist

            grad_on = (t >= keep_graph_from)
            v, q_total, i_cap = self.step(
                v, q_hist, coeff, i_hist_use, n_newton=n_newton,
                keep_graph=grad_on)

            # update history for next step (Trap needs i_prev = i_cap)
            q_hist = q_total
            i_hist = i_cap
            times.append(t)
            vprobe.append(v[probe_node])

        t_arr = torch.tensor(times, dtype=dtype, device=device)
        v_arr = torch.stack(vprobe)
        return t_arr, v_arr


# ---------------------------------------------------------------------------
# Differentiable period measurement (rising-edge midpoint crossings)
# ---------------------------------------------------------------------------
def soft_period(
    t: torch.Tensor, v: torch.Tensor, mid: float, settle: float,
) -> torch.Tensor:
    """Differentiable period from rising-edge crossings of ``mid``.

    Mirrors ``_period_from_wave``: linear-interpolate each rising crossing
    time, return mean of the inter-crossing gaps. Differentiable through the
    linear-interpolation fraction (the crossing *indices* are detached, but
    the fractional crossing time depends smoothly on v).
    """
    keep = t >= settle
    tk = t[keep]
    vk = v[keep]
    sign = torch.sign(vk - mid)
    s0 = sign[:-1]
    s1 = sign[1:]
    cross = ((s0 < 0) & (s1 >= 0)).nonzero(as_tuple=True)[0]
    if cross.numel() < 3:
        return torch.tensor(float("nan"), dtype=v.dtype, device=v.device)
    cross_times = []
    for i in cross.tolist():
        v0 = vk[i]
        v1 = vk[i + 1]
        denom = (v1 - v0)
        frac = torch.where(denom != 0, (mid - v0) / denom,
                           torch.zeros_like(denom))
        cross_times.append(tk[i] + frac * (tk[i + 1] - tk[i]))
    ct = torch.stack(cross_times)
    diffs = ct[1:] - ct[:-1]
    return diffs.mean()
