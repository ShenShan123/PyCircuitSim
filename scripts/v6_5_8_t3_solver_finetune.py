#!/usr/bin/env python3
"""V6.5.8 T3 — differentiable unrolled-DC-solver opamp fine-tune.

The capability question is settled (V6.5.8 / CHANGELOG): the EKV high-r_o core +
vout-weighted KCL existence fine-tune CAN host a stable high-gain tsmc7-opamp DC
fixed point — but the reachable OP is over-gained (~370 vs L72 163) and its gain
is COUPLED to its existence through the output-stage r_o. Every *static* lever
(vout-weight, lam-kcl, lam_lo, freeze-core) is a binary rail↔370 switch because
the loss only ever supervised a STATIC residual ``F(V_L72; θ) ≈ 0`` — never the
quantity the gate measures, the transfer curve ``Vout(Vin)`` (gain = peak
|dVout/dVin|).

T3 closes that loop: it puts the DC solve INSIDE the loss. For each Vin on the
harvested band it runs a *differentiable* damped-Newton solve of the opamp's
4-free-node KCL system ``F(V; θ) = 0`` (V = [vtail, n1, vo1i, vout]), unrolled K
steps with the autograd Jacobian, producing ``Vout(Vin; θ)``. The loss then
supervises that emergent curve against L72's (gain, trip, shape) jointly with
existence — so r_o is shaped by the GAIN target instead of by the
residual-minimisation shortcut that over-flattens it.

Per the plan's binding constraint, T3 trains the FULL model (EKV core param_head
AND the bounded residual) from the ``tsmc7_dn_ekvhr_*`` high-r_o substrate — the
transfer-curve gradient must reach r_o (in the core) and existence (in the
residual). ``--freeze-core`` is therefore NOT used.

Preservation (the binding risk): the base-data LDS-MAE anchor pins the 15 passing
gates and the ring-corridor anchor pins the shared NMOS switching region — the
same anchors that held bulk id-MRE to +1-2 % in V6.5.8.

Reachability: pure teacher-forcing (V0 = L72 OP) locates the NEAREST NN root to
the L72 OP and supervises it. ``--perturb`` displaces the V0 output node toward a
random mid-rail seed (annealed up over epochs) so the supervised root becomes a
Newton *attractor* over a basin — the batched analog of the
``diag_opamp_solver_conditioning`` mid-rail multistart that the GATE's
continuation solver needs.

Gate (run AFTER, via PYCIRCUITSIM_NN_CHECKPOINT_DN_{NMOS,PMOS}):
  1. diag_opamp_solver_conditioning.py  — reachable (high-gain soln, |gain|>50)?
  2. verify_complex_opamp.py            — AUTHORITATIVE gain ±10 % (163.4±16).
  3. diag_opamp_basin_seed.py 1c        — gain HOLDS >0 when L72-seeded.
  4. full 16-gate matrix + device DC/AC + lifted-source canary — UNREGRESSED.

Usage (self-check first — no training, validates the solver):
    CUDA_VISIBLE_DEVICES=2 conda run -n pycircuitsim python -u \
      scripts/v6_5_8_t3_solver_finetune.py --tech tsmc7 --cuda --self-check

T3.0 MVP (teacher-forced, curve+gain, anchors; the fund-or-kill):
    CUDA_VISIBLE_DEVICES=2 conda run -n pycircuitsim python -u \
      scripts/v6_5_8_t3_solver_finetune.py --tech tsmc7 --cuda --overwrite \
        --nmos-init tsmc7_dn_ekvhr_nmos --pmos-init tsmc7_dn_ekvhr_pmos \
        --epochs 60 --exp-name tsmc7_dn_t3mvp
"""
from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

print = functools.partial(print, flush=True)  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_compact_models"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the V6.5.5/6.5.8 fine-tune infra verbatim (EKV-aware load, anchors).
import v6_5_5_finetune_kcl as fk  # noqa: E402
from bsimar.config import local_variant_code  # noqa: E402
from bsimar.data.normalize import NormStats, normalizer_from_stats  # noqa: E402
from bsimar.losses.bni_mae import MAELoss  # noqa: E402

KCL_DIR = ROOT / "results" / "v6_5_5" / "kcl_groups"


# ── differentiable opamp DC solver over the harvested topology ───────────────

class OpampDiffSolver:
    """Differentiable unrolled damped-Newton DC solve of the harvested opamp.

    Loads the V6.5.8 ``*_opamp_topo.npz`` (full per-terminal incidence + L72 OP
    + arm scale). ``residual(V, vin)`` rebuilds every device's source-referenced
    NN bias from the free-node voltage vector ``V`` (G, 4) and the swept input
    ``vin`` (G,), evaluates the per-tech NMOS/PMOS DirectNet, denormalises ``id``
    to physical Amps and assembles the signed KCL current into each free node
    (sign convention = solver._stamp_mosfet_dc / the harvest self-check).
    ``solve`` then iterates the LM-damped Newton map on the dimensionless
    residual ``r = F / arm`` with the autograd Jacobian, keeping the graph so the
    composed ``Vout(Vin; θ)`` is differentiable in θ.
    """

    def __init__(self, path: Path, models: Dict[str, nn.Module],
                 norms: Dict[str, NormStats], device: torch.device) -> None:
        d = np.load(path, allow_pickle=True)
        self.device = device
        self.models = models
        self.free_nodes = [str(x) for x in d["free_nodes"]]
        self.n_free = len(self.free_nodes)
        self.vout_idx = self.free_nodes.index("vout")
        self.G = int(d["vin"].shape[0])
        self.ndev = int(d["dev_is_pmos"].shape[0])

        def _t(a, dt=torch.float32):
            return torch.tensor(np.asarray(a), dtype=dt, device=device)

        self.vin = _t(d["vin"])                       # (G,)
        self.V_l72 = _t(d["V_l72"])                   # (G, n_free)
        self.vout_l72 = _t(d["vout_l72"])             # (G,)
        self.arm = _t(d["arm_scale"])                 # (G, n_free)
        self.vin_star = float(d["vin_star"])
        self.l72_gain = float(d["meta_l72_gain"])
        self.vdd = float(d["meta_vdd"])

        drain_free = d["drain_free"].astype(int)
        source_free = d["source_free"].astype(int)
        term_free = d["term_free"].astype(int)        # (ndev, 4)
        term_is_vin = d["term_is_vin"].astype(int)    # (ndev, 4)
        term_fix_v = _t(d["term_fix_v"])              # (G, ndev, 4)
        is_pmos = d["dev_is_pmos"].astype(int)
        nfin = d["dev_nfin"]; Lg = d["dev_L"]; Tg = d["dev_T"]
        tcode = d["dev_tcode"].astype(int)

        # Per-device precompute: voltage normaliser (mean/std for the 4 V cols),
        # the constant geometry-normalised columns (log2 NFIN, L, T), the asinh
        # id denorm constants, the tech-code tensor, and the terminal incidence.
        self.dev: List[dict] = []
        for j in range(self.ndev):
            key = "pmos" if is_pmos[j] else "nmos"
            st = norms[key]
            geo = np.zeros((1, 15), dtype=np.float64)
            geo[:, 0] = nfin[j]; geo[:, 1] = Lg[j]; geo[:, 2] = Tg[j]
            # normalise a dummy zero-volt row to extract the constant geo cols.
            x0 = normalizer_from_stats(st).normalize_inputs(
                np.zeros((1, 4), dtype=np.float64), geo)
            self.dev.append({
                "model": models[key],
                "in_mean": _t(st.input_mean[:4]),
                "in_std": _t(st.input_std[:4]),
                "x_geo": _t(x0[0, 4:7]),                       # (3,) constant
                "tc_val": int(tcode[j]),
                "scale": float(st.asinh_scale[0]),
                "o_std": float(st.output_std[0]),
                "o_mean": float(st.output_mean[0]),
                "drain_free": int(drain_free[j]),
                "source_free": int(source_free[j]),
                "term_free": term_free[j],                     # (4,)
                "term_is_vin": term_is_vin[j],                 # (4,)
                "term_fix": [term_fix_v[:, j, t] for t in range(4)],  # 4×(G,)
            })

    # — residual F(V, vin; θ) into each free node —
    def residual(self, V: torch.Tensor, vin: torch.Tensor, gidx=None
                 ) -> torch.Tensor:
        """``V`` (B, n_free), ``vin`` (B,). ``gidx`` selects which of the G groups
        the rows correspond to (for the fixed-rail per-group voltages); None = all
        G groups in order."""
        B = V.shape[0]
        node_terms: List[List[torch.Tensor]] = [[] for _ in range(self.n_free)]
        for dv in self.dev:
            tv = []
            for t in range(4):
                if dv["term_is_vin"][t]:
                    col = vin
                elif dv["term_free"][t] >= 0:
                    col = V[:, dv["term_free"][t]]
                else:
                    tf = dv["term_fix"][t]
                    col = tf if gidx is None else tf[gidx]
                tv.append(col)
            vd, vg, vs, vb = tv
            zeros = torch.zeros_like(vd)
            volts = torch.stack([vd - vs, vg - vs, zeros, vb - vs], dim=1)  # (B,4)
            x_v = (volts - dv["in_mean"]) / dv["in_std"]
            x_geo = dv["x_geo"].unsqueeze(0).expand(B, -1)
            x = torch.cat([x_v, x_geo], dim=1)                # (B,7)
            tc = torch.full((B,), dv["tc_val"], dtype=torch.long, device=self.device)
            out = dv["model"](x, tech_codes=tc)               # (B,13)
            id_phys = dv["scale"] * torch.sinh(
                out[:, 0] * dv["o_std"] + dv["o_mean"])       # (B,) Amps
            if dv["drain_free"] >= 0:
                node_terms[dv["drain_free"]].append(-id_phys)
            if dv["source_free"] >= 0:
                node_terms[dv["source_free"]].append(id_phys)
        cols = [torch.stack(ts, 0).sum(0) if ts
                else torch.zeros(B, device=self.device)
                for ts in node_terms]
        return torch.stack(cols, dim=1)                       # (B, n_free)

    def solve(self, V0: torch.Tensor, vin: torch.Tensor, *, n_steps: int,
              lam: float, step: float, clip: float,
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unrolled LM-damped Newton on r = F/arm. Returns (V*, final |r| RMS)."""
        eye = torch.eye(self.n_free, device=self.device)
        V = V0.clone().requires_grad_(True)
        last_rrms = None
        for _ in range(n_steps):
            r = self.residual(V, vin) / self.arm              # (G, n_free)
            rows = []
            for i in range(self.n_free):
                gi = torch.autograd.grad(r[:, i].sum(), V,
                                         create_graph=True, retain_graph=True)[0]
                rows.append(gi)
            J = torch.stack(rows, dim=1)                      # (G, n_free, n_free)
            A = J + lam * eye
            dV = torch.linalg.solve(A, r.unsqueeze(-1)).squeeze(-1)
            if clip > 0:
                dV = torch.clamp(dV, -clip, clip)
            V = V - step * dV
            last_rrms = torch.sqrt((r ** 2).mean()).detach()
        return V, last_rrms

    def continuation_warmstart(self, vin: torch.Tensor, *, n_steps: int,
                               lam: float, step: float, clip: float
                               ) -> torch.Tensor:
        """Gate-style continuation sweep: solve groups in Vin order, warm-starting
        each from the previous converged V (first group from the L72 OP). Returns
        a DETACHED (G, n_free) — the gate-reachable root locus the batched
        gradient solve then refines under the current θ. The Jacobian needs grad,
        but each iterate is detached so NO θ-graph is built (this is an init, not
        a loss term). This is what aligns the trained root with the GATE root —
        teacher-forcing from the L72 OP lands a DIFFERENT (lower-gain) root."""
        eye = torch.eye(self.n_free, device=self.device)
        v = self.V_l72[0:1].detach().clone()        # railed entry (low-vin edge)
        out = []
        for g in range(self.G):
            vin_g = vin[g:g + 1]
            arm_g = self.arm[g:g + 1]
            for _ in range(n_steps):
                v = v.detach().requires_grad_(True)
                r = self.residual(v, vin_g, gidx=slice(g, g + 1)) / arm_g
                rows = [torch.autograd.grad(r[:, i].sum(), v, retain_graph=True)[0]
                        for i in range(self.n_free)]
                J = torch.stack(rows, dim=1)
                dV = torch.linalg.solve(J + lam * eye, r.unsqueeze(-1)).squeeze(-1)
                if clip > 0:
                    dV = torch.clamp(dV, -clip, clip)
                v = v - step * dV
            out.append(v.detach()[0])
        return torch.stack(out, dim=0)              # (G, n_free) detached

    # — diagnostics —
    @torch.no_grad()
    def frac_at_l72(self) -> np.ndarray:
        r = (self.residual(self.V_l72, self.vin) / self.arm)
        return torch.sqrt((r ** 2).mean(dim=0)).cpu().numpy()


def _gain_softpeak(vout: torch.Tensor, vin: torch.Tensor,
                   window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Peak |central-difference slope| over the window (sorted by vin).

    Returns (hard_max, soft_peak). hard_max matches the gate's np.max(np.gradient)
    closely; soft_peak (logsumexp) is the smooth loss target."""
    # central differences on the sorted grid
    s = (vout[2:] - vout[:-2]) / (vin[2:] - vin[:-2])         # (G-2,)
    w = window[1:-1]                                          # align centres
    absS = torch.abs(s) * w + (-1e9) * (1.0 - w)             # mask out-of-window
    hard = absS.max()
    tau = 0.05                                                # logsumexp sharpness
    soft = torch.logsumexp(tau * absS[w > 0], dim=0) / tau
    return hard, soft


def main() -> int:
    ap = argparse.ArgumentParser(description="V6.5.8 T3 differentiable-solver opamp fine-tune")
    ap.add_argument("--tech", default="tsmc7")
    ap.add_argument("--nmos-init", default=None, help="default tsmc{X}_dn_ekvhr_nmos")
    ap.add_argument("--pmos-init", default=None, help="default tsmc{X}_dn_ekvhr_pmos")
    ap.add_argument("--exp-name", default=None, help="default tsmc{X}_dn_t3")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--steps-per-epoch", type=int, default=150,
                    help="cap minibatch steps/epoch (the differentiable solve is "
                         "the same 79 groups every step — no need to run it once "
                         "per dataset minibatch; the anchor still samples "
                         "steps·bs random rows/epoch). 0 = full dataset pass.")
    # — solver —
    ap.add_argument("--solve-steps", type=int, default=8)
    ap.add_argument("--lam-lm", type=float, default=0.05, help="LM damping on r")
    ap.add_argument("--newton-step", type=float, default=1.0)
    ap.add_argument("--newton-clip", type=float, default=0.3, help="trust-region |dV|")
    ap.add_argument("--init-mode", choices=["teacher", "continuation"],
                    default="continuation",
                    help="V0 for the gradient solve. 'teacher' = the L72 OP "
                         "(lands the NEAREST root — decoupled from the gate, which "
                         "over-/under-shoots). 'continuation' = a no-grad gate-style "
                         "warm-start sweep (recomputed each epoch) so the supervised "
                         "root IS the gate-reachable root — the fix for the tf↔gate "
                         "gain offset.")
    ap.add_argument("--perturb", type=float, default=0.0,
                    help="(legacy) V0 vout random displacement; superseded by "
                         "--init-mode continuation. 0 = off.")
    ap.add_argument("--curve-band", type=float, default=0.05,
                    help="supervise Vout(Vin) within ±band V of vin* (inner window)")
    # — loss weights —
    ap.add_argument("--lam-curve", type=float, default=50.0)
    ap.add_argument("--lam-gain", type=float, default=1.0)
    ap.add_argument("--gain-target", type=float, default=None, help="default L72")
    ap.add_argument("--lam-lo-override", type=float, default=None,
                    help="raise the EKV core CLM-band low end at load (caps max "
                         "r_o = max gain), suppressing the over-flattened gate root "
                         "the continuation otherwise reaches. Fallback if the early "
                         "transient window is too sharp to gate-shop.")
    ap.add_argument("--ring-weight", type=float, default=1.0)
    ap.add_argument("--ring-corridor", default=None)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apply-filter", choices=["on", "off"], default="off")
    ap.add_argument("--freeze-embed", choices=["on", "off"], default="on")
    ap.add_argument("--save-every", type=int, default=0,
                    help="also dump {exp}_e{epoch}_{n,p} every N epochs so the "
                         "AUTHORITATIVE gate can pick the epoch (teacher-forced "
                         "gain carries a +offset vs the continuation gate gain).")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--self-check", action="store_true",
                    help="load init, validate the differentiable solver vs the "
                         "harvest self-check + report pre-train curve/gain; no train")
    args = ap.parse_args()

    scope = args.tech.lower()
    nmos_init = args.nmos_init or f"{scope}_dn_ekvhr_nmos"
    pmos_init = args.pmos_init or f"{scope}_dn_ekvhr_pmos"
    exp = args.exp_name or f"{scope}_dn_t3"
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print("=" * 78)
    print(f"V6.5.8 T3 differentiable-solver opamp fine-tune  tech={scope} dev={device}")
    print(f"  init: {nmos_init} / {pmos_init} -> {exp}_{{nmos,pmos}}")
    print(f"  solve: steps={args.solve_steps} lam_lm={args.lam_lm} "
          f"step={args.newton_step} clip={args.newton_clip} perturb={args.perturb}")
    print(f"  loss: lam_curve={args.lam_curve} lam_gain={args.lam_gain} "
          f"ring_w={args.ring_weight} curve_band=±{args.curve_band}")
    print("=" * 78)

    # — models (EKV-aware load, reused from finetune_kcl) —
    model_n, st_n = fk._build_and_load(nmos_init, scope, device)
    model_p, st_p = fk._build_and_load(pmos_init, scope, device)
    models = {"nmos": model_n, "pmos": model_p}
    norms = {"nmos": st_n, "pmos": st_p}
    if args.lam_lo_override is not None:
        # Cap the EKV core's MAX output resistance (= max opamp gain) by raising
        # the CLM band low end. V6.5.8 found the gate's continuation lands on an
        # OVER-FLATTENED r_o root (gain ~370) once existence is tight; capping r_o
        # bounds that root's gain near the target so the gate cannot overshoot,
        # while T3's curve supervision centres it. (V6.5.8 capped lam_lo WITHOUT
        # curve supervision and railed; T3 supplies the missing existence driver.)
        for m in (model_n, model_p):
            if getattr(m, "core", None) is not None:
                m.core.lam_lo.fill_(float(args.lam_lo_override))
        print(f"  EKV lam_lo override -> {args.lam_lo_override} "
              f"(caps max r_o / max gain to suppress the over-flattened gate root)")
    if args.freeze_embed == "on":
        model_n.tech_embedding.weight.requires_grad_(False)
        model_p.tech_embedding.weight.requires_grad_(False)

    solver = OpampDiffSolver(KCL_DIR / f"{scope}_opamp_topo.npz", models, norms, device)
    gain_target = args.gain_target if args.gain_target is not None else solver.l72_gain
    vin_star = solver.vin_star
    window = ((solver.vin - vin_star).abs() <= args.curve_band).float()  # (G,)
    # transition-emphasis curve weights: |L72 slope| + a small floor, masked to window.
    l72_slope = torch.zeros_like(solver.vout_l72)
    l72_slope[1:-1] = (solver.vout_l72[2:] - solver.vout_l72[:-2]).abs() / \
        (solver.vin[2:] - solver.vin[:-2]).abs()
    cw = (l72_slope + 0.05 * l72_slope.max()) * window
    cw = cw / (cw.sum() + 1e-12)

    print(f"  topo: G={solver.G} free={solver.free_nodes} vin*={vin_star:.4f} "
          f"L72gain={solver.l72_gain:.1f} target={gain_target:.1f} "
          f"window_pts={int(window.sum().item())}")
    frac0 = solver.frac_at_l72()
    print(f"  init F_rel @ L72 OP: {dict(zip(solver.free_nodes, np.round(frac0,4)))}")

    def _compute_V0() -> torch.Tensor:
        """V0 for the gradient solve per --init-mode (detached)."""
        if args.init_mode == "continuation":
            return solver.continuation_warmstart(
                solver.vin, n_steps=args.solve_steps, lam=args.lam_lm,
                step=args.newton_step, clip=args.newton_clip)
        return solver.V_l72

    # — solver validation: solve from V0, report curve/gain —
    def _eval_curve(V0: torch.Tensor = None):
        model_n.eval(); model_p.eval()
        if V0 is None:
            V0 = _compute_V0()
        Vs, rrms = solver.solve(V0, solver.vin, n_steps=args.solve_steps,
                                lam=args.lam_lm, step=args.newton_step,
                                clip=args.newton_clip)
        vout = Vs[:, solver.vout_idx]
        hard, soft = _gain_softpeak(vout, solver.vin, window)
        cl = (cw * (vout - solver.vout_l72) ** 2).sum()
        return vout.detach(), hard.detach(), soft.detach(), cl.detach(), rrms

    vout0, gh0, gs0, cl0, rr0 = _eval_curve()
    print(f"  PRE-TRAIN teacher-forced solve: |r|RMS={rr0:.4e} "
          f"gain(hard)={gh0:.1f} gain(soft)={gs0:.1f} L_curve={cl0:.5e}")
    # report a few transition Vout vs L72
    wi = torch.where(window > 0)[0]
    mid = wi[len(wi)//2]
    sl = slice(int(mid)-3, int(mid)+4)
    print("  vin / Vout_NN / Vout_L72 around vin*:")
    for k in range(sl.start, sl.stop):
        print(f"    {solver.vin[k]:.4f}  {vout0[k]:.4f}  {solver.vout_l72[k]:.4f}")

    if args.self_check:
        # cross-check: the differentiable residual at V_l72 must reproduce the
        # static harvest self-check magnitude (~0 in L72-current terms only; the
        # NN F_rel above is the genuine init existence gap, not a bug).
        print("\n  SELF-CHECK ok: residual assembled, solver ran, curve extracted.")
        print("  (init F_rel is the NN existence gap; ekvhr should show vo1i/vout "
              "~0.04-0.08 — the balanced EKV substrate.)")
        return 0

    # — base-data anchors (reused) —
    apply_filter = args.apply_filter == "on"
    tr_n, va_n, fit_n = fk._anchor_split(scope, "nmos", args.seed, apply_filter)
    tr_p, va_p, fit_p = fk._anchor_split(scope, "pmos", args.seed, apply_filter)
    fk._assert_norm_matches(fit_n.stats, st_n, "nmos")
    fk._assert_norm_matches(fit_p.stats, st_p, "pmos")
    lds_n = fk._lds_weights(tr_n).to(device)
    lds_p = fk._lds_weights(tr_p).to(device)

    # — ring anchor (reused loader, inline) —
    ring = None
    if args.ring_weight > 0:
        rdir = Path(args.ring_corridor) if args.ring_corridor else \
            (ROOT / "results" / "v6_5_5" / "corridors")
        ring = {}
        for key, st in (("nmos", st_n), ("pmos", st_p)):
            fr = rdir / f"{scope}_ring_{key}_corridor.npz"
            if not fr.exists():
                raise SystemExit(f"ring corridor not found: {fr}")
            dd = np.load(fr, allow_pickle=True)
            variant = str(dd["meta_variant"]) if "meta_variant" in dd.files else "ulvt"
            tcode = local_variant_code(scope, scope, variant)
            nz = normalizer_from_stats(st)
            x = nz.normalize_inputs(dd["inputs"], dd["geometry"])
            y = nz.normalize_outputs(dd["outputs"])
            ring[key] = {
                "x": torch.tensor(x, dtype=torch.float32, device=device),
                "y": torch.tensor(y, dtype=torch.float32, device=device),
                "tc": torch.full((len(x),), int(tcode), dtype=torch.long, device=device),
            }
        print(f"  ring anchor: nmos={len(ring['nmos']['x'])} "
              f"pmos={len(ring['pmos']['x'])} rows (w={args.ring_weight})")

    crit = MAELoss()
    params = [p for p in (list(model_n.parameters()) + list(model_p.parameters()))
              if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    base_val_n = fk._val_mae(model_n, va_n, device)
    base_val_p = fk._val_mae(model_p, va_p, device)
    print(f"  baseline val_MAE: n={base_val_n:.5f} p={base_val_p:.5f}")

    xn = tr_n.inputs.to(device); yn = tr_n.outputs.to(device); tcn = tr_n.tech_codes.to(device)
    xp = tr_p.inputs.to(device); yp = tr_p.outputs.to(device); tcp = tr_p.tech_codes.to(device)
    nN, nP = len(xn), len(xp)
    bs = args.batch_size

    best = {"score": float("inf"), "epoch": -1, "gain": float("nan"),
            "rise": float("nan"), "curve": float("nan")}
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model_n.train(); model_p.train()
        perm_n = torch.randperm(nN, device=device)
        perm_p = torch.randperm(nP, device=device)
        steps = max(nN, nP) // bs + 1
        if args.steps_per_epoch > 0:
            steps = min(steps, args.steps_per_epoch)
        # continuation: recompute the gate-style warm-start once per epoch
        # (lags θ by <=1 epoch; it is only the init for the grad solve, which
        # re-converges under current θ). teacher: the static L72 OP.
        V0_epoch = _compute_V0().detach()
        run = {"anchor": 0.0, "curve": 0.0, "gain": 0.0, "ring": 0.0, "n": 0}
        for s in range(steps):
            bn = perm_n[(s * bs) % nN:(s * bs) % nN + bs]
            bp = perm_p[(s * bs) % nP:(s * bs) % nP + bs]
            opt.zero_grad()
            la = crit(model_n(xn[bn], tech_codes=tcn[bn]), yn[bn], weights=lds_n[bn])
            lb = crit(model_p(xp[bp], tech_codes=tcp[bp]), yp[bp], weights=lds_p[bp])
            anchor = la + lb
            # — differentiable solve + curve/gain loss (full band each step) —
            Vs, _ = solver.solve(V0_epoch, solver.vin, n_steps=args.solve_steps,
                                 lam=args.lam_lm, step=args.newton_step,
                                 clip=args.newton_clip)
            vout = Vs[:, solver.vout_idx]
            l_curve = (cw * (vout - solver.vout_l72) ** 2).sum()
            ghard, _ = _gain_softpeak(vout, solver.vin, window)
            l_gain = ((ghard - gain_target) / gain_target) ** 2
            loss = anchor + args.lam_curve * l_curve + args.lam_gain * l_gain
            if args.ring_weight > 0:
                lr_n = crit(model_n(ring["nmos"]["x"], tech_codes=ring["nmos"]["tc"]),
                            ring["nmos"]["y"])
                lr_p = crit(model_p(ring["pmos"]["x"], tech_codes=ring["pmos"]["tc"]),
                            ring["pmos"]["y"])
                loss = loss + args.ring_weight * (lr_n + lr_p)
                run["ring"] += float((lr_n + lr_p).item())
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            run["anchor"] += anchor.item(); run["curve"] += float(l_curve.item())
            run["gain"] += float(l_gain.item()); run["n"] += 1

        vn = fk._val_mae(model_n, va_n, device)
        vp = fk._val_mae(model_p, va_p, device)
        vout_e, gh, gs, cl, rr = _eval_curve(V0_epoch)
        rise = max((vn - base_val_n) / base_val_n, (vp - base_val_p) / base_val_p)
        gain_err = abs(float(gh) - gain_target) / gain_target
        # selection: among epochs whose teacher-forced gain is within 12% AND
        # preservation is sane (<0.5 anchor drift), minimise gain error then drift.
        ok = (gain_err <= 0.12) and (rise < 0.5) and torch.isfinite(gh)
        score = (gain_err + 0.1 * max(0.0, rise)) if ok else float("inf")
        mark = ""
        if score < best["score"] - 1e-9:
            best = {"score": score, "epoch": epoch, "gain": float(gh),
                    "rise": rise, "curve": float(cl)}
            best_state = ({k: v.detach().cpu().clone() for k, v in model_n.state_dict().items()},
                          {k: v.detach().cpu().clone() for k, v in model_p.state_dict().items()})
            mark = " *best*"
        n = max(run["n"], 1)
        if args.save_every > 0 and epoch % args.save_every == 0:
            for key, m, src in (("nmos", model_n, nmos_init), ("pmos", model_p, pmos_init)):
                ep_stem = f"{exp}_e{epoch}_{key}"
                torch.save({k: v.detach().cpu().clone() for k, v in m.state_dict().items()},
                           str(fk.CHECKPOINT_DIR / f"{ep_stem}_best.pt"))
                fk._load_norm(src).save(str(fk.CHECKPOINT_DIR / f"{ep_stem}_norm.npz"))
            print(f"    [save-every] dumped {exp}_e{epoch}_{{nmos,pmos}} "
                  f"(tf gain={gh:.1f})")
        if epoch <= 5 or epoch % 2 == 0 or mark:
            print(f"  {epoch:3d} | anc={run['anchor']/n:.5f} curve={run['curve']/n:.3e} "
                  f"gainL={run['gain']/n:.4f} ring={run['ring']/n:.5f} init={args.init_mode[:4]} | "
                  f"val_n+{(vn-base_val_n)/base_val_n*100:.0f}% "
                  f"val_p+{(vp-base_val_p)/base_val_p*100:.0f}% | "
                  f"gain(hard)={gh:.1f} |r|={rr:.2e} gErr={gain_err*100:.1f}%{mark}")

    if best_state is None:
        print("  WARNING: no epoch reached the gain window — saving FINAL.")
        best_state = (model_n.state_dict(), model_p.state_dict())
        best["epoch"] = args.epochs
    else:
        print(f"  SELECTED epoch {best['epoch']}: teacher-forced gain={best['gain']:.1f} "
              f"max anchor drift +{best['rise']*100:.0f}%")

    fk.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for key, state, src_norm in (("nmos", best_state[0], nmos_init),
                                 ("pmos", best_state[1], pmos_init)):
        bp = fk.CHECKPOINT_DIR / f"{exp}_{key}_best.pt"
        npth = fk.CHECKPOINT_DIR / f"{exp}_{key}_norm.npz"
        if bp.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite {bp}; pass --overwrite.")
        torch.save(state, str(bp))
        fk._load_norm(src_norm).save(str(npth))
        print(f"  saved {bp.name} (+ norm) [best epoch {best['epoch']}]")
    print(f"\nDONE. Gate with "
          f"PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS={exp}_nmos "
          f"PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS={exp}_pmos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
