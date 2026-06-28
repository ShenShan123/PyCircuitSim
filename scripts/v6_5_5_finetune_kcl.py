#!/usr/bin/env python3
"""V6.5.5 T1 — joint NMOS+PMOS KCL-residual fine-tune (the G1 existence lever).

Routed by P0-1 (tests/diag_opamp_kcl_residual.py): the tsmc7 opamp gain->0 is an
EXISTENCE failure — the L72 high-gain OP is NOT a residual zero of the NN current
map (vo1i F_rel=0.128). The ONLY admissible lever is a net-node KCL-residual loss
that supervises the DIFFERENCE Σ(signed id) -> 0 at the true OPs (the one quantity
the absolute-id corridor never pinned), in the native-µA frame.

The opamp KCL residual at the stage-1 balance node couples an NMOS (Mn2) and a
PMOS (Mp4) — so the lever MUST hold BOTH per-tech checkpoints in one loss. This
script jointly fine-tunes the tsmc7 NMOS+PMOS DirectNet from the production
`large` checkpoints with:

    total = MAE_anchor(NMOS, base-data) + MAE_anchor(PMOS, base-data)
          + λ_kcl · mean_over_groups,free_nodes ( (Σ signed NN id) / arm )²

The base-data LDS-MAE anchor pins the 15 passing gates' surface (DC-unsafe lever
→ the anchor is load-bearing); the KCL term nudges ONLY the opamp bias locus.
KCL id is the FULL physical (native-µA) denorm of the asinh-normalised id head, so
the asinh s_id~2.6e-5 compression that killed every absolute-id lever does not
apply. Signs mirror solver._stamp_mosfet_dc:303-309 (i_leaving = -id for both
types; F[drain]+= -id, F[source]+= +id), validated by the harvest L72 self-check.

Ship gate (run AFTER this; point the gates at the new checkpoints via
PYCIRCUITSIM_NN_CHECKPOINT_DN_{NMOS,PMOS}):
  1. tests/diag_opamp_kcl_residual.py  — vo1i F_rel must drop toward 0.
  2. tests/diag_opamp_basin_seed.py    — 1c must HOLD gain>0 when seeded.
  3. tests/verify_complex_opamp.py     — the authoritative opamp gate.
  4. full 16-gate matrix + device DC/AC + lifted-source canary — UNREGRESSED.

Usage:
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
      conda run -n pycircuitsim python -u scripts/v6_5_5_finetune_kcl.py \
        --tech tsmc7 --cuda --epochs 40 --lam-kcl 1.0 --exp-name tsmc7_dn_kcl
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

from bsimar.config import (  # noqa: E402
    CHECKPOINT_DIR, tech_scope_vocab_size,
)
from bsimar.data.dataset import load_and_split_bsimar  # noqa: E402
from bsimar.data.normalize import OUTPUT_COLUMN_ORDER, NormStats  # noqa: E402
from bsimar.losses.bni_mae import MAELoss, compute_lds_weights_per_target  # noqa: E402
from bsimar.models.direct_net import DirectNet  # noqa: E402

KCL_DIR = ROOT / "results" / "v6_5_5" / "kcl_groups"
_ALL_TECHS = ("tsmc5", "tsmc7", "tsmc12", "tsmc16", "asap7")
# `large` preset (cli/train SIZE_PRESETS[('direct','large')]).
LARGE_HIDDEN, LARGE_TRUNK_LAYERS = 384, 6
NORM_MODE = "asinh"


# ── checkpoint / normalizer loading ─────────────────────────────────────────

def _load_norm(stem: str) -> NormStats:
    return NormStats.load(str(CHECKPOINT_DIR / f"{stem}_norm.npz"))


def _build_and_load(stem: str, scope: str, device: torch.device) -> Tuple[nn.Module, NormStats]:
    """Rebuild a `large` DirectNet matching the production stem and load it.

    EKV-aware (V6.5.8): a checkpoint trained with ``--ekv-core`` carries
    ``core.*`` keys (the physical id-column backbone + its norm/sign buffers).
    Rebuild the EKV core at the saved hidden size so the fine-tune optimises the
    SAME composed surface (core + bounded residual) — its buffers round-trip via
    ``load_state_dict``, so no ``set_norm`` is needed. The KCL autograd chain
    flows through the core transparently (it is part of ``forward``)."""
    st = _load_norm(stem)
    in_dim = len(st.input_mean)
    out_dim = len(OUTPUT_COLUMN_ORDER)
    vocab = tech_scope_vocab_size(scope)
    state = torch.load(str(CHECKPOINT_DIR / f"{stem}_best.pt"),
                       weights_only=True, map_location=device)
    ekv_core = any(k.startswith("core.") for k in state)
    ekv_hidden = (int(state["core.param_head.0.weight"].shape[0])
                  if ekv_core else 64)
    model = DirectNet(
        input_dim=in_dim, hidden_dim=LARGE_HIDDEN,
        n_layers=LARGE_TRUNK_LAYERS + 1, output_dim=out_dim,
        num_tech_codes=vocab, tech_embed_dim=32,
        tech_embed_dropout=0.0,                 # OFF for fine-tune (no code corruption)
        unknown_code_id=vocab - 1,
        ekv_core=ekv_core, ekv_hidden=ekv_hidden,
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"{stem}: arch mismatch missing={list(missing)} "
                         f"unexpected={list(unexpected)}")
    return model, st


# ── base-data anchor (replicates the trainer's LDS-MAE) ──────────────────────

def _anchor_split(scope: str, device_type: str, seed: int, apply_filter: bool):
    """Load + split base data exactly as the trainer; return train/val datasets."""
    excl = {t for t in _ALL_TECHS if t != scope}
    tr, va, _te, normalizer = load_and_split_bsimar(
        str(ROOT / "external_compact_models" / "bsimar" / "data" /
            "datasets" / f"{scope}_{device_type}.npz"),
        OUTPUT_COLUMN_ORDER, device_type=device_type,
        norm_mode=NORM_MODE, apply_filter=apply_filter,
        exclude_techs=excl, tech_scope=scope, seed=seed,
    )
    return tr, va, normalizer


def _assert_norm_matches(fit: NormStats, prod: NormStats, tag: str) -> None:
    """The fresh fit must equal the production normalizer (else the loaded
    weights are inconsistent with the anchor normalisation)."""
    for field in ("input_mean", "input_std", "output_mean", "output_std",
                  "asinh_scale"):
        a = np.asarray(getattr(fit, field), dtype=np.float64)
        b = np.asarray(getattr(prod, field), dtype=np.float64)
        d = float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-30))
        if d > 1e-3:
            raise SystemExit(
                f"[{tag}] re-fit normalizer field {field!r} differs from the "
                f"production checkpoint by rel {d:.2e} — base data / seed / "
                f"apply_filter do not match what produced the checkpoint.")
    print(f"  [{tag}] re-fit normalizer matches production (max rel < 1e-3) ✓")


def _lds_weights(train_ds) -> torch.Tensor:
    lds = compute_lds_weights_per_target(
        train_ds.outputs.numpy(), n_bins=100,
        lds_kernel="gaussian", lds_ks=5, lds_sigma=0.8)
    means = lds.mean(axis=0, keepdims=True)
    means[means < 1e-12] = 1.0
    return torch.tensor(lds / means, dtype=torch.float32)


# ── KCL group tensors ────────────────────────────────────────────────────────

class KCLGroups:
    """Precomputed per-device normalised inputs + free-node incidence.

    id_phys for device j = asinh_scale·sinh(id_norm·out_std + out_mean) using
    the device's own (N or P) production normalizer. F[g, node] accumulates
    -id at the drain free node and +id at the source free node (Rule 2 /
    solver._stamp_mosfet_dc). KCL loss = mean_{g,node} (F / arm_scale)².

    N2 (contraction) loss: the autograd ∂id/∂V the SOLVER consumes vs the
    device's own (accurate ~1%) predicted gm/gds COLUMNS, AT the opamp OPs. The
    1c/probe showed the now-existent OP repels because the autograd gm·ro there
    is flat (drifts from the predicted columns — the S10 deriv-fidelity gap);
    pulling autograd→columns makes the OP Newton-attracting. Localized to the 59
    opamp OPs + KCL-anchored (KCL holds the VALUE so the slope term can't move
    the FP) → avoids the S10 broad-collapse. Sign convention mirrors
    mosfet_nn._unpack_eval: gds = +∂id/∂Vd, gm = -∂id/∂Vg, gmb = -∂id/∂Vb.
    """

    def __init__(self, path: Path, models: Dict[str, nn.Module],
                 norms: Dict[str, NormStats], device: torch.device) -> None:
        d = np.load(path, allow_pickle=True)
        nn_volts = d["nn_volts"]            # (G, ndev, 4) source-ref phys V
        self.G, self.ndev, _ = nn_volts.shape
        self.arm = torch.tensor(d["arm_scale"], dtype=torch.float32, device=device)  # (G,4)
        self.drain_free = d["drain_free"].astype(int)
        self.source_free = d["source_free"].astype(int)
        is_pmos = d["dev_is_pmos"].astype(int)
        nfin = d["dev_nfin"]; Lg = d["dev_L"]; Tg = d["dev_T"]
        tcode = d["dev_tcode"].astype(int)
        self.n_free = self.arm.shape[1]
        self.device = device
        # free-node names + the two predictive indices (V6.5.7 vout-prioritized
        # lever): vo1i = stage-1 diff-pair/mirror balance; vout = output-stage
        # balance (i_Mp6 - i_Mn7) — the node T1 never supervised.
        self.free_nodes = [str(x) for x in d["free_nodes"]]
        self.vo1i_idx = self.free_nodes.index("vo1i")
        self.vout_idx = self.free_nodes.index("vout")

        def _t(a):
            return torch.tensor(np.asarray(a, dtype=np.float64),
                                dtype=torch.float32, device=device)

        self.dev: List[dict] = []
        for j in range(self.ndev):
            key = "pmos" if is_pmos[j] else "nmos"
            st = norms[key]
            volts = nn_volts[:, j, :]                       # (G,4)
            geo = np.zeros((self.G, 15), dtype=np.float64)
            geo[:, 0] = nfin[j]; geo[:, 1] = Lg[j]; geo[:, 2] = Tg[j]
            from bsimar.data.normalize import normalizer_from_stats
            x = normalizer_from_stats(st).normalize_inputs(volts, geo)
            self.dev.append({
                "model": models[key],
                "x": torch.tensor(x, dtype=torch.float32, device=device),
                "tc": torch.full((self.G,), int(tcode[j]),
                                 dtype=torch.long, device=device),
                # per-column (id,gm,gds,gmb = 0,1,2,3) asinh denorm stats.
                "o_std": _t(st.output_std[:4]),
                "o_mean": _t(st.output_mean[:4]),
                "scale": _t(st.asinh_scale[:4]),
                "in_std": _t(st.input_std[:4]),     # Vd,Vg,Vs,Vb normaliser std
                "drain_free": int(self.drain_free[j]),
                "source_free": int(self.source_free[j]),
            })

    @staticmethod
    def _denorm_col(out_col, scale, o_std, o_mean):
        return scale * torch.sinh(out_col * o_std + o_mean)

    def residual(self) -> torch.Tensor:
        """(G, n_free) signed-current residual into each free node."""
        F = torch.zeros(self.G, self.n_free, device=self.device)
        for dv in self.dev:
            out = dv["model"](dv["x"], tech_codes=dv["tc"])     # (G,13)
            id_phys = self._denorm_col(out[:, 0], dv["scale"][0],
                                       dv["o_std"][0], dv["o_mean"][0])
            if dv["drain_free"] >= 0:
                F[:, dv["drain_free"]] = F[:, dv["drain_free"]] - id_phys
            if dv["source_free"] >= 0:
                F[:, dv["source_free"]] = F[:, dv["source_free"]] + id_phys
        return F

    def loss_and_frac(self, node_w: torch.Tensor = None) -> Tuple[torch.Tensor, np.ndarray]:
        F = self.residual()
        rel = F / self.arm
        # node_w (len n_free) optionally up-weights a free node in the loss; None
        # ⇒ the V6.5.6 uniform mean (byte-identical). The returned `frac` is ALWAYS
        # the unweighted per-node RMS (used verbatim for epoch selection).
        if node_w is None:
            loss = (rel ** 2).mean()
        else:
            loss = (node_w.view(1, -1) * rel ** 2).mean()
        # per-free-node RMS fraction (diagnostic; index 2 = vo1i).
        frac = torch.sqrt((rel ** 2).mean(dim=0)).detach().cpu().numpy()
        return loss, frac

    def contraction_loss(self) -> torch.Tensor:
        """N2: relative MSE of the autograd-Jacobian the solver uses vs the
        device's own predicted gm/gds columns, summed over the opamp OPs.

        For each device: ∂id_phys/∂V_phys,c = (∂id_norm/∂x_norm,c)·factor/in_std,
        factor = scale_id·cosh(u)·o_std_id. gds_solver=+∂/∂Vd, gm_solver=-∂/∂Vg.
        Target = denorm(predicted gds/gm column), detached. Relative error so
        µS-scale conductances aren't washed out.
        """
        FLOOR = 1e-7   # S — conductance relative-error floor
        terms = []
        for dv in self.dev:
            xv = dv["x"][:, :4].detach().clone().requires_grad_(True)
            xg = dv["x"][:, 4:]
            x_full = torch.cat([xv, xg], dim=1)
            out = dv["model"](x_full, tech_codes=dv["tc"])      # (G,13)
            id_norm = out[:, 0]
            g = torch.autograd.grad(id_norm.sum(), xv, create_graph=True)[0]  # (G,4)
            u = id_norm * dv["o_std"][0] + dv["o_mean"][0]
            factor = dv["scale"][0] * torch.cosh(u) * dv["o_std"][0]          # (G,)
            gds_solver = factor * g[:, 0] / dv["in_std"][0]
            gm_solver = -factor * g[:, 1] / dv["in_std"][1]
            gds_pred = self._denorm_col(out[:, 2].detach(), dv["scale"][2],
                                        dv["o_std"][2], dv["o_mean"][2])
            gm_pred = self._denorm_col(out[:, 1].detach(), dv["scale"][1],
                                       dv["o_std"][1], dv["o_mean"][1])
            terms.append(((gds_solver - gds_pred) / (gds_pred.abs() + FLOOR)) ** 2)
            terms.append(((gm_solver - gm_pred) / (gm_pred.abs() + FLOOR)) ** 2)
        return torch.stack(terms).mean()


# ── fine-tune ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _val_mae(model, ds, device, bs=4096) -> float:
    model.eval()
    crit = MAELoss()
    tot, n = 0.0, 0
    for i in range(0, len(ds), bs):
        x = ds.inputs[i:i + bs].to(device)
        y = ds.outputs[i:i + bs].to(device)
        tc = ds.tech_codes[i:i + bs].to(device)
        tot += crit(model(x, tech_codes=tc), y).item(); n += 1
    return tot / max(n, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="V6.5.5 T1 joint KCL fine-tune")
    ap.add_argument("--tech", default="tsmc7")
    ap.add_argument("--nmos-init", default=None, help="default tsmc{X}_dn_large_nmos")
    ap.add_argument("--pmos-init", default=None, help="default tsmc{X}_dn_large_pmos")
    ap.add_argument("--exp-name", default=None, help="default tsmc{X}_dn_kcl")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lam-kcl", type=float, default=1.0)
    ap.add_argument("--lam-sob", type=float, default=0.0,
                    help="N2 contraction term weight (autograd ∂id/∂V -> predicted "
                         "gm/gds columns at the opamp OPs; 0 disables). Per-epoch "
                         "balanced like KCL.")
    ap.add_argument("--ring-weight", type=float, default=0.0,
                    help="ring-corridor anchor weight (per-step MAE on the tsmc7 "
                         "ring trajectory OSDI labels; 0 disables). Pins the shared "
                         "NMOS switching region against ring regression.")
    ap.add_argument("--ring-corridor", default=None,
                    help="dir holding {tech}_ring_{nmos,pmos}_corridor.npz "
                         "(default results/v6_5_5/corridors)")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="clip combined grad-norm (0 disables); kills the "
                         "KCL-vs-anchor tug-of-war blowups")
    ap.add_argument("--vo1i-target", type=float, default=0.02,
                    help="stage-1 balance F_rel below which KCL counts as fixed; "
                         "selection = min anchor-drift among epochs that reach it")
    ap.add_argument("--vout-weight", type=float, default=1.0,
                    help="V6.5.7 vout-prioritized lever: extra weight on the "
                         "OUTPUT-stage balance node (vout) in the KCL loss. 1.0 = "
                         "the V6.5.6 uniform mean (behavior-preserving); >1 drives "
                         "the full vout-inclusive high-gain root T1 never created.")
    ap.add_argument("--vout-target", type=float, default=None,
                    help="if set, an epoch counts as 'fixed' only when vout F_rel "
                         "is ALSO below this (AND vo1i-target). Default off = the "
                         "V6.5.6 vo1i-only selection.")
    ap.add_argument("--freeze-core", choices=["on", "off"], default="off",
                    help="V6.5.8: freeze the EKV core's param_head so its "
                         "data-true r_o (= opamp gain) is preserved while only "
                         "the bounded residual + other heads move for existence. "
                         "Targets the gain↔existence coupling (capping r_o via "
                         "lam_lo destroys the OP; this preserves r_o instead).")
    ap.add_argument("--freeze-embed", choices=["on", "off"], default="on",
                    help="freeze the tech embedding during fine-tune (the opamp "
                         "uses a single fixed code; freezing reduces drift)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apply-filter", choices=["on", "off"], default="on")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    scope = args.tech.lower()
    nmos_init = args.nmos_init or f"{scope}_dn_large_nmos"
    pmos_init = args.pmos_init or f"{scope}_dn_large_pmos"
    exp = args.exp_name or f"{scope}_dn_kcl"
    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available())
                          else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print("=" * 78)
    print(f"V6.5.5 T1 joint KCL fine-tune  tech={scope} device={device}")
    print(f"  init: {nmos_init} / {pmos_init}  -> {exp}_{{nmos,pmos}}")
    print(f"  epochs={args.epochs} lr={args.lr} wd={args.weight_decay} "
          f"λ_kcl={args.lam_kcl} grad_clip={args.grad_clip} "
          f"vo1i_target={args.vo1i_target} freeze_embed={args.freeze_embed}")
    print("=" * 78)

    # — models —
    model_n, st_n = _build_and_load(nmos_init, scope, device)
    model_p, st_p = _build_and_load(pmos_init, scope, device)
    models = {"nmos": model_n, "pmos": model_p}
    norms = {"nmos": st_n, "pmos": st_p}
    if args.freeze_embed == "on":
        model_n.tech_embedding.weight.requires_grad_(False)
        model_p.tech_embedding.weight.requires_grad_(False)
    if args.freeze_core == "on":
        n_frozen = 0
        for m in (model_n, model_p):
            if getattr(m, "core", None) is not None:
                for p in m.core.parameters():
                    p.requires_grad_(False)
                    n_frozen += 1
        print(f"  freeze-core ON: froze {n_frozen} EKV core param tensors "
              f"(data-true r_o preserved; residual+heads free)")

    # — base-data anchors —
    apply_filter = args.apply_filter == "on"
    tr_n, va_n, fit_n = _anchor_split(scope, "nmos", args.seed, apply_filter)
    tr_p, va_p, fit_p = _anchor_split(scope, "pmos", args.seed, apply_filter)
    _assert_norm_matches(fit_n.stats, st_n, "nmos")
    _assert_norm_matches(fit_p.stats, st_p, "pmos")
    lds_n = _lds_weights(tr_n).to(device)
    lds_p = _lds_weights(tr_p).to(device)

    # — KCL groups —
    kcl = KCLGroups(KCL_DIR / f"{scope}_opamp_kcl.npz", models, norms, device)
    print(f"  KCL groups: G={kcl.G} devices={kcl.ndev} free_nodes={kcl.n_free}")
    # vout-prioritized node weighting (V6.5.7). All-ones ⇒ V6.5.6 uniform mean.
    node_w = torch.ones(kcl.n_free, device=device)
    node_w[kcl.vout_idx] = args.vout_weight
    if args.vout_weight != 1.0:
        print(f"  vout-prioritized KCL: node_w={node_w.tolist()} "
              f"(vout idx={kcl.vout_idx}, vo1i idx={kcl.vo1i_idx}); "
              f"vout_target={args.vout_target}")

    # — ring-corridor anchor (Track B: pin the shared NMOS switching region) —
    from bsimar.config import local_variant_code
    from bsimar.data.normalize import normalizer_from_stats
    ring = None
    if args.ring_weight > 0:
        rdir = Path(args.ring_corridor) if args.ring_corridor else \
            (ROOT / "results" / "v6_5_5" / "corridors")
        ring = {}
        for key, st in (("nmos", st_n), ("pmos", st_p)):
            fr = rdir / f"{scope}_ring_{key}_corridor.npz"
            if not fr.exists():
                raise SystemExit(f"ring corridor not found: {fr} (harvest first)")
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
                "n": len(x),
            }
        print(f"  ring anchor: nmos={ring['nmos']['n']} pmos={ring['pmos']['n']} rows "
              f"(weight={args.ring_weight})")

    crit = MAELoss()
    params = [p for p in (list(model_n.parameters()) + list(model_p.parameters()))
              if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # baseline metrics
    base_val_n = _val_mae(model_n, va_n, device)
    base_val_p = _val_mae(model_p, va_p, device)
    with torch.no_grad():
        kloss0, frac0 = kcl.loss_and_frac()
    print(f"  baseline: val_MAE_n={base_val_n:.5f} val_MAE_p={base_val_p:.5f} "
          f"kcl={kloss0.item():.5f} frac(vo1i)={frac0[2]:.4f} frac={np.round(frac0,4).tolist()}")

    xn = tr_n.inputs.to(device); yn = tr_n.outputs.to(device); tcn = tr_n.tech_codes.to(device)
    xp = tr_p.inputs.to(device); yp = tr_p.outputs.to(device); tcp = tr_p.tech_codes.to(device)
    nN, nP = len(xn), len(xp)
    bs = args.batch_size
    # Balance the KCL pressure: it is computed on the SAME 59 groups EVERY step,
    # so without scaling it gets ~n_steps× more gradient updates per epoch than
    # its data share (the l05/l15/l40 blow-up). Scale per-step so the summed
    # per-epoch KCL weight ≈ λ_kcl (n_steps·bs ≈ n_ref).
    n_ref = float(max(nN, nP))
    kcl_step_scale = bs / n_ref
    print(f"  KCL per-step scale = bs/n_ref = {kcl_step_scale:.3e} "
          f"(per-epoch KCL weight ≈ λ_kcl={args.lam_kcl})")

    best = {"score": float("inf"), "epoch": -1, "vo1i": float("nan"),
            "rise": float("nan")}
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model_n.train(); model_p.train()
        perm_n = torch.randperm(nN, device=device)
        perm_p = torch.randperm(nP, device=device)
        steps = max(nN, nP) // bs + 1
        run_anchor, run_kcl, run_sob, ns = 0.0, 0.0, 0.0, 0
        for s in range(steps):
            bn = perm_n[(s * bs) % nN:(s * bs) % nN + bs]
            bp = perm_p[(s * bs) % nP:(s * bs) % nP + bs]
            opt.zero_grad()
            la = crit(model_n(xn[bn], tech_codes=tcn[bn]), yn[bn], weights=lds_n[bn])
            lb = crit(model_p(xp[bp], tech_codes=tcp[bp]), yp[bp], weights=lds_p[bp])
            lk, _ = kcl.loss_and_frac(node_w)
            loss = la + lb + args.lam_kcl * kcl_step_scale * lk
            if args.ring_weight > 0:
                lr_n = crit(model_n(ring["nmos"]["x"], tech_codes=ring["nmos"]["tc"]),
                            ring["nmos"]["y"])
                lr_p = crit(model_p(ring["pmos"]["x"], tech_codes=ring["pmos"]["tc"]),
                            ring["pmos"]["y"])
                loss = loss + args.ring_weight * (lr_n + lr_p)
            lsob_val = 0.0
            if args.lam_sob > 0:
                lsob = kcl.contraction_loss()
                loss = loss + args.lam_sob * kcl_step_scale * lsob
                lsob_val = float(lsob.item())
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            run_anchor += (la + lb).item(); run_kcl += lk.item()
            run_sob += lsob_val; ns += 1
        vn = _val_mae(model_n, va_n, device)
        vp = _val_mae(model_p, va_p, device)
        with torch.no_grad():
            kloss, frac = kcl.loss_and_frac()
        rise_n = (vn - base_val_n) / base_val_n
        rise_p = (vp - base_val_p) / base_val_p
        vout_ok = (args.vout_target is None) or \
            (float(frac[kcl.vout_idx]) < args.vout_target)
        fixed = (float(frac[kcl.vo1i_idx]) < args.vo1i_target) and vout_ok
        # selection = min anchor drift among epochs that FIXED the KCL balance
        # (vo1i, and vout too when --vout-target is set).
        drift = max(rise_n, rise_p)
        score = drift if fixed else float("inf")
        mark = ""
        if score < best["score"] - 1e-9:
            best = {"score": score, "epoch": epoch, "vo1i": float(frac[2]),
                    "rise": drift}
            best_state = ({k: v.detach().cpu().clone() for k, v in model_n.state_dict().items()},
                          {k: v.detach().cpu().clone() for k, v in model_p.state_dict().items()})
            mark = " *best*"
        if epoch <= 5 or epoch % 5 == 0 or mark:
            print(f"  {epoch:3d} | anchor={run_anchor/ns:.5f} kcl={run_kcl/ns:.5f} "
                  f"sob={run_sob/ns:.4f} | val_n={vn:.5f}(+{rise_n*100:.0f}%) "
                  f"val_p={vp:.5f}(+{rise_p*100:.0f}%) "
                  f"frac(vo1i)={frac[kcl.vo1i_idx]:.4f} "
                  f"frac(vout)={frac[kcl.vout_idx]:.4f} "
                  f"{'FIXED' if fixed else 'open '}{mark}")

    if best_state is None:
        print("  WARNING: no epoch reached vo1i_target — saving FINAL "
              "(gate will arbitrate; consider higher --lam-kcl / more epochs).")
        best_state = (model_n.state_dict(), model_p.state_dict())
        best["epoch"] = args.epochs
    else:
        print(f"  SELECTED epoch {best['epoch']}: vo1i={best['vo1i']:.4f} "
              f"max anchor drift +{best['rise']*100:.0f}%")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for key, state, src_norm in (("nmos", best_state[0], nmos_init),
                                 ("pmos", best_state[1], pmos_init)):
        bp = CHECKPOINT_DIR / f"{exp}_{key}_best.pt"
        npth = CHECKPOINT_DIR / f"{exp}_{key}_norm.npz"
        if bp.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite {bp}; pass --overwrite.")
        torch.save(state, str(bp))
        # normalizer is UNCHANGED (we use the production stats) — copy it.
        _load_norm(src_norm).save(str(npth))
        print(f"  saved {bp.name} (+ norm) [best epoch {best['epoch']}]")
    print(f"\nDONE. Gate with PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS={exp}_nmos_best.pt "
          f"PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS={exp}_pmos_best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
