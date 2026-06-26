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
    """Rebuild a `large` DirectNet matching the production stem and load it."""
    st = _load_norm(stem)
    in_dim = len(st.input_mean)
    out_dim = len(OUTPUT_COLUMN_ORDER)
    vocab = tech_scope_vocab_size(scope)
    model = DirectNet(
        input_dim=in_dim, hidden_dim=LARGE_HIDDEN,
        n_layers=LARGE_TRUNK_LAYERS + 1, output_dim=out_dim,
        num_tech_codes=vocab, tech_embed_dim=32,
        tech_embed_dropout=0.0,                 # OFF for fine-tune (no code corruption)
        unknown_code_id=vocab - 1,
    ).to(device)
    state = torch.load(str(CHECKPOINT_DIR / f"{stem}_best.pt"),
                       weights_only=True, map_location=device)
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
    solver._stamp_mosfet_dc). Loss = mean_{g,node} (F / arm_scale)².
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

        self.dev: List[dict] = []
        for j in range(self.ndev):
            key = "pmos" if is_pmos[j] else "nmos"
            st = norms[key]
            # build the 7-col normaliser input: [Vd,Vg,Vs,Vb, nfin_log, L, T].
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
                "out_std": float(st.output_std[0]),
                "out_mean": float(st.output_mean[0]),
                "id_s": float(st.asinh_scale[0]),
                "drain_free": int(self.drain_free[j]),
                "source_free": int(self.source_free[j]),
            })

    def residual(self) -> torch.Tensor:
        """(G, n_free) signed-current residual into each free node."""
        F = torch.zeros(self.G, self.n_free, device=self.device)
        for dv in self.dev:
            out = dv["model"](dv["x"], tech_codes=dv["tc"])     # (G,13)
            id_norm = out[:, 0]
            u = id_norm * dv["out_std"] + dv["out_mean"]
            id_phys = dv["id_s"] * torch.sinh(u)                # native Amps
            if dv["drain_free"] >= 0:
                F[:, dv["drain_free"]] = F[:, dv["drain_free"]] - id_phys
            if dv["source_free"] >= 0:
                F[:, dv["source_free"]] = F[:, dv["source_free"]] + id_phys
        return F

    def loss_and_frac(self) -> Tuple[torch.Tensor, np.ndarray]:
        F = self.residual()
        rel = F / self.arm
        loss = (rel ** 2).mean()
        # per-free-node RMS fraction (diagnostic; index 2 = vo1i).
        frac = torch.sqrt((rel ** 2).mean(dim=0)).detach().cpu().numpy()
        return loss, frac


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
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="clip combined grad-norm (0 disables); kills the "
                         "KCL-vs-anchor tug-of-war blowups")
    ap.add_argument("--vo1i-target", type=float, default=0.02,
                    help="stage-1 balance F_rel below which KCL counts as fixed; "
                         "selection = min anchor-drift among epochs that reach it")
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
        run_anchor, run_kcl, ns = 0.0, 0.0, 0
        for s in range(steps):
            bn = perm_n[(s * bs) % nN:(s * bs) % nN + bs]
            bp = perm_p[(s * bs) % nP:(s * bs) % nP + bs]
            opt.zero_grad()
            la = crit(model_n(xn[bn], tech_codes=tcn[bn]), yn[bn], weights=lds_n[bn])
            lb = crit(model_p(xp[bp], tech_codes=tcp[bp]), yp[bp], weights=lds_p[bp])
            lk, _ = kcl.loss_and_frac()
            loss = la + lb + args.lam_kcl * kcl_step_scale * lk
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            run_anchor += (la + lb).item(); run_kcl += lk.item(); ns += 1
        vn = _val_mae(model_n, va_n, device)
        vp = _val_mae(model_p, va_p, device)
        with torch.no_grad():
            kloss, frac = kcl.loss_and_frac()
        rise_n = (vn - base_val_n) / base_val_n
        rise_p = (vp - base_val_p) / base_val_p
        fixed = float(frac[2]) < args.vo1i_target
        # selection = min anchor drift among epochs that FIXED the KCL balance.
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
                  f"| val_n={vn:.5f}(+{rise_n*100:.0f}%) val_p={vp:.5f}(+{rise_p*100:.0f}%) "
                  f"frac(vo1i)={frac[2]:.4f} {'FIXED' if fixed else 'open '}{mark}")

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
