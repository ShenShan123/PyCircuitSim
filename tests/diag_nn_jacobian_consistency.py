"""V5 Phase C — C0 diagnostic: FD vs autograd Jacobian consistency.

For a grid of (Vgs, Vds, Vbs, NFIN, L, T) operating points spanning
in-distribution (ID) and just-outside-distribution (OOD) regions of
the training box, compare ``torch.autograd.grad(out, V)`` against a
5-point central-finite-difference reference for each of the 8
Jacobian channels:

    ∂id/∂Vgs → gm        ∂id/∂Vds → gds        ∂id/∂Vbs → gmb
    ∂qg/∂Vgs → cgg       ∂qg/∂Vds → cgd        ∂qg/∂Vbs → cgs
    ∂qd/∂Vgs → cdg       ∂qd/∂Vds → cdd

The diagnostic operates **directly on the trained DirectNet checkpoint**
(no Vds correction, no PMOS frame shift, no clamping, no denormalisation
— pure model autograd vs FD on normalised inputs/outputs). This isolates
the network's intrinsic Jacobian self-consistency from the inference-time
correction layers in ``mosfet_directnet.py``.

Outputs:
- A single CSV at ``results/v5_phase_c_c0_jacobian_diag/<exp_name>.csv``
  with one row per (op-point, jacobian channel) and columns:
  ``v_gs, v_ds, v_bs, region, channel, autograd, fd, abs_err, rel_err, flag``
  where ``flag`` is "OK" / "BAD" (BAD if ``|FD - autograd| > 0.1·max(|FD|, eps)``).
- A summary line per checkpoint printed to stdout and appended to
  ``results/v5_phase_c_c0_jacobian_diag.md``.

Usage:
    python tests/diag_nn_jacobian_consistency.py \
        --checkpoint v4_dn_universal_nmos --polarity nmos
    python tests/diag_nn_jacobian_consistency.py \
        --checkpoint v5_dn_s_nmos_mae --polarity nmos
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from bsimar.config import CHECKPOINT_DIR, tech_variant_to_code, UNKNOWN_CODE_ID
from bsimar.data.normalize import BSIMARNormStats
from bsimar.models.direct_net import DirectNet


# ── op-point grid ──────────────────────────────────────────────────────────

def build_op_grid(
    vdd: float,
    polarity: str,
) -> List[Tuple[Dict[str, float], str]]:
    """Build a list of (op_dict, region) tuples.

    Op-points cover both in-distribution (ID, |V|<=VDD) and just-OOD
    (|V|<=1.5*VDD).  PMOS is in source-relative frame, so the script
    flips the sign of Vds/Vgs/Vbs externally before passing in.
    """
    sign = -1.0 if polarity == "pmos" else 1.0

    # Voltage grid: 5 levels each, ID+OOD.
    vgs_levels = sign * np.array([0.05, 0.25, 0.50, 0.85, 1.30]) * vdd  # last is 1.3*VDD = OOD
    vds_levels = sign * np.array([0.05, 0.25, 0.50, 0.85, 1.30]) * vdd
    vbs_levels = sign * np.array([-0.30, 0.0, 0.30]) * vdd

    nfin_levels = [3.0, 10.0]
    l_levels = [16e-9, 20e-9]
    t_levels = [300.15]

    grid = []
    for vgs in vgs_levels:
        for vds in vds_levels:
            for vbs in vbs_levels:
                for nfin in nfin_levels:
                    for L in l_levels:
                        for T in t_levels:
                            mag = max(abs(vgs), abs(vds), abs(vbs)) / (vdd + 1e-9)
                            region = "ID" if mag <= 1.0 else "OOD"
                            grid.append(({
                                "vgs": float(vgs), "vds": float(vds),
                                "vbs": float(vbs), "nfin": float(nfin),
                                "L": float(L), "T": float(T),
                            }, region))
    return grid


# ── load model ─────────────────────────────────────────────────────────────

def _build_inputs(
    op: Dict[str, float],
    stats: BSIMARNormStats,
) -> torch.Tensor:
    """Build a normalised 7-dim input from an op-point dict.

    Frame: model expects [Vd, Vg, Vs, Vb, NFIN_log, L, T] in that order.
    The op-point gives (Vgs, Vds, Vbs); we set Vs=0 and back out Vd, Vg,
    Vb so that Vds=Vd-Vs, Vgs=Vg-Vs, Vbs=Vb-Vs.
    """
    v_s = 0.0
    v_d = op["vds"] + v_s
    v_g = op["vgs"] + v_s
    v_b = op["vbs"] + v_s

    nfin_log = float(np.log2(max(op["nfin"], 1.0)))
    raw = np.array([v_d, v_g, v_s, v_b, nfin_log, op["L"], op["T"]],
                   dtype=np.float64)

    in_std = stats.input_std.copy()
    in_std[in_std < 1e-12] = 1.0
    norm = (raw - stats.input_mean) / in_std
    return torch.tensor(norm, dtype=torch.float32).unsqueeze(0)


def _model_output(
    model: DirectNet,
    x_norm: torch.Tensor,
    tech_code: int,
) -> torch.Tensor:
    """Forward pass at a single point, returns (1, 13) normalised output."""
    tc = torch.tensor([tech_code], dtype=torch.long, device=x_norm.device)
    return model(x_norm, tech_codes=tc)


# ── jacobian computation ────────────────────────────────────────────────────

# (output_idx, input_idx, name) — input_idx is normalised input column.
# 0=Vd, 1=Vg, 2=Vs, 3=Vb. The op-point fixes Vs=0, so:
#   ∂/∂Vgs = ∂/∂Vg (Vs constant)
#   ∂/∂Vds = ∂/∂Vd (Vs constant)
#   ∂/∂Vbs = ∂/∂Vb (Vs constant)
JAC_CHANNELS = [
    (0, 1, "did_dVgs"),    # gm
    (0, 0, "did_dVds"),    # gds
    (0, 3, "did_dVbs"),    # gmb
    (4, 1, "dqg_dVgs"),    # cgg
    (4, 0, "dqg_dVds"),    # cgd
    (4, 3, "dqg_dVbs"),    # cgs
    (5, 1, "dqd_dVgs"),    # cdg
    (5, 0, "dqd_dVds"),    # cdd
]


def autograd_jacobian(
    model: DirectNet,
    x_norm: torch.Tensor,
    tech_code: int,
) -> Dict[str, float]:
    """Return ∂(out_norm)/∂(in_norm) for the 8 channels, in normalised space.

    Operates entirely on the normalised model input — no inference-time
    Vds correction, no clamping.  This is the raw autograd Jacobian the
    NR solver would consume.
    """
    x = x_norm.clone().detach().requires_grad_(True)
    out = _model_output(model, x, tech_code)
    jac: Dict[str, float] = {}
    for out_idx, in_idx, name in JAC_CHANNELS:
        grad = torch.autograd.grad(
            out[0, out_idx], x, create_graph=False, retain_graph=True,
        )[0]
        jac[name] = float(grad[0, in_idx].item())
    return jac


def fd_jacobian(
    model: DirectNet,
    x_norm: torch.Tensor,
    tech_code: int,
    h: float = 1e-3,
) -> Dict[str, float]:
    """5-point central FD reference, in normalised space."""
    fd: Dict[str, float] = {}
    for out_idx, in_idx, name in JAC_CHANNELS:
        # 5-point stencil: f(x-2h), f(x-h), f(x+h), f(x+2h)
        coeffs = [(+2, -1.0 / 12.0), (+1, +8.0 / 12.0),
                  (-1, -8.0 / 12.0), (-2, +1.0 / 12.0)]
        deriv = 0.0
        for step, c in coeffs:
            xp = x_norm.clone()
            xp[0, in_idx] = xp[0, in_idx] + step * h
            with torch.no_grad():
                out = _model_output(model, xp, tech_code)
            deriv += c * float(out[0, out_idx].item())
        fd[name] = deriv / h
    return fd


# ── main ────────────────────────────────────────────────────────────────────

def load_directnet(
    ckpt_prefix: str,
) -> Tuple[DirectNet, BSIMARNormStats]:
    """Load a DirectNet checkpoint by ``<ckpt_prefix>_best.pt`` resolution."""
    ckpt_path = CHECKPOINT_DIR / f"{ckpt_prefix}_best.pt"
    norm_path = CHECKPOINT_DIR / f"{ckpt_prefix}_norm.npz"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"Norm stats not found: {norm_path}")

    state = torch.load(str(ckpt_path), weights_only=True, map_location="cpu")
    net_keys = sorted(
        [k for k in state.keys() if k.startswith("net.") and k.endswith(".weight")],
        key=lambda k: int(k.split(".")[1]),
    )
    output_dim = state[net_keys[-1]].shape[0]
    hidden_dim = state[net_keys[-1]].shape[1]
    n_layers = len(net_keys) - 1
    num_tech_codes = state["tech_embedding.weight"].shape[0]
    tech_embed_dim = state["tech_embedding.weight"].shape[1]
    input_dim = state[net_keys[0]].shape[1] - tech_embed_dim

    model = DirectNet(
        input_dim=input_dim, hidden_dim=hidden_dim, n_layers=n_layers,
        output_dim=output_dim, num_tech_codes=num_tech_codes,
        tech_embed_dim=tech_embed_dim,
    )
    model.load_state_dict(state)
    model.eval()
    stats = BSIMARNormStats.load(str(norm_path))
    return model, stats


def run_diagnostic(
    ckpt_prefix: str,
    polarity: str,
    techs: List[str],
    out_csv: Path,
    rel_tol: float = 0.10,
) -> Dict[str, float]:
    """Run the diagnostic on one checkpoint, write per-row CSV.

    Returns a summary dict with overall PASS/BAD counts per region.
    """
    model, stats = load_directnet(ckpt_prefix)

    # Estimate VDD from training input bounds (matches mosfet_directnet.py).
    vd_range = max(abs(float(stats.input_max[0])),
                   abs(float(stats.input_min[0])))
    vdd_train = vd_range / 2.0

    print(f"[{ckpt_prefix}] polarity={polarity}, "
          f"vdd_train≈{vdd_train:.3f}V, "
          f"techs={techs}")

    rows: List[Dict[str, object]] = []
    n_total = 0
    n_bad = {"ID": 0, "OOD": 0}
    n_seen = {"ID": 0, "OOD": 0}
    rel_err_sum = {"ID": 0.0, "OOD": 0.0}

    EPS = 1e-6  # absolute floor for relative-error normalisation

    for tech in techs:
        # Try svt first, fall back to lvt — both are representative in TSMC
        tech_code = tech_variant_to_code(tech.lower(), "svt")
        if tech_code == UNKNOWN_CODE_ID:
            tech_code = tech_variant_to_code(tech.lower(), "lvt")
        if tech_code == UNKNOWN_CODE_ID:
            print(f"  skip {tech}: no tech-code in vocab")
            continue

        grid = build_op_grid(vdd_train, polarity)
        for op, region in grid:
            x_norm = _build_inputs(op, stats)
            ag = autograd_jacobian(model, x_norm, tech_code)
            fd = fd_jacobian(model, x_norm, tech_code, h=1e-3)
            for ch_name in [c[2] for c in JAC_CHANNELS]:
                a = ag[ch_name]
                f = fd[ch_name]
                abs_err = abs(a - f)
                rel = abs_err / max(abs(f), EPS)
                flag = "BAD" if rel > rel_tol else "OK"
                rows.append({
                    "tech": tech, "v_gs": op["vgs"], "v_ds": op["vds"],
                    "v_bs": op["vbs"], "nfin": op["nfin"], "L": op["L"],
                    "region": region, "channel": ch_name,
                    "autograd": a, "fd": f, "abs_err": abs_err,
                    "rel_err": rel, "flag": flag,
                })
                n_total += 1
                n_seen[region] += 1
                rel_err_sum[region] += rel
                if flag == "BAD":
                    n_bad[region] += 1

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "tech", "v_gs", "v_ds", "v_bs", "nfin", "L",
            "region", "channel", "autograd", "fd",
            "abs_err", "rel_err", "flag",
        ])
        w.writeheader()
        w.writerows(rows)

    bad_pct_id = (100.0 * n_bad["ID"] / max(n_seen["ID"], 1))
    bad_pct_ood = (100.0 * n_bad["OOD"] / max(n_seen["OOD"], 1))
    mean_rel_id = rel_err_sum["ID"] / max(n_seen["ID"], 1)
    mean_rel_ood = rel_err_sum["OOD"] / max(n_seen["OOD"], 1)

    summary = {
        "ckpt": ckpt_prefix,
        "polarity": polarity,
        "n_total": n_total,
        "n_bad_id": n_bad["ID"],
        "n_seen_id": n_seen["ID"],
        "bad_pct_id": bad_pct_id,
        "n_bad_ood": n_bad["OOD"],
        "n_seen_ood": n_seen["OOD"],
        "bad_pct_ood": bad_pct_ood,
        "mean_rel_err_id": mean_rel_id,
        "mean_rel_err_ood": mean_rel_ood,
        "csv_path": str(out_csv),
    }
    print(f"  total cells: {n_total}, "
          f"BAD ID: {n_bad['ID']}/{n_seen['ID']} ({bad_pct_id:.1f}%), "
          f"BAD OOD: {n_bad['OOD']}/{n_seen['OOD']} ({bad_pct_ood:.1f}%), "
          f"mean_rel_err ID/OOD: {mean_rel_id:.4f}/{mean_rel_ood:.4f}")
    print(f"  CSV: {out_csv}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Checkpoint save_prefix, e.g. v4_dn_universal_nmos")
    ap.add_argument("--polarity", choices=["nmos", "pmos"], required=True)
    ap.add_argument("--techs", default="tsmc5,tsmc7,tsmc12,tsmc16",
                    help="Comma-separated TSMC techs to evaluate")
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "results" /
                            "v5_phase_c_c0_jacobian_diag")
    ap.add_argument("--rel-tol", type=float, default=0.10)
    ap.add_argument("--summary-md", type=Path,
                    default=PROJECT_ROOT / "results" /
                            "v5_phase_c_c0_jacobian_diag.md",
                    help="Append-mode markdown summary file")
    args = ap.parse_args()

    techs = [t.strip() for t in args.techs.split(",") if t.strip()]
    out_csv = args.out_dir / f"{args.checkpoint}.csv"
    summary = run_diagnostic(
        args.checkpoint, args.polarity, techs, out_csv,
        rel_tol=args.rel_tol,
    )

    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.summary_md.exists()
    with args.summary_md.open("a") as f:
        if new_file:
            f.write("# V5 Phase C — C0 FD-vs-autograd Jacobian diagnostic\n\n")
            f.write("Tolerance: BAD when |FD - autograd| > "
                    "rel_tol * max(|FD|, 1e-6).\n\n")
            f.write("| ckpt | polarity | n_total | BAD% (ID) | BAD% (OOD) "
                    "| mean rel.err (ID) | mean rel.err (OOD) |\n")
            f.write("|---|---|---|---|---|---|---|\n")
        f.write(f"| `{summary['ckpt']}` | {summary['polarity']} | "
                f"{summary['n_total']} | "
                f"{summary['bad_pct_id']:.1f} | "
                f"{summary['bad_pct_ood']:.1f} | "
                f"{summary['mean_rel_err_id']:.4f} | "
                f"{summary['mean_rel_err_ood']:.4f} |\n")

    # Also dump JSON for programmatic consumption
    json_path = out_csv.with_suffix(".json")
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
