"""Diagnostic: NN drive current vs L72, on-grid versus between grid knots.

**Not a gate.** This is the measurement that identified the V7.4.2 root
cause, kept so the claim can be re-checked against any checkpoint set.

The NN datasets sample L at PDK length-bin corners. Wherever a bin's
interior is unsampled, nothing constrains the fit between knots — and
capacity makes that unconstrained interpolant drift rather than converge.
Run against V7.4.0 checkpoints, TSMC7 NMOS lands within 0.2 % at the
sampled 8 / 11 / 20 / 36 nm and is 7.5 -> 13.2 % weak (small -> xl) at the
benchmark's 16 nm. A weak NMOS lengthens an inverter's falling edge, which
is exactly the ring-oscillator period bias.

Ground truth is BSIM-CMG through the OSDI binary (never an analytic
approximation), evaluated over the corridor a falling edge integrates:
gate at the rail, drain swept VDD -> VDD/2.

Usage:
    conda run -n pycircuitsim python tests/diag/diag_l_grid_interp.py \
        --tech tsmc7 --tag tf
    ... --tiers small,large --lengths 8,11,13,16,20,36
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))
sys.path.insert(0, str(PROJECT_ROOT))

from tests.common.complex import BENCH  # noqa: E402
from bsimar.config import (  # noqa: E402
    CHECKPOINT_DIR, TECH_CONFIGS, local_variant_code,
)
from bsimar.data.normalize import (  # noqa: E402
    BSIMAR_COLUMN_ORDER, OUTPUT_COLUMN_ORDER, NormStats,
)
from pycmg.nn_generate import (  # noqa: E402
    _create_model_and_instance, eval_single_point,
)
from pycmg.parser import _scan_all_variants  # noqa: E402
from pycmg.tech import _resolve_path  # noqa: E402

DEFAULT_TIERS = ("small", "medium", "large", "xl")
_AR = {c: i for i, c in enumerate(BSIMAR_COLUMN_ORDER)}
_STD = {c: i for i, c in enumerate(OUTPUT_COLUMN_ORDER)}


def _load(stem: str, device: torch.device):
    """Return (module, NormStats, output_layout) for a checkpoint stem."""
    state = torch.load(str(CHECKPOINT_DIR / f"{stem}_best.pt"),
                       weights_only=True, map_location=device)
    cfg_path = CHECKPOINT_DIR / f"{stem}_config.npz"
    if cfg_path.exists():
        from bsimar.models.transformer import TransformerEncoderModel
        c = np.load(str(cfg_path))
        model = TransformerEncoderModel(
            input_dim=int(c["input_dim"]), target_dim=int(c["target_dim"]),
            d_model=int(c["d_model"]), nhead=int(c["nhead"]),
            num_layers=int(c["num_layers"]),
            dim_feedforward=int(c["dim_feedforward"]),
            dropout=float(c["dropout"]),
            num_tech_codes=int(c["num_tech_codes"]),
            unknown_code_id=int(c["num_tech_codes"]) - 1)
        layout = "bsimar"
    else:
        # DirectNet ships no arch sidecar — the simulator shape-infers it.
        # Reuse that exact inference so this diagnostic and the solver
        # rebuild the same module (incl. the optional mono./core. paths).
        from bsimar.models.direct_net import DirectNet
        net_keys = [k for k in state
                    if k.startswith("net.") and k.endswith(".weight")]
        emb = state["tech_embedding.weight"]
        tech_embed_dim = int(emb.shape[1])
        model = DirectNet(
            input_dim=int(state[net_keys[0]].shape[1]) - tech_embed_dim,
            hidden_dim=int(state[net_keys[-1]].shape[1]),
            n_layers=len(net_keys) - 1,
            output_dim=int(state[net_keys[-1]].shape[0]),
            num_tech_codes=int(emb.shape[0]),
            tech_embed_dim=tech_embed_dim,
            monotonic=any(k.startswith("mono.") for k in state),
            monotone_sign=(float(state["mono.sign"].item())
                           if "mono.sign" in state else 1.0),
            monotone_hidden=(int(state["mono.w_vg_raw"].shape[0])
                             if "mono.w_vg_raw" in state else 64),
            ekv_core=any(k.startswith("core.") for k in state),
            ekv_hidden=(int(state["core.param_head.0.weight"].shape[0])
                        if "core.param_head.0.weight" in state else 64))
        layout = "standard"
    model.load_state_dict(state)
    model.to(device).eval()
    return model, NormStats.load(str(CHECKPOINT_DIR / f"{stem}_norm.npz")), layout


def _nn_id(model, stats: NormStats, layout: str, device: torch.device,
           tech_code: int, vg: np.ndarray, vd: np.ndarray, vb: np.ndarray,
           nfin: float, L: float, T: float) -> np.ndarray:
    """Physical id (A). Mirrors mosfet_nn's input prep and asinh denorm."""
    raw = np.zeros((len(vg), 7), dtype=np.float64)
    raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3] = vd, vg, 0.0, vb
    raw[:, 4] = np.log2(max(nfin, 1.0))          # normalize._build_combined_input
    raw[:, 5], raw[:, 6] = L, T
    in_std = np.asarray(stats.input_std, dtype=np.float64)
    x = torch.tensor(
        (raw - np.asarray(stats.input_mean)) / np.where(in_std < 1e-12, 1.0,
                                                        in_std),
        dtype=torch.float32, device=device)
    codes = torch.full((len(vg),), tech_code, dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(x, tech_codes=codes).double().cpu().numpy()
    col = _AR["id"] if layout == "bsimar" else _STD["id"]
    cols = list(stats.output_columns or OUTPUT_COLUMN_ORDER)
    j = cols.index("id")
    return (np.asarray(stats.asinh_scale)[j]
            * np.sinh(out[:, col] * np.asarray(stats.output_std)[j]
                      + np.asarray(stats.output_mean)[j]))


def _sampled_lengths(tech: str, dev: str, vt: str) -> Tuple[np.ndarray,
                                                            List[Tuple]]:
    """(L knots present in the dataset, PDK bin [lmin, lmax] list)."""
    from bsimar.config import DATA_DIR
    path = DATA_DIR / f"{tech}_{dev}.npz"
    knots = np.array([])
    if path.exists():
        with np.load(str(path), allow_pickle=True) as d:
            knots = np.unique(d["geometry"][:, 1])
    cfg = TECH_CONFIGS[tech]
    pdk = str(_resolve_path(str(cfg.pycmg_tech.pdk_path)))
    dev_cfg = cfg.pycmg_tech.get_device(f"{dev}_{vt}")
    bins = [(v.lmin, v.lmax)
            for v in _scan_all_variants(pdk, dev_cfg.pdk_device)]
    return knots, bins


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tech", default="tsmc7")
    p.add_argument("--dev", default="nmos")
    p.add_argument("--tag", default="tf", help="tf (LEVEL=74) or dn (73)")
    p.add_argument("--tiers", default=",".join(DEFAULT_TIERS))
    p.add_argument("--lengths", default="",
                   help="Comma-separated L in nm; default = the tech's own "
                        "sampled knots plus the benchmark length")
    p.add_argument("--points", type=int, default=24)
    a = p.parse_args()

    bt = BENCH[a.tech.upper()]
    vt = bt.effective_nmos_vt if a.dev == "nmos" else bt.effective_pmos_vt
    nfin = float(bt.nfin if a.dev == "nmos" else bt.effective_nfin_p)
    L_bench = bt.l_nmos if a.dev == "nmos" else bt.l_pmos
    vdd, T = bt.vdd, 300.15
    tiers = [t for t in a.tiers.split(",") if t]

    knots, _bins = _sampled_lengths(a.tech, a.dev, vt)
    if a.lengths:
        lengths = [float(z) * 1e-9 for z in a.lengths.split(",")]
    else:
        lengths = sorted(set(knots.tolist()) | {L_bench})
        lengths = [L for L in lengths if L <= 40e-9]

    def _on_grid(L: float) -> bool:
        """Within 0.1 % of a sampled knot.

        Exact float equality would mislabel a hand-typed --lengths value
        (7.63 vs the grid's 7.6296...) as off-grid, which is precisely the
        distinction this diagnostic exists to report.
        """
        return bool(len(knots)) and bool(
            np.min(np.abs(knots - L)) <= 1e-3 * L)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 96)
    print(f"L-grid interpolation: {a.tech} {a.dev} {vt} NFIN={nfin:g} "
          f"T=27C, Vg=VDD={vdd}, Vd swept VDD -> VDD/2   [{a.tag}]")
    print(f"  sampled L knots (nm): "
          f"{sorted(round(k * 1e9, 2) for k in knots.tolist())}")
    print(f"  benchmark L = {L_bench * 1e9:.0f} nm")
    print("=" * 96)

    vd = np.linspace(vdd, 0.5 * vdd, a.points)
    vg = np.full_like(vd, vdd)
    vb = np.zeros_like(vd)
    code = local_variant_code(a.tech, a.tech, vt)
    models: Dict[str, Tuple] = {}
    for t in tiers:
        stem = f"{a.tech}_{a.tag}_{t}_{a.dev}"
        if not (CHECKPOINT_DIR / f"{stem}_best.pt").exists():
            print(f"  [skip] {stem}: no checkpoint")
            continue
        models[t] = _load(stem, device)
    if not models:
        print("No checkpoints found — nothing to compare.")
        return 1

    have = list(models)
    print(f"\n{'L (nm)':>8s} {'on grid':>8s} | {'L72 mean |id|':>13s} | "
          + " | ".join(f"{t:>12s}" for t in have))
    print(f"{'':>8s} {'':>8s} | {'':>13s} | "
          + " | ".join(f"{'mean err %':>12s}" for _ in have))
    print("-" * (35 + 15 * len(have)))
    cfg = TECH_CONFIGS[a.tech]
    for L in lengths:
        built = _create_model_and_instance(cfg, a.dev, vt, L, nfin, T)
        if built is None:
            print(f"{L * 1e9:8.2f}   (L72 bin unstable — skipped)")
            continue
        inst = built[1]
        ref = np.array([
            (lambda r: np.nan if r is None else r["id"])(
                eval_single_point(inst, float(vd[k]), float(vg[k]), 0.0,
                                  float(vb[k]), _silent=True))
            for k in range(len(vd))])
        good = np.isfinite(ref) & (np.abs(ref) > 1e-9)
        if not good.any():
            continue
        cells = []
        for t in have:
            m, st, lay = models[t]
            pred = _nn_id(m, st, lay, device, code, vg, vd, vb, nfin, L, T)
            e = ((np.abs(pred[good]) - np.abs(ref[good]))
                 / np.abs(ref[good]))
            cells.append(f"{100 * e.mean():+12.2f}")
        on = "yes" if _on_grid(L) else "NO"
        mark = "  <-- benchmark" if abs(L - L_bench) < 1e-13 else ""
        print(f"{L * 1e9:8.2f} {on:>8s} | {np.abs(ref[good]).mean():13.4e} | "
              + " | ".join(cells) + mark)

    print("\nA large off-grid error that GROWS with tier is the V7.4.2 "
          "signature:\ncapacity converging the knots while the unconstrained "
          "interpolant drifts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
