"""V6.4.7 S10 (P4) — Sobolev id-derivative SIGN-CONVENTION verification.

The P0-I §2 trap: the autograd ∂id/∂V the Sobolev term supervises must be
mapped to the stored OSDI gm/gds/gmb columns with the CORRECT per-channel
sign before training, or the term pulls the surface the wrong way. The
deriv-fidelity scorer settled this empirically (uniform negation of all three
channels; float64-FD selfcheck < 0.5 %); this script confirms the LOSS's own
target transform (``bsimar.losses.SobolevIdLoss``) uses that same convention
by comparing, on strong-inversion rows of a *well-trained* control-v2 net:

    residual_uniform = median| autograd ∂id/∂V  -  (-target·chain) |
    residual_930     = median| autograd ∂id/∂V  -  (±target·chain) |   (gds NOT
                                                       flipped — the 930c274 rule)

A net that already value-matches OSDI gm/gds/gmb (control-v2: NMOS gds ~7 %)
must have residual_uniform << residual_930 for gds (the channel where the two
conventions disagree), and residual_uniform ~ residual_930 for gm/gmb (where
they agree). If uniform negation is right, gds residual_uniform is small while
residual_930 ≈ 2·|target| (sign-flipped → adds instead of cancels).

Usage:
    conda run -n pycircuitsim python scripts/v6_4_7_s10_sign_check.py \
        --nmos v6_4_7_ctlv2_s42_tsmc7_nmos --pmos v6_4_7_ctlv2_s42_tsmc7_pmos \
        --tech tsmc7 --data-suffix v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "external_compact_models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the gate's validated forward/autograd chain so the LOSS target is
# checked against exactly the surface the gate measures.
from v6_4_7_deriv_fidelity import (  # noqa: E402
    _build_directnet_from_state, _normalize_inputs_ck, _forward_id_grads,
    _denorm_id, CKPT_DIR, DATA_DIR,
)
from bsimar.losses.bni_mae import SOBOLEV_ID_CHANNELS  # noqa: E402
from bsimar.data.normalize import NormStats, OUTPUT_COLUMN_ORDER  # noqa: E402

_CHAN_TRUE_COL = {"gm": 1, "gds": 2, "gmb": 3}
_STRONG = 1e-6


def _check_device(tech: str, dev: str, stem: str, suffix: str | None) -> None:
    pt = CKPT_DIR / f"{stem}_best.pt"
    nz = CKPT_DIR / f"{stem}_norm.npz"
    infix = f"_{suffix}" if suffix else ""
    data_path = DATA_DIR / f"{tech.lower()}{infix}_{dev}.npz"
    state = torch.load(str(pt), weights_only=True, map_location="cpu")
    model = _build_directnet_from_state(state)
    stats = NormStats.load(str(nz))
    id_col = (stats.output_columns or OUTPUT_COLUMN_ORDER).index("id")

    data = np.load(data_path, allow_pickle=True)
    inputs, geometry, outputs = data["inputs"], data["geometry"], data["outputs"]
    from bsimar.eval.loo_labels import get_or_build_tech_variant_labels
    from bsimar.config import (CODE_TO_TECH_VARIANT, LOCAL_VARIANT_CODES,
                               LOCAL_UNKNOWN_CODE_ID)
    codes_u = np.asarray(get_or_build_tech_variant_labels(
        str(data_path), dev, verbose=False), dtype=np.int64)
    table = LOCAL_VARIANT_CODES[tech.lower()]
    unk = LOCAL_UNKNOWN_CODE_ID[tech.lower()]
    codes = np.array([table.get(CODE_TO_TECH_VARIANT.get(int(c), ("", "")), unk)
                      for c in codes_u], dtype=np.int64)

    strong = np.flatnonzero(np.abs(outputs[:, 0]) > _STRONG)
    rng = np.random.default_rng(0)
    pick = strong[rng.choice(len(strong), size=min(20000, len(strong)),
                             replace=False)]

    x_norm = _normalize_inputs_ck(stats, inputs[pick], geometry[pick])
    idn, gn = _forward_id_grads(model, x_norm, codes[pick], id_col,
                                torch.device("cpu"))
    id_phys_pred, factor = _denorm_id(stats, idn, id_col)
    in_std = stats.input_std.copy(); in_std[in_std < 1e-12] = 1.0
    out_std_id = float(stats.output_std[id_col])

    print(f"\n[{dev}] stem={stem}  n_strong_rows={len(pick)}")
    print(f"  {'chan':5s} {'med|grad|':>11s} {'res_uniform':>12s} "
          f"{'res_930':>11s}  verdict")
    for tgt_name, vcol in SOBOLEV_ID_CHANNELS:
        tcol = _CHAN_TRUE_COL[tgt_name]
        # npz outputs are PHYSICAL; map to normalized-derivative units via the
        # same asinh chain factor the loss uses (in_std/out_std_id/factor).
        tgt_phys = outputs[pick, tcol].astype(np.float64)
        chain = in_std[vcol] / out_std_id / factor
        grad = gn[:, vcol].astype(np.float64)
        tgt_uniform = -tgt_phys * chain
        sign_930 = -1.0 if tgt_name in ("gm", "gmb") else +1.0
        tgt_930 = sign_930 * tgt_phys * chain
        res_u = float(np.median(np.abs(grad - tgt_uniform)))
        res_9 = float(np.median(np.abs(grad - tgt_930)))
        med_g = float(np.median(np.abs(grad)))
        if abs(res_u - res_9) <= 1e-9 * max(res_u, res_9, 1e-30):
            verdict = "tie (conventions agree)"
        elif res_u < res_9:
            verdict = "uniform ✓"
        else:
            verdict = "930 ✓ (!! investigate)"
        print(f"  {tgt_name:5s} {med_g:11.4e} {res_u:12.4e} "
              f"{res_9:11.4e}  {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tech", required=True)
    ap.add_argument("--nmos", required=True)
    ap.add_argument("--pmos", required=True)
    ap.add_argument("--data-suffix", default=None)
    args = ap.parse_args()
    _check_device(args.tech, "nmos", args.nmos, args.data_suffix)
    _check_device(args.tech, "pmos", args.pmos, args.data_suffix)
    print("\nExpect: gds res_uniform << res_930 (the two conventions only "
          "disagree on gds); gm/gmb ~equal. 'uniform ✓' on gds confirms the "
          "SobolevIdLoss sign map.")


if __name__ == "__main__":
    main()
