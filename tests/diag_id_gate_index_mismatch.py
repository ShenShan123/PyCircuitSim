"""Diagnostic: reproduce the apply_id_gate index-mismatch (Bug A).

Loads the on-disk `v5c_universal_pmos_norm.npz`, builds a synthetic
single-row input matching the inverter operating point that exposed
the bug (Vd=-0.6, Vg=-0.5, Vs=0, Vb=0, NFIN=10, L=16e-9, T=300), runs
`apply_id_gate` in both the buggy form (`id_idx_in_stats=4`) and the
fixed form (`id_idx_in_stats=0`), and prints the gated id_phys
magnitudes side by side.

Verdict: buggy lands in charge-scale (~1e-16); fixed lands in
current-scale (~1e-5). The ~10⁹× gap reproduces plan §1A.

Run: ``python tests/diag_id_gate_index_mismatch.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from bsimar.data.normalize import BSIMARNormalizer, BSIMARNormStats
from bsimar.models.id_gate import apply_id_gate


CKPT_DIR = PROJECT_ROOT / "external_compact_models" / "bsimar" / "checkpoints"
NORM_PATH = CKPT_DIR / "v5c_universal_pmos_norm.npz"


def main() -> int:
    if not NORM_PATH.exists():
        print(f"SKIP: {NORM_PATH} not found")
        return 0

    stats = BSIMARNormStats.load(str(NORM_PATH))
    if stats.mode != "asinh":
        print(f"SKIP: norm mode={stats.mode!r}, expected asinh")
        return 0
    norm = BSIMARNormalizer(mode="asinh", stats=stats)

    print(f"Loaded stats from: {NORM_PATH.name}")
    print(f"  output_mean[0]    = {float(stats.output_mean[0]):+.4e} (id slot)")
    print(f"  output_std[0]     = {float(stats.output_std[0]):+.4e}")
    print(f"  asinh_scale[0]    = {float(stats.asinh_scale[0]):+.4e} (id)")
    print(f"  output_mean[4]    = {float(stats.output_mean[4]):+.4e} (qg slot)")
    print(f"  output_std[4]     = {float(stats.output_std[4]):+.4e}")
    print(f"  asinh_scale[4]    = {float(stats.asinh_scale[4]):+.4e} (qg)")

    # Synthetic input matching plan §1A: PMOS inverter operating point.
    vd, vg, vs, vb = -0.6, -0.5, 0.0, 0.0
    nfin_log = np.log2(10.0)
    L = 16e-9
    T = 300.0
    geo_phys = np.array([vd, vg, vs, vb, nfin_log, L, T], dtype=np.float64)
    in_std = stats.input_std.copy()
    in_std[in_std < 1e-12] = 1.0
    x_norm_np = (geo_phys - stats.input_mean) / in_std
    x = torch.from_numpy(x_norm_np[None, :])

    # Synthetic id_raw_norm at slot 4 (BSIMAR_COLUMN_ORDER).
    out_norm = np.zeros((1, 13), dtype=np.float64)
    out_norm[0, 4] = 0.5
    o = torch.from_numpy(out_norm)

    # Buggy form
    out_bug = apply_id_gate(
        x, o, norm,
        id_idx_in_output=4, id_idx_in_stats=4, vt_arch=0.04,
    )
    id_norm_bug = float(out_bug[0, 4].item())
    u_bug = id_norm_bug * float(stats.output_std[4]) + float(stats.output_mean[4])
    id_phys_bug = float(stats.asinh_scale[4]) * float(np.sinh(u_bug))

    # Fixed form
    out_fix = apply_id_gate(
        x, o, norm,
        id_idx_in_output=4, id_idx_in_stats=0, vt_arch=0.04,
    )
    id_norm_fix = float(out_fix[0, 4].item())
    u_fix = id_norm_fix * float(stats.output_std[0]) + float(stats.output_mean[0])
    id_phys_fix = float(stats.asinh_scale[0]) * float(np.sinh(u_fix))

    print()
    print(f"Input       : Vd={vd}, Vg={vg}, Vs={vs}, Vb={vb}, NFIN=10, L=16nm")
    print(f"id_raw_norm at slot 4 : 0.5")
    print()
    print(f"BUGGY  (id_idx_in_stats=4, qg slot): id_gated_phys = {id_phys_bug:+.4e} A")
    print(f"FIXED  (id_idx_in_stats=0, id slot): id_gated_phys = {id_phys_fix:+.4e} A")
    print()
    print(
        f"VERDICT: buggy: charge-scale ~1e-16 / fixed: current-scale ~1e-5 — "
        f"ratio |fixed/bug| = {abs(id_phys_fix) / max(abs(id_phys_bug), 1e-300):.3e}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
