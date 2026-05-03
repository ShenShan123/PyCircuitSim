"""Unit tests for the structural Vds gate (B3, Sprint S-ARCH-A).

Verifies:
  (a) Id_gated(Vds=0) == 0 to numerical zero (zscore mode).
  (b) Same property for asinh mode.
  (c) Gate is monotone in |Vds|.
  (d) DirectNet output of the id slot post-gate matches the formula.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from bsimar.data.normalize import BSIMARNormalizer, BSIMARNormStats
from bsimar.models.id_gate import apply_id_gate


N_OUTPUTS = 13
ID_IDX_DIRECT = 0
ID_IDX_BSIMAR = 4
VT_ARCH = 0.04


def _zscore_stats(seed: int = 0) -> BSIMARNormStats:
    """Synthetic zscore stats with non-trivial mean/std for inputs+outputs."""
    rng = np.random.default_rng(seed)
    in_mean = rng.normal(size=7).astype(np.float64) * 0.1
    in_std = (rng.uniform(0.5, 2.0, size=7)).astype(np.float64)
    in_min = in_mean - 3.0 * in_std
    in_max = in_mean + 3.0 * in_std
    out_mean = rng.normal(size=N_OUTPUTS).astype(np.float64) * 1e-5
    out_std = (rng.uniform(1e-6, 1e-3, size=N_OUTPUTS)).astype(np.float64)
    return BSIMARNormStats(
        mode="zscore",
        output_mean=out_mean, output_std=out_std,
        input_mean=in_mean, input_std=in_std,
        input_min=in_min, input_max=in_max,
    )


def _asinh_stats(seed: int = 1) -> BSIMARNormStats:
    """Synthetic asinh+zscore stats with non-trivial scale."""
    rng = np.random.default_rng(seed)
    in_mean = rng.normal(size=7).astype(np.float64) * 0.1
    in_std = (rng.uniform(0.5, 2.0, size=7)).astype(np.float64)
    in_min = in_mean - 3.0 * in_std
    in_max = in_mean + 3.0 * in_std
    asinh_scale = (10.0 ** rng.uniform(-19.0, -3.0, size=N_OUTPUTS)).astype(np.float64)
    out_mean = rng.normal(size=N_OUTPUTS).astype(np.float64) * 0.5
    out_std = (rng.uniform(0.5, 1.5, size=N_OUTPUTS)).astype(np.float64)
    return BSIMARNormStats(
        mode="asinh",
        output_mean=out_mean, output_std=out_std,
        input_mean=in_mean, input_std=in_std,
        input_min=in_min, input_max=in_max,
        asinh_scale=asinh_scale,
    )


def _denorm_id(val_norm: float, stats: BSIMARNormStats, idx: int) -> float:
    u = float(val_norm) * float(stats.output_std[idx]) + float(stats.output_mean[idx])
    if stats.mode == "asinh":
        return float(stats.asinh_scale[idx]) * float(np.sinh(u))
    return u


# ── (a) zscore: Id_gated(Vds=0) == 0 to numerical zero ──────────────────────

def test_id_gated_zero_at_vds_zero_zscore():
    stats = _zscore_stats(seed=0)
    norm = BSIMARNormalizer(mode="zscore", stats=stats)

    rng = np.random.default_rng(42)
    B = 32
    # The simplest way to guarantee Vds_phys == 0 *exactly* under
    # float32 arithmetic is to use the same value for Vd_n and Vs_n
    # AND make sure (Vd_n*std+mean) for both columns equals each other.
    # We achieve this by setting Vd_n and Vs_n such that the resulting
    # Vd_phys == Vs_phys at float64 precision, then casting to float64
    # tensors to avoid the float32 ULP drift in Vd_phys - Vs_phys.
    x_np = rng.standard_normal((B, 7)).astype(np.float64)
    a = x_np[:, 0]
    x_np[:, 2] = (
        (a * stats.input_std[0] + stats.input_mean[0] - stats.input_mean[2])
        / stats.input_std[2]
    )

    x = torch.from_numpy(x_np)  # float64 tensor
    out_norm = torch.from_numpy(rng.standard_normal((B, N_OUTPUTS)).astype(np.float64))

    out_gated = apply_id_gate(
        x, out_norm, norm,
        id_idx_in_output=ID_IDX_DIRECT, vt_arch=VT_ARCH,
    )

    # With float64 inputs, Vds_phys == 0 holds to ~1e-16 and the gate
    # is exactly tanh(0) = 0 → id_gated_phys == 0 to numerical zero.
    id_norm_out = out_gated[:, ID_IDX_DIRECT].detach().cpu().numpy()
    id_phys = id_norm_out * stats.output_std[ID_IDX_DIRECT] + stats.output_mean[ID_IDX_DIRECT]
    np.testing.assert_allclose(id_phys, 0.0, atol=1e-12)


# ── (b) asinh: Id_gated(Vds=0) == 0 to numerical zero ───────────────────────

def test_id_gated_zero_at_vds_zero_asinh():
    stats = _asinh_stats(seed=1)
    norm = BSIMARNormalizer(mode="asinh", stats=stats)

    rng = np.random.default_rng(43)
    B = 32
    # See zscore test for why float64.
    x_np = rng.standard_normal((B, 7)).astype(np.float64)
    a = x_np[:, 0]
    x_np[:, 2] = (
        (a * stats.input_std[0] + stats.input_mean[0] - stats.input_mean[2])
        / stats.input_std[2]
    )

    x = torch.from_numpy(x_np)
    # Use BSIMAR id index (4) here to also exercise that path.
    out_norm = torch.from_numpy(rng.standard_normal((B, N_OUTPUTS)).astype(np.float64))

    out_gated = apply_id_gate(
        x, out_norm, norm,
        id_idx_in_output=ID_IDX_BSIMAR, vt_arch=VT_ARCH,
    )

    # Denormalised (asinh chain) id should be exactly zero.
    id_norm_out = out_gated[:, ID_IDX_BSIMAR].detach().cpu().numpy().astype(np.float64)
    u = id_norm_out * stats.output_std[ID_IDX_BSIMAR] + stats.output_mean[ID_IDX_BSIMAR]
    id_phys = stats.asinh_scale[ID_IDX_BSIMAR] * np.sinh(u)
    # asinh(0) = 0 → u = out_mean / out_std * out_std + out_mean? No: when
    # id_gated_phys = 0, asinh(0/s) = 0 → u_target = 0 →
    # id_norm_out = (0 - out_mean) / out_std. Then the line above recovers
    # u = -out_mean + out_mean = 0 → id_phys = s * sinh(0) = 0. Good.
    np.testing.assert_allclose(id_phys, 0.0, atol=1e-12)


# ── (c) Gate is monotone in |Vds| (synthetic monotone-id model) ─────────────

def test_gate_monotone_in_abs_vds():
    """For a synthetic model whose id_phys is constant >0 (or <0), the
    gated id_phys magnitude must be monotone non-decreasing in |Vds|."""
    stats = _zscore_stats(seed=2)
    norm = BSIMARNormalizer(mode="zscore", stats=stats)

    # Pick a single non-Vd/Vs row and sweep Vd_n only (Vs_n fixed at 0).
    base = np.zeros(7, dtype=np.float32)
    # Set Vs_n so Vs_phys = 0
    base[2] = -float(stats.input_mean[2]) / float(stats.input_std[2])

    n = 401
    vds_phys = np.linspace(-2.0, 2.0, n).astype(np.float64)
    # Vd_n such that Vd_phys = vds_phys (since Vs_phys=0, Vds_phys = Vd_phys).
    vd_n = (vds_phys - stats.input_mean[0]) / stats.input_std[0]
    x_np = np.tile(base, (n, 1))
    x_np[:, 0] = vd_n.astype(np.float32)

    # Synthetic id slot: a constant in normalised space corresponding to
    # +1e-4 A in physical space.
    target_id_phys = 1e-4
    id_norm_const = (target_id_phys - stats.output_mean[ID_IDX_DIRECT]) / stats.output_std[ID_IDX_DIRECT]
    out_norm = np.zeros((n, N_OUTPUTS), dtype=np.float32)
    out_norm[:, ID_IDX_DIRECT] = id_norm_const

    x = torch.from_numpy(x_np)
    o = torch.from_numpy(out_norm)
    out_gated = apply_id_gate(
        x, o, norm, id_idx_in_output=ID_IDX_DIRECT, vt_arch=VT_ARCH,
    )

    id_norm_out = out_gated[:, ID_IDX_DIRECT].detach().cpu().numpy()
    id_phys_out = (
        id_norm_out * stats.output_std[ID_IDX_DIRECT]
        + stats.output_mean[ID_IDX_DIRECT]
    )

    # Split sweep into negative-Vds half and positive-Vds half; magnitude
    # should be monotone non-decreasing in |Vds| on each half.
    mid = n // 2
    neg_half = np.abs(id_phys_out[:mid + 1][::-1])  # |Vds| increasing
    pos_half = np.abs(id_phys_out[mid:])
    assert np.all(np.diff(neg_half) >= -1e-9), "neg-half not monotone"
    assert np.all(np.diff(pos_half) >= -1e-9), "pos-half not monotone"


# ── (d) Hand-computed match for one row (zscore + asinh) ────────────────────

def test_formula_zscore_one_row():
    stats = _zscore_stats(seed=3)
    norm = BSIMARNormalizer(mode="zscore", stats=stats)

    # Single row, fixed values.
    x_row = np.array([0.5, 0.0, -0.3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    out_row = np.linspace(-1.0, 1.0, N_OUTPUTS, dtype=np.float32)

    x = torch.from_numpy(x_row).unsqueeze(0)
    o = torch.from_numpy(out_row).unsqueeze(0)
    out_gated = apply_id_gate(
        x, o, norm, id_idx_in_output=ID_IDX_DIRECT, vt_arch=VT_ARCH,
    )

    # Hand compute.
    vd_phys = x_row[0] * stats.input_std[0] + stats.input_mean[0]
    vs_phys = x_row[2] * stats.input_std[2] + stats.input_mean[2]
    vds = vd_phys - vs_phys
    gate = np.tanh(vds / VT_ARCH)
    id_raw_phys = (
        out_row[ID_IDX_DIRECT] * stats.output_std[ID_IDX_DIRECT]
        + stats.output_mean[ID_IDX_DIRECT]
    )
    id_gated_phys_expected = id_raw_phys * gate
    id_gated_norm_expected = (
        id_gated_phys_expected - stats.output_mean[ID_IDX_DIRECT]
    ) / stats.output_std[ID_IDX_DIRECT]

    actual = float(out_gated[0, ID_IDX_DIRECT].item())
    np.testing.assert_allclose(actual, id_gated_norm_expected, atol=1e-5, rtol=1e-5)

    # Other columns must be passed through unchanged.
    for j in range(N_OUTPUTS):
        if j == ID_IDX_DIRECT:
            continue
        assert float(out_gated[0, j].item()) == pytest.approx(float(out_row[j]))


def test_formula_asinh_one_row():
    stats = _asinh_stats(seed=4)
    norm = BSIMARNormalizer(mode="asinh", stats=stats)

    x_row = np.array([0.7, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    out_row = np.linspace(-0.8, 0.8, N_OUTPUTS, dtype=np.float32)

    x = torch.from_numpy(x_row).unsqueeze(0)
    o = torch.from_numpy(out_row).unsqueeze(0)
    out_gated = apply_id_gate(
        x, o, norm, id_idx_in_output=ID_IDX_BSIMAR, vt_arch=VT_ARCH,
    )

    vd_phys = x_row[0] * stats.input_std[0] + stats.input_mean[0]
    vs_phys = x_row[2] * stats.input_std[2] + stats.input_mean[2]
    vds = vd_phys - vs_phys
    gate = np.tanh(vds / VT_ARCH)
    id_idx = ID_IDX_BSIMAR
    # With the post-fix default `id_idx_in_stats=0`, the gate must read
    # id's stats from the OUTPUT_COLUMN_ORDER index 0 — independent of
    # ID_IDX_BSIMAR=4.
    stats_idx = 0
    s_id = stats.asinh_scale[stats_idx]
    u = (out_row[id_idx] * stats.output_std[stats_idx]
         + stats.output_mean[stats_idx])
    id_raw_phys = s_id * np.sinh(u)
    id_gated_phys_expected = id_raw_phys * gate
    id_gated_norm_expected = (
        np.arcsinh(id_gated_phys_expected / s_id) - stats.output_mean[stats_idx]
    ) / stats.output_std[stats_idx]

    actual = float(out_gated[0, id_idx].item())
    np.testing.assert_allclose(actual, id_gated_norm_expected, atol=1e-5, rtol=1e-5)

    # Other columns passed through unchanged.
    for j in range(N_OUTPUTS):
        if j == id_idx:
            continue
        assert float(out_gated[0, j].item()) == pytest.approx(float(out_row[j]))


# ── (e) BSIMAR layout: stats lookup must NOT use the model-output index ─────

def _bsimar_layout_stats(seed: int = 7) -> BSIMARNormStats:
    """asinh stats with id-like scale at index 0 and qg-like at index 4.

    Mirrors the real `v5c_universal_pmos_norm.npz` shape that triggered
    Bug A in the 2026-05-03 plan:
      [0] id : asinh_scale ~5.5e-5  (current scale)
      [4] qg : asinh_scale ~1.1e-16 (charge scale)
    Other indices get arbitrary middle-of-the-road values.
    """
    rng = np.random.default_rng(seed)
    in_mean = rng.normal(size=7).astype(np.float64) * 0.1
    in_std = (rng.uniform(0.5, 2.0, size=7)).astype(np.float64)
    in_min = in_mean - 3.0 * in_std
    in_max = in_mean + 3.0 * in_std
    asinh_scale = np.full(N_OUTPUTS, 1e-10, dtype=np.float64)
    asinh_scale[0] = 5.5e-5    # id (current scale)
    asinh_scale[4] = 1.1e-16   # qg (charge scale, ~10⁹× smaller)
    out_mean = np.zeros(N_OUTPUTS, dtype=np.float64)
    out_mean[0] = 1.213
    out_mean[4] = -1.030
    out_std = np.ones(N_OUTPUTS, dtype=np.float64)
    out_std[0] = 2.076
    out_std[4] = 1.287
    return BSIMARNormStats(
        mode="asinh",
        output_mean=out_mean, output_std=out_std,
        input_mean=in_mean, input_std=in_std,
        input_min=in_min, input_max=in_max,
        asinh_scale=asinh_scale,
    )


def test_bsimar_gate_uses_stats_index_zero_for_id():
    """Fixed form: id_idx_in_output=4, id_idx_in_stats=0 — gated id must
    land in the current-scale (~5.5e-5 A range), not the charge-scale.
    """
    stats = _bsimar_layout_stats(seed=7)
    norm = BSIMARNormalizer(mode="asinh", stats=stats)

    rng = np.random.default_rng(11)
    B = 64
    # Pick Vds_phys ≈ -0.6V (PMOS-like) and a non-trivial out_norm.
    x_np = rng.standard_normal((B, 7)).astype(np.float64)
    # Force Vds_phys = -0.6 deterministically: Vd_phys = -0.6, Vs_phys = 0.
    x_np[:, 0] = (-0.6 - stats.input_mean[0]) / stats.input_std[0]
    x_np[:, 2] = (0.0 - stats.input_mean[2]) / stats.input_std[2]
    # Use moderate id_raw_norm at slot 4 (BSIMAR layout).
    out_norm = np.zeros((B, N_OUTPUTS), dtype=np.float64)
    out_norm[:, 4] = 0.5  # id slot in BSIMAR_COLUMN_ORDER

    x = torch.from_numpy(x_np)
    o = torch.from_numpy(out_norm)
    out_gated_fixed = apply_id_gate(
        x, o, norm,
        id_idx_in_output=4, id_idx_in_stats=0, vt_arch=VT_ARCH,
    )
    id_norm_out = (
        out_gated_fixed[:, 4].detach().cpu().numpy().astype(np.float64))
    # Denormalise via index 0 (the right stats slot).
    u = id_norm_out * stats.output_std[0] + stats.output_mean[0]
    id_phys = stats.asinh_scale[0] * np.sinh(u)
    # Magnitude must be in current scale (~ 1e-7 .. 1e-3 A range).
    max_abs = float(np.max(np.abs(id_phys)))
    assert max_abs > 1e-7, (
        f"fixed gate produced max|id|={max_abs:e}, expected current scale")
    assert max_abs < 1e-2, (
        f"fixed gate produced max|id|={max_abs:e}, expected current scale")


def test_bsimar_gate_buggy_form_lands_in_charge_scale():
    """Regression guard: the *buggy* call form (id_idx_in_stats=4, the
    qg slot) reconstructs id with the charge-scale ~1.1e-16, ~10⁹×
    smaller than the true current scale. This test asserts the bug
    would be caught if anyone re-introduces it.
    """
    stats = _bsimar_layout_stats(seed=7)
    norm = BSIMARNormalizer(mode="asinh", stats=stats)

    rng = np.random.default_rng(11)
    B = 64
    x_np = rng.standard_normal((B, 7)).astype(np.float64)
    x_np[:, 0] = (-0.6 - stats.input_mean[0]) / stats.input_std[0]
    x_np[:, 2] = (0.0 - stats.input_mean[2]) / stats.input_std[2]
    out_norm = np.zeros((B, N_OUTPUTS), dtype=np.float64)
    out_norm[:, 4] = 0.5

    x = torch.from_numpy(x_np)
    o = torch.from_numpy(out_norm)
    out_gated_bug = apply_id_gate(
        x, o, norm,
        id_idx_in_output=4, id_idx_in_stats=4,  # buggy: qg slot
        vt_arch=VT_ARCH,
    )
    id_norm_out = (
        out_gated_bug[:, 4].detach().cpu().numpy().astype(np.float64))
    # Denormalise via index 4 (the slot the buggy call used).
    u = id_norm_out * stats.output_std[4] + stats.output_mean[4]
    id_phys_charge = stats.asinh_scale[4] * np.sinh(u)
    max_abs = float(np.max(np.abs(id_phys_charge)))
    # Must be in charge-scale (~1e-19 .. 1e-13 C range).
    assert max_abs < 1e-12, (
        f"buggy gate produced max|id|={max_abs:e}, expected charge scale")


# ── Smoke: gate works end-to-end with DirectNet random model ────────────────

def test_directnet_end_to_end_gate_zero_vds():
    """Run a randomly-initialised DirectNet through apply_id_gate and check
    that Id(Vds=0) = 0 still holds."""
    from bsimar.models.direct_net import DirectNet

    torch.manual_seed(0)
    stats = _zscore_stats(seed=5)
    norm = BSIMARNormalizer(mode="zscore", stats=stats)

    model = DirectNet(
        input_dim=7, hidden_dim=32, n_layers=2, output_dim=N_OUTPUTS,
        num_tech_codes=4, tech_embed_dim=4, tech_embed_dropout=0.0,
    )
    model.eval()

    B = 16
    rng = np.random.default_rng(99)
    # The DirectNet model runs in float32; we still get the structural
    # property to within a few ULPs of Vds residual times the model
    # output scale. Loosened tolerance reflects this float32 ceiling.
    x_np = rng.standard_normal((B, 7)).astype(np.float64)
    a = x_np[:, 0]
    x_np[:, 2] = (
        (a * stats.input_std[0] + stats.input_mean[0] - stats.input_mean[2])
        / stats.input_std[2]
    )
    x = torch.from_numpy(x_np.astype(np.float32))
    tc = torch.zeros(B, dtype=torch.long)

    with torch.no_grad():
        out = model(x, tech_codes=tc)
        out_gated = apply_id_gate(
            x, out, norm, id_idx_in_output=ID_IDX_DIRECT, vt_arch=VT_ARCH,
        )
    id_norm_out = out_gated[:, ID_IDX_DIRECT].cpu().numpy()
    id_phys = (
        id_norm_out * stats.output_std[ID_IDX_DIRECT]
        + stats.output_mean[ID_IDX_DIRECT]
    )
    # float32 ULP drift in Vd_phys - Vs_phys × output_std → ~1e-9 .. 1e-7.
    np.testing.assert_allclose(id_phys, 0.0, atol=1e-7)
