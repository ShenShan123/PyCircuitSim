"""Smoke tests for SlopeMatchLoss (B2 v5 Phase B).

Verifies that the slope-match loss is near zero when the model's
autograd-derived ``dId/dVg`` matches the data's ``gm`` exactly, in
both ``zscore`` and ``asinh`` normaliser modes. Also checks that an
all-non-grid batch returns zero.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# Bootstrap the bsimar package onto sys.path (mirrors tests/common/nn.py).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXTERNAL_DIR = PROJECT_ROOT / "external_compact_models"
for _p in (PROJECT_ROOT, _EXTERNAL_DIR):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

from bsimar.data.normalize import (  # noqa: E402
    BSIMARNormalizer,
    OUTPUT_COLUMN_ORDER,
)
from bsimar.losses import SlopeMatchLoss  # noqa: E402


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_synthetic_batch(
    n: int = 64,
    a: float = 1.5e-4,
    b: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic linear (id = a*Vg + b) NMOS-like batch.

    Returns (inputs[N,4], geometry[N,15], outputs[N,13]) in physical
    units. The id, gm columns are populated; the rest are filled with
    physically plausible values to exercise normalisation but are not
    used by the slope loss.
    """
    rng = np.random.default_rng(42)
    Vd = rng.uniform(0.0, 0.7, n)
    Vg = rng.uniform(0.0, 0.7, n)
    Vs = np.zeros(n)
    Vb = np.zeros(n)
    inputs = np.column_stack([Vd, Vg, Vs, Vb])

    NFIN = np.full(n, 10.0)
    L = np.full(n, 16e-9)
    T = np.full(n, 300.0)
    proc = np.zeros((n, 12))
    geometry = np.column_stack([NFIN, L, T, proc])

    id_phys = a * Vg + b
    gm_phys = np.full(n, a)
    gds_phys = np.full(n, 1e-6)
    gmb_phys = np.full(n, 1e-7)
    qg = np.full(n, 1e-17)
    qd = np.full(n, 1e-17)
    qs = np.full(n, 1e-17)
    qb = np.full(n, 1e-17)
    cgg = np.full(n, 1e-18)
    cgd = np.full(n, 1e-18)
    cgs = np.full(n, 1e-18)
    cdg = np.full(n, 1e-18)
    cdd = np.full(n, 1e-18)

    cols = [id_phys, gm_phys, gds_phys, gmb_phys,
            qg, qd, qs, qb, cgg, cgd, cgs, cdg, cdd]
    outputs = np.column_stack(cols)
    assert [c for c in OUTPUT_COLUMN_ORDER] == [
        "id", "gm", "gds", "gmb", "qg", "qd", "qs", "qb",
        "cgg", "cgd", "cgs", "cdg", "cdd"]
    return inputs, geometry, outputs


class _LinearIdModel(nn.Module):
    """Tiny model whose normalised id is exactly target_slope_norm * x_vg.

    Concretely: out[..., id_idx] = slope_norm * x[..., vg_idx].
    All other output columns are filled with the corresponding target
    column so the slope-loss-internal denormalisation of (id, gm) sees
    the exact target value.
    """

    def __init__(self, slope_norm: float, id_idx: int):
        super().__init__()
        self.slope_norm = float(slope_norm)
        self.id_idx = int(id_idx)
        # Learnable param so autograd can flow but we won't actually
        # update; we want a deterministic prediction.
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x_norm: torch.Tensor,
        target_norm: torch.Tensor,
    ) -> torch.Tensor:
        # Build a prediction that equals target everywhere EXCEPT for the
        # id column, where we put slope_norm * x_vg + 0.0. We use the
        # target's gm (id+1) so the slope loss can use it as ground truth.
        pred = target_norm.clone() + 0.0 * self.dummy  # connect to graph
        # Replace id column with the linear function of x_vg.
        vg = x_norm[:, 1]
        new_id = self.slope_norm * vg
        pred = pred.clone()
        pred[:, self.id_idx] = new_id
        return pred


# ── Tests ───────────────────────────────────────────────────────────────────


def _setup(mode: str):
    inputs, geometry, outputs = _make_synthetic_batch(n=128, a=2.0e-4)
    normalizer = BSIMARNormalizer(mode=mode)
    normalizer.fit(inputs, geometry, outputs)
    x_norm = normalizer.normalize_inputs(inputs, geometry)
    y_norm = normalizer.normalize_outputs(outputs)
    return inputs, geometry, outputs, normalizer, x_norm, y_norm


def _expected_target_slope(
    mode: str,
    normalizer: BSIMARNormalizer,
    id_phys: float,
    gm_phys: float,
) -> float:
    id_idx = OUTPUT_COLUMN_ORDER.index("id")
    vg_std = float(normalizer.stats.input_std[1])
    id_std = float(normalizer.stats.output_std[id_idx])
    if mode == "zscore":
        return (vg_std / id_std) * gm_phys
    s_id = float(normalizer.stats.asinh_scale[id_idx])
    return vg_std * gm_phys / (id_std * np.sqrt(s_id ** 2 + id_phys ** 2))


def test_slope_loss_zscore_near_zero():
    """zscore mode: when pred slope == target slope, loss is ~0."""
    mode = "zscore"
    _, _, _, normalizer, x_norm, y_norm = _setup(mode)

    # The data is exactly id = a*Vg + b with constant gm = a, so for
    # zscore mode the *normalised* target slope is the same constant
    # (vg_std/id_std)*a for every row. Use that as the model's slope.
    a = 2.0e-4
    b = 0.0
    target_slope_norm = _expected_target_slope(mode, normalizer, b, a)

    id_idx = 0  # OUTPUT_COLUMN_ORDER -> id at index 0
    model = _LinearIdModel(slope_norm=target_slope_norm, id_idx=id_idx)

    sample_class = torch.full((x_norm.shape[0],), 4, dtype=torch.int8)

    x_t = torch.tensor(x_norm, dtype=torch.float32, requires_grad=True)
    y_t = torch.tensor(y_norm, dtype=torch.float32)

    pred = model(x_t, y_t)

    loss_fn = SlopeMatchLoss(
        normalizer=normalizer, mode=mode,
        id_idx_in_output=id_idx, vg_idx_in_input=1,
        max_samples=128,
    )
    # With b=0 (constant id_phys=0) this is exact in zscore mode.
    # However, our synthetic id_phys = a*Vg varies across rows; in
    # zscore mode the slope is independent of id_phys so the loss is
    # still exact.
    loss = loss_fn(x_t, pred, y_t, sample_class)
    assert loss.item() < 1e-4, f"Expected ~0 loss, got {loss.item()}"


def test_slope_loss_asinh_near_zero():
    """asinh mode: per-row target slope reproduces dependence on id_phys."""
    mode = "asinh"
    _, _, outputs, normalizer, x_norm, y_norm = _setup(mode)

    # For asinh mode the target slope depends on id_phys, so a single
    # constant model slope cannot match every row. Instead, build a
    # model whose normalised id value reproduces the *normalised target*
    # exactly: the slope of the prediction in normalised space equals
    # the target slope in normalised space, by construction.
    # Strategy: set pred[:, id_idx] = (vg_std/id_std)*gm_phys / row_factor * x_vg
    # ... that's awkward. Instead, use a model that returns
    # pred_id_norm(x) = c0 + slope_per_row * x_vg, where slope_per_row
    # is computed from ground truth. Then the autograd slope d(pred)/d(x_vg)
    # is slope_per_row.
    id_idx = OUTPUT_COLUMN_ORDER.index("id")
    a = 2.0e-4
    # Per-row target slope (asinh chain rule).
    id_phys = outputs[:, id_idx]
    gm_phys = np.full(len(outputs), a)
    vg_std = float(normalizer.stats.input_std[1])
    id_std = float(normalizer.stats.output_std[id_idx])
    s_id = float(normalizer.stats.asinh_scale[id_idx])
    target_slope_per_row = (
        vg_std * gm_phys / (id_std * np.sqrt(s_id ** 2 + id_phys ** 2))
    )

    class _PerRowSlopeModel(nn.Module):
        def __init__(self, slopes: np.ndarray, id_idx: int):
            super().__init__()
            self.slopes = nn.Parameter(
                torch.tensor(slopes, dtype=torch.float32),
                requires_grad=False,
            )
            self.id_idx = id_idx
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, x_norm, target_norm):
            pred = target_norm.clone() + 0.0 * self.dummy
            vg = x_norm[:, 1]
            pred = pred.clone()
            pred[:, self.id_idx] = self.slopes * vg
            return pred

    model = _PerRowSlopeModel(target_slope_per_row, id_idx)
    sample_class = torch.full((x_norm.shape[0],), 4, dtype=torch.int8)

    x_t = torch.tensor(x_norm, dtype=torch.float32, requires_grad=True)
    y_t = torch.tensor(y_norm, dtype=torch.float32)

    pred = model(x_t, y_t)
    loss_fn = SlopeMatchLoss(
        normalizer=normalizer, mode=mode,
        id_idx_in_output=id_idx, vg_idx_in_input=1,
        max_samples=128,
    )
    loss = loss_fn(x_t, pred, y_t, sample_class)
    assert loss.item() < 1e-4, f"Expected ~0 loss, got {loss.item()}"


def test_slope_loss_no_grid_returns_zero():
    """All-non-grid batch returns torch.zeros."""
    mode = "zscore"
    _, _, _, normalizer, x_norm, y_norm = _setup(mode)

    id_idx = 0
    model = _LinearIdModel(slope_norm=0.5, id_idx=id_idx)
    # All rows are class 5 (hot), not 4 (grid).
    sample_class = torch.full((x_norm.shape[0],), 5, dtype=torch.int8)

    x_t = torch.tensor(x_norm, dtype=torch.float32, requires_grad=True)
    y_t = torch.tensor(y_norm, dtype=torch.float32)

    pred = model(x_t, y_t)

    loss_fn = SlopeMatchLoss(
        normalizer=normalizer, mode=mode,
        id_idx_in_output=id_idx, vg_idx_in_input=1,
        max_samples=128,
    )
    loss = loss_fn(x_t, pred, y_t, sample_class)
    assert loss.item() == 0.0, f"Expected exact 0 (no grid rows), got {loss.item()}"


def test_slope_loss_requires_grad():
    """Forgetting requires_grad on x raises a clear error."""
    mode = "zscore"
    _, _, _, normalizer, x_norm, y_norm = _setup(mode)

    id_idx = 0
    model = _LinearIdModel(slope_norm=0.5, id_idx=id_idx)
    sample_class = torch.full((x_norm.shape[0],), 4, dtype=torch.int8)

    x_t = torch.tensor(x_norm, dtype=torch.float32, requires_grad=False)
    y_t = torch.tensor(y_norm, dtype=torch.float32)

    pred = model(x_t, y_t)
    loss_fn = SlopeMatchLoss(
        normalizer=normalizer, mode=mode,
        id_idx_in_output=id_idx, vg_idx_in_input=1,
        max_samples=128,
    )
    with pytest.raises(RuntimeError, match="requires_grad"):
        loss_fn(x_t, pred, y_t, sample_class)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
