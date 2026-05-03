"""Phys-score robustness tests (Phase B-B1 / Bug B regression guard).

Asserts that the median-based phys-score is dominated by the bulk of
well-behaved outputs, while the mean-based phys-score is destroyed by
a single outlier (the id-column AR-rollout sinh blowup discovered
2026-05-03 in plan §2B).

Also (when the on-disk v5c PMOS checkpoint pair is present) asserts
that the median score correctly ranks `_best.pt` lower than the buggy
`_best.phys.pt` — i.e. the fix really would have selected the right
checkpoint.
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


def _phys_score(nrmse_arr: np.ndarray, r2_arr: np.ndarray, agg) -> float:
    n = float(agg(nrmse_arr))
    r = float(agg(r2_arr))
    if np.isnan(n) or np.isnan(r):
        return float("inf")
    return n + 0.1 * (1.0 - r)


def test_median_robust_to_id_blowup():
    """Synthesise 13 per-output NRMSE/R² values where 12 are well-behaved
    and one (the id slot at index 0) blows up. Median must be dominated
    by the 12 well-behaved outputs; mean must be dominated by the
    outlier.
    """
    rng = np.random.default_rng(0)
    nrmse = rng.uniform(0.05, 1.0, size=13)   # all sub-1 %
    r2 = rng.uniform(0.99, 1.0, size=13)
    nrmse[0] = 1e7    # id blow-up: 10 million %
    r2[0] = -1e10     # negative R² as in plan §1B

    score_med = _phys_score(nrmse, r2, np.nanmedian)
    score_mean = _phys_score(nrmse, r2, np.nanmean)

    assert score_med < 5.0, (
        f"median score should be dominated by well-behaved outputs, "
        f"got {score_med:e}")
    assert score_mean > 1e6, (
        f"mean score should be destroyed by the outlier, "
        f"got {score_mean:e}")
    # Sanity: ratio must be huge.
    assert score_mean / score_med > 1e5


def test_median_ranks_v5c_pmos_plain_below_phys_bug():
    """Real-data check: with the on-disk v5c PMOS checkpoint pair, the
    median-based score on AR-rollout test predictions must rank
    `_best.pt` lower (better) than the (buggy) `_best.phys.pt` snapshot.

    Skips cleanly if any of the four required files is missing.
    """
    ckpt_dir = (
        PROJECT_ROOT / "external_compact_models" / "bsimar"
        / "checkpoints")
    plain = ckpt_dir / "v5c_universal_pmos_best.pt"
    phys_renamed = ckpt_dir / "v5c_universal_pmos_best.phys.bug.pt"
    phys_live = ckpt_dir / "v5c_universal_pmos_best.phys.pt"
    norm = ckpt_dir / "v5c_universal_pmos_norm.npz"
    config = ckpt_dir / "v5c_universal_pmos_config.npz"

    phys = phys_renamed if phys_renamed.exists() else phys_live
    dataset = (
        PROJECT_ROOT / "external_compact_models" / "bsimar" / "data"
        / "datasets" / "universal_pmos.npz")
    if not (plain.exists() and phys.exists() and norm.exists()
            and config.exists() and dataset.exists()):
        pytest.skip("v5c PMOS checkpoint set + dataset not present")

    from bsimar.config import OUTPUT_COLUMNS
    from bsimar.data.dataset import load_and_split_bsimar
    from bsimar.data.normalize import (
        reorder_outputs, unreorder_outputs,
    )
    from bsimar.eval.metrics import compute_physical_metrics
    from bsimar.models.transformer import TransformerEncoderModel
    from bsimar.training.trainer import test_model

    cfg = np.load(str(config))
    num_tech_codes = (
        int(cfg["num_tech_codes"]) if "num_tech_codes" in cfg.files else 22)

    _, _, test_ds, normalizer = load_and_split_bsimar(
        str(dataset),
        column_names=OUTPUT_COLUMNS,
        device_type="pmos",
        apply_filter=True,
        exclude_techs={"asap7"},
        norm_mode="asinh",
    )
    test_ds.outputs = torch.tensor(
        reorder_outputs(test_ds.outputs.numpy()), dtype=torch.float32)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _score_for(ckpt_path: Path) -> float:
        model = TransformerEncoderModel(
            input_dim=int(cfg["input_dim"]),
            target_dim=int(cfg["target_dim"]),
            d_model=int(cfg["d_model"]),
            nhead=int(cfg["nhead"]),
            num_layers=int(cfg["num_layers"]),
            dim_feedforward=int(cfg["dim_feedforward"]),
            dropout=float(cfg["dropout"]),
            num_tech_codes=num_tech_codes,
        )
        state = torch.load(
            str(ckpt_path), weights_only=True, map_location="cpu")
        model.load_state_dict(state)
        model.to(device).eval()
        pred_norm, true_norm = test_model(model, test_loader, device)
        pred_norm = unreorder_outputs(pred_norm)
        true_norm = unreorder_outputs(true_norm)
        m = compute_physical_metrics(pred_norm, true_norm, normalizer)
        nrmse_arr = np.array(
            [v["NRMSE(%)"] for v in m.values()], dtype=np.float64)
        r2_arr = np.array(
            [v["R2"] for v in m.values()], dtype=np.float64)
        return _phys_score(nrmse_arr, r2_arr, np.nanmedian)

    s_plain = _score_for(plain)
    s_phys = _score_for(phys)
    print(f"median score: plain={s_plain:.4e}  phys={s_phys:.4e}")
    assert s_plain < s_phys, (
        f"median score should rank plain ({s_plain:.4e}) below "
        f"phys ({s_phys:.4e})")
