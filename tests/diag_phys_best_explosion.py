"""Diagnostic: AR-rollout phys-metric comparison between `_best.pt`
and `_best.phys.pt` (or its renamed `.phys.bug.pt` form).

Reproduces plan §1B: prints the per-output NRMSE/R² table that exposed
Bug B (the mean-aggregated phys-score rewarding an early epoch whose
id slot was already broken under AR rollout).

Usage::

    python tests/diag_phys_best_explosion.py --prefix v5c_universal

Skips cleanly if any required file (config, norm, both checkpoints, or
the on-disk universal dataset) is missing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from bsimar.config import OUTPUT_COLUMNS  # noqa: E402
from bsimar.data.dataset import load_and_split_bsimar  # noqa: E402
from bsimar.data.normalize import (  # noqa: E402
    BSIMARNormStats, BSIMARNormalizer, reorder_outputs, unreorder_outputs,
)
from bsimar.eval.metrics import compute_physical_metrics  # noqa: E402
from bsimar.models.transformer import TransformerEncoderModel  # noqa: E402
from bsimar.training.trainer import test_model  # noqa: E402

CKPT_DIR = PROJECT_ROOT / "external_compact_models" / "bsimar" / "checkpoints"
DATASETS_DIR = (
    PROJECT_ROOT / "external_compact_models" / "bsimar" / "data" / "datasets")


def _maybe_path(p: Path) -> Optional[Path]:
    return p if p.exists() else None


def _build_model(config_path: Path, num_tech_codes: int) -> torch.nn.Module:
    cfg = np.load(str(config_path))
    return TransformerEncoderModel(
        input_dim=int(cfg["input_dim"]),
        target_dim=int(cfg["target_dim"]),
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_layers=int(cfg["num_layers"]),
        dim_feedforward=int(cfg["dim_feedforward"]),
        dropout=float(cfg["dropout"]),
        num_tech_codes=num_tech_codes,
    )


def _eval_checkpoint(
    ckpt_path: Path,
    config_path: Path,
    norm_path: Path,
    test_loader,
    normalizer: BSIMARNormalizer,
    device: torch.device,
    *,
    id_gate: bool,
    label: str,
) -> dict:
    cfg = np.load(str(config_path))
    num_tech_codes = (
        int(cfg["num_tech_codes"]) if "num_tech_codes" in cfg.files else 22)
    model = _build_model(config_path, num_tech_codes)
    state = torch.load(str(ckpt_path), weights_only=True, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    pred_norm, true_norm = test_model(
        model, test_loader, device,
        id_gate=id_gate, normalizer=normalizer)
    pred_norm = unreorder_outputs(pred_norm)
    true_norm = unreorder_outputs(true_norm)
    metrics = compute_physical_metrics(pred_norm, true_norm, normalizer)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="v5c_universal")
    parser.add_argument("--device-type", default="pmos",
                        choices=("nmos", "pmos"))
    args = parser.parse_args()

    prefix = args.prefix
    dt = args.device_type

    plain = _maybe_path(CKPT_DIR / f"{prefix}_{dt}_best.pt")
    phys = _maybe_path(CKPT_DIR / f"{prefix}_{dt}_best.phys.pt")
    phys_bug = _maybe_path(CKPT_DIR / f"{prefix}_{dt}_best.phys.bug.pt")
    norm = _maybe_path(CKPT_DIR / f"{prefix}_{dt}_norm.npz")
    config = _maybe_path(CKPT_DIR / f"{prefix}_{dt}_config.npz")
    dataset = _maybe_path(DATASETS_DIR / f"universal_{dt}.npz")

    # Use the renamed .phys.bug.pt if present, otherwise the live .phys.pt.
    phys_used = phys if phys is not None else phys_bug

    missing = []
    for name, p in [
        ("plain (_best.pt)", plain),
        ("phys (_best.phys.pt or .phys.bug.pt)", phys_used),
        ("norm.npz", norm),
        ("config.npz", config),
        ("dataset", dataset),
    ]:
        if p is None:
            missing.append(name)
    if missing:
        print(f"SKIP: missing {missing}")
        return 0

    print(f"prefix          : {prefix}")
    print(f"device_type     : {dt}")
    print(f"plain ckpt      : {plain.name}")
    print(f"phys ckpt       : {phys_used.name}")
    print(f"norm.npz        : {norm.name}")
    print(f"dataset         : {dataset.name}")

    stats = BSIMARNormStats.load(str(norm))
    print(f"phys_best_metric: {getattr(stats, 'phys_best_metric', 'n/a')}")
    print(f"id_gate         : {getattr(stats, 'id_gate', False)}")

    # Build test dataset with same loader the trainer uses.
    train_ds, val_ds, test_ds, normalizer = load_and_split_bsimar(
        str(dataset),
        column_names=OUTPUT_COLUMNS,
        device_type=dt,
        apply_filter=True,
        exclude_techs={"asap7"},
        norm_mode="asinh",
    )
    # Match BSIMAR layout: reorder targets.
    test_ds.outputs = torch.tensor(
        reorder_outputs(test_ds.outputs.numpy()), dtype=torch.float32)

    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_ds, batch_size=2048, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device          : {device}")

    id_gate_flag = bool(getattr(stats, "id_gate", False))
    print()
    print("Evaluating PLAIN (_best.pt)...")
    m_plain = _eval_checkpoint(
        plain, config, norm, test_loader, normalizer, device,
        id_gate=id_gate_flag, label="plain")
    print("Evaluating PHYS (_best.phys.pt)...")
    m_phys = _eval_checkpoint(
        phys_used, config, norm, test_loader, normalizer, device,
        id_gate=id_gate_flag, label="phys")

    # Comparison table per plan §1B.
    cols = list(m_plain.keys())
    print()
    header = (f"{'Output':>6s} | "
              f"{'NRMSE_plain%':>14s} | {'R2_plain':>10s} | "
              f"{'NRMSE_phys%':>14s} | {'R2_phys':>10s}")
    print(header)
    print("-" * len(header))
    nrmse_plain_arr, r2_plain_arr = [], []
    nrmse_phys_arr, r2_phys_arr = [], []
    for c in cols:
        np_ = float(m_plain[c]["NRMSE(%)"])
        rp = float(m_plain[c]["R2"])
        nh = float(m_phys[c]["NRMSE(%)"])
        rh = float(m_phys[c]["R2"])
        print(f"{c:>6s} | {np_:14.3e} | {rp:10.3e} | {nh:14.3e} | {rh:10.3e}")
        nrmse_plain_arr.append(np_)
        r2_plain_arr.append(rp)
        nrmse_phys_arr.append(nh)
        r2_phys_arr.append(rh)

    # Aggregates: mean (legacy bug) and median (fix).
    nrmse_p = np.array(nrmse_plain_arr, dtype=np.float64)
    r2_p = np.array(r2_plain_arr, dtype=np.float64)
    nrmse_h = np.array(nrmse_phys_arr, dtype=np.float64)
    r2_h = np.array(r2_phys_arr, dtype=np.float64)

    def _score(nrmse_arr, r2_arr, agg):
        n = float(agg(nrmse_arr))
        r = float(agg(r2_arr))
        return n + 0.1 * (1.0 - r), n, r

    print("-" * len(header))
    s_p_mean, n_p_mean, r_p_mean = _score(nrmse_p, r2_p, np.nanmean)
    s_h_mean, n_h_mean, r_h_mean = _score(nrmse_h, r2_h, np.nanmean)
    s_p_med, n_p_med, r_p_med = _score(nrmse_p, r2_p, np.nanmedian)
    s_h_med, n_h_med, r_h_med = _score(nrmse_h, r2_h, np.nanmedian)
    print(f"{'mean':>6s} | nrmse={n_p_mean:.3e} r2={r_p_mean:.3e} "
          f"score_plain={s_p_mean:.3e}  ||  "
          f"nrmse={n_h_mean:.3e} r2={r_h_mean:.3e} score_phys={s_h_mean:.3e}")
    print(f"{'median':>6s} | nrmse={n_p_med:.3e} r2={r_p_med:.3e} "
          f"score_plain={s_p_med:.3e}  ||  "
          f"nrmse={n_h_med:.3e} r2={r_h_med:.3e} score_phys={s_h_med:.3e}")

    print()
    if s_p_med < s_h_med:
        print(
            "VERDICT: median-based score correctly RANKS PLAIN < PHYS "
            "(plain is the better checkpoint, as expected).")
    else:
        print(
            "VERDICT: median-based score still ranks PHYS as better — "
            "investigate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
