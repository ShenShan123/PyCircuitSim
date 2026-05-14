"""Evaluate the 8 V6.2.1 per-tech DirectNet checkpoints on their test splits.

Reports per-target NRMSE, MRE, Max-MRE, R² in physical units, plus
normalized-space metrics (no inverse-transform conditioning).

Outputs:
    results/v6_2_1_metrics_report/report.md
    results/v6_2_1_metrics_report/per_target_<tech>_<size>_<dev>.csv
    results/v6_2_1_metrics_report/summary.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))

from bsimar.config import (  # noqa: E402
    CHECKPOINT_DIR,
    DATA_DIR,
    OUTPUT_COLUMNS,
    tech_scope_vocab_size,
)
from bsimar.data.dataset import load_and_split_bsimar  # noqa: E402
from bsimar.data.normalize import NormStats, normalizer_for  # noqa: E402
from bsimar.eval.metrics import compute_physical_metrics  # noqa: E402
from bsimar.models.direct_net import DirectNet  # noqa: E402

OUTPUT_COLUMN_ORDER = OUTPUT_COLUMNS  # 13 targets


# Mirrors the (model, size) preset from bsimar/cli/train.py.
SIZE_ARCH = {
    "small":  dict(trunk_hidden=128, trunk_layers=3, batch_size=2048),
    "medium": dict(trunk_hidden=256, trunk_layers=5, batch_size=2048),
}

TECHS = ("tsmc5", "tsmc7", "tsmc12", "tsmc16")
SIZES = ("small", "medium")
DEVICES = ("nmos", "pmos")

REPORT_DIR = PROJECT_ROOT / "results" / "v6_2_1_metrics_report"


def _all_tech_names() -> List[str]:
    return ["asap7", "tsmc5", "tsmc7", "tsmc12", "tsmc16"]


def _build_model(
    in_dim: int, out_dim: int, size: str, num_tech_codes: int,
) -> DirectNet:
    arch = SIZE_ARCH[size]
    return DirectNet(
        input_dim=in_dim,
        hidden_dim=arch["trunk_hidden"],
        n_layers=arch["trunk_layers"] + 1,
        output_dim=out_dim,
        num_tech_codes=num_tech_codes,
        tech_embed_dim=32,
        tech_embed_dropout=0.0,
        unknown_code_id=num_tech_codes - 1,
    )


@torch.no_grad()
def _collect(
    model: DirectNet, loader: DataLoader, device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for x, y, tc in loader:
        x, tc = x.to(device), tc.to(device)
        preds.append(model(x, tech_codes=tc).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def _max_mre_per_target(
    pred_norm: np.ndarray, true_norm: np.ndarray, normalizer,
    mre_threshold_pct: float = 0.001,
) -> Dict[str, float]:
    """Worst-case per-sample MRE (%) on the same valid mask as the average."""
    pred_phys = normalizer.denormalize_outputs(pred_norm)
    true_phys = normalizer.denormalize_outputs(true_norm)
    cols = (normalizer.stats.output_columns
            if normalizer.stats.output_columns else OUTPUT_COLUMN_ORDER)
    out: Dict[str, float] = {}
    for i, name in enumerate(cols):
        y_t, y_p = true_phys[:, i], pred_phys[:, i]
        peak = float(np.abs(y_t).max())
        if peak == 0:
            out[name] = float("nan")
            continue
        mask = np.abs(y_t) > peak * mre_threshold_pct
        if not mask.any():
            out[name] = float("nan")
            continue
        rel = np.abs((y_t[mask] - y_p[mask]) / y_t[mask]) * 100.0
        out[name] = float(rel.max())
    return out


def evaluate_one(
    tech: str, size: str, device_type: str, device: torch.device,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], int]:
    save_prefix = f"{tech}_dn_{size}_{device_type}"
    ckpt = CHECKPOINT_DIR / f"{save_prefix}_best.pt"
    norm_path = CHECKPOINT_DIR / f"{save_prefix}_norm.npz"
    data_path = DATA_DIR / f"{tech}_{device_type}.npz"
    if not (ckpt.exists() and norm_path.exists() and data_path.exists()):
        raise FileNotFoundError(
            f"Missing one of: {ckpt}, {norm_path}, {data_path}")

    # Reproduce the trainer's split (seed=42, train=0.8, val=0.1).
    exclude = {t for t in _all_tech_names() if t != tech}
    print(f"\n=== {save_prefix} ===")
    train_ds, val_ds, test_ds, normalizer = load_and_split_bsimar(
        str(data_path), list(OUTPUT_COLUMN_ORDER), device_type=device_type,
        norm_mode="asinh", train_ratio=0.8, val_ratio=0.1, seed=42,
        apply_filter=True, exclude_techs=exclude, tech_scope=tech,
    )
    # Replace with the saved normalizer stats so we evaluate against the
    # exact transform used at training time (defensive — should match).
    saved_stats = NormStats.load(str(norm_path))
    normalizer = normalizer_for(saved_stats.mode)
    normalizer.stats = saved_stats

    # Rebuild normalized inputs/outputs with the saved stats. The split
    # indices are deterministic via seed=42 so test_ds points at the
    # right rows; we just renormalize on top.
    test_ds.inputs = torch.tensor(
        normalizer.normalize_inputs(
            normalizer.denormalize_inputs(test_ds.inputs.numpy())
            if hasattr(normalizer, "denormalize_inputs") else
            test_ds.inputs.numpy(),
            np.zeros((len(test_ds.inputs), 0)),
        ),
        dtype=torch.float32,
    ) if False else test_ds.inputs  # noqa: E501  (skip — splits are identical)

    in_dim = test_ds.inputs.shape[1]
    out_dim = test_ds.outputs.shape[1]
    num_tech_codes = tech_scope_vocab_size(tech)

    model = _build_model(in_dim, out_dim, size, num_tech_codes).to(device)
    model.load_state_dict(torch.load(str(ckpt), weights_only=True, map_location=device))

    loader = DataLoader(test_ds, batch_size=4096, shuffle=False,
                        num_workers=0, pin_memory=False)
    pred_norm, true_norm = _collect(model, loader, device)

    metrics = compute_physical_metrics(pred_norm, true_norm, normalizer)
    max_mre = _max_mre_per_target(pred_norm, true_norm, normalizer)
    return metrics, max_mre, len(test_ds)


def _write_per_target_csv(
    path: Path, metrics: Dict[str, Dict[str, float]],
    max_mre: Dict[str, float],
) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "target", "NRMSE(%)", "MRE(%)", "MaxMRE(%)", "R2",
            "NRMSE_norm(%)", "R2_norm", "MAE_norm",
            "n_valid", "n_total",
        ])
        for name, m in metrics.items():
            w.writerow([
                name,
                f"{m['NRMSE(%)']:.4f}",
                f"{m['MRE(%)']:.3f}",
                f"{max_mre.get(name, float('nan')):.2f}",
                f"{m['R2']:.6f}",
                f"{m['NRMSE_norm(%)']:.4f}",
                f"{m['R2_norm']:.6f}",
                f"{m['MAE_norm']:.6f}",
                m["n_valid"], m["n_total"],
            ])


def _aggregate(
    metrics: Dict[str, Dict[str, float]], max_mre: Dict[str, float],
) -> Dict[str, float]:
    def _mean(key: str) -> float:
        vals = [m[key] for m in metrics.values()
                if not np.isnan(m[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def _id_only(key: str) -> float:
        return metrics["id"][key]

    max_mre_avg_vals = [v for v in max_mre.values() if not np.isnan(v)]
    return {
        "NRMSE_avg(%)": _mean("NRMSE(%)"),
        "MRE_avg(%)": _mean("MRE(%)"),
        "MaxMRE_avg(%)": float(np.mean(max_mre_avg_vals))
        if max_mre_avg_vals else float("nan"),
        "R2_avg": _mean("R2"),
        "NRMSE_norm_avg(%)": _mean("NRMSE_norm(%)"),
        "R2_norm_avg": _mean("R2_norm"),
        # Id is the headline target for circuit simulation.
        "Id_NRMSE(%)": _id_only("NRMSE(%)"),
        "Id_MRE(%)": _id_only("MRE(%)"),
        "Id_MaxMRE(%)": max_mre.get("id", float("nan")),
        "Id_R2": _id_only("R2"),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eval device: {device}")

    rows: List[Tuple[str, str, str, int, Dict[str, float]]] = []
    detail: Dict[str, Tuple[Dict[str, Dict[str, float]], Dict[str, float]]] = {}

    for tech in TECHS:
        for size in SIZES:
            for dev in DEVICES:
                try:
                    metrics, max_mre, n_test = evaluate_one(
                        tech, size, dev, device)
                except FileNotFoundError as e:
                    print(f"\n=== {tech}_dn_{size}_{dev} === SKIP "
                          f"(missing artifact: {e})")
                    continue
                agg = _aggregate(metrics, max_mre)
                rows.append((tech, size, dev, n_test, agg))
                tag = f"{tech}_{size}_{dev}"
                detail[tag] = (metrics, max_mre)
                _write_per_target_csv(
                    REPORT_DIR / f"per_target_{tag}.csv",
                    metrics, max_mre)
                print(f"  Id: NRMSE={agg['Id_NRMSE(%)']:.3f}% "
                      f"MRE={agg['Id_MRE(%)']:.2f}% "
                      f"MaxMRE={agg['Id_MaxMRE(%)']:.1f}% "
                      f"R²={agg['Id_R2']:.4f} | "
                      f"avg R²={agg['R2_avg']:.4f}")

    # ── summary.csv ──────────────────────────────────────────────────
    with (REPORT_DIR / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tech", "size", "device", "n_test",
            "NRMSE_avg(%)", "MRE_avg(%)", "MaxMRE_avg(%)", "R2_avg",
            "NRMSE_norm_avg(%)", "R2_norm_avg",
            "Id_NRMSE(%)", "Id_MRE(%)", "Id_MaxMRE(%)", "Id_R2",
        ])
        for tech, size, dev, n_test, agg in rows:
            w.writerow([
                tech, size, dev, n_test,
                f"{agg['NRMSE_avg(%)']:.4f}",
                f"{agg['MRE_avg(%)']:.3f}",
                f"{agg['MaxMRE_avg(%)']:.2f}",
                f"{agg['R2_avg']:.6f}",
                f"{agg['NRMSE_norm_avg(%)']:.4f}",
                f"{agg['R2_norm_avg']:.6f}",
                f"{agg['Id_NRMSE(%)']:.4f}",
                f"{agg['Id_MRE(%)']:.3f}",
                f"{agg['Id_MaxMRE(%)']:.2f}",
                f"{agg['Id_R2']:.6f}",
            ])

    # ── report.md ────────────────────────────────────────────────────
    lines: List[str] = []
    lines.append("# DirectNet Per-tech Metrics Report (all TSMC techs)\n")
    lines.append(
        "Test-split metrics for every shipping V6.2/V6.2.1 dedicated "
        "DirectNet checkpoint across TSMC5/7/12/16. Combos without a "
        "checkpoint on disk (currently TSMC7 small) are skipped. "
        "Split is deterministic (seed=42, train/val/test = 0.8/0.1/0.1).\n")
    lines.append(
        "Physical-space NRMSE/MRE/MaxMRE/R² use the per-target valid mask "
        "(|y_true| > 0.1% of peak |y|, the PyCMG numerical-noise floor). "
        "Normalized-space metrics use the full set (no mask). "
        "**MaxMRE** is the worst-case per-sample MRE on the same valid mask "
        "and is dominated by near-floor samples that pass the threshold.\n")

    lines.append("\n## Summary (averaged over 13 targets)\n")
    lines.append(
        "| Tech | Size | Dev | n_test | NRMSE% | MRE% | MaxMRE% | R² | "
        "Id NRMSE% | Id MRE% | Id MaxMRE% | Id R² |\n"
        "|------|------|-----|-------:|-------:|-----:|--------:|---:|"
        "---------:|--------:|-----------:|------:|")
    for tech, size, dev, n_test, agg in rows:
        lines.append(
            f"| {tech.upper()} | {size} | {dev} | {n_test} | "
            f"{agg['NRMSE_avg(%)']:.3f} | {agg['MRE_avg(%)']:.2f} | "
            f"{agg['MaxMRE_avg(%)']:.1f} | {agg['R2_avg']:.4f} | "
            f"{agg['Id_NRMSE(%)']:.3f} | {agg['Id_MRE(%)']:.2f} | "
            f"{agg['Id_MaxMRE(%)']:.1f} | {agg['Id_R2']:.4f} |")

    lines.append("\n## Normalized-space metrics (full set, no mask)\n")
    lines.append(
        "| Tech | Size | Dev | NRMSE_norm% | R²_norm |\n"
        "|------|------|-----|------------:|--------:|")
    for tech, size, dev, _n, agg in rows:
        lines.append(
            f"| {tech.upper()} | {size} | {dev} | "
            f"{agg['NRMSE_norm_avg(%)']:.3f} | "
            f"{agg['R2_norm_avg']:.4f} |")

    lines.append("\n## Per-target breakdown (production size = `medium`)\n")
    for tech in TECHS:
        for dev in DEVICES:
            tag = f"{tech}_medium_{dev}"
            if tag not in detail:
                continue
            metrics, max_mre = detail[tag]
            lines.append(f"\n### {tech.upper()} / medium / {dev.upper()}\n")
            lines.append(
                "| Target | NRMSE% | MRE% | MaxMRE% | R² | n_valid/n_total |\n"
                "|--------|-------:|-----:|--------:|---:|----------------:|")
            for name, m in metrics.items():
                nrmse = m["NRMSE(%)"]
                mre = m["MRE(%)"]
                r2 = m["R2"]
                mxm = max_mre.get(name, float("nan"))
                nv, nt = m["n_valid"], m["n_total"]
                lines.append(
                    f"| {name} | "
                    f"{'N/A' if np.isnan(nrmse) else f'{nrmse:.3f}'} | "
                    f"{'N/A' if np.isnan(mre) else f'{mre:.2f}'} | "
                    f"{'N/A' if np.isnan(mxm) else f'{mxm:.1f}'} | "
                    f"{'N/A' if np.isnan(r2) else f'{r2:.4f}'} | "
                    f"{nv}/{nt} |")

    lines.append("\n## Files in this report\n")
    lines.append("- `summary.csv` — one row per (tech, size, dev)\n"
                 "- `per_target_<tech>_<size>_<dev>.csv` — full 13-target "
                 "breakdown per cell\n"
                 "- `report.md` — this file")

    (REPORT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {REPORT_DIR}/report.md")


if __name__ == "__main__":
    main()
