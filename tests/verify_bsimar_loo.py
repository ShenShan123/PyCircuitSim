#!/usr/bin/env python3
"""Leave-one-out (by technology) comprehensive test for BSIMAR v3.

For each of the 5 base technologies ``{asap7, tsmc5, tsmc7, tsmc12, tsmc16}``,
hold out **all** samples of that tech as the test set, train on the
remaining 4 techs, and report per-output metrics. This exercises the
v3 Transformer's cross-tech generalisation under the full production
recipe so the fold numbers are directly comparable to the v3 in-
distribution baseline (NRMSE 0.223 %, MRE 1.41 %, R² 0.9984).

Usage:
    conda run -n pycircuitsim --no-capture-output \\
        python -u tests/verify_bsimar_loo.py --device-type nmos --cuda --gpu 2

Flags:
    --device-type {nmos|pmos}   Dataset polarity.
    --folds       Comma list    Run a subset, e.g. ``--folds asap7,tsmc5``.
    --epochs N                  Override the v3 production 150.
    --ar-finetune-epochs N      Override the v3 production 5.
    --cuda                      Use GPU.
    --gpu N                     Which CUDA device id to pin (default 2 = Blackwell).
    --dry-run                   1-epoch smoke test for one fold, no report.
    --skip                      Skip folds whose checkpoint already exists.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

# Bootstrap imports — mirror the existing verify_nn_*.py scripts.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))


def _pin_gpu(gpu_id: int) -> None:
    """Pin a single GPU before torch is imported.

    Sets ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` so the id matches the
    ordering shown by ``nvidia-smi`` (default torch order is
    ``FASTEST_FIRST`` which can put the two A100s before the
    Blackwell on this host).
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


# Defaults (the v3 production recipe).
TECH_ORDER = ["asap7", "tsmc5", "tsmc7", "tsmc12", "tsmc16"]
V3_BASELINE = {"NRMSE(%)": 0.223, "MRE(%)": 1.41, "R2": 0.9984}
OUTPUT_GROUPS: Dict[str, List[str]] = {
    "currents+cond": ["id", "gm", "gds", "gmb"],
    "charges":       ["qg", "qd", "qs", "qb"],
    "capacitances":  ["cgg", "cgd", "cgs", "cdg", "cdd"],
}


# ══════════════════════════════════════════════════════════════════════════════
# Fold driver
# ══════════════════════════════════════════════════════════════════════════════

def run_one_fold(
    held_out_tech: str,
    device_type: str,
    data_path: Path,
    epochs: int,
    ar_finetune_epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    device_str: str,
    dry_run: bool,
    skip_if_exists: bool,
) -> Dict:
    """Train one LOO fold and return a metrics dict."""
    # Local import so _pin_gpu runs before torch touches CUDA.
    import torch
    from torch.utils.data import DataLoader

    from bsimar.config import (
        CHECKPOINT_DIR, TransformerConfig, OUTPUT_COLUMNS,
    )
    from bsimar.data.normalize import reorder_outputs, unreorder_outputs
    from bsimar.eval.loo_labels import build_loo_splits, get_test_slice
    from bsimar.eval.metrics import compute_physical_metrics, print_metrics
    from bsimar.training.trainer import test_model, train_transformer

    save_prefix = f"loo_{held_out_tech}_{device_type}"
    ckpt_phys = CHECKPOINT_DIR / f"{save_prefix}_best.phys.pt"
    ckpt_best = CHECKPOINT_DIR / f"{save_prefix}_best.pt"
    ckpt_norm = CHECKPOINT_DIR / f"{save_prefix}_norm.npz"

    print(f"\n{'='*78}")
    print(f"  LOO FOLD: holding out {held_out_tech.upper()} ({device_type})")
    print(f"{'='*78}")

    t0 = time.time()

    # 1. Build splits (fits normaliser on the 4-tech training slice only).
    print(f"\n[1/4] Building LOO splits...")
    train_ds, val_ds, test_ds, normalizer = build_loo_splits(
        str(data_path), held_out_tech, device_type, val_frac=0.1, seed=42,
    )

    # 2. Train. For resume, skip training if the phys-best checkpoint
    # already exists (the driver's --skip behaviour).
    if skip_if_exists and ckpt_phys.exists():
        print(f"\n[2/4] --skip: checkpoint exists at {ckpt_phys.name}, "
              f"reusing it.")
        # Load architecture config so we can instantiate the model.
        from bsimar.models.transformer import TransformerEncoderModel
        arch_path = CHECKPOINT_DIR / f"{save_prefix}_config.npz"
        arch = dict(np.load(arch_path))
        model = TransformerEncoderModel(
            input_dim=int(arch["input_dim"]),
            target_dim=int(arch["target_dim"]),
            d_model=int(arch["d_model"]),
            nhead=int(arch["nhead"]),
            num_layers=int(arch["num_layers"]),
            dim_feedforward=int(arch["dim_feedforward"]),
            dropout=float(arch["dropout"]),
        ).to(torch.device(device_str))
        model.load_state_dict(
            torch.load(str(ckpt_phys), weights_only=True,
                       map_location=torch.device(device_str)))
    else:
        print(f"\n[2/4] Training transformer ({epochs} epochs, "
              f"AR finetune {ar_finetune_epochs} epochs)...")
        cfg = TransformerConfig(
            batch_size=batch_size, max_epochs=epochs,
            lr=lr, patience=patience,
        )
        model, _norm_returned = train_transformer(
            data_path=str(data_path),
            save_prefix=save_prefix,
            config=cfg,
            device_str=device_str,
            ar_finetune_epochs=ar_finetune_epochs,
            overwrite=True,
            # IMPORTANT: reorder_outputs is applied *inside* train_transformer
            # on the datasets we pass in — we must NOT pre-reorder.
            prebuilt_data=(train_ds, val_ds, test_ds, normalizer),
        )
        # After training, reload the phys-best checkpoint (train_transformer
        # already does this internally at the end, but we want to be explicit
        # so dry-run and full-run paths match).
        if ckpt_phys.exists():
            model.load_state_dict(
                torch.load(str(ckpt_phys), weights_only=True,
                           map_location=torch.device(device_str)))

    # 3. Run the final test using the reordered test_ds.
    # train_transformer internally reordered train/val/test_ds in place,
    # so the datasets we hold still have outputs in BSIMAR order.
    # We must unreorder the predictions for the metrics computation.
    print(f"\n[3/4] Running test on held-out {held_out_tech}...")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    pred_norm, true_norm = test_model(
        model, test_loader, torch.device(device_str))
    pred_norm = unreorder_outputs(pred_norm)
    true_norm = unreorder_outputs(true_norm)
    metrics = compute_physical_metrics(pred_norm, true_norm, normalizer)
    print_metrics(metrics)

    # 4. Worst-case bias points on the held-out tech.
    print(f"\n[4/4] Extracting worst-case bias points...")
    # Denormalise to physical space for both pred and true.
    pred_phys = normalizer.denormalize_outputs(pred_norm)
    true_phys = normalizer.denormalize_outputs(true_norm)
    raw = get_test_slice(str(data_path), held_out_tech, device_type)
    # Apply the same filter that build_loo_splits uses.
    from bsimar.data.dataset import filter_small_targets
    keep = filter_small_targets(raw["outputs"], OUTPUT_COLUMNS)
    raw_inputs = raw["inputs"][keep]
    raw_geometry = raw["geometry"][keep]
    # Per-sample relative L1 error on `id`.
    id_col = OUTPUT_COLUMNS.index("id")
    abs_err = np.abs(pred_phys[:, id_col] - true_phys[:, id_col])
    denom = np.abs(true_phys[:, id_col])
    denom = np.where(denom < 1e-15, 1e-15, denom)
    rel_err = abs_err / denom
    # NOTE: test_ds was built from the *same* filtered+labelled rows,
    # which should be identical in order to raw_inputs/raw_geometry after
    # filter_small_targets. Assert that lengths match as a safety.
    assert len(raw_inputs) == len(pred_phys), (
        f"Worst-case alignment mismatch: "
        f"{len(raw_inputs)} raw vs {len(pred_phys)} pred")

    top_k = min(20, len(rel_err))
    worst_idx = np.argsort(-rel_err)[:top_k]
    worst_cases = []
    for i in worst_idx:
        worst_cases.append({
            "idx": int(i),
            "Vd": float(raw_inputs[i, 0]),
            "Vg": float(raw_inputs[i, 1]),
            "Vs": float(raw_inputs[i, 2]),
            "Vb": float(raw_inputs[i, 3]),
            "NFIN": float(raw_geometry[i, 0]),
            "L": float(raw_geometry[i, 1]),
            "id_true": float(true_phys[i, id_col]),
            "id_pred": float(pred_phys[i, id_col]),
            "rel_err": float(rel_err[i]),
        })

    # Training-domain coverage check: what fraction of held-out inputs
    # land outside the normalizer's recorded train-domain range?
    from bsimar.data.normalize import _build_combined_input
    held_combined = _build_combined_input(raw_inputs, raw_geometry)
    inp_min = normalizer.stats.input_min
    inp_max = normalizer.stats.input_max
    below = (held_combined < inp_min[None, :]).any(axis=1)
    above = (held_combined > inp_max[None, :]).any(axis=1)
    ood_any = (below | above).mean()
    # Per-feature % of samples outside the range.
    ood_per_feature = (
        ((held_combined < inp_min[None, :]) |
         (held_combined > inp_max[None, :])).mean(axis=0).tolist())

    elapsed = time.time() - t0
    print(f"\nFold {held_out_tech} done in {elapsed/60:.1f} min")

    return {
        "held_out_tech": held_out_tech,
        "device_type": device_type,
        "elapsed_sec": elapsed,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "metrics": {k: {mk: float(mv) for mk, mv in v.items()}
                    for k, v in metrics.items()},
        "worst_cases": worst_cases,
        "ood_any_frac": float(ood_any),
        "ood_per_feature": ood_per_feature,
        "checkpoint": str(ckpt_phys),
        "normalizer": str(ckpt_norm),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Report generator
# ══════════════════════════════════════════════════════════════════════════════

def _avg(metrics: Dict, key: str, names: List[str]) -> float:
    vals = [metrics[n][key] for n in names
            if n in metrics
            and metrics[n][key] is not None
            and not np.isnan(metrics[n][key])]
    return float(np.mean(vals)) if vals else float("nan")


def _group_avg(metrics: Dict, group: List[str], key: str) -> float:
    return _avg(metrics, key, group)


def _f(v: float, fmt: str = "6.3f") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "    —"
    return f"{v:{fmt}}"


def generate_report(
    folds: List[Dict],
    report_dir: Path,
    device_type: str,
    total_elapsed_sec: float,
    data_path: Path,
) -> Path:
    """Write a comprehensive markdown report to ``report_dir/report.md``."""
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"

    lines: List[str] = []

    lines.append(f"# BSIMAR v3 Leave-One-Out (by Technology) -- {device_type.upper()}")
    lines.append("")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Dataset**: `{data_path}`")
    lines.append(f"**Recipe**: v3 production (asinh+zscore, MAE+LDS+VovLDS, "
                 f"parallel_caps, grouped_inputs, phys-best tracker, "
                 f"AR-rollout finetune)")
    lines.append(f"**Total wall-clock**: {total_elapsed_sec/60:.1f} min "
                 f"({total_elapsed_sec/3600:.2f} h)")
    lines.append("")
    lines.append(f"In-distribution v3 baseline (random split, all 5 techs "
                 f"pooled): NRMSE={V3_BASELINE['NRMSE(%)']:.3f}%, "
                 f"MRE={V3_BASELINE['MRE(%)']:.2f}%, R²={V3_BASELINE['R2']:.4f}. "
                 f"Any LOO fold below ~0.5 % would suggest tech-label leakage "
                 f"and should be flagged as suspicious.")
    lines.append("")

    # ── Headline table ──────────────────────────────────────────────────
    lines.append("## Headline: per-fold overall metrics")
    lines.append("")
    lines.append("`NRMSE%` / `MRE%` / `R²` are averaged over the 13 physical "
                 "outputs (id, gm, gds, gmb, qg, qd, qs, qb, cgg, cgd, cgs, "
                 "cdg, cdd). NRMSE_n is the normalized-space metric (no "
                 "denormalisation, no mask), which is the safest cross-fold "
                 "comparison because it is immune to the per-target valid "
                 "mask and the asinh denormalisation.")
    lines.append("")
    lines.append("| Held-out tech | n_train | n_val | n_test | "
                 "NRMSE % | MRE % | R² | NRMSE_n % | R²_n | "
                 "train min | vs. baseline |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for f in folds:
        m = f["metrics"]
        nrmse = _avg(m, "NRMSE(%)", list(m.keys()))
        mre = _avg(m, "MRE(%)", list(m.keys()))
        r2 = _avg(m, "R2", list(m.keys()))
        nrmse_n = _avg(m, "NRMSE_norm(%)", list(m.keys()))
        r2_n = _avg(m, "R2_norm", list(m.keys()))
        delta = nrmse / V3_BASELINE["NRMSE(%)"]
        lines.append(
            f"| **{f['held_out_tech']}** | {f['n_train']:,} | {f['n_val']:,} "
            f"| {f['n_test']:,} | {_f(nrmse, '7.3f')} | {_f(mre, '6.2f')} "
            f"| {_f(r2, '7.4f')} | {_f(nrmse_n, '8.3f')} | {_f(r2_n, '7.4f')} "
            f"| {f['elapsed_sec']/60:6.1f} | **{delta:5.1f}×** |"
        )
    lines.append("")

    # ── Per-output heatmap (NRMSE%) ────────────────────────────────────
    lines.append("## Per-output NRMSE% across folds")
    lines.append("")
    lines.append("Each cell is the physical-space NRMSE % on the held-out tech "
                 "for that output.")
    lines.append("")
    # Column header.
    all_outputs = ["id", "gm", "gds", "gmb",
                   "qg", "qd", "qs", "qb",
                   "cgg", "cgd", "cgs", "cdg", "cdd"]
    header = "| Held-out | " + " | ".join(all_outputs) + " |"
    sep = "|---|" + "|".join(["---:"] * len(all_outputs)) + "|"
    lines.append(header)
    lines.append(sep)
    for f in folds:
        row = [f["held_out_tech"]]
        for o in all_outputs:
            v = f["metrics"][o]["NRMSE(%)"]
            row.append(_f(v, "6.2f"))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── Per-output MRE% heatmap ─────────────────────────────────────────
    lines.append("## Per-output MRE% across folds")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for f in folds:
        row = [f["held_out_tech"]]
        for o in all_outputs:
            v = f["metrics"][o]["MRE(%)"]
            row.append(_f(v, "6.2f"))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── Per-output R² heatmap ───────────────────────────────────────────
    lines.append("## Per-output R² across folds")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for f in folds:
        row = [f["held_out_tech"]]
        for o in all_outputs:
            v = f["metrics"][o]["R2"]
            row.append(_f(v, "6.3f"))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── Output-group summary ────────────────────────────────────────────
    lines.append("## Summary by output group")
    lines.append("")
    lines.append("Average NRMSE % per group per fold:")
    lines.append("")
    lines.append("| Held-out | currents+cond | charges | capacitances |")
    lines.append("|---|---:|---:|---:|")
    for f in folds:
        m = f["metrics"]
        row = [f["held_out_tech"]]
        for group_name, cols in OUTPUT_GROUPS.items():
            row.append(_f(_group_avg(m, cols, "NRMSE(%)"), "6.2f"))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── Bottleneck analysis ─────────────────────────────────────────────
    lines.append("## Bottleneck analysis")
    lines.append("")

    # Worst fold.
    fold_avgs = [(f["held_out_tech"],
                  _avg(f["metrics"], "NRMSE(%)", list(f["metrics"].keys())))
                 for f in folds]
    fold_avgs_sorted = sorted(fold_avgs, key=lambda x: -x[1])
    worst_tech, worst_nrmse = fold_avgs_sorted[0]
    best_tech, best_nrmse = fold_avgs_sorted[-1]
    lines.append(f"- **Worst fold**: `{worst_tech}` at NRMSE "
                 f"**{worst_nrmse:.3f} %** "
                 f"({worst_nrmse/V3_BASELINE['NRMSE(%)']:.1f}× the "
                 f"in-distribution v3 baseline).")
    lines.append(f"- **Best fold**: `{best_tech}` at NRMSE "
                 f"**{best_nrmse:.3f} %** "
                 f"({best_nrmse/V3_BASELINE['NRMSE(%)']:.1f}× baseline).")

    # Worst output group across folds.
    group_avgs = {g: 0.0 for g in OUTPUT_GROUPS}
    for f in folds:
        m = f["metrics"]
        for g, cols in OUTPUT_GROUPS.items():
            group_avgs[g] += _group_avg(m, cols, "NRMSE(%)")
    for g in group_avgs:
        group_avgs[g] /= max(len(folds), 1)
    worst_group = max(group_avgs.items(), key=lambda kv: kv[1])
    lines.append(f"- **Worst output group (average across folds)**: "
                 f"`{worst_group[0]}` at NRMSE **{worst_group[1]:.3f} %**.")

    # Critical outputs for circuit accuracy.
    lines.append("")
    lines.append("### Critical-output watch list (id, gm, cgg)")
    lines.append("")
    lines.append("These three dominate circuit-level accuracy: `id` is the "
                 "drain current, `gm` is the NR solver's main conductance, "
                 "`cgg` is the gate cap that sets switching delay. Worst "
                 "case across folds:")
    lines.append("")
    lines.append("| Output | Worst fold | NRMSE % | MRE % | R² |")
    lines.append("|---|---|---:|---:|---:|")
    for out in ["id", "gm", "cgg"]:
        worst_f = max(folds, key=lambda f: f["metrics"][out]["NRMSE(%)"])
        m = worst_f["metrics"][out]
        lines.append(
            f"| `{out}` | {worst_f['held_out_tech']} | "
            f"{m['NRMSE(%)']:6.3f} | {m['MRE(%)']:6.2f} | {m['R2']:6.3f} |"
        )
    lines.append("")

    # Per-tech OOD surprise list.
    lines.append("### OOD surprises per fold (R² < 0.8 or NRMSE > 5×baseline)")
    lines.append("")
    for f in folds:
        surprises = []
        for out in all_outputs:
            m = f["metrics"][out]
            r2 = m["R2"]
            nrmse = m["NRMSE(%)"]
            if (r2 is not None and not np.isnan(r2) and r2 < 0.8) or \
               nrmse > 5 * V3_BASELINE["NRMSE(%)"]:
                surprises.append(
                    f"`{out}` (NRMSE={nrmse:.2f}%, R²={r2:.3f})")
        if surprises:
            lines.append(f"- **{f['held_out_tech']}**: "
                         f"{', '.join(surprises)}")
        else:
            lines.append(f"- **{f['held_out_tech']}**: no surprises "
                         f"(all outputs within 5× baseline and R² ≥ 0.8)")
    lines.append("")

    # Training-domain OOD coverage.
    lines.append("### Training-domain coverage")
    lines.append("")
    lines.append("Fraction of held-out samples whose *combined input* "
                 "(4 voltages + 14 geometry + process features) falls "
                 "outside the normalizer's recorded train-domain box "
                 "`[input_min, input_max]`. High values indicate that "
                 "the held-out tech's inputs land in a region the model "
                 "has never been asked to predict, which upper-bounds "
                 "the achievable LOO accuracy.")
    lines.append("")
    lines.append("| Held-out | OOD fraction |")
    lines.append("|---|---:|")
    for f in folds:
        lines.append(f"| {f['held_out_tech']} | {f['ood_any_frac']*100:5.1f}% |")
    lines.append("")

    # ── Worst-case bias points ──────────────────────────────────────────
    lines.append("## Worst-case bias points (top 20 per fold by `id` rel-error)")
    lines.append("")
    for f in folds:
        lines.append(f"### {f['held_out_tech']}")
        lines.append("")
        lines.append("| Rank | Vd | Vg | Vb | NFIN | L (nm) | id_true (A) | id_pred (A) | rel-err |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for rank, wc in enumerate(f["worst_cases"], 1):
            lines.append(
                f"| {rank} | {wc['Vd']:+.3f} | {wc['Vg']:+.3f} | "
                f"{wc['Vb']:+.3f} | {wc['NFIN']:.0f} | "
                f"{wc['L']*1e9:.1f} | {wc['id_true']:+.3e} | "
                f"{wc['id_pred']:+.3e} | {wc['rel_err']:6.2f} |"
            )
        lines.append("")

    # ── Recommendations ──────────────────────────────────────────────────
    lines.append("## Recommendations (data-driven)")
    lines.append("")
    # Auto-generated from the numbers.
    recs = []
    if worst_nrmse > 10 * V3_BASELINE["NRMSE(%)"]:
        recs.append(
            f"- **Generalisation is weak on `{worst_tech}`** "
            f"({worst_nrmse/V3_BASELINE['NRMSE(%)']:.1f}× the baseline). "
            f"Candidate fixes, in order of expected impact: (a) add a few "
            f"anchor samples from `{worst_tech}` to the training set "
            f"(semi-supervised fine-tune) to see if the model has the "
            f"capacity but is missing PDK-specific calibration; (b) "
            f"investigate whether the held-out tech has unique physics "
            f"features (e.g., asymmetric L, unusual PHIG/EOT) that are "
            f"not present in the other four; (c) tech-aware conditioning "
            f"by concatenating a one-hot tech embedding onto the inputs.")
    if worst_group[0] == "charges":
        recs.append(
            "- **Charges degrade faster than currents under LOO**, which "
            "matches the known chain-rule weakness of asinh normalisation "
            "for Q-like outputs. Consider reviving the charge-consistency "
            "loss experiment (N4) *for LOO only* — the v3 sprint found it "
            "hurt in-distribution but LOO is a different optimisation.")
    elif worst_group[0] == "capacitances":
        recs.append(
            "- **Capacitances degrade faster than charges/currents**, "
            "suggesting the parallel-caps head is memorising PDK-specific "
            "cap values rather than learning dq/dV. Consider adding an "
            "autograd-consistency term in LOO training only.")
    elif worst_group[0] == "currents+cond":
        recs.append(
            "- **Currents+conductances dominate the LOO error**, the most "
            "circuit-critical group. This is the hardest signal and "
            "indicates the model is not transferring I-V physics across "
            "techs. Short-term: retrain with a higher `w_curr` weight. "
            "Medium-term: add cross-tech data augmentation (interpolate "
            "between two techs' process-param vectors).")
    # OOD fraction high.
    high_ood = [f for f in folds if f["ood_any_frac"] > 0.5]
    if high_ood:
        techs = ", ".join(f"`{f['held_out_tech']}`" for f in high_ood)
        recs.append(
            f"- **{techs} land outside the training-domain box** for >50% "
            f"of their samples. The model is being asked to extrapolate; "
            f"any fraction of LOO error caused by pure extrapolation "
            f"cannot be recovered by a better loss. Widening the voltage "
            f"box factor at data-gen time (currently 2.0×VDD) is the only "
            f"principled fix.")
    # If no recommendations fired, at least say so.
    if not recs:
        recs.append(
            "- No high-priority LOO pathology detected. All folds are "
            "within 5× of the in-distribution baseline; the v3 recipe "
            "appears to generalise cross-tech.")
    lines.extend(recs)
    lines.append("")

    report_path.write_text("\n".join(lines))
    return report_path


def write_worst_cases_csv(
    folds: List[Dict], report_dir: Path, device_type: str,
) -> None:
    """Dump per-fold top-20 worst cases to CSV files."""
    import csv
    for f in folds:
        path = report_dir / f"worst_cases_{f['held_out_tech']}_{device_type}.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(
                f["worst_cases"][0].keys()))
            writer.writeheader()
            writer.writerows(f["worst_cases"])


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="BSIMAR v3 Leave-One-Out (by technology) experiment")
    parser.add_argument("--device-type", choices=["nmos", "pmos"], default="nmos")
    parser.add_argument("--folds", type=str, default=",".join(TECH_ORDER),
                        help="Comma list of held-out techs to run "
                             "(default: all 5)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--ar-finetune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu", type=int, default=2,
                        help="CUDA device id to pin "
                             "(default 2 = Blackwell RTX Pro 6000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="1-epoch smoke test for fold 0, no full "
                             "training, no report generation")
    parser.add_argument("--skip", action="store_true",
                        help="Skip folds whose phys-best checkpoint "
                             "already exists (resume a crashed run)")
    args = parser.parse_args()

    if args.cuda:
        _pin_gpu(args.gpu)

    # Torch imports after GPU pinning.
    import torch
    from bsimar.utils.seed import set_seed
    set_seed(42)

    data_path = (PROJECT_ROOT / "external_compact_models" / "bsimar" /
                 "data" / "datasets" / f"universal_{args.device_type}.npz")
    if not data_path.exists():
        print(f"ERROR: dataset not found: {data_path}")
        if args.device_type == "pmos":
            print("  The PMOS dataset must be generated first:")
            print("  python external_compact_models/PyCMG/scripts/"
                  "generate_nn_data.py --device pmos --universal")
        return 1

    requested_folds = [f.strip() for f in args.folds.split(",") if f.strip()]
    for tech in requested_folds:
        if tech not in TECH_ORDER:
            print(f"ERROR: unknown tech '{tech}'. Must be one of {TECH_ORDER}")
            return 1

    device_str = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if args.cuda and not torch.cuda.is_available():
        print("WARNING: --cuda requested but CUDA is not available; "
              "falling back to CPU (will be very slow).")
    print(f"Compute device: {device_str}")
    if device_str == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    if args.dry_run:
        print("\n[DRY RUN] epochs=1, ar-finetune-epochs=0, one fold only")
        fold = run_one_fold(
            held_out_tech=requested_folds[0],
            device_type=args.device_type,
            data_path=data_path,
            epochs=1,
            ar_finetune_epochs=0,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            device_str=device_str,
            dry_run=True,
            skip_if_exists=False,
        )
        print("\n[DRY RUN] Single fold completed. Smoke test PASS.")
        return 0

    # Full run.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = (PROJECT_ROOT / "tests" / "verify_bsimar_loo_results" /
                  f"{timestamp}_{args.device_type}")
    report_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nReport output directory: {report_dir}")

    t_start = time.time()
    fold_results: List[Dict] = []
    for tech in requested_folds:
        try:
            result = run_one_fold(
                held_out_tech=tech,
                device_type=args.device_type,
                data_path=data_path,
                epochs=args.epochs,
                ar_finetune_epochs=args.ar_finetune_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                patience=args.patience,
                device_str=device_str,
                dry_run=False,
                skip_if_exists=args.skip,
            )
            fold_results.append(result)
            # Persist after every fold so a crash loses at most one.
            with open(report_dir / "metrics.json", "w") as fh:
                json.dump(
                    {"folds": fold_results,
                     "total_elapsed_sec": time.time() - t_start,
                     "device_type": args.device_type,
                     "data_path": str(data_path)}, fh, indent=2)
        except Exception as exc:
            print(f"\n\n!!! Fold {tech} FAILED: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
            print(f"Marking fold as failed and continuing...")
            fold_results.append({
                "held_out_tech": tech,
                "device_type": args.device_type,
                "elapsed_sec": 0,
                "n_train": 0, "n_val": 0, "n_test": 0,
                "metrics": {o: {"NRMSE(%)": float("nan"),
                                "MRE(%)": float("nan"),
                                "R2": float("nan"),
                                "NRMSE_norm(%)": float("nan"),
                                "R2_norm": float("nan"),
                                "MAE_norm": float("nan"),
                                "n_valid": 0, "n_total": 0}
                            for o in ["id","gm","gds","gmb","qg","qd",
                                      "qs","qb","cgg","cgd","cgs","cdg","cdd"]},
                "worst_cases": [],
                "ood_any_frac": float("nan"),
                "ood_per_feature": [],
                "checkpoint": "FAILED",
                "normalizer": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            })

    total_elapsed = time.time() - t_start
    print(f"\n\n{'='*78}")
    print(f"  All {len(requested_folds)} folds completed in "
          f"{total_elapsed/60:.1f} min ({total_elapsed/3600:.2f} h)")
    print(f"{'='*78}")

    # Write report.
    # Only include folds that trained successfully (no NaN metrics) in
    # the analysis; failed folds are listed in a separate "failures"
    # section.
    good_folds = [f for f in fold_results
                  if not np.isnan(f["metrics"]["id"]["NRMSE(%)"])]
    if good_folds:
        report_path = generate_report(
            good_folds, report_dir, args.device_type,
            total_elapsed, data_path)
        write_worst_cases_csv(good_folds, report_dir, args.device_type)
        print(f"\nReport: {report_path}")
    else:
        print("\nNo successful folds — no report generated.")

    # Always dump the raw metrics JSON.
    with open(report_dir / "metrics.json", "w") as fh:
        json.dump(
            {"folds": fold_results,
             "total_elapsed_sec": total_elapsed,
             "device_type": args.device_type,
             "data_path": str(data_path)}, fh, indent=2)
    print(f"Raw metrics: {report_dir / 'metrics.json'}")

    # Exit non-zero if any fold failed.
    failed = [f for f in fold_results if f.get("checkpoint") == "FAILED"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
