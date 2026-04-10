#!/usr/bin/env python3
"""BSIMAR v4 zero-shot evaluation on ASAP7.

Tests the TSMC-only pretrained v4 model on ASAP7 data without any fine-tuning.
The model uses the UNKNOWN tech code for ASAP7 samples.

After fine-tuning, re-run with --finetuned to use the expanded model with
ASAP7-specific tech codes.

Usage:
    # Zero-shot (UNKNOWN code)
    conda run -n pycircuitsim python tests/verify_bsimar_v4_zeroshot.py

    # After fine-tuning
    conda run -n pycircuitsim python tests/verify_bsimar_v4_zeroshot.py --finetuned
"""

import sys
import argparse
from pathlib import Path

import numpy as np

# Path bootstrap
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "external_compact_models"
PYCMG_DIR = EXTERNAL_DIR / "PyCMG"
for p in (PROJECT_ROOT, EXTERNAL_DIR, PYCMG_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bsimar.config import (
    CHECKPOINT_DIR, DATA_DIR, OUTPUT_COLUMNS,
    TECH_VARIANT_CODES, UNKNOWN_CODE_ID,
    tech_variant_to_code,
)
from bsimar.data.normalize import (
    BSIMARNormStats, BSIMARNormalizer,
    reorder_outputs, unreorder_outputs,
)
from bsimar.models.transformer import TransformerEncoderModel
from bsimar.eval.loo_labels import get_or_build_tech_variant_labels
from bsimar.data.dataset import filter_small_targets, MOSFETDatasetV4
from bsimar.eval.metrics import compute_physical_metrics, print_metrics

import torch
from torch.utils.data import DataLoader


def evaluate_v4_on_tech(
    model_path: str,
    data_path: str,
    device_type: str,
    target_techs: set,
    use_true_codes: bool = False,
) -> None:
    """Evaluate a v4 model on specific techs.

    Args:
        use_true_codes: If True, use the actual tech codes (post-finetune).
            If False, use UNKNOWN_CODE_ID (zero-shot).
    """
    model_path = Path(model_path)
    config_path = model_path.parent / (
        model_path.stem.replace("_best", "_config")
        .replace(".phys", "").replace(".ar", "") + ".npz")
    norm_path = model_path.parent / (
        model_path.stem.replace("_best", "_norm")
        .replace(".phys", "").replace(".ar", "") + ".npz")

    # Load model
    cfg = np.load(str(config_path))
    model = TransformerEncoderModel(
        input_dim=int(cfg["input_dim"]),
        target_dim=int(cfg["target_dim"]),
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_layers=int(cfg["num_layers"]),
        dim_feedforward=int(cfg["dim_feedforward"]),
        dropout=float(cfg["dropout"]),
        use_tech_codes=True,
        num_tech_codes=int(cfg["num_tech_codes"]),
    )
    state = torch.load(str(model_path), weights_only=True, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Load normalizer
    normalizer = BSIMARNormalizer(
        mode="asinh",
        stats=BSIMARNormStats.load(str(norm_path)),
        include_proc_params=False,
    )

    # Load data
    raw = np.load(data_path, allow_pickle=True)
    inputs = raw["inputs"]
    geometry = raw["geometry"]
    outputs = raw["outputs"]

    keep = filter_small_targets(outputs, OUTPUT_COLUMNS)
    inputs, geometry, outputs = inputs[keep], geometry[keep], outputs[keep]

    tech_codes = get_or_build_tech_variant_labels(
        data_path, device_type, verbose=False)
    tech_codes = tech_codes[keep]

    # Filter for target techs
    target_code_set = {
        code for (tech, _), code in TECH_VARIANT_CODES.items()
        if tech in target_techs
    }
    mask = np.array([int(c) in target_code_set for c in tech_codes], dtype=bool)
    inputs = inputs[mask]
    geometry = geometry[mask]
    outputs = outputs[mask]
    tech_codes = tech_codes[mask]

    print(f"  Evaluating on {len(inputs)} samples from {target_techs}")

    # Override tech codes for zero-shot
    if not use_true_codes:
        tech_codes = np.full_like(tech_codes, UNKNOWN_CODE_ID)
        print(f"  Using UNKNOWN code ({UNKNOWN_CODE_ID}) for all samples")
    else:
        print(f"  Using true tech codes: {sorted(set(tech_codes.tolist()))}")

    # Normalize
    x = normalizer.normalize_inputs(inputs, geometry)
    y = normalizer.normalize_outputs(outputs)
    ds = MOSFETDatasetV4(x, y, tech_codes)

    # Reorder
    ds.outputs = torch.tensor(
        reorder_outputs(ds.outputs.numpy()), dtype=torch.float32)

    loader = DataLoader(ds, batch_size=2048, shuffle=False)

    # Inference
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb, tc in loader:
            pred = model(xb, tech_codes=tc)
            all_pred.append(pred.numpy())
            all_true.append(yb.numpy())

    pred_norm = unreorder_outputs(np.concatenate(all_pred))
    true_norm = unreorder_outputs(np.concatenate(all_true))

    metrics = compute_physical_metrics(pred_norm, true_norm, normalizer)
    print_metrics(metrics)


def main():
    parser = argparse.ArgumentParser(description="BSIMAR v4 zero-shot evaluation")
    parser.add_argument("--finetuned", action="store_true",
                        help="Use finetuned model with true ASAP7 codes")
    parser.add_argument("--device-type", default="nmos", choices=["nmos", "pmos"])
    parser.add_argument("--model-path", default=None,
                        help="Override model path (default: auto-detect)")
    args = parser.parse_args()

    data_path = str(DATA_DIR / f"universal_{args.device_type}.npz")
    if not Path(data_path).exists():
        print(f"Dataset not found: {data_path}")
        sys.exit(1)

    if args.model_path:
        model_path = args.model_path
    elif args.finetuned:
        model_path = str(CHECKPOINT_DIR / f"v4_ft_asap7_{args.device_type}_best.phys.pt")
    else:
        model_path = str(CHECKPOINT_DIR / f"v4_universal_{args.device_type}_best.phys.pt")

    if not Path(model_path).exists():
        # Try without .phys suffix
        alt = model_path.replace(".phys.pt", "_best.pt")
        if Path(alt).exists():
            model_path = alt
        else:
            print(f"Model not found: {model_path}")
            sys.exit(1)

    mode = "finetuned (true codes)" if args.finetuned else "zero-shot (UNKNOWN code)"
    print(f"=== BSIMAR v4 {mode} evaluation on ASAP7 ===")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print()

    evaluate_v4_on_tech(
        model_path=model_path,
        data_path=data_path,
        device_type=args.device_type,
        target_techs={"asap7"},
        use_true_codes=args.finetuned,
    )


if __name__ == "__main__":
    main()
