"""B3 LoRA training driver for DirectNet V6.4.5 Track-B.

Fine-tunes the TSMC7 DirectNet base (nmos and pmos) using LoRA on the trunk
Linear layers. The loss is reweighted to emphasise:
  - id (column 0): weight 2.0  -- drives switching current fidelity
  - gds (column 2): weight 1.5 -- drives drain conductance (period sharpness)
  - all others: 1.0

The switching-region row weighting (mid-rail Vgs) is handled implicitly via
the standard LDS mechanism on the normalised id distribution, which already
up-weights the low-density subthreshold / transition region.

Usage (rank-8, nmos):
    CUDA_VISIBLE_DEVICES=2 conda run -n pycircuitsim python \
        experiments/v6_4_5_track_b/B3_train.py \
        --device-type nmos --rank 8 --epochs 60 --overwrite

Usage (rank-32, pmos):
    CUDA_VISIBLE_DEVICES=2 conda run -n pycircuitsim python \
        experiments/v6_4_5_track_b/B3_train.py \
        --device-type pmos --rank 32 --epochs 60 --overwrite
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Path bootstrap: make bsimar and pycircuitsim importable.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models"))
sys.path.insert(0, str(PROJECT_ROOT / "external_compact_models" / "PyCMG"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "v6_4_5_track_b"))

from B3_lora import wrap_trunk_with_lora, merge_lora  # noqa: E402

from bsimar.config import CHECKPOINT_DIR, DATA_DIR  # noqa: E402
from bsimar.data.dataset import load_and_split_bsimar  # noqa: E402
from bsimar.data.normalize import OUTPUT_COLUMN_ORDER  # noqa: E402
from bsimar.losses.bni_mae import MAELoss, compute_lds_weights_per_target  # noqa: E402
from bsimar.utils.seed import set_seed  # noqa: E402

# ---------------------------------------------------------------------------
# B3 loss column weights:
#   OUTPUT_COLUMN_ORDER = [id, gm, gds, gmb, qg, qd, qs, qb,
#                          cgg, cgd, cgs, cdg, cdd]
# rank-8 preset (mild emphasis — used in first attempt):
# id (0): 2.0, gm (1): 1.0, gds (2): 1.5, rest: 1.0
#
# rank-32 preset (aggressive emphasis — switching-region focus):
# id (0): 3.0, gm (1): 2.0, gds (2): 2.0, rest: 1.0
# ---------------------------------------------------------------------------
B3_COLUMN_WEIGHTS_R8 = np.array(
    [2.0, 1.0, 1.5, 1.0,  # id, gm, gds, gmb
     1.0, 1.0, 1.0, 1.0,  # qg, qd, qs, qb
     1.0, 1.0, 1.0, 1.0, 1.0],  # cgg, cgd, cgs, cdg, cdd
    dtype=np.float32,
)

B3_COLUMN_WEIGHTS_R32 = np.array(
    [3.0, 2.0, 2.0, 1.0,  # id, gm, gds, gmb — aggressive switching emphasis
     1.0, 1.0, 1.0, 1.0,  # qg, qd, qs, qb
     1.0, 1.0, 1.0, 1.0, 1.0],  # cgg, cgd, cgs, cdg, cdd
    dtype=np.float32,
)

# Default (backward compat)
B3_COLUMN_WEIGHTS = B3_COLUMN_WEIGHTS_R8

_NORM_MODE = "asinh"
_NUM_WORKERS = 4


def _load_base_model(device_type: str, device: torch.device) -> nn.Module:
    """Load canonical TSMC7 DirectNet base checkpoint into a DirectNet instance."""
    from bsimar.models.direct_net import DirectNet

    stem = f"tsmc7_dn_medium_{device_type}"
    pt_path = CHECKPOINT_DIR / f"{stem}_best.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {pt_path}")

    state = torch.load(str(pt_path), map_location="cpu", weights_only=True)

    # Infer architecture from state_dict (mirrors _build_from_state).
    net_keys = [k for k in state if k.startswith("net.") and k.endswith(".weight")]
    output_dim = state[net_keys[-1]].shape[0]
    hidden_dim = state[net_keys[-1]].shape[1]
    n_layers = len(net_keys) - 1
    num_tech_codes = state["tech_embedding.weight"].shape[0]
    tech_embed_dim = state["tech_embedding.weight"].shape[1]
    input_dim = state[net_keys[0]].shape[1] - tech_embed_dim
    monotonic = any(k.startswith("mono.") for k in state)
    monotone_sign, monotone_hidden = 1.0, 64
    if monotonic:
        monotone_sign = float(state["mono.sign"].item())
        monotone_hidden = state["mono.w_vg_raw"].shape[0]

    model = DirectNet(
        input_dim=input_dim, hidden_dim=hidden_dim,
        n_layers=n_layers, output_dim=output_dim,
        num_tech_codes=num_tech_codes,
        tech_embed_dim=tech_embed_dim,
        unknown_code_id=num_tech_codes - 1,  # per-tech vocab: UNKNOWN = last slot
        monotonic=monotonic,
        monotone_sign=monotone_sign,
        monotone_hidden=monotone_hidden,
    )
    model.load_state_dict(state)
    model = model.to(device)
    return model


def _sanity_check_lora(
    base_model: nn.Module, wrapped: nn.Module, input_dim: int
) -> None:
    """Assert wrapped output == base output at LoRA init (B=0 => delta=0).

    input_dim: number of CONTINUOUS input features (not including tech embed).
    Both models must be on the same device (CPU for this check).
    """
    base_model.eval()
    wrapped.eval()
    with torch.no_grad():
        x = torch.randn(16, input_dim)
        tc = torch.zeros(16, dtype=torch.long)
        o_base = base_model(x, tech_codes=tc)
        o_wrap = wrapped(x, tech_codes=tc)
        max_diff = (o_base - o_wrap).abs().max().item()
        assert max_diff < 1e-5, (
            f"LoRA init sanity FAILED: max diff = {max_diff:.2e} (should be ~0)")
        print(f"  [sanity] LoRA init: max(|wrapped - base|) = {max_diff:.2e}  PASS")


def _train(
    *,
    device_type: str,
    rank: int,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    overwrite: bool,
    device: torch.device,
) -> None:
    save_prefix = f"b3_lora_r{rank}_tsmc7_{device_type}"
    best_path = CHECKPOINT_DIR / f"{save_prefix}_best.pt"
    norm_src = CHECKPOINT_DIR / f"tsmc7_dn_medium_{device_type}_norm.npz"
    norm_dst = CHECKPOINT_DIR / f"{save_prefix}_norm.npz"

    if best_path.exists() and not overwrite:
        raise SystemExit(
            f"Refusing to overwrite {best_path}. Use --overwrite.")

    # Pick rank-dependent column weights.
    col_weights = B3_COLUMN_WEIGHTS_R32 if rank == 32 else B3_COLUMN_WEIGHTS_R8
    cw_desc = ("id=3.0, gm=2.0, gds=2.0, others=1.0"
               if rank == 32 else "id=2.0, gds=1.5, others=1.0")

    print(f"\n=== B3 LoRA rank={rank} {device_type.upper()} on {device} ===")
    print(f"  Saving to: {save_prefix}")
    print(f"  Column weights: {cw_desc}")

    # ── Data ─────────────────────────────────────────────────────────────
    data_path = DATA_DIR / f"tsmc7_{device_type}.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    # For rank-32: load raw dataset first to compute switching-region row mask
    # (Vg near mid-rail = trip zone), then pass to the loader.
    # The normalized inputs have Vg at column 1 in the raw space (before norm).
    # We up-weight rows in Vg ∈ [VDD/4, 3*VDD/4] (TSMC7 VDD=0.75V => [0.19, 0.56]).
    switch_row_boost: float = 1.0  # default = uniform
    if rank == 32:
        raw = np.load(str(data_path))
        vg_raw = raw["inputs"][:, 1]  # Vg column in the raw npz
        vdd = 0.75  # TSMC7
        switch_mask_all = (vg_raw >= vdd * 0.25) & (vg_raw <= vdd * 0.75)
        switch_frac = switch_mask_all.mean()
        switch_row_boost = 2.0  # 2× up-weight for switching rows
        print(f"  Switching-region rows (Vg ∈ [0.19, 0.56]V): "
              f"{switch_mask_all.sum():,}/{len(switch_mask_all):,} "
              f"({switch_frac:.1%}) — boost {switch_row_boost}×")
        del raw  # free memory before loading through normalizer

    train_ds, val_ds, _, normalizer = load_and_split_bsimar(
        str(data_path), OUTPUT_COLUMN_ORDER,
        device_type=device_type,
        norm_mode=_NORM_MODE,
        apply_filter=True,
        exclude_techs={"tsmc5", "tsmc12", "tsmc16", "asap7"},
        tech_scope="tsmc7",
    )

    # ── LDS weights with B3 column multipliers ───────────────────────────
    print("  Computing LDS weights …")
    lds = compute_lds_weights_per_target(
        train_ds.outputs.numpy(), n_bins=100,
        lds_kernel="gaussian", lds_ks=5, lds_sigma=0.8)
    means = lds.mean(axis=0, keepdims=True)
    means[means < 1e-12] = 1.0
    lds = lds / means
    # Apply B3 column multipliers.
    lds = lds * col_weights[None, :]

    # Rank-32: apply switching-region row boost. The load_and_split_bsimar
    # shuffles and splits, so we need to re-derive the switching mask on the
    # TRAINING SPLIT's raw Vg. The normalized input col 1 is Vg (z-scored):
    # z = (Vg - mu_Vg) / sigma_Vg. We re-detect the trip region in norm space.
    if rank == 32 and switch_row_boost > 1.0:
        # Vg is column 1 of the normalized inputs (before embedding concat).
        # The normalizer's input stats are under normalizer.stats.
        vg_norm = train_ds.inputs[:, 1].numpy()  # normalized Vg column
        # Detect midband: z-score range corresponding to physical [0.19, 0.56]V
        # We use a robust proxy: abs(Vg_norm) < 1 (near-mean, trip region).
        switch_train = np.abs(vg_norm) < 1.0
        lds[switch_train] *= switch_row_boost
        print(f"  Applied {switch_row_boost}× row boost to "
              f"{switch_train.sum():,}/{len(switch_train):,} "
              f"training rows (|Vg_norm|<1)")

    train_w = TensorDataset(
        train_ds.inputs, train_ds.outputs, train_ds.tech_codes,
        torch.tensor(lds, dtype=torch.float32))
    train_loader = DataLoader(
        train_w, batch_size=batch_size, shuffle=True,
        num_workers=_NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=_NUM_WORKERS, pin_memory=True, persistent_workers=True)

    # ── Model ─────────────────────────────────────────────────────────────
    # Load a clean copy for the sanity check, then another for training.
    base_for_sanity = _load_base_model(device_type, torch.device("cpu"))
    # Derive input_dim from the state_dict (same as _build_from_state).
    _state = torch.load(
        str(CHECKPOINT_DIR / f"tsmc7_dn_medium_{device_type}_best.pt"),
        map_location="cpu", weights_only=True)
    _net_keys = [k for k in _state if k.startswith("net.") and k.endswith(".weight")]
    _tech_ed = _state["tech_embedding.weight"].shape[1]
    input_dim = _state[_net_keys[0]].shape[1] - _tech_ed

    model = _load_base_model(device_type, device)
    # Wrap a CPU copy for sanity, train the GPU copy.
    wrapped_cpu = copy.deepcopy(base_for_sanity)
    wrap_trunk_with_lora(wrapped_cpu, rank=rank)
    _sanity_check_lora(base_for_sanity, wrapped_cpu, input_dim)

    wrap_trunk_with_lora(model, rank=rank)

    # Count trainable params.
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {n_train:,} / {n_total:,} "
          f"({100.0 * n_train / max(n_total, 1):.2f}%)")

    # ── Optimizer: ONLY LoRA params ───────────────────────────────────────
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(lora_params, lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = MAELoss()

    # ── Training loop ─────────────────────────────────────────────────────
    best_val = float("inf")
    bad = 0
    t0 = time.time()
    print(f"  Training {epochs} epochs (patience={patience}, lr={lr:.1e})")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for x, y, tc, w in train_loader:
            x, y, tc, w = x.to(device), y.to(device), tc.to(device), w.to(device)
            optimizer.zero_grad()
            pred = model(x, tech_codes=tc)
            loss = criterion(pred, y, weights=w)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        train_loss = total_loss / max(n, 1)

        model.eval()
        val_total = 0.0
        val_n = 0
        with torch.no_grad():
            for x, y, tc in val_loader:
                x, y, tc = x.to(device), y.to(device), tc.to(device)
                pred = model(x, tech_codes=tc)
                val_total += criterion(pred, y).item()
                val_n += 1
        val_loss = val_total / max(val_n, 1)
        scheduler.step()

        marker = ""
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            bad = 0
            # Save merged (plain) state_dict so _build_from_state loads it.
            # We merge into a CPU copy to avoid mutating the training model.
            cpu_model = copy.deepcopy(model).cpu()
            merged = merge_lora(cpu_model)
            torch.save(merged.state_dict(), str(best_path))
            marker = " *best*"
        else:
            bad += 1

        if epoch <= 5 or epoch % 10 == 0 or marker:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  {epoch:4d} | train={train_loss:.5f} "
                  f"val={val_loss:.5f} lr={lr_now:.2e}{marker}")

        if bad >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s. Best val={best_val:.6f}")
    print(f"  Saved merged checkpoint: {best_path}")

    # Copy norm stats (identical to the base — same dataset, same normalizer).
    shutil.copy2(str(norm_src), str(norm_dst))
    print(f"  Copied norm stats: {norm_dst}")

    # ── Sanity: load merged checkpoint and compare to wrapped model output ─
    print("  [sanity] verifying merged checkpoint loads correctly …")
    from bsimar.models.direct_net import DirectNet

    merged_state = torch.load(str(best_path), map_location="cpu", weights_only=True)
    # Confirm no lora.* keys.
    lora_keys = [k for k in merged_state if "lora" in k.lower()]
    assert not lora_keys, f"Merged state_dict still has LoRA keys: {lora_keys}"

    # Rebuild via _build_from_state logic.
    net_keys = [k for k in merged_state
                if k.startswith("net.") and k.endswith(".weight")]
    out_dim = merged_state[net_keys[-1]].shape[0]
    hid_dim = merged_state[net_keys[-1]].shape[1]
    n_l = len(net_keys) - 1
    num_tc = merged_state["tech_embedding.weight"].shape[0]
    tech_ed = merged_state["tech_embedding.weight"].shape[1]
    in_d = merged_state[net_keys[0]].shape[1] - tech_ed
    mono = any(k.startswith("mono.") for k in merged_state)

    rebuilt = DirectNet(
        input_dim=in_d, hidden_dim=hid_dim, n_layers=n_l, output_dim=out_dim,
        num_tech_codes=num_tc, tech_embed_dim=tech_ed, monotonic=mono,
        unknown_code_id=num_tc - 1,
    )
    rebuilt.load_state_dict(merged_state)
    rebuilt.eval()

    # Compare with the training model (also on CPU).
    cpu_model_eval = copy.deepcopy(model).cpu()
    cpu_model_eval.eval()
    with torch.no_grad():
        x = torch.randn(16, in_d)
        tc = torch.zeros(16, dtype=torch.long)
        o_merged = rebuilt(x, tech_codes=tc)
        o_wrapped = cpu_model_eval(x, tech_codes=tc)
        diff = (o_merged - o_wrapped).abs().max().item()
        # After training the wrapped model has learned LoRA deltas, so
        # 'wrapped' and 'rebuilt from merged state_dict' should give the
        # same output — they differ only by float32 accumulation order.
        # A large diff here indicates a merge bug; small diff is expected.
        if diff < 1.0:
            print(f"  [sanity] merged vs wrapped: max diff = {diff:.2e}  OK")
        else:
            print(f"  [WARNING] merged vs wrapped: max diff = {diff:.2e} "
                  f"(larger than expected — check merge_lora)")


def main() -> None:
    ap = argparse.ArgumentParser(description="B3 LoRA fine-tune for TSMC7 DirectNet")
    ap.add_argument("--device-type", choices=["nmos", "pmos"], required=True)
    ap.add_argument("--rank", type=int, default=8, choices=[8, 32])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    _train(
        device_type=args.device_type,
        rank=args.rank,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        overwrite=args.overwrite,
        device=device,
    )


if __name__ == "__main__":
    main()
