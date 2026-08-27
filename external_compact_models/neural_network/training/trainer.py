"""Single training loop for DirectNet (MLP) and BSIMAR (Transformer).

Both architectures share the same data, normaliser, LDS-MAE loss, cosine
schedule, and early-stop pattern. The only differences:

* the Transformer's ``forward`` takes a teacher-forced ``y`` argument
  during training and runs autoregressively at eval time;
* the Transformer trains in ``BSIMAR_COLUMN_ORDER`` and saves an
  architecture sidecar so the simulator can rebuild the model.

Both differences are gated on a single ``is_transformer`` flag inside
``_train_loop``. Public entry points: ``train_directnet`` and
``train_transformer``.
"""

from __future__ import annotations

import os
import time
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from neural_network.config import (
    DirectNetConfig, TransformerConfig,
    CHECKPOINT_DIR, RESULTS_DIR,
    CODE_TO_TECH_VARIANT,
    NUM_TSMC_CODES_WITH_UNKNOWN,
)
from neural_network.data.dataset import MOSFETDataset, load_and_split_bsimar
from neural_network.data.normalize import (
    FULL_TERMINAL_OUTPUT_COLUMN_ORDER, OUTPUT_COLUMN_ORDER,
    reorder_outputs, unreorder_outputs,
    _NormalizerBase,
)
from neural_network.losses.bni_mae import MAELoss, compute_lds_weights_per_target

# V6 Tier 2 (2026-05-09): both DirectNet and the Transformer train with
# asinh + z-score outputs. Concentrates loss on the small-Id band that
# dominates inverter trip-point NRMSE.
_NORM_MODE = "asinh"
_NUM_WORKERS = 8


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ── Batch iteration (V7.0.2) ───────────────────────────────────────────────

class _DeviceBatches:
    """Epoch iterator over tensors parked on the training device.

    Drop-in for ``DataLoader`` in this trainer: same yielded tuples, same
    tail-batch behaviour, same shuffle-per-epoch semantics.

    Why it exists: every split is already a set of in-memory tensors, but
    the shipped path wrapped them in a ``TensorDataset`` and handed that to
    a ``DataLoader`` with 8 worker processes. Per batch that is 2048
    individual ``__getitem__`` calls, a collate, and an IPC copy — to
    deliver what is one contiguous slice of a tensor that already exists.
    Measured on DirectNet-large / batch 2048: **11.0 ms/step -> 3.3 ms/step**.

    The whole train split is small enough to park on the GPU (1.6M rows x
    34 float32 columns = 218 MB); ``_pick_loader`` checks free memory and
    falls back to the ``DataLoader`` when it would not fit comfortably.
    """

    def __init__(
        self, tensors: Sequence[torch.Tensor], batch_size: int,
        shuffle: bool, device: torch.device,
    ) -> None:
        self.tensors = [t.to(device, non_blocking=True) for t in tensors]
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.device = device
        self.n = int(self.tensors[0].shape[0])

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        bs = self.batch_size
        if self.shuffle:
            perm = torch.randperm(self.n, device=self.device)
            for i in range(0, self.n, bs):
                idx = perm[i:i + bs]
                yield tuple(t[idx] for t in self.tensors)
        else:
            # Contiguous views — no gather, no copy.
            for i in range(0, self.n, bs):
                yield tuple(t[i:i + bs] for t in self.tensors)


def _device_resident_bytes(tensors: Sequence[torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def _pick_loader(
    tensors: Sequence[torch.Tensor], batch_size: int, shuffle: bool,
    device: torch.device, label: str,
):
    """Return a GPU-resident iterator when it fits, else a ``DataLoader``.

    Override with ``BSIMAR_LOADER=torch`` to force the legacy path (e.g.
    to reproduce a historical run) or ``=device`` to force residency.
    """
    choice = os.environ.get("BSIMAR_LOADER", "auto").lower()
    if choice not in ("auto", "torch", "device"):
        raise ValueError(
            f"BSIMAR_LOADER must be auto|torch|device, got {choice!r}")

    need = _device_resident_bytes(tensors)
    if choice == "device":
        fits = True
    elif choice == "torch" or device.type != "cuda":
        fits = False
    else:
        free, _total = torch.cuda.mem_get_info(device)
        # Half of free memory: the model, optimizer state, activations and
        # any co-tenant training stream still need room.
        fits = need < 0.5 * free

    if fits:
        print(f"  Loader[{label}]: device-resident "
              f"({need / 1e6:.0f} MB on {device})")
        return _DeviceBatches(tensors, batch_size, shuffle, device)

    print(f"  Loader[{label}]: torch DataLoader "
          f"({need / 1e6:.0f} MB, num_workers={_NUM_WORKERS})")
    return DataLoader(
        TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
        num_workers=_NUM_WORKERS, pin_memory=True, persistent_workers=True)


# ── Pre-flight checks ──────────────────────────────────────────────────────

def _assert_codes_in_vocab(
    datasets: Sequence[MOSFETDataset], num_tech_codes: int, tech_scope: str,
) -> None:
    """Raise if any split carries a tech code the embedding cannot index.

    audit C6q: ``nn.Embedding(num_tech_codes)`` has no range check of its
    own — an out-of-range code is an IndexError on CPU and an opaque
    device-side assert on CUDA, both surfacing an epoch into a multi-hour
    run with nothing naming the offending tech. The reachable case is the
    universal scope, whose vocabulary is TSMC-only (0-16 + UNKNOWN 17)
    while the registry also numbers ASAP7 at 18-21.
    """
    maxima = [int(ds.tech_codes.max()) for ds in datasets if len(ds) > 0]
    if not maxima or max(maxima) < num_tech_codes:
        return
    offenders = sorted({
        int(c)
        for ds in datasets if len(ds) > 0
        for c in torch.unique(ds.tech_codes).tolist()
        if int(c) >= num_tech_codes
    })
    named = ", ".join(
        f"{c}={CODE_TO_TECH_VARIANT.get(c, ('?', '?'))}" for c in offenders)
    raise ValueError(
        f"tech_scope={tech_scope!r}: dataset carries tech code(s) outside "
        f"the embedding vocabulary (num_tech_codes={num_tech_codes}, valid "
        f"0..{num_tech_codes - 1}): {named}. Drop those rows with "
        f"--exclude-techs, or size the vocabulary with --num-tech-codes "
        f"and warm-start from a checkpoint that has the extra rows.")


# ── Per-epoch helpers ──────────────────────────────────────────────────────

def _epoch_train(
    model: nn.Module, loader: DataLoader,
    criterion: MAELoss, optimizer: optim.Optimizer,
    device: torch.device, is_transformer: bool,
    ema_model: Optional[nn.Module] = None,
    sobolev_loss: Optional[nn.Module] = None,
    sobolev_norm: Optional[Dict[str, torch.Tensor]] = None,
    subthresh_loss: Optional[nn.Module] = None,
    aux_norm: Optional[Dict[str, torch.Tensor]] = None,
    charge_sobolev_loss: Optional[nn.Module] = None,
    charge_sobolev_norm: Optional[Dict[str, torch.Tensor]] = None,
    amp: bool = False,
    clip_grad: bool = False,
) -> Tuple[float, float]:
    """One training epoch. Returns (mean_total_loss, mean_aux_term).

    When ``sobolev_loss`` is given (DirectNet / P4 only), the autograd
    ∂id/∂V consistency term is added to the MAE; ``x`` carries gradient so
    ``torch.autograd.grad`` inside the term can build the second-order graph.
    When ``subthresh_loss`` is given (DirectNet / P3), the subthreshold id
    value+ceiling term is added (no second-order graph needed). ``mean_aux_term``
    sums whichever auxiliary terms are active.
    """
    model.train()
    # V7.0.2 — accumulate on-device. The old ``float(loss.item())`` per
    # batch forced a host sync every step (~800/epoch) for a scalar only
    # printed once per epoch.
    total = torch.zeros((), device=device)
    total_aux = torch.zeros((), device=device)
    n = 0
    # V6.8: the Sobolev / charge-Sobolev terms build a second-order graph
    # (autograd.grad with create_graph=True). PyTorch's fused SDPA kernels
    # have no double-backward (`aten::_scaled_dot_product_efficient_attention
    # _backward is not implemented`), so the Transformer forward must record
    # the MATH attention backend when those terms are active. ~2-3x slower
    # attention; DirectNet and aux-free Transformer paths are untouched.
    needs_double_bwd = (
        sobolev_loss is not None or charge_sobolev_loss is not None)
    if is_transformer and needs_double_bwd:
        from torch.nn.attention import sdpa_kernel, SDPBackend

        def _fwd_ctx():
            return sdpa_kernel(SDPBackend.MATH)
    else:
        import contextlib

        def _fwd_ctx():
            return contextlib.nullcontext()
    for x, y, tc, w in loader:
        x, y, tc, w = (x.to(device), y.to(device),
                       tc.to(device), w.to(device))
        optimizer.zero_grad()
        if needs_double_bwd:
            x = x.requires_grad_(True)
        # V6.8 opt-in bf16 autocast (Transformer wall-clock). Incompatible
        # with the double-backward aux terms — the CLI guards that combo.
        with _fwd_ctx(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            pred = (model(x, y, tech_codes=tc) if is_transformer
                    else model(x, tech_codes=tc))
            loss = criterion(pred.float(), y, weights=w)
        if sobolev_loss is not None:
            sob = sobolev_loss(
                x_norm=x, y_pred_norm=pred, y_true_norm=y,
                in_std=sobolev_norm["in_std"],
                out_std=sobolev_norm["out_std"],
                out_mean=sobolev_norm["out_mean"],
                asinh_scale=sobolev_norm["asinh_scale"],
                weights=w)
            loss = loss + sob
            total_aux += sob.detach()
        if subthresh_loss is not None:
            sub = subthresh_loss(
                x_norm=x, y_pred_norm=pred, y_true_norm=y,
                in_mean=aux_norm["in_mean"], in_std=aux_norm["in_std"],
                out_std=aux_norm["out_std"], out_mean=aux_norm["out_mean"],
                asinh_scale=aux_norm["asinh_scale"])
            loss = loss + sub
            total_aux += sub.detach()
        if charge_sobolev_loss is not None:
            csob = charge_sobolev_loss(
                x_norm=x, y_pred_norm=pred, y_true_norm=y,
                in_std=charge_sobolev_norm["in_std"],
                out_std=charge_sobolev_norm["out_std"],
                out_mean=charge_sobolev_norm["out_mean"],
                asinh_scale=charge_sobolev_norm["asinh_scale"],
                weights=w)
            loss = loss + csob
            total_aux += csob.detach()
        loss.backward()
        if is_transformer or clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if ema_model is not None:
            ema_model.update_parameters(model)
        total += loss.detach()
        n += 1
    n = max(n, 1)
    return float(total) / n, float(total_aux) / n


@torch.no_grad()
def _epoch_eval(
    model: nn.Module, loader: DataLoader, criterion: MAELoss,
    device: torch.device, is_transformer: bool,
    amp: bool = False,
) -> float:
    model.eval()
    total = torch.zeros((), device=device)
    n = 0
    for x, y, tc in loader:
        x, y, tc = x.to(device), y.to(device), tc.to(device)
        # Teacher-forced eval for the Transformer (val loss aligned with train).
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=amp):
            pred = (model(x, y, tech_codes=tc) if is_transformer
                    else model(x, tech_codes=tc))
        total += criterion(pred.float(), y).detach()
        n += 1
    return float(total) / max(n, 1)


@torch.no_grad()
def _collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device,
    is_transformer: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect (pred_norm, true_norm, tech_codes) on a loader (AR for TF)."""
    model.eval()
    all_pred, all_true, all_tc = [], [], []
    for x, y, tc in loader:
        x, tc = x.to(device), tc.to(device)
        pred = model(x, tech_codes=tc)  # AR inference for the Transformer
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())
        all_tc.append(tc.cpu().numpy())
    return (np.concatenate(all_pred),
            np.concatenate(all_true),
            np.concatenate(all_tc))


def _per_tech_report(
    pred_norm: np.ndarray, true_norm: np.ndarray,
    tech_codes: np.ndarray, normalizer: _NormalizerBase,
) -> None:
    from neural_network.config import CODE_TO_TECH_VARIANT
    from neural_network.eval.metrics import compute_physical_metrics

    print(f"\n{'Tech':>15s} | {'n_test':>6s} | "
          f"{'NRMSE%':>8s} | {'R2':>8s}")
    print("-" * 50)
    for code in sorted(np.unique(tech_codes)):
        mask = tech_codes == code
        tech, variant = CODE_TO_TECH_VARIANT.get(int(code), ("unk", "unk"))
        m = compute_physical_metrics(
            pred_norm[mask], true_norm[mask], normalizer)
        nr = [v["NRMSE(%)"] for v in m.values()
              if not np.isnan(v["NRMSE(%)"])]
        r2 = [v["R2"] for v in m.values() if not np.isnan(v["R2"])]
        print(f"{tech}:{variant:>9s} | {mask.sum():6d} | "
              f"{(np.mean(nr) if nr else float('nan')):8.3f} | "
              f"{(np.mean(r2) if r2 else float('nan')):8.4f}")


# ── Generic train loop ─────────────────────────────────────────────────────

def _train_loop(
    *,
    model: nn.Module,
    train_ds, val_ds, test_ds,
    normalizer: _NormalizerBase,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    save_prefix: str,
    device: torch.device,
    overwrite: bool,
    is_transformer: bool,
    arch_config: Optional[dict] = None,
    column_weights: Optional[np.ndarray] = None,
    class_weights: Optional[Dict[str, float]] = None,
    swa_mode: str = "none",
    ema_decay: float = 0.999,
    sobolev: bool = False,
    lam_sobolev: float = 0.1,
    sobolev_floor: float = 1e-12,
    sobolev_strong_boost: float = 1.0,
    sobolev_corridor_only: bool = False,
    subthresh: bool = False,
    lam_subthresh: float = 0.05,
    subthresh_s2: float = 1e-9,
    subthresh_upper: float = 1e-6,
    subthresh_floor: float = 1e-12,
    subthresh_off_floor: float = 1e-10,
    subthresh_ceiling_k: float = 1.0,
    subthresh_ceiling_w: float = 1.0,
    charge_sobolev: bool = False,
    lam_charge_sobolev: float = 0.05,
    charge_sobolev_floor: float = 1e-19,
    amp: bool = False,
    clip_grad: bool = False,
) -> Tuple[nn.Module, _NormalizerBase]:
    if amp and (sobolev or subthresh or charge_sobolev):
        raise ValueError(
            "--amp is incompatible with the double-backward aux losses "
            "(sobolev / subthresh / charge-sobolev)")
    if amp:
        print("  AMP: bf16 autocast ON (train + teacher-forced val)")
    if is_transformer:
        for ds in (train_ds, val_ds, test_ds):
            ds.outputs = torch.tensor(
                reorder_outputs(ds.outputs.numpy()), dtype=torch.float32)
        print("  Outputs reordered to BSIMAR_COLUMN_ORDER")

    print("  Computing LDS weights …")
    lds = compute_lds_weights_per_target(
        train_ds.outputs.numpy(), n_bins=100,
        lds_kernel="gaussian", lds_ks=5, lds_sigma=0.8)
    means = lds.mean(axis=0, keepdims=True)
    means[means < 1e-12] = 1.0
    lds = lds / means

    # V6.4.7 S9b (plan P5 plumbing, pulled forward): per-sample-class
    # multipliers folded into the LDS tensor AFTER the per-target
    # mean-normalization above, then the product is RENORMALIZED to unit
    # mean per target — otherwise the effective LR changes and confounds
    # every A/B against control. Runs BEFORE the column-weight presets so
    # their deliberately non-unit means (e.g. qs=0) are preserved.
    if class_weights is not None:
        names = getattr(train_ds, "sample_class_names", None)
        sc = getattr(train_ds, "sample_class", None)
        if names is None or sc is None:
            raise ValueError(
                "class_weights requires a dataset with sample_class and "
                "meta_sample_class_names (regenerate the .npz with a "
                "sample_class-aware generator).")
        unknown = sorted(set(class_weights) - set(names))
        if unknown:
            raise ValueError(
                f"class_weights names {unknown} not in dataset "
                f"sample_class names {names}")
        w = np.ones(len(sc), dtype=np.float32)
        sc_np = sc.numpy()
        for name, mult in class_weights.items():
            w[sc_np == names.index(name)] = float(mult)
        lds = lds * w[:, None]
        prod_means = lds.mean(axis=0, keepdims=True)
        prod_means[prod_means < 1e-12] = 1.0
        lds = lds / prod_means
        print(f"  Class weights: {class_weights}")
        print(f"  LDS x class-weight product renormalized; post-product "
              f"per-target mean = {np.round(lds.mean(axis=0), 6).tolist()}")

    if column_weights is not None:
        cw = np.asarray(column_weights, dtype=np.float32)
        if cw.shape != (lds.shape[1],):
            raise ValueError(
                f"column_weights shape {cw.shape} does not match "
                f"output dim {lds.shape[1]}")
        lds = lds * cw[None, :]
        print(f"  Column-weight preset: {cw.tolist()}")

    train_loader = _pick_loader(
        (train_ds.inputs, train_ds.outputs, train_ds.tech_codes,
         torch.tensor(lds, dtype=torch.float32)),
        batch_size, True, device, "train")
    val_loader = _pick_loader(
        (val_ds.inputs, val_ds.outputs, val_ds.tech_codes),
        batch_size, False, device, "val")
    test_loader = _pick_loader(
        (test_ds.inputs, test_ds.outputs, test_ds.tech_codes),
        batch_size, False, device, "test")

    # V7.0.2 — fused AdamW folds the whole parameter update into one
    # kernel. At these model sizes the step is launch-overhead-bound, so
    # this is the single largest remaining training win after the loader:
    # 3.29 -> 1.77 ms/step on DirectNet-large. CUDA-only; the CPU path
    # keeps the reference implementation.
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
        fused=(device.type == "cuda"))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = MAELoss()

    # Aux-loss plumbing shared by the Sobolev / subthreshold / charge-Sobolev
    # terms. The column order and the per-OUTPUT stat arrays must match the
    # layout of pred/y: the Transformer trains in BSIMAR_COLUMN_ORDER (the
    # outputs were reordered above) while the normalizer stats stay in the
    # canonical OUTPUT_COLUMN_ORDER — so for the Transformer the output-side
    # arrays are permuted to BSIMAR order. Input-side stats are order-invariant.
    # The column-sum autograd trick inside the losses stays valid for the
    # Transformer (batch rows never attend to each other). Caveat: under
    # teacher forcing the differentiated head is conditioned on the TRUE
    # prefix tokens, so the supervised slope approximates the AR-inference
    # Jacobian (exact for qg — the first AR token).
    def _aux_cols_and_stats() -> Tuple[list, np.ndarray, np.ndarray, np.ndarray]:
        st = normalizer.stats
        if st.mode != "asinh" or st.asinh_scale is None:
            raise ValueError(
                "Aux losses require asinh output normalization")
        base_cols = (st.output_columns if st.output_columns is not None
                     else OUTPUT_COLUMN_ORDER)
        if not is_transformer:
            return base_cols, st.output_std, st.output_mean, st.asinh_scale
        from neural_network.data.normalize import BSIMAR_COLUMN_ORDER
        perm = [base_cols.index(c) for c in BSIMAR_COLUMN_ORDER]
        return (list(BSIMAR_COLUMN_ORDER), st.output_std[perm],
                st.output_mean[perm], st.asinh_scale[perm])

    def _nt(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=device)

    # V6.4.7 S10 (P4) — Sobolev id-derivative consistency term (asinh output
    # norm); couples autograd ∂id/∂V to the supervised gm/gds/gmb columns the
    # deriv-fidelity gate measures. V6.8: also available for the Transformer
    # (teacher-forced slope; see _aux_cols_and_stats caveat).
    sobolev_loss: Optional[nn.Module] = None
    sobolev_norm: Optional[Dict[str, torch.Tensor]] = None
    if sobolev:
        from neural_network.losses.bni_mae import SobolevIdLoss
        st = normalizer.stats
        cols, out_std_a, out_mean_a, asinh_a = _aux_cols_and_stats()
        sobolev_loss = SobolevIdLoss(
            lam=lam_sobolev, column_order=cols,
            id_floor=sobolev_floor, strong_boost=sobolev_strong_boost,
            corridor_only=sobolev_corridor_only)
        sobolev_norm = {
            "in_std": _nt(st.input_std),
            "out_std": _nt(out_std_a),
            "out_mean": _nt(out_mean_a),
            "asinh_scale": _nt(asinh_a),
        }
        print(f"  Sobolev id-deriv loss ON (λ={lam_sobolev}, "
              f"floor={sobolev_floor:g}, strong_boost={sobolev_strong_boost})")

    # V6.4.7 S11 (P3) — subthreshold id value+ceiling term (asinh output
    # norm); re-scales the sub-uA roll-off (asinh s2) so the SRAM force_ic
    # weak-inversion band carries loss mass. Shares aux_norm (in_mean/in_std
    # for the per-fin OFF ceiling; out/asinh for denorm).
    subthresh_loss: Optional[nn.Module] = None
    aux_norm: Optional[Dict[str, torch.Tensor]] = None
    if subthresh:
        from neural_network.losses.bni_mae import SubthresholdIdLoss
        st = normalizer.stats
        cols, out_std_a, out_mean_a, asinh_a = _aux_cols_and_stats()
        subthresh_loss = SubthresholdIdLoss(
            lam=lam_subthresh, column_order=cols, s2=subthresh_s2,
            upper=subthresh_upper, id_floor=subthresh_floor,
            off_floor=subthresh_off_floor, ceiling_k=subthresh_ceiling_k,
            ceiling_w=subthresh_ceiling_w)
        aux_norm = {
            "in_mean": _nt(st.input_mean),
            "in_std": _nt(st.input_std),
            "out_std": _nt(out_std_a),
            "out_mean": _nt(out_mean_a),
            "asinh_scale": _nt(asinh_a),
        }
        print(f"  Subthreshold id loss ON (λ={lam_subthresh}, s2={subthresh_s2:g}, "
              f"upper={subthresh_upper:g}, off_floor={subthresh_off_floor:g}, "
              f"ceiling_k={subthresh_ceiling_k}, ceiling_w={subthresh_ceiling_w})")

    # V6.5.2 — charge-derivative (cap) Sobolev consistency term (asinh output
    # norm); couples the autograd ∂q/∂V the AC/transient solvers consume to
    # the supervised cgg/cgd/cdg/cdd columns (the cap analogue of the S10
    # id-slope Sobolev). Shares the same x.requires_grad path as sobolev.
    # V6.8: available for the Transformer — for qg (the first AR token) the
    # teacher-forced ∂qg/∂V IS the AR-inference Jacobian the AC solver reads.
    charge_sobolev_loss: Optional[nn.Module] = None
    charge_sobolev_norm: Optional[Dict[str, torch.Tensor]] = None
    if charge_sobolev:
        from neural_network.losses.bni_mae import ChargeSobolevLoss
        st = normalizer.stats
        cols, out_std_a, out_mean_a, asinh_a = _aux_cols_and_stats()
        charge_sobolev_loss = ChargeSobolevLoss(
            lam=lam_charge_sobolev, column_order=cols,
            cap_floor=charge_sobolev_floor)
        charge_sobolev_norm = {
            "in_std": _nt(st.input_std),
            "out_std": _nt(out_std_a),
            "out_mean": _nt(out_mean_a),
            "asinh_scale": _nt(asinh_a),
        }
        print(f"  Charge-Sobolev (cap dQ/dV) loss ON (λ={lam_charge_sobolev}, "
              f"cap_floor={charge_sobolev_floor:g})")

    # V6.4.7 S9 — within-run weight averaging (P6, pulled forward).
    # "ema": per-step exponential moving average from epoch 1.
    # "swa": equal-weight averaging from 75% of max_epochs.
    # In either mode, val selection and the saved checkpoint use the
    # AVERAGED weights once averaging is active; the state_dict is taken
    # from avg_model.module so the on-disk key format is unchanged.
    if swa_mode not in ("none", "ema", "swa"):
        raise ValueError(f"swa_mode must be none|ema|swa, got {swa_mode!r}")
    avg_model = None
    swa_start = 1
    if swa_mode == "ema":
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        avg_model = AveragedModel(
            model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay),
            use_buffers=True)
        print(f"  SWA/EMA: per-step EMA, decay={ema_decay}")
    elif swa_mode == "swa":
        from torch.optim.swa_utils import AveragedModel
        avg_model = AveragedModel(model, use_buffers=True)
        swa_start = max(1, int(0.75 * epochs))
        print(f"  SWA/EMA: equal-weight SWA from epoch {swa_start}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINT_DIR / f"{save_prefix}_best.pt"
    norm_path = CHECKPOINT_DIR / f"{save_prefix}_norm.npz"

    if best_path.exists() and not overwrite:
        raise SystemExit(
            f"Refusing to overwrite {best_path}. "
            "Pass --overwrite or pick a unique --exp-name.")

    # Trustworthy phys-best aggregator (post-fix; see plan §2B).
    normalizer.stats.phys_best_metric = "median"

    best_val = float("inf")
    bad = 0
    print(f"  Training {save_prefix} for {epochs} epochs "
          f"(patience={patience})")
    t0 = time.time()
    epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_aux = _epoch_train(
            model, train_loader, criterion, optimizer, device, is_transformer,
            ema_model=avg_model if swa_mode == "ema" else None,
            sobolev_loss=sobolev_loss, sobolev_norm=sobolev_norm,
            subthresh_loss=subthresh_loss, aux_norm=aux_norm,
            charge_sobolev_loss=charge_sobolev_loss,
            charge_sobolev_norm=charge_sobolev_norm,
            amp=amp, clip_grad=clip_grad)
        if swa_mode == "swa" and epoch >= swa_start:
            avg_model.update_parameters(model)
            if epoch == swa_start:
                best_val = float("inf")  # selection switches candidates here
                bad = 0
        # Candidate = averaged weights once averaging is active, else raw.
        avg_active = (swa_mode == "ema"
                      or (swa_mode == "swa" and epoch >= swa_start))
        eval_model = avg_model if avg_active else model
        val_loss = _epoch_eval(
            eval_model, val_loader, criterion, device, is_transformer,
            amp=amp)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        marker = ""
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            bad = 0
            state_src = avg_model.module if avg_active else model
            torch.save(state_src.state_dict(), str(best_path))
            normalizer.stats.save(str(norm_path))
            marker = " *best*"
        else:
            bad += 1

        if epoch <= 5 or epoch % 10 == 0 or marker:
            extra = (f" aux={train_aux:.5f}"
                     if (sobolev_loss is not None or subthresh_loss is not None
                         or charge_sobolev_loss is not None)
                     else "")
            print(f"  {epoch:4d} | train={train_loss:.5f}{extra} "
                  f"val={val_loss:.5f} lr={lr_now:.2e}{marker}")

        if bad >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s "
          f"({elapsed / max(epoch, 1):.1f}s/epoch). Best val={best_val:.6f}")

    # Architecture sidecar for models the simulator can't shape-infer
    # (Transformer). DirectNet passes arch_config=None — unchanged.
    if arch_config is not None:
        np.savez(
            str(CHECKPOINT_DIR / f"{save_prefix}_config.npz"),
            **{k: np.array(v) for k, v in arch_config.items()})

    # Final test eval — use AR inference for the Transformer.
    model.load_state_dict(torch.load(str(best_path), weights_only=True))
    pred_norm, true_norm, test_tc = _collect_predictions(
        model, test_loader, device, is_transformer)
    if is_transformer:
        pred_norm = unreorder_outputs(pred_norm)
        true_norm = unreorder_outputs(true_norm)

    from neural_network.eval.metrics import compute_physical_metrics, print_metrics
    metrics = compute_physical_metrics(pred_norm, true_norm, normalizer)
    print("\nPhysical metrics (test set):")
    print_metrics(metrics)
    _per_tech_report(pred_norm, true_norm, test_tc, normalizer)
    print(f"\nSaved checkpoint: {best_path}")
    print(f"Saved norm stats: {norm_path}")
    return model, normalizer


# ── Public entry points ────────────────────────────────────────────────────

def train_directnet(
    data_path: str,
    device_type: str = "nmos",
    config: DirectNetConfig = DirectNetConfig(),
    device_str: str = "cpu",
    save_prefix: str = "v4_re_dn_universal_nmos",
    exclude_techs: Optional[Set[str]] = None,
    num_tech_codes: int = NUM_TSMC_CODES_WITH_UNKNOWN,
    p_unknown: float = 0.1,
    max_rows: Optional[int] = None,
    overwrite: bool = False,
    column_weights: Optional[np.ndarray] = None,
    output_subset: Optional[list[str]] = None,
    output_columns: Optional[list[str]] = None,
    tech_scope: str = "universal",
    monotonic: bool = False,
    ekv_core: bool = False,
    ekv_alpha: float = 0.5,
    ekv_hidden: int = 64,
    ekv_lam_lo: float = 0.05,
    swa_mode: str = "none",
    ema_decay: float = 0.999,
    apply_filter: bool = True,
    class_weights: Optional[Dict[str, float]] = None,
    sobolev: bool = False,
    lam_sobolev: float = 0.1,
    sobolev_floor: float = 1e-12,
    sobolev_strong_boost: float = 1.0,
    sobolev_corridor_only: bool = False,
    subthresh: bool = False,
    lam_subthresh: float = 0.05,
    subthresh_s2: float = 1e-9,
    subthresh_upper: float = 1e-6,
    subthresh_floor: float = 1e-12,
    subthresh_off_floor: float = 1e-10,
    subthresh_ceiling_k: float = 1.0,
    subthresh_ceiling_w: float = 1.0,
    charge_sobolev: bool = False,
    lam_charge_sobolev: float = 0.05,
    charge_sobolev_floor: float = 1e-19,
    init_from: Optional[str] = None,
    amp: bool = False,
    split_mode: str = "combo",
    **_: object,  # swallow legacy kwargs
) -> Tuple[nn.Module, _NormalizerBase]:
    """DirectNet MLP training pipeline.

    ``sobolev`` (V6.4.7 S10 / P4, default False): add the Sobolev
    id-derivative consistency term (``lam_sobolev`` weight, ``sobolev_floor``
    trust-floor mask on |id_true|, ``sobolev_strong_boost`` upweight for
    |id_true| > 1uA). Couples autograd ∂id/∂V to the OSDI gm/gds/gmb columns.

    ``init_from`` (V6.4.7 S10 / P4 fine-tune): checkpoint stem (resolved
    under ``CHECKPOINT_DIR/<stem>_best.pt``) or an explicit ``*.pt`` path to
    warm-start the trunk + embedding from. The architecture must match (same
    --size / --tech-scope). Used for the fine-tune λ-screen from control-v2.

    ``apply_filter`` (V6.4.7 S9b, default True = legacy): drop rows with
    |id| <= 1e-15 before splitting. V6.4.7 arms pass False — small-current
    rows stay in training (plan rev-3 ruling 3).

    ``class_weights`` (V6.4.7 S9b): sample_class-name → loss multiplier,
    folded into the LDS tensor and renormalized to unit mean per target.

    ``column_weights`` (length = output dim): per-target multiplier on
    the loss (combined with LDS). Use to down-weight or zero out targets
    the simulator does not consume — e.g. ``qs`` (always replaced by KCL).

    ``output_columns`` declares the complete dataset/model contract. It
    defaults to the checkpoint-compatible 13-column DirectNet layout; the
    separate full-terminal family supplies its six independent surfaces.

    ``output_subset`` (list of column names): if given, train only on
    this subset of the 13 outputs (E2 4-output head). The model's
    ``output_dim`` becomes ``len(output_subset)`` and the saved norm
    stats record which columns were kept.

    ``monotonic`` (Phase 7a, default False): build the DirectNet with a
    residual sub-network monotone in ``Vg`` added to the ``id`` column.
    The base trunk is unchanged; the extra ``mono.*`` parameters are
    auto-detected at inference. No loss term is added.
    """
    from neural_network.models.direct_net import DirectNet

    device = torch.device(device_str)
    print(f"DirectNet on {device}; tech codes={num_tech_codes}, "
          f"p_unknown={p_unknown}")
    if exclude_techs:
        print(f"  Excluding techs: {exclude_techs}")

    training_columns = (
        list(output_columns) if output_columns is not None
        else list(OUTPUT_COLUMN_ORDER)
    )
    if tuple(training_columns) == FULL_TERMINAL_OUTPUT_COLUMN_ORDER:
        if (output_subset is not None or monotonic or ekv_core or sobolev
                or subthresh or charge_sobolev):
            raise ValueError(
                "The full-terminal family is a separate six-surface "
                "DirectNet contract and cannot use reduced-head subsets, "
                "backbones, or legacy auxiliary losses."
            )
        if apply_filter:
            raise ValueError(
                "The full-terminal family requires apply_filter=False; "
                "the legacy filter is defined only for the id column."
            )
    train_ds, val_ds, test_ds, normalizer = load_and_split_bsimar(
        data_path, training_columns, device_type=device_type,
        train_ratio=config.train_ratio, val_ratio=config.val_ratio,
        apply_filter=apply_filter, exclude_techs=exclude_techs,
        norm_mode=_NORM_MODE, max_rows=max_rows,
        output_subset=output_subset,
        tech_scope=tech_scope,
        split_mode=split_mode,
    )
    _assert_codes_in_vocab(
        (train_ds, val_ds, test_ds), num_tech_codes, tech_scope)
    in_dim = train_ds.inputs.shape[1]
    out_dim = train_ds.outputs.shape[1]
    if output_subset is not None:
        print(f"  Output subset: {output_subset} (output_dim={out_dim})")

    # Place UNKNOWN at the last slot of whatever vocab we have. Universal
    # vocab=18 keeps unknown=17 (existing convention); per-tech vocab=5
    # (TSMC5) → unknown=4 / vocab=4 (TSMC7) → unknown=3. Without this,
    # `p_unknown` training-time dropout would write code 17 into a 5-row
    # embedding and trigger a CUDA assert.
    #
    # Phase 7a monotone residual: the id column is monotone-increasing in
    # Vgs for NMOS (terminal-current convention, ∂id/∂Vg > 0) and
    # decreasing for PMOS in the source-relative training frame. The sign
    # is verified against the PyCMG datasets. Only valid when `id` is the
    # first output column (i.e. no `output_subset` reshuffle).
    if monotonic and output_subset is not None:
        raise ValueError(
            "--monotonic is incompatible with output_subset: the monotone "
            "residual targets id at column 0.")
    monotone_sign = 1.0 if device_type == "nmos" else -1.0
    if ekv_core and output_subset is not None:
        raise ValueError(
            "--ekv-core is incompatible with output_subset: the EKV core "
            "composes id at column 0.")
    model = DirectNet(
        input_dim=in_dim, hidden_dim=config.trunk_hidden,
        n_layers=config.trunk_layers + 1, output_dim=out_dim,
        num_tech_codes=num_tech_codes,
        tech_embed_dim=32, tech_embed_dropout=p_unknown,
        unknown_code_id=num_tech_codes - 1,
        monotonic=monotonic, monotone_sign=monotone_sign,
        ekv_core=ekv_core, ekv_alpha=ekv_alpha, ekv_hidden=ekv_hidden,
        ekv_lam_lo=ekv_lam_lo,
    ).to(device)
    if monotonic:
        print(f"  Phase 7a monotone-Vg residual ON (sign={monotone_sign:+.0f})")
    if ekv_core:
        # The id column is column 0 of OUTPUT_COLUMN_ORDER; seed the core's
        # norm/sign buffers from the fitted normalizer so it composes in
        # physical Amps and re-encodes to the standardised-asinh id space.
        st = normalizer.stats
        if st.mode != "asinh" or st.asinh_scale is None:
            raise ValueError("--ekv-core requires asinh output normalization")
        model.core.set_norm(
            v_mean=st.input_mean[:4], v_std=st.input_std[:4],
            id_s=float(st.asinh_scale[0]),
            id_mean=float(st.output_mean[0]), id_std=float(st.output_std[0]),
            device_type=device_type,
        )
        model.core.to(device)
        print(f"  S3 EKV core ON (alpha={ekv_alpha}, hidden={ekv_hidden}, "
              f"i0_scale={float(st.asinh_scale[0]):.3e}, "
              f"sign={'NMOS' if device_type == 'nmos' else 'PMOS'})")
    print(f"  Params: {model.count_parameters():,}")

    if init_from is not None:
        init_path = Path(init_from)
        if not init_path.suffix:
            init_path = CHECKPOINT_DIR / f"{init_from}_best.pt"
        if not init_path.exists():
            raise FileNotFoundError(
                f"init_from checkpoint not found: {init_path}")
        init_state = torch.load(str(init_path), weights_only=True,
                                map_location=device)
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"init_from architecture mismatch for {init_path.name}: "
                f"missing={list(missing)} unexpected={list(unexpected)}")
        print(f"  Warm-started from {init_path.name}")

    trained = _train_loop(
        model=model, is_transformer=False,
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        normalizer=normalizer,
        epochs=config.max_epochs, batch_size=config.batch_size,
        lr=config.lr, weight_decay=config.weight_decay,
        patience=config.patience, save_prefix=save_prefix,
        device=device, overwrite=overwrite,
        column_weights=column_weights,
        class_weights=class_weights,
        swa_mode=swa_mode, ema_decay=ema_decay,
        sobolev=sobolev, lam_sobolev=lam_sobolev,
        sobolev_floor=sobolev_floor,
        sobolev_strong_boost=sobolev_strong_boost,
        sobolev_corridor_only=sobolev_corridor_only,
        subthresh=subthresh, lam_subthresh=lam_subthresh,
        subthresh_s2=subthresh_s2, subthresh_upper=subthresh_upper,
        subthresh_floor=subthresh_floor,
        subthresh_off_floor=subthresh_off_floor,
        subthresh_ceiling_k=subthresh_ceiling_k,
        subthresh_ceiling_w=subthresh_ceiling_w,
        charge_sobolev=charge_sobolev,
        lam_charge_sobolev=lam_charge_sobolev,
        charge_sobolev_floor=charge_sobolev_floor,
        amp=amp,
    )
    if tuple(training_columns) == FULL_TERMINAL_OUTPUT_COLUMN_ORDER:
        checkpoint_path = CHECKPOINT_DIR / f"{save_prefix}_best.pt"
        norm_path = CHECKPOINT_DIR / f"{save_prefix}_norm.npz"
        marker_path = checkpoint_path.with_suffix(".pt.complete")
        marker_path.write_text(json.dumps({
            "family": "directnet-full",
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "normalization": norm_path.name,
            "normalization_sha256": _sha256_file(norm_path),
            "output_columns": list(FULL_TERMINAL_OUTPUT_COLUMN_ORDER),
        }, sort_keys=True, indent=2) + "\n")
    return trained


def train_transformer(
    data_path: str,
    save_prefix: str,
    device_type: str = "nmos",
    config: TransformerConfig = TransformerConfig(),
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    patience: Optional[int] = None,
    lr: Optional[float] = None,
    device_str: str = "cpu",
    overwrite: bool = False,
    exclude_techs: Optional[Set[str]] = None,
    num_tech_codes: int = NUM_TSMC_CODES_WITH_UNKNOWN,
    p_unknown: float = 0.1,
    max_rows: Optional[int] = None,
    tech_scope: str = "universal",
    swa_mode: str = "none",
    ema_decay: float = 0.999,
    apply_filter: bool = True,
    class_weights: Optional[Dict[str, float]] = None,
    sobolev: bool = False,
    lam_sobolev: float = 0.1,
    sobolev_floor: float = 1e-12,
    sobolev_strong_boost: float = 1.0,
    sobolev_corridor_only: bool = False,
    subthresh: bool = False,
    lam_subthresh: float = 0.05,
    subthresh_s2: float = 1e-9,
    subthresh_upper: float = 1e-6,
    subthresh_floor: float = 1e-12,
    subthresh_off_floor: float = 1e-10,
    subthresh_ceiling_k: float = 1.0,
    subthresh_ceiling_w: float = 1.0,
    charge_sobolev: bool = False,
    lam_charge_sobolev: float = 0.05,
    charge_sobolev_floor: float = 1e-19,
    init_from: Optional[str] = None,
    amp: bool = False,
    split_mode: str = "combo",
    **_: object,  # swallow legacy kwargs
) -> Tuple[nn.Module, _NormalizerBase]:
    """BSIMAR Transformer training pipeline.

    V6.8 recipe-parity port: ``init_from`` (curriculum warm-start),
    ``class_weights`` (corridor / inv_trip class multipliers), the
    Sobolev / subthreshold / charge-Sobolev aux terms (BSIMAR column
    order handled inside ``_train_loop``), and opt-in bf16 ``amp``.
    """
    from neural_network.models.transformer import TransformerEncoderModel

    epochs = epochs if epochs is not None else config.max_epochs
    batch_size = batch_size if batch_size is not None else config.batch_size
    patience = patience if patience is not None else config.patience
    lr = lr if lr is not None else config.lr

    device = torch.device(device_str)
    print(f"BSIMAR Transformer on {device}; tech codes={num_tech_codes}, "
          f"p_unknown={p_unknown}")
    if exclude_techs:
        print(f"  Excluding techs: {exclude_techs}")

    train_ds, val_ds, test_ds, normalizer = load_and_split_bsimar(
        data_path, OUTPUT_COLUMN_ORDER, device_type=device_type,
        apply_filter=apply_filter, exclude_techs=exclude_techs,
        norm_mode=_NORM_MODE, max_rows=max_rows,
        tech_scope=tech_scope,
        split_mode=split_mode,
    )
    _assert_codes_in_vocab(
        (train_ds, val_ds, test_ds), num_tech_codes, tech_scope)
    in_dim = train_ds.inputs.shape[1]
    out_dim = train_ds.outputs.shape[1]

    model = TransformerEncoderModel(
        input_dim=in_dim, target_dim=out_dim,
        d_model=config.d_model, nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        num_tech_codes=num_tech_codes,
        tech_embed_dropout=p_unknown,
        # Rule 16: UNKNOWN at the tail of the (possibly local) vocab.
        unknown_code_id=num_tech_codes - 1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")

    if init_from is not None:
        init_path = Path(init_from)
        if not init_path.suffix:
            init_path = CHECKPOINT_DIR / f"{init_from}_best.pt"
        if not init_path.exists():
            raise FileNotFoundError(
                f"init_from checkpoint not found: {init_path}")
        init_state = torch.load(str(init_path), weights_only=True,
                                map_location=device)
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"init_from architecture mismatch for {init_path.name}: "
                f"missing={list(missing)} unexpected={list(unexpected)}")
        print(f"  Warm-started from {init_path.name}")

    arch_config = {
        "input_dim": in_dim, "target_dim": out_dim,
        "d_model": config.d_model, "nhead": config.nhead,
        "num_layers": config.num_layers,
        "dim_feedforward": config.dim_feedforward,
        "dropout": config.dropout,
        "num_tech_codes": num_tech_codes,
    }
    return _train_loop(
        model=model, is_transformer=True,
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        normalizer=normalizer,
        epochs=epochs, batch_size=batch_size,
        lr=lr, weight_decay=config.weight_decay,
        patience=patience, save_prefix=save_prefix,
        device=device, overwrite=overwrite,
        arch_config=arch_config,
        class_weights=class_weights,
        swa_mode=swa_mode, ema_decay=ema_decay,
        sobolev=sobolev, lam_sobolev=lam_sobolev,
        sobolev_floor=sobolev_floor,
        sobolev_strong_boost=sobolev_strong_boost,
        sobolev_corridor_only=sobolev_corridor_only,
        subthresh=subthresh, lam_subthresh=lam_subthresh,
        subthresh_s2=subthresh_s2, subthresh_upper=subthresh_upper,
        subthresh_floor=subthresh_floor,
        subthresh_off_floor=subthresh_off_floor,
        subthresh_ceiling_k=subthresh_ceiling_k,
        subthresh_ceiling_w=subthresh_ceiling_w,
        charge_sobolev=charge_sobolev,
        lam_charge_sobolev=lam_charge_sobolev,
        charge_sobolev_floor=charge_sobolev_floor,
        amp=amp,
    )
