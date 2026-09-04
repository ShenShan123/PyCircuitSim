"""Single full-terminal training loop for DirectNet and BSIM-AR.

Both architectures share the same data, normaliser, LDS-MAE loss, cosine
schedule, and early-stop pattern. The only differences:

* the Transformer's ``forward`` uses teacher forcing by default, with an
  opt-in deployed-rollout path for full-terminal fine-tuning, and runs
  autoregressively at eval time;
* the Transformer trains in its declared charge-first order and saves an
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
from neural_network.data.contracts import (
    BSIMAR_FULL_TERMINAL_COLUMN_ORDER,
    FULL_TERMINAL_OUTPUT_COLUMN_ORDER,
    FULL_TERMINAL_OUTPUT_CONTRACT,
)
from neural_network.data.normalize import (
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


def _full_terminal_dataset_provenance(data_path: str) -> Dict[str, object]:
    """Return the immutable dataset identity embedded in a model bundle."""
    path = Path(data_path)
    marker_path = path.with_suffix(path.suffix + ".complete")
    if not marker_path.is_file():
        raise ValueError(
            f"full-terminal dataset completion marker is missing: {marker_path}"
        )
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"invalid full-terminal dataset marker: {marker_path}"
        ) from exc
    if not isinstance(marker, dict) or marker.get("dataset") != path.name:
        raise ValueError("full-terminal dataset marker names a different file")
    dataset_sha256 = _sha256_file(path)
    if marker.get("dataset_sha256") != dataset_sha256:
        raise ValueError("full-terminal dataset checksum does not match marker")
    source_commit = marker.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ValueError("full-terminal dataset source commit is invalid")
    if marker.get("source_dirty") is not False:
        raise ValueError("full-terminal dataset came from a dirty source tree")
    return {
        "dataset": path.name,
        "dataset_sha256": dataset_sha256,
        "dataset_completion_marker": marker_path.name,
        "dataset_completion_marker_sha256": _sha256_file(marker_path),
        "dataset_source_commit": source_commit,
    }


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
    subthresh_loss: Optional[nn.Module] = None,
    aux_norm: Optional[Dict[str, torch.Tensor]] = None,
    amp: bool = False,
    clip_grad: bool = False,
    autoregressive_training: bool = False,
) -> Tuple[float, float]:
    """Run one epoch and return mean total and auxiliary loss."""
    model.train()
    # V7.0.2 — accumulate on-device. The old ``float(loss.item())`` per
    # batch forced a host sync every step (~800/epoch) for a scalar only
    # printed once per epoch.
    total = torch.zeros((), device=device)
    total_aux = torch.zeros((), device=device)
    n = 0
    for x, y, tc, w in loader:
        x, y, tc, w = (x.to(device), y.to(device),
                       tc.to(device), w.to(device))
        optimizer.zero_grad()
        with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            pred = (
                model(
                    x,
                    None if autoregressive_training else y,
                    tech_codes=tc,
                )
                if is_transformer else model(x, tech_codes=tc)
            )
            loss = criterion(pred.float(), y, weights=w)
        if subthresh_loss is not None:
            sub = subthresh_loss(
                x_norm=x, y_pred_norm=pred, y_true_norm=y,
                in_mean=aux_norm["in_mean"], in_std=aux_norm["in_std"],
                out_std=aux_norm["out_std"], out_mean=aux_norm["out_mean"],
                asinh_scale=aux_norm["asinh_scale"])
            loss = loss + sub
            total_aux += sub.detach()
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
    autoregressive: bool = False,
) -> float:
    """Evaluate one epoch, optionally matching Transformer deployment."""
    model.eval()
    total = torch.zeros((), device=device)
    n = 0
    for x, y, tc in loader:
        x, y, tc = x.to(device), y.to(device), tc.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=amp):
            if is_transformer:
                pred = model(
                    x,
                    None if autoregressive else y,
                    tech_codes=tc,
                )
            else:
                pred = model(x, tech_codes=tc)
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
    class_weights: Optional[Dict[str, float]] = None,
    swa_mode: str = "none",
    ema_decay: float = 0.999,
    subthresh: bool = False,
    lam_subthresh: float = 0.05,
    subthresh_s2: float = 1e-9,
    subthresh_upper: float = 1e-6,
    subthresh_floor: float = 1e-12,
    subthresh_off_floor: float = 1e-10,
    subthresh_ceiling_k: float = 1.0,
    subthresh_ceiling_w: float = 1.0,
    amp: bool = False,
    clip_grad: bool = False,
    autoregressive_validation: bool = False,
    autoregressive_training: bool = False,
) -> Tuple[nn.Module, _NormalizerBase]:
    if amp:
        print("  AMP: bf16 autocast ON (train + validation)")
    if autoregressive_validation and not is_transformer:
        raise ValueError(
            "autoregressive_validation requires a Transformer model")
    if autoregressive_training and not is_transformer:
        raise ValueError(
            "autoregressive_training requires a Transformer model")
    normalized_columns = list(
        normalizer.stats.output_columns
        if normalizer.stats.output_columns is not None
        else FULL_TERMINAL_OUTPUT_COLUMN_ORDER
    )
    if normalized_columns != list(FULL_TERMINAL_OUTPUT_COLUMN_ORDER):
        raise ValueError(
            "normalizer columns do not match the full-terminal contract: "
            f"{normalized_columns}"
        )
    ordered_transformer_columns: Optional[list[str]] = None
    if is_transformer:
        ordered_transformer_columns = list(BSIMAR_FULL_TERMINAL_COLUMN_ORDER)
        if (len(ordered_transformer_columns) != len(normalized_columns)
                or set(ordered_transformer_columns) != set(normalized_columns)):
            raise ValueError(
                "Transformer target columns must be a permutation of the "
                f"normalizer columns: target={ordered_transformer_columns}, "
                f"normalizer={normalized_columns}")
        permutation = [
            normalized_columns.index(name)
            for name in ordered_transformer_columns
        ]
        for ds in (train_ds, val_ds, test_ds):
            ds.outputs = torch.tensor(
                ds.outputs.numpy()[:, permutation], dtype=torch.float32)
        print(f"  Transformer target order: {ordered_transformer_columns}")
        validation_mode = (
            "autoregressive" if autoregressive_validation else "teacher-forced"
        )
        training_mode = (
            "autoregressive" if autoregressive_training else "teacher-forced"
        )
        print(f"  Transformer training mode: {training_mode}")
        print(f"  Transformer validation mode: {validation_mode}")

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

    # The subthreshold loss reads physical-space stats. Transformer output
    # tensors use their charge-first order, so permute only those stats.
    def _aux_cols_and_stats() -> Tuple[list, np.ndarray, np.ndarray, np.ndarray]:
        st = normalizer.stats
        if st.mode != "asinh" or st.asinh_scale is None:
            raise ValueError(
                "Aux losses require asinh output normalization")
        base_cols = (st.output_columns if st.output_columns is not None
                     else list(FULL_TERMINAL_OUTPUT_COLUMN_ORDER))
        if not is_transformer:
            return base_cols, st.output_std, st.output_mean, st.asinh_scale
        assert ordered_transformer_columns is not None
        perm = [base_cols.index(c) for c in ordered_transformer_columns]
        return (ordered_transformer_columns, st.output_std[perm],
                st.output_mean[perm], st.asinh_scale[perm])

    def _nt(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=device)

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
            subthresh_loss=subthresh_loss, aux_norm=aux_norm,
            amp=amp, clip_grad=clip_grad,
            autoregressive_training=autoregressive_training)
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
            amp=amp, autoregressive=autoregressive_validation)
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
            extra = (
                f" aux={train_aux:.5f}"
                if subthresh_loss is not None else ""
            )
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
        assert ordered_transformer_columns is not None
        inverse = [
            ordered_transformer_columns.index(name)
            for name in normalized_columns
        ]
        pred_norm = pred_norm[:, inverse]
        true_norm = true_norm[:, inverse]

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
    save_prefix: str = "refac_dnf_medium_nmos",
    exclude_techs: Optional[Set[str]] = None,
    num_tech_codes: int = NUM_TSMC_CODES_WITH_UNKNOWN,
    p_unknown: float = 0.1,
    max_rows: Optional[int] = None,
    overwrite: bool = False,
    tech_scope: str = "universal",
    swa_mode: str = "none",
    ema_decay: float = 0.999,
    class_weights: Optional[Dict[str, float]] = None,
    init_from: Optional[str] = None,
    amp: bool = False,
    split_mode: str = "combo",
    training_overlay_classes: Optional[Set[str]] = None,
) -> Tuple[nn.Module, _NormalizerBase]:
    """Train a six-surface DirectNet-Full checkpoint bundle."""
    from neural_network.models.direct_net import DirectNet

    device = torch.device(device_str)
    print(f"DirectNet on {device}; tech codes={num_tech_codes}, "
          f"p_unknown={p_unknown}")
    if exclude_techs:
        print(f"  Excluding techs: {exclude_techs}")

    dataset_provenance = _full_terminal_dataset_provenance(data_path)
    train_ds, val_ds, test_ds, normalizer = load_and_split_bsimar(
        data_path, device_type=device_type,
        train_ratio=config.train_ratio, val_ratio=config.val_ratio,
        exclude_techs=exclude_techs,
        norm_mode=_NORM_MODE, max_rows=max_rows,
        tech_scope=tech_scope,
        split_mode=split_mode,
        training_overlay_classes=training_overlay_classes,
    )
    _assert_codes_in_vocab(
        (train_ds, val_ds, test_ds), num_tech_codes, tech_scope)
    in_dim = train_ds.inputs.shape[1]
    out_dim = train_ds.outputs.shape[1]
    # Place UNKNOWN at the last slot of whatever vocab we have. Universal
    # vocab=18 keeps unknown=17 (existing convention); per-tech vocab=5
    # (TSMC5) → unknown=4 / vocab=4 (TSMC7) → unknown=3. Without this,
    # `p_unknown` training-time dropout would write code 17 into a 5-row
    # embedding and trigger a CUDA assert.
    model = DirectNet(
        input_dim=in_dim, hidden_dim=config.trunk_hidden,
        n_layers=config.trunk_layers + 1, output_dim=out_dim,
        num_tech_codes=num_tech_codes,
        tech_embed_dim=32, tech_embed_dropout=p_unknown,
        unknown_code_id=num_tech_codes - 1,
    ).to(device)
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
        class_weights=class_weights,
        swa_mode=swa_mode, ema_decay=ema_decay,
        amp=amp,
    )
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
        **dataset_provenance,
    }, sort_keys=True, indent=2) + "\n")
    return trained


def train_transformer(
    data_path: str,
    save_prefix: str = "refac_tff_medium_nmos",
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
    class_weights: Optional[Dict[str, float]] = None,
    subthresh: bool = False,
    lam_subthresh: float = 0.05,
    subthresh_s2: float = 1e-9,
    subthresh_upper: float = 1e-6,
    subthresh_floor: float = 1e-12,
    subthresh_off_floor: float = 1e-10,
    subthresh_ceiling_k: float = 1.0,
    subthresh_ceiling_w: float = 1.0,
    init_from: Optional[str] = None,
    amp: bool = False,
    split_mode: str = "combo",
    training_overlay_classes: Optional[Set[str]] = None,
    full_terminal_ar_target_dim: Optional[int] = None,
    autoregressive_training: bool = False,
) -> Tuple[nn.Module, _NormalizerBase]:
    """Train a six-surface BSIM-AR-Full checkpoint bundle."""
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

    dataset_provenance = _full_terminal_dataset_provenance(data_path)
    ar_target_dim = (
        len(BSIMAR_FULL_TERMINAL_COLUMN_ORDER)
        if full_terminal_ar_target_dim is None
        else int(full_terminal_ar_target_dim)
    )
    if ar_target_dim not in (3, 6):
        raise ValueError(
            "BSIM-AR-Full supports 3 or 6 autoregressive targets, got "
            f"{ar_target_dim}"
        )
    target_columns = list(BSIMAR_FULL_TERMINAL_COLUMN_ORDER)

    train_ds, val_ds, test_ds, normalizer = load_and_split_bsimar(
        data_path, device_type=device_type,
        exclude_techs=exclude_techs,
        norm_mode=_NORM_MODE, max_rows=max_rows,
        tech_scope=tech_scope,
        split_mode=split_mode,
        training_overlay_classes=training_overlay_classes,
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
        ar_target_dim=ar_target_dim,
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
    arch_config.update({
        "ar_target_dim": ar_target_dim,
        "output_contract": FULL_TERMINAL_OUTPUT_CONTRACT,
        "target_columns": target_columns,
        "training_mode": (
            "autoregressive" if autoregressive_training
            else "teacher-forced"
        ),
        "validation_mode": "autoregressive",
    })
    trained = _train_loop(
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
        subthresh=subthresh, lam_subthresh=lam_subthresh,
        subthresh_s2=subthresh_s2, subthresh_upper=subthresh_upper,
        subthresh_floor=subthresh_floor,
        subthresh_off_floor=subthresh_off_floor,
        subthresh_ceiling_k=subthresh_ceiling_k,
        subthresh_ceiling_w=subthresh_ceiling_w,
        amp=amp,
        autoregressive_validation=True,
        autoregressive_training=autoregressive_training,
    )
    checkpoint_path = CHECKPOINT_DIR / f"{save_prefix}_best.pt"
    norm_path = CHECKPOINT_DIR / f"{save_prefix}_norm.npz"
    config_path = CHECKPOINT_DIR / f"{save_prefix}_config.npz"
    marker_path = checkpoint_path.with_suffix(".pt.complete")
    marker_path.write_text(json.dumps({
        "family": "bsimar-full",
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "normalization": norm_path.name,
        "normalization_sha256": _sha256_file(norm_path),
        "configuration": config_path.name,
        "configuration_sha256": _sha256_file(config_path),
        "output_columns": list(FULL_TERMINAL_OUTPUT_COLUMN_ORDER),
        "target_columns": target_columns,
        "ar_target_dim": ar_target_dim,
        "training_mode": (
            "autoregressive" if autoregressive_training
            else "teacher-forced"
        ),
        **dataset_provenance,
    }, sort_keys=True, indent=2) + "\n")
    return trained
