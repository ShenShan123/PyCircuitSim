"""V6.4.7 S9b (plan rev-3 ruling 4) — derivative-fidelity + off-state probe.

For one candidate (tech, nmos_stem, pmos_stem) this module compares the RAW
NETWORK's autograd derivatives of the ``id`` head against the dataset's
analytic OSDI columns ``gm/gds/gmb`` on the held-out TEST split, and probes
the hard-OFF region (the Mpl-class +0.50 uA-vs-~0 failure). It is the
scorer-side measurement for plan P4's derivative-fidelity gate (P0-B
baseline: gds NRMSE 20-23 %).

Train-frame metric (plan P4): the forward chain is
``raw V -> z-score with the CHECKPOINT's saved norm stats -> network ->
id head (normalized asinh space) -> denorm``, with NO inference-time
softplus clamp and NO Rule-15 Vds correction — this is the surface the
Sobolev term would supervise, not the post-corrected one NR consumes.

Sign convention (the P0-I §2 trap, settled empirically + per the P0-B
header ``scripts/v6_4_6_p0b_ro_overlay.py:15-18``):

    stored id  : NMOS conducting id < 0 ; PMOS conducting id > 0
    stored gm  = -d(id_stored)/dVg   (positive at strong inversion)
    stored gds = -d(id_stored)/dVd   (true physical, positive)
    stored gmb = -d(id_stored)/dVb   (positive)

i.e. the OSDI opvars are positive-magnitude conductances while the stored
``id`` keeps the PyCMG terminal sign, so the convention map from autograd
to the OSDI columns is a uniform NEGATION of all three channels for BOTH
device types. Verified empirically on TSMC7 (strong inversion, |id|>1e-5):
NMOS id 94 % negative with gm/gds/gmb 92-100 % positive; PMOS id 94 %
positive with gm/gds/gmb 95-100 % positive. The negation is a documented,
sign-preserving convention map (NOT an abs()); sign-agreement fractions are
reported both raw-frame and convention-mapped so a *residual* systematic
flip (a real checkpoint defect) stays visible.

Split faithfulness: the test split is re-derived with the EXACT loader
logic (``filter_small_targets`` + exclude-techs mask -> combined boolean
keep -> ``np.random.default_rng(42).permutation`` -> tail past
train/val at DirectNetConfig ratios 0.8/0.1) so that RAW rows are
addressable, and then CROSS-CHECKED against an actual
``load_and_split_bsimar`` call (same args ``train_directnet`` uses): split
sizes, normalized inputs/outputs and local tech codes must match. Both
paths are run because the loader only returns normalized tensors while the
metrics need physical units; the cross-check makes the re-derivation
provably split-faithful instead of assumed.

Reported populations per channel (all Rule-16 quartets):
  overall            — full held-out test split (the scorer headline
                       ``deriv_*_nrmse`` keys; spec-literal).
  strong_gt_1uA      — |id_true| > 1 uA.
  fwd_inrail         — polarity-correct Vd (NMOS Vd>=0 / PMOS Vd<=0),
                       |Vd|,|Vg| <= VDD, |Vb| <= 0.1*VDD — the
                       operating-corridor population comparable to the
                       P0-B RO-trajectory numbers (gds 20-23 %); merged
                       as ``deriv_*_nrmse_fwd`` (the A/B-sensitive keys).
  fwd_inrail_strong  — fwd_inrail AND |id_true| > 1 uA (opamp-gain rows).

KNOWN SIGNAL (anchor run 2026-06-12, debugged before shipping): the
full-split aggregates are dominated by a small population of high-|id|
BIAS-FLAT rows (largely T=398 K leakage plateaus, e.g. TSMC7 PMOS
Vd=0/Vg=-0.75/NFIN=25/L=11n: id=-25.17 mA with gm=2.3e-7 S — re-verified
EXACTLY against a fresh OSDI eval, so they are genuine ground truth, not
generator junk; ~32-36 % of |id|>1uA rows sit below g_sum/|id|=1 /V as a
continuum, so no clean exclusion exists). The network value-matches id on
those rows to <1 % but carries O(1) asinh-space slope where OSDI is flat,
inflating overall NRMSE (PMOS ~880 % vs NMOS gds ~15-20 % which lands in
the P0-B band). Compare candidates on the SAME population; use
``fwd_inrail`` for P0-B-anchored corridor reads.

Off-state probe eligibility: |Vg| <= 0.02*VDD, polarity-correct Vd with
|Vd| in [0.4*VDD, 1.05*VDD], |Vb| <= 0.05*VDD, and NOT in train/val.
(The spec's plain ``|Vd| >= 0.4*VDD`` mask was measured to admit
overshoot/body-diode rows with |id_true| up to ~1 A, which would swamp
the Mpl-class hard-OFF signal — the plan §S19/254 wording is
"Vgs=0/Vds=VDD biases", i.e. the conducting-direction in-rail cell.)
Rows dropped by the |id|<=1e-15 loader filter never reached training, so
they are held-out by construction and are INCLUDED — they are precisely
the hard-OFF class the probe targets.

Usage:
    conda run -n pycircuitsim python scripts/v6_4_7_deriv_fidelity.py \
        --tech TSMC7 --nmos tsmc7_dn_medium_nmos --pmos tsmc7_dn_medium_pmos \
        --selfcheck --json

All bsimar imports are lazy (inside functions) per the documented
BSIMAR_CHECKPOINT_DIR env-var ordering constraint in
``scripts/eval_v6_4_5_candidate.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

if TYPE_CHECKING:  # runtime bsimar imports stay lazy (module docstring)
    from bsimar.data.normalize import NormStats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "external_compact_models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CKPT_DIR = PROJECT_ROOT / "external_compact_models" / "bsimar" / "checkpoints"
DATA_DIR = (PROJECT_ROOT / "external_compact_models" / "bsimar"
            / "data" / "datasets")

# Mirrors bsimar/cli/train.py:_ALL_TECH_NAMES (per-tech auto-exclude set).
_ALL_TECH_NAMES = ("tsmc5", "tsmc7", "tsmc12", "tsmc16", "asap7")

# channel -> voltage input column of the 4-col [Vd, Vg, Vs, Vb] block.
_CHAN_VCOL: Dict[str, int] = {"gm": 1, "gds": 0, "gmb": 3}
# channel -> OUTPUT_COLUMN_ORDER index of the analytic OSDI column.
_CHAN_TRUE_COL: Dict[str, int] = {"gm": 1, "gds": 2, "gmb": 3}

_SI_ID_FLOOR = 1e-5        # strong-inversion |id| floor for sign agreement
_STRONG_ID_FLOOR = 1e-6    # |id| > 1 uA subset (opamp-gain-relevant rows)
_EVAL_SEED = 123           # deterministic test-split subsample
_OFFSTATE_SEED = 124       # deterministic off-state subsample
_SELFCHECK_SEED = 7        # deterministic FD row pick
_BATCH = 16384
# Checkpoint-vs-fresh-fit norm-stats agreement tolerance (relative,
# array-wide). A checkpoint trained on exactly this (dataset, filter,
# exclude) reproduces the fresh fit bit-equal (measured 0.0 on the V6.4.4
# anchors); any dataset/filter mismatch moves the train-split stats by
# orders of magnitude more than this. Exceeding it means the candidate is
# being scored against a population it was NOT trained on — refuse to emit
# silent numbers (see ``allow_stats_mismatch``).
_STATS_REL_TOL = 1e-6


# ── Model / checkpoint loading ─────────────────────────────────────────────

def _build_directnet_from_state(
    state: Dict[str, torch.Tensor],
) -> torch.nn.Module:
    """Rebuild a DirectNet from its state_dict shapes.

    Mirrors ``pycircuitsim/models/mosfet_directnet.py:_build_from_state``
    (incl. Phase-7a ``mono.*`` auto-detection) so the measured surface is
    the same module the simulator runs.
    """
    from bsimar.models.direct_net import DirectNet

    net_keys = [k for k in state
                if k.startswith("net.") and k.endswith(".weight")]
    output_dim = state[net_keys[-1]].shape[0]
    hidden_dim = state[net_keys[-1]].shape[1]
    n_layers = len(net_keys) - 1
    num_tech_codes = state["tech_embedding.weight"].shape[0]
    tech_embed_dim = state["tech_embedding.weight"].shape[1]
    input_dim = state[net_keys[0]].shape[1] - tech_embed_dim
    monotonic = any(k.startswith("mono.") for k in state)
    monotone_sign = 1.0
    monotone_hidden = 64
    if monotonic:
        monotone_sign = float(state["mono.sign"].item())
        monotone_hidden = state["mono.w_vg_raw"].shape[0]
    model = DirectNet(
        input_dim=input_dim, hidden_dim=hidden_dim,
        n_layers=n_layers, output_dim=output_dim,
        num_tech_codes=num_tech_codes, tech_embed_dim=tech_embed_dim,
        monotonic=monotonic, monotone_sign=monotone_sign,
        monotone_hidden=monotone_hidden,
    )
    model.load_state_dict(state)
    model.eval()
    return model


def _infer_scope(tech: str, num_tech_codes: int) -> str:
    """Embedding vocab size -> tech scope (Rule 19)."""
    from bsimar.config import LOCAL_VOCAB_SIZE, NUM_TSMC_CODES_WITH_UNKNOWN

    scope = tech.lower()
    if num_tech_codes == LOCAL_VOCAB_SIZE.get(scope):
        return scope
    if num_tech_codes == NUM_TSMC_CODES_WITH_UNKNOWN:
        return "universal"
    raise ValueError(
        f"Cannot infer tech scope for tech={tech}: embedding vocab "
        f"{num_tech_codes} matches neither local "
        f"{LOCAL_VOCAB_SIZE.get(scope)} nor universal "
        f"{NUM_TSMC_CODES_WITH_UNKNOWN}")


def _local_code_lut(scope: str, max_code: int) -> np.ndarray:
    """universal tech-code -> local-vocab code LUT.

    Mirrors ``bsimar/data/dataset.py`` tech_scope remap (out-of-scope
    rows collapse to the local UNKNOWN slot).
    """
    from bsimar.config import (
        CODE_TO_TECH_VARIANT, LOCAL_UNKNOWN_CODE_ID, LOCAL_VARIANT_CODES)

    table = LOCAL_VARIANT_CODES[scope]
    unk = LOCAL_UNKNOWN_CODE_ID[scope]
    lut = np.full(max_code + 1, unk, dtype=np.int64)
    for c in range(max_code + 1):
        tv = CODE_TO_TECH_VARIANT.get(c)
        if tv is not None and tv[0] == scope:
            lut[c] = table.get(tv, unk)
    return lut


# ── Split re-derivation (loader-faithful) ──────────────────────────────────

def _derive_split(
    data_path: Path,
    device_type: str,
    apply_filter: bool,
    exclude_techs: Optional[set],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicate ``load_and_split_bsimar``'s row selection on raw arrays.

    Returns (test_global_idx, trainval_global_idx, keep_mask). The loader
    applies the filter then the exclude mask sequentially; a combined
    boolean keep is row-set identical and order-preserving, and the split
    permutation is the loader's exact single
    ``np.random.default_rng(seed).permutation(n_kept)`` call (no other RNG
    draw precedes it when max_rows is None — asserted by the caller's
    cross-check against the real loader).
    """
    from bsimar.data.dataset import filter_small_targets
    from bsimar.data.normalize import OUTPUT_COLUMN_ORDER
    from bsimar.eval.loo_labels import get_or_build_tech_variant_labels

    data = np.load(data_path, allow_pickle=True)
    outputs = data["outputs"]
    keep = np.ones(len(outputs), dtype=bool)
    if apply_filter:
        keep &= filter_small_targets(outputs, OUTPUT_COLUMN_ORDER)
    if exclude_techs:
        from bsimar.config import TECH_VARIANT_CODES
        codes = get_or_build_tech_variant_labels(
            str(data_path), device_type, verbose=False)
        excl = {code for (t, _), code in TECH_VARIANT_CODES.items()
                if t in exclude_techs}
        keep &= np.array([int(c) not in excl for c in codes], dtype=bool)

    kept_global = np.flatnonzero(keep)
    n = len(kept_global)
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    test_global = kept_global[perm[n_train + n_val:]]
    trainval_global = kept_global[perm[:n_train + n_val]]
    return test_global, trainval_global, keep


# ── Metrics ────────────────────────────────────────────────────────────────

def _rule16(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """Rule-16 quartet, identical to v6_4_6_p0b_ro_overlay.rule16 so the
    gds NRMSE is directly comparable to the P0-B 20-23 % baseline."""
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    diff = pred - true
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-300 else float("nan")
    ptp = float(np.ptp(true))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    nrmse = rmse / ptp * 100.0 if ptp > 1e-300 else float("nan")
    scale = max(float(np.max(np.abs(true))), 1e-30)
    denom = np.maximum(np.abs(true), 1e-3 * scale)
    mre = float(np.mean(np.abs(diff) / denom)) * 100.0
    return {"mre_pct": mre, "r2": r2, "nrmse_pct": nrmse,
            "max_err": float(np.max(np.abs(diff)))}


# ── Forward / autograd chain ───────────────────────────────────────────────

def _normalize_inputs_ck(
    stats: "NormStats", inputs: np.ndarray, geometry: np.ndarray,
) -> np.ndarray:
    """Raw [Vd,Vg,Vs,Vb] + geometry -> 7-col z-score with CHECKPOINT stats
    (train-frame: no softplus clamp)."""
    from bsimar.data.normalize import _build_combined_input

    combined = _build_combined_input(inputs, geometry)
    in_std = stats.input_std.copy()
    in_std[in_std < 1e-12] = 1.0  # defensive, mirrors _setup_gpu
    return (combined - stats.input_mean) / in_std


def _forward_id_grads(
    model: torch.nn.Module,
    x_norm: np.ndarray,
    tech_codes: np.ndarray,
    id_col: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    need_grad: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Batched forward of the id head; optional per-row d(id_norm)/d(x_norm)
    over the 4 voltage columns (rows are independent, so the grad of the
    column-sum is the per-row gradient — same trick as
    ``_MOSFETNNBase._eval``)."""
    id_out: List[np.ndarray] = []
    grads: List[np.ndarray] = []
    n = len(x_norm)
    for i in range(0, n, _BATCH):
        xv = torch.tensor(x_norm[i:i + _BATCH, :4], dtype=dtype,
                          device=device, requires_grad=need_grad)
        xg = torch.tensor(x_norm[i:i + _BATCH, 4:], dtype=dtype,
                          device=device)
        tc = torch.tensor(tech_codes[i:i + _BATCH], dtype=torch.long,
                          device=device)
        x_full = torch.cat([xv, xg], dim=1)
        if need_grad:
            with torch.enable_grad():
                out = model(x_full, tech_codes=tc)
                idn = out[:, id_col]
                g = torch.autograd.grad(idn.sum(), xv)[0]
            grads.append(g.detach().cpu().numpy())
        else:
            with torch.no_grad():
                out = model(x_full, tech_codes=tc)
                idn = out[:, id_col]
        id_out.append(idn.detach().cpu().numpy())
    return (np.concatenate(id_out),
            np.concatenate(grads) if need_grad else None)


def _denorm_id(
    stats: "NormStats", id_norm: np.ndarray, id_col: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """id head normalized -> (id_phys, output jacobian factor).

    Vectorized mirror of ``_NormalizerBase.denormalize_derivative``'s
    output side: factor = sqrt(s^2 + id_phys^2) for asinh, 1 for zscore.
    """
    out_std = float(stats.output_std[id_col])
    out_mean = float(stats.output_mean[id_col])
    u = id_norm.astype(np.float64) * out_std + out_mean
    if stats.mode == "asinh":
        s = float(stats.asinh_scale[id_col])
        id_phys = s * np.sinh(u)
        factor = np.sqrt(s * s + id_phys * id_phys)
    else:
        id_phys = u
        factor = np.ones_like(u)
    return id_phys, factor


def _phys_derivs(
    stats: "NormStats", grads_norm: np.ndarray, id_norm: np.ndarray,
    id_col: int,
) -> Dict[str, np.ndarray]:
    """d(id_norm)/d(x_norm) -> OSDI-convention physical gm/gds/gmb.

    Chain rule per ``denormalize_derivative``:
        d id_phys / d V_j = g_norm[:, j] * out_std_id * factor / in_std[j]
    then the documented convention map (module docstring): stored
    gm/gds/gmb = -d(id_stored)/dV{g,d,b}, so all three channels are
    NEGATED — a sign-preserving map, not an abs().
    """
    out_std = float(stats.output_std[id_col])
    _, factor = _denorm_id(stats, id_norm, id_col)
    in_std = stats.input_std.copy()
    in_std[in_std < 1e-12] = 1.0
    derivs: Dict[str, np.ndarray] = {}
    for ch, vcol in _CHAN_VCOL.items():
        d_phys = (grads_norm[:, vcol].astype(np.float64)
                  * out_std * factor / float(in_std[vcol]))
        derivs[ch] = -d_phys  # convention map -> OSDI column frame
    return derivs


# ── Self-check: float64 central finite differences ─────────────────────────

def _selfcheck_fd(
    state: Dict[str, torch.Tensor],
    stats: "NormStats",
    raw_inputs: np.ndarray,
    raw_geometry: np.ndarray,
    tech_codes: np.ndarray,
    id_col: int,
    n_rows: int = 200,
    h: float = 1e-4,
) -> Dict[str, float]:
    """Central FD of the full physical chain vs autograd, in float64.

    float64 (model cloned + ``.double()``) so the FD cancellation noise of
    the float32 forward (~1e-7 relative) cannot pollute the <0.5 % gate;
    the check targets CHAIN correctness (normalization, autograd wiring,
    denorm jacobian), which is dtype-independent.
    """
    model64 = _build_directnet_from_state(state).double()
    rng = np.random.default_rng(_SELFCHECK_SEED)
    pick = rng.choice(len(raw_inputs), size=min(n_rows, len(raw_inputs)),
                      replace=False)
    v = raw_inputs[pick].astype(np.float64)
    geo = raw_geometry[pick]
    tc = tech_codes[pick]
    dev = torch.device("cpu")

    def f_idphys(v_block: np.ndarray) -> np.ndarray:
        xn = _normalize_inputs_ck(stats, v_block, geo)
        idn, _ = _forward_id_grads(
            model64, xn, tc, id_col, dev, dtype=torch.float64,
            need_grad=False)
        id_phys, _ = _denorm_id(stats, idn, id_col)
        return id_phys

    # autograd reference in the same float64 chain
    xn0 = _normalize_inputs_ck(stats, v, geo)
    idn0, g0 = _forward_id_grads(
        model64, xn0, tc, id_col, dev, dtype=torch.float64, need_grad=True)
    ag = _phys_derivs(stats, g0, idn0, id_col)

    out: Dict[str, float] = {}
    for ch, vcol in _CHAN_VCOL.items():
        vp = v.copy(); vp[:, vcol] += h
        vm = v.copy(); vm[:, vcol] -= h
        fd = (f_idphys(vp) - f_idphys(vm)) / (2.0 * h)
        fd_conv = -fd  # same convention map as _phys_derivs
        denom = np.maximum(np.maximum(np.abs(ag[ch]), np.abs(fd_conv)),
                           1e-20)
        rel = np.abs(fd_conv - ag[ch]) / denom
        out[f"fd_med_rel_err_{ch}_pct"] = float(np.median(rel) * 100.0)
    out["n_rows"] = int(len(pick))
    out["h_volt"] = h
    return out


# ── Per-device evaluation ──────────────────────────────────────────────────

def _eval_device(
    tech: str,
    device_type: str,
    stem: str,
    ckpt_dir: Path,
    data_dir: Path,
    apply_filter: bool,
    max_eval_rows: int,
    selfcheck: bool,
    torch_device: str,
    verify_split: bool,
    data_suffix: Optional[str],
    allow_stats_mismatch: bool,
) -> Dict[str, object]:
    from bsimar.config import DirectNetConfig
    from bsimar.data.dataset import load_and_split_bsimar
    from bsimar.data.normalize import (
        NormStats, OUTPUT_COLUMN_ORDER)

    pt = ckpt_dir / f"{stem}_best.pt"
    nz = ckpt_dir / f"{stem}_norm.npz"
    infix = f"_{data_suffix}" if data_suffix else ""
    data_path = data_dir / f"{tech.lower()}{infix}_{device_type}.npz"
    for p in (pt, nz, data_path):
        if not p.exists():
            raise FileNotFoundError(str(p))

    state = torch.load(str(pt), weights_only=True, map_location="cpu")
    model = _build_directnet_from_state(state)
    dev = torch.device(torch_device)
    model.to(dev)
    stats = NormStats.load(str(nz))
    cols = (stats.output_columns if stats.output_columns is not None
            else OUTPUT_COLUMN_ORDER)
    if "id" not in cols:
        raise ValueError(f"checkpoint {stem} has no id head: {cols}")
    id_col = cols.index("id")

    scope = _infer_scope(tech, state["tech_embedding.weight"].shape[0])
    exclude = ({t for t in _ALL_TECH_NAMES if t != scope}
               if scope != "universal" else None)

    cfg = DirectNetConfig()
    seed = 42  # load_and_split_bsimar default; train_directnet never overrides

    # 1) loader-faithful raw split re-derivation
    test_idx, trainval_idx, keep = _derive_split(
        data_path, device_type, apply_filter, exclude,
        cfg.train_ratio, cfg.val_ratio, seed)

    data = np.load(data_path, allow_pickle=True)
    inputs = data["inputs"]
    geometry = data["geometry"]
    outputs = data["outputs"]
    vdd = float(data["meta_vdd"])

    # local tech codes for every global row (off-state rows included)
    from bsimar.eval.loo_labels import get_or_build_tech_variant_labels
    codes_u = np.asarray(get_or_build_tech_variant_labels(
        str(data_path), device_type, verbose=False), dtype=np.int64)
    if scope != "universal":
        lut = _local_code_lut(scope, int(codes_u.max()))
        codes_all = lut[codes_u]
    else:
        codes_all = codes_u

    # 2) cross-check vs the actual loader (split-faithfulness proof)
    split_check: Dict[str, object] = {}
    if verify_split:
        _, _, test_ds, fitted = load_and_split_bsimar(
            str(data_path), OUTPUT_COLUMN_ORDER, device_type=device_type,
            norm_mode="asinh", train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio, seed=seed, apply_filter=apply_filter,
            exclude_techs=exclude, tech_scope=scope)
        if len(test_ds) != len(test_idx):
            raise AssertionError(
                f"split mismatch: loader test={len(test_ds)} vs "
                f"re-derived={len(test_idx)}")
        x_chk = fitted.normalize_inputs(
            inputs[test_idx], geometry[test_idx]).astype(np.float32)
        d_in = float(np.max(np.abs(x_chk - test_ds.inputs.numpy())))
        y_chk = fitted.normalize_outputs(
            outputs[test_idx]).astype(np.float32)
        d_out = float(np.max(np.abs(y_chk - test_ds.outputs.numpy())))
        d_tc = int(np.sum(codes_all[test_idx]
                          != test_ds.tech_codes.numpy()))
        if d_in > 1e-6 or d_out > 1e-6 or d_tc != 0:
            raise AssertionError(
                f"split content mismatch: d_in={d_in} d_out={d_out} "
                f"tech_code_diffs={d_tc}")
        # Checkpoint stats vs freshly fit stats: bit-equal when the
        # checkpoint was trained on exactly this (dataset, filter); a
        # nonzero delta means the candidate is being scored against a
        # population it was NOT trained on (e.g. a v2-trained candidate
        # against the v1 npz, or the wrong --apply-filter) — refuse to
        # produce silent numbers unless explicitly overridden.
        fs = fitted.stats

        def _maxabs(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.max(np.abs(np.asarray(a, dtype=np.float64)
                                       - np.asarray(b, dtype=np.float64))))

        def _rel(a: np.ndarray, b: np.ndarray) -> float:
            scale = max(float(np.max(np.abs(
                np.asarray(b, dtype=np.float64)))), 1e-30)
            return _maxabs(a, b) / scale

        rel_deltas = {
            "input_mean": _rel(stats.input_mean, fs.input_mean),
            "input_std": _rel(stats.input_std, fs.input_std),
            "output_mean": _rel(stats.output_mean, fs.output_mean),
            "output_std": _rel(stats.output_std, fs.output_std),
        }
        stats_mismatch = any(v > _STATS_REL_TOL for v in rel_deltas.values())
        split_check = {
            "norm_input_check_maxabs": d_in,
            "norm_output_check_maxabs": d_out,
            "ck_vs_fit_input_mean_maxabs": _maxabs(
                stats.input_mean, fs.input_mean),
            "ck_vs_fit_input_std_maxabs": _maxabs(
                stats.input_std, fs.input_std),
            "ck_vs_fit_output_mean_maxabs": _maxabs(
                stats.output_mean, fs.output_mean),
            "ck_vs_fit_output_std_maxabs": _maxabs(
                stats.output_std, fs.output_std),
            "ck_vs_fit_rel_max": max(rel_deltas.values()),
            "stats_mismatch": bool(stats_mismatch),
        }
        if stats_mismatch:
            msg = (
                f"norm-stats mismatch for {stem}: checkpoint stats differ "
                f"from a fresh fit on {data_path.name} "
                f"(apply_filter={apply_filter}); rel deltas "
                + ", ".join(f"{k}={v:.3e}" for k, v in rel_deltas.items())
                + f" exceed tol {_STATS_REL_TOL:g}. The candidate was NOT "
                f"trained on this (dataset, filter) population — pass the "
                f"matching --data-suffix/--apply-filter (v2-trained arms "
                f"need --data-suffix v2 --no-apply-filter), or "
                f"--allow-stats-mismatch for an explicit cross-population "
                f"read.")
            if not allow_stats_mismatch:
                raise ValueError(msg)
            print(f"[deriv-fidelity] WARNING: {msg}", file=sys.stderr)
        del test_ds, fitted

    # 3) deterministic subsample of the test split
    rng = np.random.default_rng(_EVAL_SEED)
    if len(test_idx) > max_eval_rows:
        sub = np.sort(rng.choice(len(test_idx), size=max_eval_rows,
                                 replace=False))
        eval_idx = test_idx[sub]
    else:
        eval_idx = test_idx

    # 4) forward + autograd with the CHECKPOINT's stats (the network was
    # trained in that frame; it is also the frame NR consumes)
    x_norm = _normalize_inputs_ck(
        stats, inputs[eval_idx], geometry[eval_idx])
    idn, gn = _forward_id_grads(
        model, x_norm, codes_all[eval_idx], id_col, dev)
    id_pred, _ = _denorm_id(stats, idn, id_col)
    derivs = _phys_derivs(stats, gn, idn, id_col)

    id_true = outputs[eval_idx, 0]
    strong = np.abs(id_true) > _STRONG_ID_FLOOR
    si = np.abs(id_true) > _SI_ID_FLOOR
    # operating-corridor population (module docstring): conducting-direction
    # Vd, in-rail Vd/Vg, near-nominal body.
    pol = 1.0 if device_type == "nmos" else -1.0
    v_ev = inputs[eval_idx]
    fwd_inrail = ((pol * v_ev[:, 0] >= 0.0)
                  & (np.abs(v_ev[:, 0]) <= vdd)
                  & (np.abs(v_ev[:, 1]) <= vdd)
                  & (np.abs(v_ev[:, 3]) <= 0.1 * vdd))
    fwd_strong = fwd_inrail & strong

    def _sub(pred: np.ndarray, true: np.ndarray,
             mask: np.ndarray) -> Optional[Dict[str, float]]:
        return _rule16(pred[mask], true[mask]) if mask.sum() > 1 else None

    channels: Dict[str, Dict[str, object]] = {}
    for ch in _CHAN_VCOL:
        true = outputs[eval_idx, _CHAN_TRUE_COL[ch]]
        pred = derivs[ch]
        entry: Dict[str, object] = {
            "overall": _rule16(pred, true),
            "strong_gt_1uA": _sub(pred, true, strong),
            "fwd_inrail": _sub(pred, true, fwd_inrail),
            "fwd_inrail_strong": _sub(pred, true, fwd_strong),
            "n_strong": int(strong.sum()),
            "n_fwd_inrail": int(fwd_inrail.sum()),
            "n_fwd_inrail_strong": int(fwd_strong.sum()),
        }
        if si.sum() > 0:
            s_pred = np.sign(pred[si])
            s_true = np.sign(true[si])
            entry["sign_agree_si"] = float(np.mean(s_pred == s_true))
            # raw frame (pre-negation) for the P0-I §2 trap audit
            entry["sign_agree_si_rawframe"] = float(
                np.mean(np.sign(-pred[si]) == s_true))
            entry["n_si"] = int(si.sum())
        channels[ch] = entry

    # 5) off-state probe (tightened mask — module docstring): Vgs~0,
    # conducting-direction in-rail Vds, nominal body, not in train/val
    off_mask = ((np.abs(inputs[:, 1]) <= 0.02 * vdd)
                & (pol * inputs[:, 0] >= 0.4 * vdd)
                & (np.abs(inputs[:, 0]) <= 1.05 * vdd)
                & (np.abs(inputs[:, 3]) <= 0.05 * vdd))
    in_trainval = np.zeros(len(outputs), dtype=bool)
    in_trainval[trainval_idx] = True
    off_idx = np.flatnonzero(off_mask & ~in_trainval)
    rng_off = np.random.default_rng(_OFFSTATE_SEED)
    if len(off_idx) > max_eval_rows:
        off_idx = off_idx[np.sort(rng_off.choice(
            len(off_idx), size=max_eval_rows, replace=False))]
    offstate: Dict[str, object] = {"n_rows": int(len(off_idx))}
    if len(off_idx) > 0:
        x_off = _normalize_inputs_ck(
            stats, inputs[off_idx], geometry[off_idx])
        idn_off, _ = _forward_id_grads(
            model, x_off, codes_all[off_idx], id_col, dev, need_grad=False)
        id_off_pred, _ = _denorm_id(stats, idn_off, id_col)
        id_off_true = outputs[off_idx, 0]
        excess = np.maximum(np.abs(id_off_pred) - np.abs(id_off_true), 0.0)
        offstate.update({
            "id_pred_mean_abs_a": float(np.mean(np.abs(id_off_pred))),
            "id_pred_max_abs_a": float(np.max(np.abs(id_off_pred))),
            "id_true_mean_abs_a": float(np.mean(np.abs(id_off_true))),
            "id_true_max_abs_a": float(np.max(np.abs(id_off_true))),
            # the Mpl-class signal proper: predicted current IN EXCESS of
            # the true leakage (T=398 rows carry genuine mA-scale true
            # leakage even at Vgs~0, so max|id_pred| alone tracks them)
            "id_excess_mean_a": float(np.mean(excess)),
            "id_excess_max_a": float(np.max(excess)),
        })

    result: Dict[str, object] = {
        "stem": stem, "scope": scope, "apply_filter": apply_filter,
        "n_test": int(len(test_idx)), "n_eval": int(len(eval_idx)),
        "vdd": vdd,
        "split_check": split_check or None,
        "channels": channels,
        "offstate": offstate,
        "id_value": {
            "overall": _rule16(id_pred, id_true),
            "strong_gt_1uA": (_rule16(id_pred[strong], id_true[strong])
                              if strong.sum() > 1 else None),
        },
    }
    if selfcheck:
        result["selfcheck"] = _selfcheck_fd(
            state, stats, inputs[eval_idx], geometry[eval_idx],
            codes_all[eval_idx], id_col)
    return result


# ── Public API ─────────────────────────────────────────────────────────────

def compute_deriv_fidelity(
    tech: str,
    nmos_stem: str,
    pmos_stem: str,
    ckpt_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    apply_filter: bool = True,
    max_eval_rows: int = 50000,
    selfcheck: bool = False,
    torch_device: str = "cpu",
    verify_split: bool = True,
    data_suffix: Optional[str] = None,
    allow_stats_mismatch: bool = False,
) -> Dict[str, object]:
    """Derivative-fidelity + off-state metrics for one candidate pair.

    ``data_suffix``: dataset filename infix — ``"v2"`` selects
    ``{tech}_v2_{nmos,pmos}.npz`` (the S9b regen-v2 stems written by
    ``scripts/v6_4_7_s9b_regen_v2.sh``); None selects the v1
    ``{tech}_{nmos,pmos}.npz``. v2-trained candidates (control-v2 + all
    campaign arms, trained with the loader filter OFF) must be scored
    with ``data_suffix="v2", apply_filter=False`` — a mismatch is
    detected via the checkpoint-vs-fresh-fit norm-stats delta and RAISES
    unless ``allow_stats_mismatch=True`` (which downgrades it to a
    stderr warning + ``deriv_split_mismatch=1``).

    Returns a dict carrying the scorer summary keys:
      ``deriv_{gm,gds,gmb}_nrmse``      — worst-of NMOS/PMOS overall NRMSE %
      ``deriv_{gm,gds,gmb}_nrmse_fwd``  — same on the fwd_inrail corridor
                                          (P0-B-comparable; the full-split
                                          numbers are plateau-dominated, see
                                          KNOWN SIGNAL in the module
                                          docstring — use these for A/B)
      ``offstate_id_excess_max/mean``   — worst-of predicted |id| IN EXCESS
                                          of true leakage on hard-OFF rows
                                          (the Mpl-class +0.50 uA-vs-~0
                                          signal; headline off-state key)
      ``offstate_id_pred_max``          — worst-of max |id_pred| (secondary;
                                          tracks genuine T=398K leakage
                                          ground truth, NOT the NN's error)
      ``deriv_split_mismatch``          — 1 if either device's checkpoint
                                          norm stats mismatch the fresh fit
                                          (None if verify_split=False)
    plus full ``per_device`` detail.
    """
    ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else CKPT_DIR
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR

    per_device: Dict[str, Dict[str, object]] = {}
    for device_type, stem in (("nmos", nmos_stem), ("pmos", pmos_stem)):
        per_device[device_type] = _eval_device(
            tech, device_type, stem, ckpt_dir, data_dir, apply_filter,
            max_eval_rows, selfcheck, torch_device, verify_split,
            data_suffix, allow_stats_mismatch)

    def _worst(fn: Callable[[Dict[str, object]], float]) -> float:
        return max(fn(per_device["nmos"]), fn(per_device["pmos"]))

    out: Dict[str, object] = {
        "tech": tech, "nmos": nmos_stem, "pmos": pmos_stem,
        "apply_filter": apply_filter, "data_suffix": data_suffix,
    }
    for ch in _CHAN_VCOL:
        out[f"deriv_{ch}_nrmse"] = _worst(
            lambda d, c=ch: float(d["channels"][c]["overall"]["nrmse_pct"]))

    def _fwd_nrmse(d: Dict[str, object], c: str) -> float:
        m = d["channels"][c]["fwd_inrail"]  # type: ignore[index]
        return float(m["nrmse_pct"]) if m is not None else float("nan")

    for ch in _CHAN_VCOL:
        out[f"deriv_{ch}_nrmse_fwd"] = _worst(
            lambda d, c=ch: _fwd_nrmse(d, c))
    out["offstate_id_excess_max"] = _worst(
        lambda d: float(d["offstate"].get("id_excess_max_a", float("nan"))))
    out["offstate_id_excess_mean"] = _worst(
        lambda d: float(d["offstate"].get("id_excess_mean_a", float("nan"))))
    out["offstate_id_pred_max"] = _worst(
        lambda d: float(d["offstate"].get("id_pred_max_abs_a", float("nan"))))
    checks = [per_device[dv].get("split_check") for dv in ("nmos", "pmos")]
    out["deriv_split_mismatch"] = (
        int(any(bool(c["stats_mismatch"]) for c in checks))  # type: ignore[index]
        if all(checks) else None)
    out["per_device"] = per_device
    return out


# ── CLI ────────────────────────────────────────────────────────────────────

def _print_human(res: Dict[str, object]) -> None:
    suffix = res.get("data_suffix")
    print(f"\n=== deriv fidelity  {res['tech']}  nmos={res['nmos']}  "
          f"pmos={res['pmos']}  (filter={'on' if res['apply_filter'] else 'off'}, "
          f"data={'v1' if not suffix else suffix}) ===")
    for dev_name, d in res["per_device"].items():  # type: ignore[union-attr]
        print(f"\n  [{dev_name}] stem={d['stem']} scope={d['scope']} "
              f"n_test={d['n_test']} n_eval={d['n_eval']}")
        if d.get("split_check"):
            sc = d["split_check"]
            print(f"    split-check: norm-in maxabs={sc['norm_input_check_maxabs']:.2e} "
                  f"norm-out maxabs={sc['norm_output_check_maxabs']:.2e} "
                  f"ck-vs-fit rel_max={sc['ck_vs_fit_rel_max']:.2e} "
                  f"mismatch={int(sc['stats_mismatch'])}")
        hdr = (f"    {'chan':5s} {'MRE%':>10s} {'R2':>8s} {'NRMSE%':>8s} "
               f"{'MaxErr(S)':>11s} {'signSI':>7s}")
        print(hdr + "   (overall)")
        for ch, e in d["channels"].items():
            m = e["overall"]
            print(f"    {ch:5s} {m['mre_pct']:10.2f} {m['r2']:8.4f} "
                  f"{m['nrmse_pct']:8.2f} {m['max_err']:11.3e} "
                  f"{e.get('sign_agree_si', float('nan')):7.3f}")
        for sub, nkey, label in (
                ("strong_gt_1uA", "n_strong", "|id| > 1uA"),
                ("fwd_inrail", "n_fwd_inrail", "fwd in-rail (P0-B corridor)"),
                ("fwd_inrail_strong", "n_fwd_inrail_strong",
                 "fwd in-rail & |id|>1uA")):
            print(f"    --- {label} (n={d['channels']['gm'][nkey]}) ---")
            for ch, e in d["channels"].items():
                m = e[sub]
                if m is None:
                    continue
                print(f"    {ch:5s} {m['mre_pct']:10.2f} {m['r2']:8.4f} "
                      f"{m['nrmse_pct']:8.2f} {m['max_err']:11.3e}")
        off = d["offstate"]
        print(f"    off-state (n={off['n_rows']}): "
              f"|id_pred| mean={off.get('id_pred_mean_abs_a', float('nan')):.3e} A "
              f"max={off.get('id_pred_max_abs_a', float('nan')):.3e} A | "
              f"|id_true| mean={off.get('id_true_mean_abs_a', float('nan')):.3e} A "
              f"max={off.get('id_true_max_abs_a', float('nan')):.3e} A | "
              f"excess mean={off.get('id_excess_mean_a', float('nan')):.3e} A "
              f"max={off.get('id_excess_max_a', float('nan')):.3e} A")
        if d.get("selfcheck"):
            s = d["selfcheck"]
            print(f"    selfcheck FD(f64,h={s['h_volt']:g}): med rel err "
                  f"gm={s['fd_med_rel_err_gm_pct']:.2e}% "
                  f"gds={s['fd_med_rel_err_gds_pct']:.2e}% "
                  f"gmb={s['fd_med_rel_err_gmb_pct']:.2e}%  (gate <0.5%)")
    print(f"\n  summary: deriv_gm_nrmse={res['deriv_gm_nrmse']:.2f}% "
          f"deriv_gds_nrmse={res['deriv_gds_nrmse']:.2f}% "
          f"deriv_gmb_nrmse={res['deriv_gmb_nrmse']:.2f}%")
    print(f"           fwd_inrail: gm={res['deriv_gm_nrmse_fwd']:.2f}% "
          f"gds={res['deriv_gds_nrmse_fwd']:.2f}% "
          f"gmb={res['deriv_gmb_nrmse_fwd']:.2f}%")
    print(f"           offstate_id_excess max={res['offstate_id_excess_max']:.3e} A "
          f"mean={res['offstate_id_excess_mean']:.3e} A | "
          f"id_pred_max={res['offstate_id_pred_max']:.3e} A | "
          f"split_mismatch={res['deriv_split_mismatch']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tech", required=True)
    ap.add_argument("--nmos", required=True, help="nmos checkpoint stem")
    ap.add_argument("--pmos", required=True, help="pmos checkpoint stem")
    ap.add_argument("--ckpt-dir", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--data-suffix", default=None,
                    help="dataset filename infix: {tech}_{suffix}_{dev}.npz "
                         "— pass 'v2' for the S9b regen-v2 datasets "
                         "(default: none, the v1 {tech}_{dev}.npz)")
    ap.add_argument("--apply-filter", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="match the loader's |id|<=1e-15 row filter "
                         "(default on, for comparability with current "
                         "checkpoints; --no-apply-filter for regen-v2 arms)")
    ap.add_argument("--allow-stats-mismatch", action="store_true",
                    help="downgrade the checkpoint-vs-dataset norm-stats "
                         "mismatch error to a warning (explicit "
                         "cross-population read; sets "
                         "deriv_split_mismatch=1)")
    ap.add_argument("--max-rows", type=int, default=50000,
                    help="deterministic test-split subsample cap")
    ap.add_argument("--selfcheck", action="store_true",
                    help="run the float64 central-FD chain check")
    ap.add_argument("--device", default="cpu", help="torch device")
    ap.add_argument("--json", action="store_true",
                    help="emit one RESULT JSON line (key 'deriv_fidelity')")
    args = ap.parse_args()

    res = compute_deriv_fidelity(
        tech=args.tech, nmos_stem=args.nmos, pmos_stem=args.pmos,
        ckpt_dir=args.ckpt_dir, data_dir=args.data_dir,
        apply_filter=args.apply_filter, max_eval_rows=args.max_rows,
        selfcheck=args.selfcheck, torch_device=args.device,
        data_suffix=args.data_suffix,
        allow_stats_mismatch=args.allow_stats_mismatch)

    _print_human(res)
    if args.json:
        print("RESULT " + json.dumps({"deriv_fidelity": res}))


if __name__ == "__main__":
    main()
